"""Carga do indice servido pela API: `records.jsonl` + `matrix.npy`.

O indice e um DIRETORIO com exatamente dois arquivos, escritos por `api/build_index.py`
(offline, com GPU) e publicados pela infra [3]:

    <indice>/records.jsonl   PointerRecord por linha (ensemble.records)
    <indice>/matrix.npy      [n, dim] float32 L2-normalizado, na ordem dos records

`path` pode ser local ou `gs://bucket/prefixo` -- no segundo caso os dois arquivos sao
baixados para um diretorio temporario antes da leitura. O `google-cloud-storage` e
importado DENTRO da funcao: o caminho local (testes, dev) nao depende dele.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from ensemble.records import PointerRecord, load_records

log = logging.getLogger(__name__)

RECORDS_NAME = "records.jsonl"
MATRIX_NAME = "matrix.npy"
GS_PREFIX = "gs://"


def load_index(path: str) -> tuple[list[PointerRecord], np.ndarray]:
    """Le o indice de `path` (local ou `gs://`) e valida o casamento records/matriz.

    Levanta `FileNotFoundError` nomeando o arquivo que falta e `ValueError` quando a
    matriz nao e `[n, dim]` ou tem um numero de linhas diferente do de records (indice
    desalinhado serve o item errado em silencio -- tem de ser erro na carga).
    """
    local = _baixar_do_gcs(path) if path.startswith(GS_PREFIX) else Path(path)

    records_path, matrix_path = local / RECORDS_NAME, local / MATRIX_NAME
    for p in (records_path, matrix_path):
        if not p.is_file():
            raise FileNotFoundError(f"indice sem {p.name}: {p}")

    records = load_records(records_path)
    matrix = np.asarray(np.load(matrix_path), dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(
            f"{MATRIX_NAME} precisa ser [n, dim]; recebi shape {matrix.shape} em {matrix_path}")
    if matrix.shape[0] != len(records):
        raise ValueError(
            f"indice desalinhado em {local}: {len(records)} records para "
            f"{matrix.shape[0]} linhas de {MATRIX_NAME}")

    log.info("indice carregado de %s: %d itens, dim=%d", path, matrix.shape[0], matrix.shape[1])
    return records, matrix


def _baixar_do_gcs(uri: str) -> Path:
    """Baixa `<uri>/records.jsonl` e `<uri>/matrix.npy` para um diretorio temporario.

    Import guardado: `google-cloud-storage` so e exigido de quem aponta `INDEX_PATH` para
    um bucket (Cloud Run); rodar com indice local nao instala nada disso.
    """
    try:
        from google.cloud import storage  # type: ignore[attr-defined]
    except ImportError as e:  # pragma: no cover - caminho de deploy, nao dos testes
        raise RuntimeError(
            f"indice em {uri} exige o pacote google-cloud-storage "
            "(pip install google-cloud-storage) ou um caminho local em INDEX_PATH") from e

    bucket_name, _, prefixo = uri[len(GS_PREFIX):].partition("/")
    if not bucket_name:
        raise ValueError(f"URI gs:// sem bucket: {uri!r}")

    destino = Path(tempfile.mkdtemp(prefix="indice-consulta-"))
    bucket = storage.Client().bucket(bucket_name)
    for nome in (RECORDS_NAME, MATRIX_NAME):
        blob_name = f"{prefixo.rstrip('/')}/{nome}" if prefixo else nome
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"indice sem {nome}: gs://{bucket_name}/{blob_name}")
        log.info("baixando gs://%s/%s", bucket_name, blob_name)
        blob.download_to_filename(str(destino / nome))
    return destino
