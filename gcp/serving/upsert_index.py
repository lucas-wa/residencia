"""Upsert incremental e IDEMPOTENTE dos records do indice servido, entre ciclos.

Cada ciclo do fluxo event-driven [3] (`gcp/serving/trigger_cycle.sh`) captura o corpus
numa Spot VM que e apagada ao fim. O que sobrevive ao ciclo e o `records.jsonl` que vive
no GCS: este modulo mescla os records recem-capturados com os que ja estavam la.

    novo ciclo ----> upsert_records(existentes, novos) ----> records.jsonl no GCS

A chave e o `id` do `PointerRecord` (o `item_id` da captura, hash estavel da URL): o
record NOVO sobrescreve o antigo de mesmo id, e a ordem dos existentes e preservada. Dai
a idempotencia -- repetir um ciclo (preempcao do spot, retry do scheduler, cron que
disparou duas vezes) nao duplica item nem embaralha o indice.

DIVISAO DE TRABALHO COM `api/build_index.py` -- este modulo NAO embedda nada
--------------------------------------------------------------------------------------
O indice que a API [4] le (`api/index_io.py`) e um DIRETORIO com dois arquivos:

    <indice>/records.jsonl   PointerRecord por linha
    <indice>/matrix.npy      [n, dim] float32 na ORDEM dos records

Aqui so se mexe no primeiro. A matriz e recomputada por `python -m api.build_index` a
partir dos records ja mesclados -- so ela sabe embeddar pelo lado de itens do baseline A,
e so ela tem GPU. Por isso o ciclo guarda o corpus acumulado num prefixo PROPRIO
(`CORPUS_URI`), separado do indice servido (`INDEX_URI`): o indice servido e escrito
INTEIRO pelo `build_index` (records + matriz juntos, consistentes) e nunca pela metade.
Fosse o upsert direto no prefixo servido, a API acharia `n` records e `m` linhas de
matriz e recusaria a carga ate o proximo rebuild (`api.index_io.load_index` levanta
`ValueError` -- falha fechada, mas ainda assim indisponibilidade).

O caminho `gs://` importa `google-cloud-storage` DENTRO da funcao: um caminho local
(testes, dev, a propria VM antes de subir) nao depende do pacote.
"""
from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from ensemble.records import PointerRecord, load_records, save_records

log = logging.getLogger(__name__)

RECORDS_NAME = "records.jsonl"
GS_PREFIX = "gs://"


def upsert_records(existing: list[PointerRecord],
                   novos: list[PointerRecord]) -> list[PointerRecord]:
    """Mescla `novos` em `existing` deduplicando por `id`, com o novo vencendo.

    Ordem ESTAVEL: os ids que ja existiam ficam onde estavam (com o valor atualizado) e
    os ineditos entram ao fim, na ordem em que aparecem em `novos`. A estabilidade nao e
    estetica -- a linha `i` de `matrix.npy` e o embedding do record `i`, entao um indice
    reordenado a cada ciclo invalidaria toda matriz ja calculada.

    Id repetido DENTRO de `novos` segue a mesma regra do upsert (o ultimo vence); id
    repetido dentro de `existing` (indice ja corrompido) e colapsado na primeira posicao.
    Funcao pura: nao toca disco nem rede.
    """
    atualizacao = {r.id: r for r in novos}      # dentro de `novos`, o ultimo vence
    saida: list[PointerRecord] = []
    vistos: set[str] = set()

    for r in existing:
        if r.id in vistos:
            continue
        vistos.add(r.id)
        saida.append(atualizacao.get(r.id, r))

    for r in novos:
        if r.id not in vistos:
            vistos.add(r.id)
            saida.append(atualizacao[r.id])

    return saida


def load_index_gcs(uri: str) -> list[PointerRecord]:
    """Le `<uri>/records.jsonl` (local ou `gs://`); indice inexistente devolve `[]`.

    Lista vazia e a resposta certa para o PRIMEIRO ciclo: nao ha nada no bucket ainda e o
    upsert tem de comecar do zero em vez de quebrar a rodada.
    """
    if uri.startswith(GS_PREFIX):
        local = _baixar_do_gcs(uri)
        if local is None:
            log.info("indice ainda nao existe em %s: comecando do zero", uri)
            return []
    else:
        local = Path(uri) / RECORDS_NAME
        if not local.is_file():
            log.info("indice ainda nao existe em %s: comecando do zero", uri)
            return []

    records = load_records(local)
    log.info("indice lido de %s: %d records", uri, len(records))
    return records


def save_index_gcs(uri: str, records: list[PointerRecord]) -> None:
    """Escreve `records` em `<uri>/records.jsonl` (local ou `gs://`), criando o destino."""
    if uri.startswith(GS_PREFIX):
        with tempfile.TemporaryDirectory(prefix="upsert-indice-") as tmp:
            local = Path(tmp) / RECORDS_NAME
            save_records(records, local)
            _subir_para_o_gcs(uri, local)
    else:
        destino = Path(uri)
        destino.mkdir(parents=True, exist_ok=True)
        save_records(records, destino / RECORDS_NAME)
    log.info("indice gravado em %s: %d records", uri, len(records))


def upsert_index(uri: str, novos: list[PointerRecord]) -> list[PointerRecord]:
    """Le o indice de `uri`, mescla `novos` e grava de volta. Devolve o resultado."""
    mesclado = upsert_records(load_index_gcs(uri), novos)
    save_index_gcs(uri, mesclado)
    return mesclado


# -- ramo gs:// (import guardado) ------------------------------------------------------

def _bucket_e_blob(uri: str) -> tuple[str, str]:
    bucket_name, _, prefixo = uri[len(GS_PREFIX):].partition("/")
    if not bucket_name:
        raise ValueError(f"URI gs:// sem bucket: {uri!r}")
    prefixo = prefixo.strip("/")
    return bucket_name, f"{prefixo}/{RECORDS_NAME}" if prefixo else RECORDS_NAME


def _cliente_gcs():
    try:
        from google.cloud import storage  # type: ignore[attr-defined]
    except ImportError as e:  # pragma: no cover - caminho de deploy, nao dos testes
        raise RuntimeError(
            "indice em gs:// exige o pacote google-cloud-storage "
            "(pip install google-cloud-storage) ou um caminho local") from e
    return storage.Client()


def _baixar_do_gcs(uri: str) -> Path | None:  # pragma: no cover - exige bucket real
    """Baixa `<uri>/records.jsonl` para um temporario; `None` se o blob nao existe."""
    bucket_name, blob_name = _bucket_e_blob(uri)
    blob = _cliente_gcs().bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        return None
    destino = Path(tempfile.mkdtemp(prefix="upsert-indice-")) / RECORDS_NAME
    log.info("baixando gs://%s/%s", bucket_name, blob_name)
    blob.download_to_filename(str(destino))
    return destino


def _subir_para_o_gcs(uri: str, local: Path) -> None:  # pragma: no cover - idem
    bucket_name, blob_name = _bucket_e_blob(uri)
    log.info("subindo %s -> gs://%s/%s", local, bucket_name, blob_name)
    _cliente_gcs().bucket(bucket_name).blob(blob_name).upload_from_filename(str(local))


def main():
    ap = argparse.ArgumentParser(
        description="Mescla os records de um ciclo no indice persistido (local ou gs://).")
    ap.add_argument("--index", required=True,
                    help="prefixo do indice: gs://bucket/prefixo ou um diretorio local")
    ap.add_argument("--novos", required=True,
                    help="records.jsonl capturado neste ciclo")
    ap.add_argument("--out-local", default=None,
                    help="opcional: grava tambem o resultado mesclado neste diretorio "
                         "(o build_index da matriz le dali, sem baixar de novo)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    novos = load_records(a.novos)
    antes = load_index_gcs(a.index)
    depois = upsert_records(antes, novos)
    save_index_gcs(a.index, depois)
    if a.out_local:
        save_index_gcs(a.out_local, depois)

    ids_antes = {r.id for r in antes}
    ids_novos = {r.id for r in novos}
    print(f"indice {a.index}: {len(antes)} -> {len(depois)} records "
          f"({len(novos)} do ciclo, {len(ids_novos - ids_antes)} ineditos, "
          f"{len(ids_novos & ids_antes)} atualizados)")


if __name__ == "__main__":
    main()
