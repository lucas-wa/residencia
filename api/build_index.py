"""Constroi OFFLINE o indice servido por `api/app.py` -- precisa de GPU (staged).

Embedda os itens do corpus v4 pelo lado de ITENS do baseline A (`build_unified_a_embedder`,
que ja roteia cada item pela modalidade: `documento`/`imagem`/`video` por `embed_images`,
`texto`/`audio` por `embed_texts`) e escreve o diretorio de indice que a API consome:

    <out>/records.jsonl   copia fiel dos records de entrada (a ORDEM define as linhas)
    <out>/matrix.npy      [n, dim] float32 L2-normalizado

Uso (H100/4090, com o adaptador da Semana 10):

    python -m api.build_index \
        --records data/govexe/ensemble_orcamento_v4/records.jsonl \
        --out data/govexe/indice_api_v4 \
        --adapter finetune/out/qwen3vl-lora

Depois o diretorio sobe para o GCS (infra [3]) e vira o `INDEX_PATH` do Cloud Run.

Este modulo NAO tem teste de unidade por decisao de escopo: ele e fiacao fina entre
`load_records`, o embedder real (que so existe com GPU) e `np.save`. A peca testavel --
casamento records/matriz -- e validada na LEITURA, em `api.index_io.load_index`.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from api.index_io import MATRIX_NAME, RECORDS_NAME
from ensemble.native.specialist import _l2_normalize
from ensemble.native.unified_a import a_repr, a_route
from ensemble.records import load_records

log = logging.getLogger(__name__)

DEFAULT_RECORDS = "data/govexe/ensemble_orcamento_v4/records.jsonl"
DEFAULT_OUT = "data/govexe/indice_api_v4"


def build_index(records_path: str | Path, out_dir: str | Path, *,
                adapter_path: str | None = None, embed_items=None) -> Path:
    """Embedda os itens e escreve `<out_dir>/{records.jsonl,matrix.npy}`.

    `embed_items` injeta o embedder (duble em dev); sem ele, carrega o Qwen3-VL real via
    `build_unified_a_embedder(adapter_path)` -- import tardio, so quem roda de fato paga.
    """
    records = load_records(records_path)
    if not records:
        raise ValueError(f"nenhum record em {records_path}: nao ha indice a construir")

    if embed_items is None:
        from ensemble.native.embedders import build_unified_a_embedder
        embed_items, _ = build_unified_a_embedder(adapter_path=adapter_path)

    pares = [(a_route(r), a_repr(r)) for r in records]
    log.info("embeddando %d itens (%d pela rota de imagem)", len(pares),
             sum(1 for rota, _ in pares if rota == "image"))
    matrix = _l2_normalize(embed_items(pares))
    if matrix.shape[0] != len(records):
        raise ValueError(
            f"{len(records)} records para {matrix.shape[0]} embeddings "
            "(a linha i tem de ser o embedding do record i)")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / MATRIX_NAME, matrix)
    shutil.copyfile(records_path, out / RECORDS_NAME)
    log.info("indice: %d itens, dim=%d -> %s", matrix.shape[0], matrix.shape[1], out)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Embedda o corpus v4 pelo baseline A e salva o indice servido pela API.")
    ap.add_argument("--records", default=DEFAULT_RECORDS,
                    help="records.jsonl do corpus v4")
    ap.add_argument("--out", default=DEFAULT_OUT, help="diretorio do indice de saida")
    ap.add_argument("--adapter", default=None,
                    help="diretorio do adaptador LoRA (omitido = checkpoint publico)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out = build_index(a.records, a.out, adapter_path=a.adapter)
    print(f"indice em {out}")


if __name__ == "__main__":
    main()
