"""Entrypoint CLI do ensemble cross-modal: le records + queries, roda a demo, grava o
output anotado por modalidade.

E a ponta executavel da STAGE `ensemble` do `slurm_run.sh`: no cluster o `--records`
aponta para o `records.jsonl` dos descritores-ponteiro (um por binario), o `--queries`
para o arquivo de consultas (uma por linha) e o `--out` para o `.txt` em `logs/`.

O embedder e o MESMO BGE-M3 denso usado no indice textual unico (`fusao/ensemble.md`):
ele e construido por IMPORT TARDIO em `_build_embedder` -- como o
`doc_specialist._build_retriever` faz com o ColVision -- para que importar/testar este
modulo numa maquina sem GPU nao arraste FlagEmbedding/torch nem baixe checkpoint. Nos
testes o `embedder` e injetado por um duble, entao `_build_embedder` nunca roda.

Uso (no container do cluster, via slurm_run.sh):
    python -m ensemble.run_ensemble \
        --records data/govexe/ensemble_orcamento/records.jsonl \
        --queries ensemble/queries_orcamento.txt \
        --out logs/ensemble_demo.txt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from ensemble.index import Embedder, build_index
from ensemble.records import load_records

__all__ = ["main"]

log = logging.getLogger(__name__)

# Modelo denso do indice textual unico (`fusao/ensemble.md`): BGE-M3, dim 1024.
_BGE_M3_MODEL = "BAAI/bge-m3"
_TOP_K = 10


def _load_queries(path: str | Path) -> list[str]:
    """Le as consultas do arquivo, uma por linha; ignora linhas em branco.

    Mantem os acentos (UTF-8) -- o corpus e as consultas sao em pt-BR.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _build_embedder() -> Embedder:
    """Constroi o embedder denso BGE-M3 -- import TARDIO, como `doc_specialist`.

    O `from FlagEmbedding import BGEM3FlagModel` arrasta torch/FlagEmbedding e, na
    primeira chamada, baixa o checkpoint do Hub. Mante-lo AQUI DENTRO e o que permite
    importar (e testar) este modulo numa maquina sem GPU, com o embedder injetado por um
    duble. So roda de verdade no cluster, dentro do `main`, quando nenhum duble foi dado.

    Devolve `encode(list[str]) -> np.ndarray [n, d]` com os vetores densos; a
    normalizacao L2 e o cosine ficam no `ensemble.index` (nao aqui).
    """
    from FlagEmbedding import BGEM3FlagModel

    log.info("carregando embedder denso %s (use_fp16=True)", _BGE_M3_MODEL)
    model = BGEM3FlagModel(_BGE_M3_MODEL, use_fp16=True)

    def encode(texts: list[str]) -> np.ndarray:
        out = model.encode(texts, return_dense=True, return_sparse=False,
                           return_colbert_vecs=False)
        return np.asarray(out["dense_vecs"], dtype=np.float32)

    return encode


def main(argv: list[str] | None = None, *, embedder: Embedder | None = None) -> int:
    """Roda a demo cross-modal e grava o output anotado. Retorna 0 em sucesso, !=0 em erro.

    `embedder` injetado pula o `_build_embedder` (BGE-M3) -- e o que os testes usam para
    nao baixar modelo. No cluster ele vem vazio e o embedder real e construido por import
    tardio.
    """
    from ensemble.demo import format_cli, run_demo

    parser = argparse.ArgumentParser(
        prog="python -m ensemble.run_ensemble",
        description="Demo cross-modal do ensemble: ranking unico anotado por modalidade.",
    )
    parser.add_argument("--records", required=True,
                        help="records.jsonl com os descritores-ponteiro (um por linha)")
    parser.add_argument("--queries", required=True,
                        help="arquivo de consultas, uma por linha")
    parser.add_argument("--out", required=True,
                        help="arquivo .txt de saida (ranking anotado)")
    parser.add_argument("--k", type=int, default=_TOP_K,
                        help=f"top-k por consulta (default: {_TOP_K})")
    args = parser.parse_args(argv)

    records_path = Path(args.records)
    queries_path = Path(args.queries)

    if not records_path.is_file():
        print(f"erro: records nao encontrado: {records_path}", file=sys.stderr)
        return 2
    if not queries_path.is_file():
        print(f"erro: queries nao encontrado: {queries_path}", file=sys.stderr)
        return 2

    records = load_records(records_path)
    if not records:
        print(f"erro: nenhum record em {records_path}", file=sys.stderr)
        return 2

    queries = _load_queries(queries_path)
    if not queries:
        print(f"erro: nenhuma consulta em {queries_path}", file=sys.stderr)
        return 2

    if embedder is None:
        embedder = _build_embedder()

    index = build_index(records, embedder=embedder)
    results = run_demo(index, queries, embedder=embedder, k=args.k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(format_cli(r) for r in results) + "\n")

    print(f"ok: {len(queries)} consulta(s), {len(records)} record(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
