"""Consulta fim a fim da fatia GCP: texto -> embedding -> top-k paginas no pgvector.

Fecha o circuito aberto por `collector.py` (Bronze) e `embed_job.py` (indice): a consulta
pt-BR e embutida no **mesmo espaco** das paginas por
`benchmark.models.qwen3vl_embed.embed_texts` e o ranking sai por **cosseno**.

Como os dois lados saem L2-normalizados, cosseno = produto interno; no Cloud SQL quem
ordena e o operador `<=>` do pgvector (`PgVectorStore.search`), e na logica testavel
(`rank_top_k` / `InMemoryVectorStore`) e um produto interno em memoria -- mesmo ranking.

Uso:
    export PGPASSWORD="$(gcloud secrets versions access latest --secret=govexe-pg-password)"
    python -m gcp.slice.query --text "metas para a saude publica" --k 5
    python -m gcp.slice.query --text "educacao" --vectors data/embeddings/planos_qwen3vl  # dev
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# permite tanto `python gcp/slice/query.py` quanto `python -m gcp.slice.query`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.models.qwen3vl_embed import DEFAULT_MODEL_ID, embed_texts  # noqa: E402
from gcp.slice.collector import DEFAULT_SOURCE  # noqa: E402
from gcp.slice.embed_job import (  # noqa: E402
    TOP_K,
    Hit,
    InMemoryVectorStore,
    VectorStore,
    add_pg_arguments,
    build_rows,
    load_vectors,
    open_vector_store,
    rank_top_k,
)

log = logging.getLogger(__name__)

# `rank_top_k` (produto interno em memoria) mora no modulo do indice, ao lado dos dois
# backends; reexportado aqui porque e a logica de ranking que esta CLI exercita.
__all__ = ["main", "rank_top_k", "search", "store_from_vectors"]


def store_from_vectors(prefix: str, source: str = DEFAULT_SOURCE) -> InMemoryVectorStore:
    """Modo dev: carrega `<prefixo>.npy` + manifest num indice em memoria (sem Cloud SQL)."""
    manifest, vecs = load_vectors(prefix)
    rows = build_rows(manifest["items"], vecs,
                      model_id=manifest.get("model_id", DEFAULT_MODEL_ID), source=source)
    return InMemoryVectorStore(rows)


def search(text: str, store: VectorStore, *, k: int = TOP_K, model_id: str = DEFAULT_MODEL_ID,
           device: str = "cuda", dtype: str = "bfloat16",
           dim: int | None = None) -> list[Hit]:
    """Embute a consulta e devolve as `k` paginas mais proximas."""
    q = embed_texts([text], model_id=model_id, device=device, dtype=dtype, dim=dim)[0]
    return store.search(q, k)


def main(argv: list[str] | None = None, *, store: VectorStore | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="consulta em texto (pt-BR)")
    ap.add_argument("--k", type=int, default=TOP_K, help=f"quantas paginas (default: {TOP_K})")
    ap.add_argument("--vectors", default=None,
                    help="modo dev: prefixo .npy/.manifest.json em vez do Cloud SQL")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help=f"checkpoint do embedder (default: {DEFAULT_MODEL_ID})")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--dim", type=int, default=None,
                    help="trunca o vetor da consulta para N dimensoes (MRL)")
    add_pg_arguments(ap)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if store is None:
        store = store_from_vectors(a.vectors) if a.vectors else open_vector_store(a)

    hits = search(a.text, store, k=a.k, model_id=a.model_id, device=a.device,
                  dtype=a.dtype, dim=a.dim)
    if not hits:
        log.warning("nenhuma pagina indexada para a consulta %r", a.text)
        return 1
    print(f"consulta: {a.text!r}  (top-{a.k})")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank:2d}. {hit.score:.4f}  {hit.page_id}  {hit.uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
