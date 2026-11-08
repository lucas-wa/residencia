"""Consulta fim a fim da fatia GCP: texto -> embedding -> top-k itens no pgvector.

Fecha o circuito aberto por `collector.py` (Bronze) e `embed_job.py` (indice): a consulta
pt-BR e embutida no **mesmo espaco** dos itens indexados por
`benchmark.models.qwen3vl_embed.embed_texts` e o ranking sai por **cosseno**.

A busca e **cross-modal**: como as quatro modalidades (`doc_image`, `image`, `text`,
`audio_asr`) foram embutidas pelo MESMO checkpoint no MESMO espaco, uma consulta textual
recupera pagina de documento, foto, texto nativo e audio transcrito do MESMO indice.
`--modality` restringe o ranking a uma delas (`WHERE modality = ...`, apoiado pelo btree
criado junto da tabela).

Como os dois lados saem L2-normalizados, cosseno = produto interno; no Cloud SQL quem
ordena e o operador `<=>` do pgvector (`PgVectorStore.search`), e na logica testavel
(`rank_top_k` / `InMemoryVectorStore`) e um produto interno em memoria -- mesmo ranking.

Acima disso ha a consulta MULTI-ESPACO (`search_architecture`): a arquitetura `A+audio`
soma ao Qwen3-VL o indice do CLAP (audio nativo), e os dois rankings se fundem por RRF
(`gcp/slice/fusion.py`) -- posicao, nunca score, porque a similaridade de espacos
diferentes nao se compara. Por default ela funde TODOS os espacos da arquitetura (a busca
cross-modal); `only_spaces` restringe a fusao a um subconjunto, que e como a avaliacao
mede UMA tarefa por vez (`gcp/slice/eval.py`). A CLI abaixo continua sendo a de espaco
unico.

Uso:
    export PGPASSWORD="$(gcloud secrets versions access latest --secret=govexe-pg-password)"
    python -m gcp.slice.query --text "metas para a saude publica" --k 5
    python -m gcp.slice.query --text "sessao sobre saneamento" --modality audio_asr
    python -m gcp.slice.query --text "educacao" --vectors data/embeddings/planos_qwen3vl  # dev
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

# permite tanto `python gcp/slice/query.py` quanto `python -m gcp.slice.query`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.models.qwen3vl_embed import DEFAULT_MODEL_ID, embed_texts  # noqa: E402
from gcp.slice.collector import DEFAULT_SOURCE  # noqa: E402
from gcp.slice.embed_job import (  # noqa: E402
    DEFAULT_DIM,
    MODALITIES,
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
from gcp.slice.fusion import rrf_fuse  # noqa: E402
from gcp.slice.spaces import architecture_spaces  # noqa: E402

log = logging.getLogger(__name__)

# `rank_top_k` (produto interno em memoria) mora no modulo do indice, ao lado dos dois
# backends; reexportado aqui porque e a logica de ranking que esta CLI exercita.
__all__ = ["MULTI_TOP_K", "main", "rank_top_k", "search", "search_architecture",
           "store_from_vectors"]

# Profundidade pedida a CADA espaco antes da fusao. Maior que o `TOP_K` da CLI de espaco
# unico de proposito: o RRF so pode promover um item que apareca na lista de algum espaco,
# entao um pool raso por espaco joga fora justamente os itens que a fusao existe para
# resgatar (o bem colocado em dois espacos, nunca no topo de nenhum).
MULTI_TOP_K = 10


def store_from_vectors(prefix: str, source: str = DEFAULT_SOURCE) -> InMemoryVectorStore:
    """Modo dev: carrega `<prefixo>.npy` + manifest num indice em memoria (sem Cloud SQL)."""
    manifest, vecs = load_vectors(prefix)
    rows = build_rows(manifest["items"], vecs,
                      model_id=manifest.get("model_id", DEFAULT_MODEL_ID), source=source)
    return InMemoryVectorStore(rows)


def search(text: str, store: VectorStore, *, k: int = TOP_K, model_id: str = DEFAULT_MODEL_ID,
           device: str = "cuda", dtype: str = "bfloat16", dim: int | None = None,
           modality: str | None = None) -> list[Hit]:
    """Embute a consulta e devolve os `k` itens mais proximos (de qualquer modalidade)."""
    q = embed_texts([text], model_id=model_id, device=device, dtype=dtype, dim=dim)[0]
    return store.search(q, k, modality=modality)


def search_architecture(arch: str, query_text: str, *, k: int = MULTI_TOP_K,
                        stores: dict[str, VectorStore],
                        embedders: dict[str, Callable[[list[str]], np.ndarray]],
                        only_spaces: Iterable[str] | None = None,
                        ) -> list[tuple[str, float]]:
    """Consulta textual nos espacos da arquitetura e funde os rankings por RRF.

    Um espaco por vez: a consulta e embutida pelo embedder de texto DAQUELE espaco (e so
    assim o vetor cai na geometria certa -- Qwen3-VL e CLAP nem tem a mesma dimensao) e o
    top-`k` sai do store DAQUELE espaco. O que se funde sao as listas de `item_id`, nunca
    os scores: entre espacos a escala da similaridade nao se compara (ver `fusion.py`).

    Nao filtra por modalidade de proposito: cada espaco ja indexa so as modalidades que
    declara (`Space.modalities`), e e a busca cross-modal que se quer aqui -- em `A+audio`
    o mesmo texto recupera pagina pelo Qwen3-VL e audio NATIVO pelo CLAP.

    `only_spaces` restringe a fusao a um SUBCONJUNTO dos espacos da arquitetura; o default
    (`None`) funde todos, que e a busca cross-modal da CLI. Quem o usa e a AVALIACAO: uma
    tarefa doc-visual nao pode fundir o espaco do CLAP, que so indexa clipes de audio e
    entremearia audio entre as paginas -- faria `A+audio` diferir de `A` numa tarefa em
    que a spec (secao 5) as declara identicas. Nome fora da arquitetura (ou filtro que nao
    sobra espaco nenhum) e `ValueError` na hora: avaliar sobre indice nenhum devolveria um
    zero que parece metrica.

    `stores` e `embedders` sao injetados por `name` de espaco (producao monta a partir de
    `SPACES`; o teste passa fakes), o que mantem esta funcao sem carregar checkpoint nenhum.
    Devolve `[(item_id, score_rrf), ...]` com o melhor primeiro -- o score e o do RRF, nao
    um cosseno, e so faz sentido DENTRO deste ranking.
    """
    espacos = architecture_spaces(arch)           # ValueError na hora se `arch` nao existe
    if only_spaces is not None:
        pedidos = set(only_spaces)
        da_arquitetura = {espaco.name for espaco in espacos}
        if not pedidos <= da_arquitetura:
            raise ValueError(
                f"only_spaces={sorted(pedidos)} pede espaco que nao esta na arquitetura "
                f"{arch!r} (ela tem: {sorted(da_arquitetura)})")
        espacos = [espaco for espaco in espacos if espaco.name in pedidos]
        if not espacos:
            raise ValueError(f"only_spaces vazio para a arquitetura {arch!r}: nao ha "
                             f"espaco nenhum a consultar")

    rankings: list[list[str]] = []
    for espaco in espacos:
        if espaco.name not in stores or espaco.name not in embedders:
            faltando = "store" if espaco.name not in stores else "embedder"
            raise KeyError(f"{faltando} ausente para o espaco {espaco.name!r}, exigido pela "
                           f"arquitetura {arch!r} (tabela {espaco.table})")
        # `atleast_2d` aceita tanto o (1, dim) usual quanto um (dim,) de embedder simples.
        q = np.atleast_2d(np.asarray(embedders[espaco.name]([query_text])))[0]
        hits = stores[espaco.name].search(q, k)
        rankings.append([hit.item_id for hit in hits])
        log.debug("espaco %s: %d hits", espaco.name, len(hits))
    return rrf_fuse(rankings)


def main(argv: list[str] | None = None, *, store: VectorStore | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="consulta em texto (pt-BR)")
    ap.add_argument("--k", type=int, default=TOP_K, help=f"quantos itens (default: {TOP_K})")
    ap.add_argument("--modality", default=None, choices=list(MODALITIES),
                    help="restringe a busca a uma modalidade (default: todas -- cross-modal)")
    ap.add_argument("--vectors", default=None,
                    help="modo dev: prefixo .npy/.manifest.json em vez do Cloud SQL")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help=f"checkpoint do embedder (default: {DEFAULT_MODEL_ID})")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--dim", type=int, default=None,
                    help="trunca o vetor da consulta para N dimensoes (MRL); default: a "
                         f"dimensao do indice ({DEFAULT_DIM} no Cloud SQL, a do .npy em "
                         "--vectors)")
    add_pg_arguments(ap)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if store is None:
        store = store_from_vectors(a.vectors) if a.vectors else open_vector_store(a)

    # a consulta tem de sair na MESMA dimensao do indice: no Cloud SQL quem manda e o
    # `--dim` do embed_job (DEFAULT_DIM, teto do HNSW); em --vectors, a do .npy ja gravado.
    dim = a.dim if a.dim is not None else (None if a.vectors else DEFAULT_DIM)
    hits = search(a.text, store, k=a.k, model_id=a.model_id, device=a.device,
                  dtype=a.dtype, dim=dim, modality=a.modality)
    if not hits:
        log.warning("nenhum item indexado para a consulta %r (modalidade: %s)",
                    a.text, a.modality or "todas")
        return 1
    filtro = f"  modalidade: {a.modality}" if a.modality else ""
    print(f"consulta: {a.text!r}  (top-{a.k}){filtro}")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank:2d}. {hit.score:.4f}  [{hit.modality}]  {hit.item_id}  {hit.uri}")
        # o credito e o pointer do binario original acompanham o item em toda consulta
        # (requisito de licenca das fontes de `fusao/coleta-gcp.md` secao 6).
        if hit.credit:
            print(f"      credito: {hit.credit}")
        if hit.pointer_uri:
            print(f"      pointer: {hit.pointer_uri}")
        if hit.content_text:
            trecho = " ".join(hit.content_text.split())[:120]
            print(f"      texto:   {trecho}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
