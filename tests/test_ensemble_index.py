"""Indice em memoria do ensemble: embeddings L2-normalizados + cosine.

Sem GPU e sem baixar BGE-M3: o embedder e um duble deterministico (bag-of-words),
no mesmo padrao do retriever falso de `test_doc_specialist.py`.
"""

import numpy as np
import pytest

from ensemble.index import build_index
from ensemble.records import PointerRecord

VOCAB = ["orçamento", "da", "saúde", "foto", "de", "pônei", "áudio", "reunião"]


def fake_embedder(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            if w in VOCAB:
                out[i, VOCAB.index(w)] += 1.0
    return out


def _rec(id_: str, descriptor: str) -> PointerRecord:
    return PointerRecord(
        id=id_,
        modality="texto",
        descriptor=descriptor,
        pointer_uri=f"gs://b/{id_}",
        source="t",
    )


def _records() -> list[PointerRecord]:
    return [
        _rec("pony", "foto de pônei"),
        _rec("saude", "orçamento da saúde"),
        _rec("reuniao", "áudio de reunião"),
    ]


def test_search_ordena_por_cosine():
    idx = build_index(_records(), embedder=fake_embedder)
    hits = idx.search("orçamento saúde", embedder=fake_embedder)
    assert hits[0][0].id == "saude"
    ids = [r.id for r, _ in hits]
    assert ids.index("saude") < ids.index("pony")
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_corpus_embutido_uma_vez():
    calls: list[list[str]] = []

    def spy(texts):
        calls.append(list(texts))
        return fake_embedder(texts)

    recs = _records()
    idx = build_index(recs, embedder=spy)
    assert calls == [[r.descriptor for r in recs]]
    idx.search("orçamento", embedder=spy)
    idx.search("foto", embedder=spy)
    # build: 1 chamada com o corpus; cada search: 1 chamada so com a query
    assert len(calls) == 3
    assert calls[1] == ["orçamento"] and calls[2] == ["foto"]


def test_build_index_corpus_vazio_levanta():
    with pytest.raises(ValueError):
        build_index([], embedder=fake_embedder)


def test_search_query_vazia_levanta():
    idx = build_index(_records(), embedder=fake_embedder)
    with pytest.raises(ValueError):
        idx.search("  ", embedder=fake_embedder)


def test_search_k_maior_que_corpus():
    idx = build_index(_records(), embedder=fake_embedder)
    assert len(idx.search("orçamento", k=100, embedder=fake_embedder)) == 3
