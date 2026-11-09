from __future__ import annotations

import numpy as np

from ensemble.native.unified_a import UnifiedRetrieverA, a_repr
from ensemble.records import PointerRecord

VOCAB = ["orçamento", "saúde", "educação", "estrada", "praia"]


def _bow(texts: list[str]) -> np.ndarray:
    return np.array(
        [[float(w in t.lower().split()) for w in VOCAB] for t in texts], dtype=np.float32
    )


def _rec(id_: str, modality: str, a: str | None, descriptor: str = "x") -> PointerRecord:
    meta = {"a_repr": a} if a is not None else {}
    return PointerRecord(
        id=id_, modality=modality, descriptor=descriptor,
        pointer_uri=f"file://{id_}", source="t", meta=meta,
    )


def _recs() -> list[PointerRecord]:
    return [
        _rec("doc1", "documento", "orçamento saúde"),
        _rec("aud1", "audio", "orçamento da saúde"),  # ASR->texto
        _rec("vid1", "video", "estrada praia"),
        _rec("img1", "imagem", "educação"),
    ]


def test_a_espaco_unico_ranqueia_cross_modal():
    r = UnifiedRetrieverA(_recs(), embed_items=_bow, embed_query=_bow)
    res = r.search("orçamento saúde", 2)
    assert {i for i, _ in res} == {"doc1", "aud1"}
    mods = {x.id: x.modality for x in _recs()}
    assert len({mods[i] for i, _ in res}) == 2  # top-k mistura modalidades


def test_a_ranking_global_nao_por_eixo():
    r = UnifiedRetrieverA(_recs(), embed_items=_bow, embed_query=_bow)
    res = r.search("orçamento saúde", 10)
    assert len(res) == 4
    assert {i for i, _ in res} == {"doc1", "aud1", "vid1", "img1"}
    assert [s for _, s in res] == sorted((s for _, s in res), reverse=True)


def test_a_embed_items_uma_vez():
    calls = []

    def counting(texts):
        calls.append(list(texts))
        return _bow(texts)

    r = UnifiedRetrieverA(_recs(), embed_items=counting, embed_query=_bow)
    r.search("saúde", 3)
    r.search("praia", 3)
    assert len(calls) == 1
    assert len(calls[0]) == 4


def test_a_repr_fallback_path_depois_descriptor():
    both = PointerRecord("a", "imagem", "desc", "file://a", "t", {"path": "p.png"})
    none = PointerRecord("b", "texto", "desc", "file://b", "t", {})
    assert a_repr(_rec("c", "audio", "asr")) == "asr"
    assert a_repr(both) == "p.png"
    assert a_repr(none) == "desc"
