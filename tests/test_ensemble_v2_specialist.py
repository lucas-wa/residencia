from __future__ import annotations

import numpy as np

from ensemble.native.specialist import DenseSpecialist, Specialist
from ensemble.records import PointerRecord

VOCAB = ["orçamento", "saúde", "educação", "estrada", "praia", "da"]


def _bow(texts: list[str]) -> np.ndarray:
    return np.array(
        [[float(w in t.lower().split()) for w in VOCAB] for t in texts], dtype=np.float32
    )


def _rec(id_: str, modality: str, descriptor: str, path: str | None = None) -> PointerRecord:
    meta = {"path": path} if path else {}
    return PointerRecord(
        id=id_, modality=modality, descriptor=descriptor,
        pointer_uri=f"file://{id_}", source="t", meta=meta,
    )


def test_dense_ranqueia_por_cosine():
    recs = [
        _rec("t1", "texto", "orçamento da saúde"),
        _rec("t2", "texto", "estrada praia"),
        _rec("t3", "texto", "educação"),
    ]
    sp = DenseSpecialist("texto", recs, embed_items=_bow, embed_query=_bow)
    res = sp.search("orçamento saúde", 3)
    assert res[0][0] == "t1"
    assert [s for _, s in res] == sorted((s for _, s in res), reverse=True)
    assert isinstance(sp, Specialist)
    assert sp.modality == "texto"


def test_dense_so_indexa_sua_modalidade():
    # imagem usa meta["path"] como representacao do item
    recs = [
        _rec("t1", "texto", "orçamento"),
        _rec("i1", "imagem", "x", path="orçamento saúde"),
        _rec("i2", "imagem", "x", path="estrada"),
        _rec("a1", "audio", "x", path="saúde"),
    ]
    sp = DenseSpecialist("imagem", recs, embed_items=_bow, embed_query=_bow)
    res = sp.search("orçamento saúde", 10)
    assert {i for i, _ in res} == {"i1", "i2"}
    assert res[0][0] == "i1"


def test_dense_modalidade_vazia_retorna_lista_vazia():
    recs = [_rec("t1", "texto", "orçamento")]
    sp = DenseSpecialist("video", recs, embed_items=_bow, embed_query=_bow)
    assert sp.search("orçamento", 5) == []


def test_dense_embed_items_chamado_uma_vez():
    calls = []

    def spy(texts):
        calls.append(list(texts))
        return _bow(texts)

    recs = [_rec("t1", "texto", "orçamento"), _rec("t2", "texto", "saúde")]
    sp = DenseSpecialist("texto", recs, embed_items=spy, embed_query=_bow)
    sp.search("orçamento", 2)
    sp.search("saúde", 2)
    assert len(calls) == 1
    assert calls[0] == ["orçamento", "saúde"]
