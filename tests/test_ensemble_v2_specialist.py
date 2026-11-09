from __future__ import annotations

import numpy as np

from ensemble.native.specialist import DenseSpecialist, Specialist
from ensemble.records import PointerRecord

VOCAB = ["orçamento", "saúde", "educação", "estrada", "praia", "da"]


def _bow(texts: list[str]) -> np.ndarray:
    return np.array(
        [[float(w in t.lower().split()) for w in VOCAB] for t in texts], dtype=np.float32
    )


def _rec(
    id_: str,
    modality: str,
    descriptor: str,
    path: str | None = None,
    a_repr: str | None = None,
) -> PointerRecord:
    meta: dict = {}
    if path:
        meta["path"] = path
    if a_repr:
        meta["a_repr"] = a_repr
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


def test_dense_video_usa_keyframe_por_janela():
    # Duas janelas do MESMO video (mesmo meta["path"]) com keyframes distintos em a_repr.
    # O especialista de video deve embutir o keyframe de cada janela, nao o path comum,
    # senao as duas janelas colapsam na mesma representacao.
    calls = []

    def spy(texts):
        calls.append(list(texts))
        return _bow(texts)

    recs = [
        _rec("v1", "video", "x", path="video.mp4", a_repr="orçamento saúde"),
        _rec("v2", "video", "x", path="video.mp4", a_repr="estrada praia"),
    ]
    sp = DenseSpecialist("video", recs, embed_items=spy, embed_query=_bow)
    # embed_items recebeu os dois keyframes distintos, nao "video.mp4" duas vezes.
    assert calls == [["orçamento saúde", "estrada praia"]]
    res = sp.search("orçamento saúde", 2)
    assert res[0][0] == "v1"
    scores = {i: s for i, s in res}
    assert scores["v1"] != scores["v2"]  # nao colapsam


def test_dense_video_sem_a_repr_cai_para_path_e_avisa(caplog):
    import logging

    rec = _rec("v1", "video", "x", path="orçamento saúde")  # sem a_repr
    with caplog.at_level(logging.WARNING):
        sp = DenseSpecialist("video", [rec], embed_items=_bow, embed_query=_bow)
    res = sp.search("orçamento saúde", 1)
    assert res[0][0] == "v1"
    assert any("v1" in r.message for r in caplog.records)


def test_dense_embed_items_contagem_de_linhas_diverge_falha():
    # embed_items que devolve menos linhas que ids -> ValueError claro.
    def bad(texts):
        return _bow(texts)[:1]

    recs = [_rec("t1", "texto", "orçamento"), _rec("t2", "texto", "saúde")]
    try:
        DenseSpecialist("texto", recs, embed_items=bad, embed_query=_bow)
    except ValueError as e:
        assert "1" in str(e) and "2" in str(e)
    else:
        raise AssertionError("esperava ValueError por divergencia de contagem")
