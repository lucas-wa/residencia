from __future__ import annotations

import numpy as np

from ensemble.native.unified_a import (
    A_IMAGE_MODALITIES,
    UnifiedRetrieverA,
    a_repr,
    a_route,
)
from ensemble.records import PointerRecord

VOCAB = ["orçamento", "saúde", "educação", "estrada", "praia"]


def _bow(texts: list[str]) -> np.ndarray:
    return np.array(
        [[float(w in t.lower().split()) for w in VOCAB] for t in texts], dtype=np.float32
    )


def _bow_itens(pares: list[tuple[str, str]]) -> np.ndarray:
    """`embed_items` do A: recebe `(rota, valor)`; o duble embute o valor em qualquer rota."""
    return _bow([valor for _, valor in pares])


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
    r = UnifiedRetrieverA(_recs(), embed_items=_bow_itens, embed_query=_bow)
    res = r.search("orçamento saúde", 2)
    assert {i for i, _ in res} == {"doc1", "aud1"}
    mods = {x.id: x.modality for x in _recs()}
    assert len({mods[i] for i, _ in res}) == 2  # top-k mistura modalidades


def test_a_ranking_global_nao_por_eixo():
    r = UnifiedRetrieverA(_recs(), embed_items=_bow_itens, embed_query=_bow)
    res = r.search("orçamento saúde", 10)
    assert len(res) == 4
    assert {i for i, _ in res} == {"doc1", "aud1", "vid1", "img1"}
    assert [s for _, s in res] == sorted((s for _, s in res), reverse=True)


def test_a_embed_items_uma_vez():
    calls = []

    def counting(pares):
        calls.append(list(pares))
        return _bow_itens(pares)

    r = UnifiedRetrieverA(_recs(), embed_items=counting, embed_query=_bow)
    r.search("saúde", 3)
    r.search("praia", 3)
    assert len(calls) == 1
    assert len(calls[0]) == 4


def test_a_route_imagem_documento_video_sao_imagem_texto_e_audio_sao_texto():
    """O A e multimodal: so quem tem `a_repr` de CAMINHO vai pela rota de imagem."""
    assert a_route(_rec("d1", "documento", "pages/d1_p1.png")) == "image"
    assert a_route(_rec("i1", "imagem", "fotos/i1.jpg")) == "image"
    assert a_route(_rec("v1", "video", "keyframes/v1.jpg")) == "image"
    assert a_route(_rec("t1", "texto", "artigo sobre orçamento")) == "text"
    assert a_route(_rec("a1", "audio", "transcrição do clipe")) == "text"
    assert A_IMAGE_MODALITIES == frozenset({"documento", "imagem", "video"})


def test_a_passa_tuplas_rota_valor_ao_embed_items():
    """`embed_items` recebe `(rota, a_repr)`, nao a string crua -- e na ordem dos records."""
    recebidos: list[list[tuple[str, str]]] = []

    def espiao(pares):
        recebidos.append(list(pares))
        return _bow_itens(pares)

    r = UnifiedRetrieverA(_recs(), embed_items=espiao, embed_query=_bow)

    assert recebidos == [[
        ("image", "orçamento saúde"),     # documento -> caminho/pagina
        ("text", "orçamento da saúde"),   # audio -> transcricao ASR
        ("image", "estrada praia"),       # video -> keyframe
        ("image", "educação"),            # imagem -> foto
    ]]
    # a matriz sai na ordem dos records: cada linha ranqueia o id do seu record
    assert r.search("educação", 1)[0][0] == "img1"   # 4a linha
    assert r.search("praia", 1)[0][0] == "vid1"      # 3a linha


def test_a_repr_fallback_path_depois_descriptor():
    both = PointerRecord("a", "imagem", "desc", "file://a", "t", {"path": "p.png"})
    none = PointerRecord("b", "texto", "desc", "file://b", "t", {})
    assert a_repr(_rec("c", "audio", "asr")) == "asr"
    assert a_repr(both) == "p.png"
    assert a_repr(none) == "desc"
