"""Runner de demonstracao cross-modal: ranking intercalado e anotado por modalidade."""

import numpy as np

from ensemble.demo import format_cli, run_demo
from ensemble.index import build_index
from ensemble.records import PointerRecord

VOCAB = ["orçamento", "saúde", "foto", "pônei", "áudio", "reunião", "clipe"]


def fake_embedder(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            if w in VOCAB:
                out[i, VOCAB.index(w)] += 1.0
    return out


def _rec(id_: str, modality: str, descriptor: str) -> PointerRecord:
    return PointerRecord(
        id=id_,
        modality=modality,
        descriptor=descriptor,
        pointer_uri=f"gs://b/{id_}",
        source="teste",
        meta={},
    )


def _corpus(with_video: bool = True) -> list[PointerRecord]:
    recs = [
        _rec("d1", "documento", "Orçamento saúde"),
        _rec("i1", "imagem", "foto pônei"),
        _rec("a1", "audio", "áudio reunião orçamento"),
        _rec("t1", "texto", "orçamento saúde reunião"),
    ]
    if with_video:
        recs.append(_rec("v1", "video", "clipe orçamento"))
    return recs


def test_demo_interleava_modalidades():
    index = build_index(_corpus(), embedder=fake_embedder)
    (res,) = run_demo(index, ["orçamento saúde"], embedder=fake_embedder, k=5)
    assert len({r.modality for r, _ in res.hits}) > 1
    out = format_cli(res)
    for r, _ in res.hits:
        assert f"[{r.modality.upper()}]" in out


def test_demo_reporta_modalidade_ausente():
    index = build_index(_corpus(with_video=False), embedder=fake_embedder)
    (res,) = run_demo(index, ["orçamento"], embedder=fake_embedder, k=5)
    assert res.modalidades_ausentes == {"video"}
    assert "video" not in res.modalidades_presentes


def test_ponteiro_indisponivel_nao_estoura():
    index = build_index(_corpus(), embedder=fake_embedder)
    (res,) = run_demo(
        index, ["orçamento"], embedder=fake_embedder, resolver=lambda u: False
    )
    assert "ponteiro indisponível" in format_cli(res)


def test_format_cli_preserva_acentos():
    index = build_index(_corpus(), embedder=fake_embedder)
    (res,) = run_demo(index, ["orçamento"], embedder=fake_embedder)
    assert "Orçamento" in format_cli(res)
