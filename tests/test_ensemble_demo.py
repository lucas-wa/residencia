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


# Embedder enviesado a texto: dim0 = relevância à query "orçamento",
# dim1 = ruído ortogonal. texto/documento alinham forte à query; audio/imagem/video
# alinham fraco — então, num top-k global pequeno, as não-textuais são empurradas para fora.
_VIES_TEXTUAL = {
    "orçamento": (1.0, 0.0),
    "texto A sobre orçamento": (1.0, 0.0),
    "texto B sobre orçamento": (1.0, 0.0),
    "documento A sobre orçamento": (0.98, 0.20),
    "documento B sobre orçamento": (0.97, 0.24),
    "áudio sobre orçamento": (0.30, 0.95),
    "imagem sobre orçamento": (0.25, 0.97),
    "vídeo sobre orçamento": (0.20, 0.98),
}


def text_biased_embedder(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), 2), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = _VIES_TEXTUAL.get(t, (0.0, 1.0))
    return out


def _corpus_vies_textual() -> list[PointerRecord]:
    return [
        _rec("t1", "texto", "texto A sobre orçamento"),
        _rec("t2", "texto", "texto B sobre orçamento"),
        _rec("d1", "documento", "documento A sobre orçamento"),
        _rec("d2", "documento", "documento B sobre orçamento"),
        _rec("a1", "audio", "áudio sobre orçamento"),
        _rec("i1", "imagem", "imagem sobre orçamento"),
        _rec("v1", "video", "vídeo sobre orçamento"),
    ]


def test_ausencia_a_nivel_de_corpus_sob_vies_textual():
    # (a) Mesmo quando o top-k global é dominado por texto/documento, uma modalidade
    # presente no corpus NÃO pode ser reportada como ausente.
    index = build_index(_corpus_vies_textual(), embedder=text_biased_embedder)
    (res,) = run_demo(index, ["orçamento"], embedder=text_biased_embedder, k=2)

    # o top-k global está de fato enviesado (não-textuais foram empurradas para fora)
    assert {r.modality for r, _ in res.hits} <= {"texto", "documento"}
    assert "audio" not in {r.modality for r, _ in res.hits}

    # ausência é a nível de corpus: todas as 5 modalidades existem no corpus
    assert res.modalidades_presentes == {
        "texto",
        "documento",
        "audio",
        "imagem",
        "video",
    }
    assert res.modalidades_ausentes == set()


def test_destaques_por_modalidade_surfaceia_melhor_de_cada():
    # (b) o melhor hit de CADA modalidade presente aparece, mesmo sob viés de score,
    # e format_cli imprime a tag de todas as modalidades presentes.
    index = build_index(_corpus_vies_textual(), embedder=text_biased_embedder)
    (res,) = run_demo(index, ["orçamento"], embedder=text_biased_embedder, k=2)

    destaques_mods = [r.modality for r, _ in res.destaques_por_modalidade]
    assert set(destaques_mods) == {"texto", "documento", "audio", "imagem", "video"}
    # um único destaque por modalidade
    assert len(destaques_mods) == len(set(destaques_mods))
    # ordenado por score decrescente
    scores = [s for _, s in res.destaques_por_modalidade]
    assert scores == sorted(scores, reverse=True)

    out = format_cli(res)
    for m in ("TEXTO", "DOCUMENTO", "AUDIO", "IMAGEM", "VIDEO"):
        assert f"[{m}]" in out


def test_destaques_fora_do_topk_marcam_ponteiro_indisponivel():
    # resolvidos deve cobrir a uniao (top-k U destaques): um ponteiro morto numa
    # modalidade empurrada para fora do top-k ainda precisa da marca de indisponivel.
    index = build_index(_corpus_vies_textual(), embedder=text_biased_embedder)
    (res,) = run_demo(
        index,
        ["orçamento"],
        embedder=text_biased_embedder,
        k=2,
        resolver=lambda u: False,
    )

    fora_do_topk = {"a1", "i1", "v1"}  # audio/imagem/video nao entram no top-k (k=2)
    assert fora_do_topk & {r.id for r, _ in res.hits} == set()
    # mas aparecem nos destaques e devem estar resolvidos como indisponiveis
    destaques_ids = {r.id for r, _ in res.destaques_por_modalidade}
    assert fora_do_topk <= destaques_ids
    for rec, _ in res.destaques_por_modalidade:
        if rec.id in fora_do_topk:
            assert res.resolvidos[rec.pointer_uri] is False

    # cada destaque fora do top-k aparece marcado como indisponivel no output
    out = format_cli(res)
    for linha in out.splitlines():
        if any(f"gs://b/{i}" in linha for i in fora_do_topk):
            assert "[ponteiro indisponível]" in linha
