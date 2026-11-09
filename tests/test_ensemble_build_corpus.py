import pytest

from ensemble import build_corpus as bc
from ensemble.records import MODALITIES, PointerRecord, load_records, validate_record

EIXOS = {
    "texto": "fetch_budget_articles",
    "documento": "fetch_budget_pages",
    "imagem": "fetch_gallery",
    "audio": "fetch_clips",
    "video": "fetch_sessions",
}


def _duble(modality, n):
    return [
        PointerRecord(
            id=f"{modality}-{i}",
            modality=modality,
            descriptor=f"descritor de orcamento {modality} {i}",
            pointer_uri=f"https://exemplo.gov.br/{modality}/{i}",
            source="duble",
        )
        for i in range(n)
    ]


def _patch(monkeypatch, vazio=None):
    chamadas = {}
    for modality, fn in EIXOS.items():
        def fake(*args, _m=modality, **kwargs):
            chamadas[_m] = kwargs.get("n")
            return [] if _m == vazio else _duble(_m, kwargs["n"])
        monkeypatch.setattr(bc, fn, fake)
    return chamadas


def test_build_corpus_concatena_e_valida(tmp_path, monkeypatch):
    chamadas = _patch(monkeypatch)
    destino = tmp_path / "sub" / "records.jsonl"
    out = bc.build_corpus(destino, por_eixo=3)
    assert out == destino
    recs = load_records(out)
    assert len(recs) == 15
    assert {r.modality for r in recs} == set(MODALITIES)
    for r in recs:
        validate_record(r)
    assert set(chamadas.values()) == {3}


@pytest.mark.parametrize("eixo", list(EIXOS))
def test_build_corpus_eixo_vazio_nomeia_eixo(tmp_path, monkeypatch, eixo):
    _patch(monkeypatch, vazio=eixo)
    with pytest.raises(RuntimeError, match=f"eixo '{eixo}' retornou 0 itens"):
        bc.build_corpus(tmp_path / "r.jsonl", por_eixo=2)
