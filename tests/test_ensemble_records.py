import dataclasses

import pytest

from ensemble.records import PointerRecord, load_records, save_records, validate_record


def _rec(**kw):
    base = dict(
        id="r1",
        modality="texto",
        descriptor="Orçamento da saúde",
        pointer_uri="gs://bucket/r1.txt",
        source="portal",
        meta={"ano": 2024},
    )
    base.update(kw)
    return PointerRecord(**base)


def test_roundtrip_jsonl_preserva_acentos(tmp_path):
    path = tmp_path / "records.jsonl"
    original = _rec()
    save_records([original], path)
    assert "Orçamento da saúde" in path.read_text(encoding="utf-8")
    loaded = load_records(path)
    assert loaded == [original]
    assert loaded[0].descriptor == "Orçamento da saúde"
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded[0].descriptor = "x"


def test_validate_rejeita_modalidade_invalida():
    with pytest.raises(ValueError):
        validate_record(_rec(modality="pdf"))


def test_validate_rejeita_descriptor_vazio():
    with pytest.raises(ValueError):
        validate_record(_rec(descriptor="  "))


def test_validate_rejeita_pointer_vazio():
    with pytest.raises(ValueError):
        validate_record(_rec(pointer_uri=""))
