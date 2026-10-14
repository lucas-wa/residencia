"""Resolucao das classes do colpali-engine por `kind` (sem instalar o pacote)."""

import sys
import types

import pytest

from benchmark.models.colvision import resolve_classes


def _install_fake_colpali(monkeypatch, names: dict[str, object]):
    """Injeta um `colpali_engine.models` falso com exatamente `names` exportado."""
    pkg = types.ModuleType("colpali_engine")
    models = types.ModuleType("colpali_engine.models")
    for k, v in names.items():
        setattr(models, k, v)
    pkg.models = models
    monkeypatch.setitem(sys.modules, "colpali_engine", pkg)
    monkeypatch.setitem(sys.modules, "colpali_engine.models", models)


class _ColPali: ...
class _ColPaliProcessor: ...
class _ColQwen2: ...
class _ColQwen2Processor: ...
class _ColQwen25: ...
class _ColQwen25Processor: ...


_LEGACY = {
    "ColPali": _ColPali, "ColPaliProcessor": _ColPaliProcessor,
    "ColQwen2": _ColQwen2, "ColQwen2Processor": _ColQwen2Processor,
}
_WITH_25 = dict(_LEGACY, ColQwen2_5=_ColQwen25, ColQwen2_5Processor=_ColQwen25Processor)


def test_colpali(monkeypatch):
    _install_fake_colpali(monkeypatch, _WITH_25)
    assert resolve_classes("colpali") == (_ColPali, _ColPaliProcessor)


def test_colqwen2(monkeypatch):
    _install_fake_colpali(monkeypatch, _WITH_25)
    assert resolve_classes("colqwen2") == (_ColQwen2, _ColQwen2Processor)


def test_colqwen25_usa_classe_dedicada_quando_existe(monkeypatch):
    _install_fake_colpali(monkeypatch, _WITH_25)
    assert resolve_classes("colqwen2.5") == (_ColQwen25, _ColQwen25Processor)


def test_colqwen25_cai_para_colqwen2_em_release_antiga(monkeypatch):
    _install_fake_colpali(monkeypatch, _LEGACY)
    assert resolve_classes("colqwen2.5") == (_ColQwen2, _ColQwen2Processor)


def test_kind_desconhecido(monkeypatch):
    _install_fake_colpali(monkeypatch, _WITH_25)
    with pytest.raises(ValueError):
        resolve_classes("qwen3vl")
