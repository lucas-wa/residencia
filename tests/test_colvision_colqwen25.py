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
# Nome do processor CONFERIDO no wheel do colpali-engine (0.3.9+) e no card do
# `vidore/colqwen2.5-v0.2`: `ColQwen2_5_Processor`, com underscore antes de "Processor"
# -- assimetrico em relacao a `ColQwen2Processor` da linha 2. Escrever
# `ColQwen2_5Processor` aqui foi o que deixou o teste verde enquanto o cluster caia no
# fallback para ColQwen2 e estourava "size mismatch" no state_dict.
_WITH_25 = dict(_LEGACY, ColQwen2_5=_ColQwen25, ColQwen2_5_Processor=_ColQwen25Processor)


def test_colpali(monkeypatch):
    _install_fake_colpali(monkeypatch, _WITH_25)
    assert resolve_classes("colpali") == (_ColPali, _ColPaliProcessor)


def test_colqwen2(monkeypatch):
    _install_fake_colpali(monkeypatch, _WITH_25)
    assert resolve_classes("colqwen2") == (_ColQwen2, _ColQwen2Processor)


def test_colqwen25_usa_classe_dedicada_quando_existe(monkeypatch):
    _install_fake_colpali(monkeypatch, _WITH_25)
    assert resolve_classes("colqwen2.5") == (_ColQwen25, _ColQwen25Processor)


def test_colqwen25_erra_claro_em_release_antiga(monkeypatch):
    """Release sem ColQwen2_5 -> erro nomeado, NAO fallback silencioso para ColQwen2.

    O checkpoint 2.5 e qwen2_5_vl e a classe ColQwen2 e qwen2_vl: carregar um pelo outro
    nao degrada, quebra no state_dict ("size mismatch", 2048 vs 1280) longe da causa.
    """
    _install_fake_colpali(monkeypatch, _LEGACY)
    with pytest.raises(ImportError, match="ColQwen2_5"):
        resolve_classes("colqwen2.5")


def test_colqwen25_nome_do_processor_tem_underscore(monkeypatch):
    """Regressao: `ColQwen2_5Processor` (sem o underscore) nao existe no colpali-engine.

    Com o nome errado o import falhava e o wrapper caia em ColQwen2 -- exatamente o bug
    que derrubava o config `v3_colqwen25` no cluster.
    """
    _install_fake_colpali(
        monkeypatch, dict(_LEGACY, ColQwen2_5=_ColQwen25, ColQwen2_5Processor=_ColQwen25Processor)
    )
    with pytest.raises(ImportError, match="ColQwen2_5"):
        resolve_classes("colqwen2.5")


def test_kind_desconhecido(monkeypatch):
    _install_fake_colpali(monkeypatch, _WITH_25)
    with pytest.raises(ValueError):
        resolve_classes("qwen3vl")
