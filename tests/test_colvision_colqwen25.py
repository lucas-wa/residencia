"""Resolucao das classes do colpali-engine por `kind` (sem instalar o pacote)."""

import sys
import types

import pytest

from benchmark.models.colvision import ColVisionRetriever, resolve_classes


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


# -- fallback do processor (bug do transformers 4.57.x) --------------------------------

_MODEL_ID = "vidore/colqwen2.5-v0.2"
_TEMPLATE_BUG = "expected str, bytes or os.PathLike object, not NoneType"


class _FakeModel:
    device = "cpu"

    def eval(self):
        return self


class _FakeModelCls:
    """Fica no lugar da classe do colpali-engine: `from_pretrained` nunca baixa pesos."""

    def from_pretrained(self, model_id, **kwargs):
        return _FakeModel()


class _FakeProcCls:
    """`from_pretrained` levanta `raises` na 1a chamada e devolve um sentinela depois."""

    def __init__(self, raises=None):
        self.raises = raises
        self.calls: list[str] = []

    def from_pretrained(self, path):
        self.calls.append(path)
        if len(self.calls) == 1 and self.raises is not None:
            raise self.raises
        return ("proc", path)


def _prepare(monkeypatch, proc_cls, snapshot=None):
    """Injeta colpali_engine/torch falsos e, se dado, um `snapshot_download` gravador."""
    import huggingface_hub

    _install_fake_colpali(
        monkeypatch, dict(_WITH_25, ColQwen2_5=_FakeModelCls(), ColQwen2_5_Processor=proc_cls)
    )
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "torch", torch)
    calls: list[dict] = []
    if snapshot is not None:
        def _fake_snapshot_download(repo_id=None, **kwargs):
            calls.append(dict(kwargs, repo_id=repo_id))
            return snapshot
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)
    return calls


def test_processor_cai_para_snapshot_local_sem_chat_templates(monkeypatch):
    """Regressao: transformers 4.57.x estoura em `open(None)` com `additional_chat_templates/`.

    `get_processor_dict` monta o dict de templates com `cached_file(...,
    _raise_exceptions_for_missing_entries=False)`; uma entrada vem `None` e a
    comprehension abre sem checar -> TypeError. Recarregar de um snapshot local sem esses
    arquivos faz o glob de templates vir vazio e o load passa.
    """
    proc = _FakeProcCls(raises=TypeError(_TEMPLATE_BUG))
    calls = _prepare(monkeypatch, proc, snapshot="/tmp/snap-colqwen25")

    r = ColVisionRetriever(_MODEL_ID, kind="colqwen2.5", device="cpu")

    assert proc.calls == [_MODEL_ID, "/tmp/snap-colqwen25"]
    assert r._proc == ("proc", "/tmp/snap-colqwen25")
    assert len(calls) == 1
    assert calls[0]["repo_id"] == _MODEL_ID
    assert "additional_chat_templates/*" in calls[0]["ignore_patterns"]


def test_typeerror_de_outra_causa_propaga_sem_snapshot(monkeypatch):
    """So o TypeError DESTE bug aciona o fallback -- qualquer outro sobe intacto."""
    proc = _FakeProcCls(raises=TypeError("got an unexpected keyword argument 'foo'"))
    calls = _prepare(monkeypatch, proc, snapshot="/tmp/snap-colqwen25")

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ColVisionRetriever(_MODEL_ID, kind="colqwen2.5", device="cpu")

    assert calls == []
    assert proc.calls == [_MODEL_ID]


def test_processor_que_carrega_direto_nao_baixa_snapshot(monkeypatch):
    """Caminho colpali/colqwen2 (e qualquer load que funciona): uma chamada, sem fallback."""
    proc = _FakeProcCls()
    calls = _prepare(monkeypatch, proc, snapshot="/tmp/snap-colqwen25")

    r = ColVisionRetriever(_MODEL_ID, kind="colqwen2.5", device="cpu")

    assert proc.calls == [_MODEL_ID]
    assert r._proc == ("proc", _MODEL_ID)
    assert calls == []
