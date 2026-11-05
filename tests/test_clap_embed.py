"""Contrato do embedder CLAP: forma, dtype e norma -- offline, sem baixar pesos.

O backbone nunca e carregado: ou o teste constroi o embedder por `__new__` e substitui
`_audio_vectors`/`_text_vectors` (contrato puro), ou injeta dubles de modelo/processor
e entra no caminho real (`ClapProcessor` -> `get_audio_features`), com `torch` e
`librosa` empurrados por `sys.modules`.

512 e a dimensao do `laion/larger_clap_general` declarada em `SPACES["clap"].dim`.
"""

import logging
import sys
import types

import numpy as np
import pytest

from benchmark.models import clap_embed


def test_embed_audio_normalizado_e_finito(monkeypatch):
    emb = clap_embed.ClapEmbedder.__new__(clap_embed.ClapEmbedder)
    monkeypatch.setattr(emb, "_audio_vectors", lambda paths: np.random.randn(len(paths), 512))
    out = emb._postprocess(emb._audio_vectors(["a.wav", "b.wav"]))
    assert out.shape == (2, 512)
    assert np.isfinite(out).all()
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_texto_e_audio_no_mesmo_dim(monkeypatch):
    emb = clap_embed.ClapEmbedder.__new__(clap_embed.ClapEmbedder)
    monkeypatch.setattr(emb, "_audio_vectors", lambda p: np.random.randn(len(p), 512))
    monkeypatch.setattr(emb, "_text_vectors", lambda t: np.random.randn(len(t), 512))
    assert emb._postprocess(emb._audio_vectors(["a.wav"])).shape[1] == \
           emb._postprocess(emb._text_vectors(["metas de saúde"])).shape[1]


def test_postprocess_devolve_float32_bidimensional():
    """Vetor solto vira `[1, dim]` float32 -- o indice sempre recebe uma matriz."""
    emb = clap_embed.ClapEmbedder.__new__(clap_embed.ClapEmbedder)
    out = emb._postprocess(np.array([3.0, 4.0], dtype="float64"))
    assert out.shape == (1, 2) and out.dtype == np.float32
    assert np.allclose(out, [[0.6, 0.8]])


def test_defaults_concordam_com_o_registro_de_espacos():
    """Divergir do registro quebra o upsert do `embed_job` (tabela criada com outra dim)."""
    from gcp.slice.spaces import SPACES

    assert clap_embed.DEFAULT_CLAP_ID == "laion/larger_clap_general"
    assert clap_embed.DEFAULT_CLAP_ID == SPACES["clap"].model_id
    assert clap_embed.CLAP_DIM == SPACES["clap"].dim == 512
    assert clap_embed.CLAP_SAMPLE_RATE == 48_000
    assert clap_embed.CLAP_MAX_SECONDS == 10.0


# -- caminho real: librosa -> ClapProcessor -> get_audio_features/get_text_features ----


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Tensor:
    """Tensor minimo: so `.to(device)`, que e o que o wrapper usa."""

    def __init__(self, data):
        self.data = np.asarray(data)
        self.movido_para = None

    def to(self, device):
        self.movido_para = device
        return self


@pytest.fixture
def fake_libs(monkeypatch):
    """`torch.no_grad` e `librosa.load` dublados; devolve as chamadas de `load`.

    A onda devolvida dura `sys.modules["librosa"].segundos` (1 s por padrao) -- o teste do
    aviso de truncacao aumenta esse valor para simular um clipe longo.
    """
    torch = types.ModuleType("torch")
    torch.no_grad = _NoGrad
    monkeypatch.setitem(sys.modules, "torch", torch)

    chamadas: list[dict] = []

    def load(path, sr=None, mono=True):
        chamadas.append({"path": str(path), "sr": sr, "mono": mono})
        return np.zeros(int((sr or 16_000) * librosa.segundos), dtype="float32"), sr

    librosa = types.ModuleType("librosa")
    librosa.load = load
    librosa.segundos = 1.0
    monkeypatch.setitem(sys.modules, "librosa", librosa)
    return chamadas


class _FeatureExtractor:
    sampling_rate = 48_000
    max_length_s = 10          # janela do CLAP: acima disso o extractor trunca


class _Proc:
    """ClapProcessor dublado: `audios=` -> features de audio; `text=` -> tokens."""

    feature_extractor = _FeatureExtractor()

    def __init__(self):
        self.chamadas: list[dict] = []

    def __call__(self, **kwargs):
        self.chamadas.append(kwargs)
        if kwargs.get("audios") is not None:
            n = len(kwargs["audios"])
            return {"input_features": _Tensor(np.zeros((n, 1, 4, 2))),
                    "is_longer": _Tensor(np.zeros((n, 1)))}
        n = len(kwargs["text"])
        return {"input_ids": _Tensor([[1, 2, 3]] * n),
                "attention_mask": _Tensor([[1, 1, 1]] * n)}


class _Model:
    """ClapModel dublado: devolve vetores NAO normalizados (norma 5 e 13)."""

    def __init__(self):
        self.audio_batches: list[dict] = []
        self.text_batches: list[dict] = []

    def get_audio_features(self, **inputs):
        if "input_features" not in inputs:
            raise AssertionError("get_audio_features sem input_features")
        self.audio_batches.append(inputs)
        n = len(inputs["input_features"].data)
        return np.array([[0.0, 5.0, 12.0, 0.0]] * n, dtype="float32")

    def get_text_features(self, **inputs):
        if "input_ids" not in inputs:
            raise AssertionError("get_text_features sem input_ids")
        self.text_batches.append(inputs)
        n = len(inputs["input_ids"].data)
        return np.array([[3.0, 4.0, 0.0, 0.0]] * n, dtype="float32")


def _bare(model=None, proc=None, batch_size=2) -> clap_embed.ClapEmbedder:
    """Instancia sem __init__, com modelo/processor ja postos (`_load` nao carrega nada)."""
    emb = clap_embed.ClapEmbedder.__new__(clap_embed.ClapEmbedder)
    emb.name = "clap:test"
    emb.model_id = "test/clap"
    emb.device = "cpu"
    emb.batch_size = batch_size
    emb._model = model
    emb._proc = proc
    return emb


def test_embed_audio_batela_e_alimenta_o_processor(fake_libs):
    model, proc = _Model(), _Proc()
    emb = _bare(model, proc, batch_size=2)

    vecs = clap_embed.embed_audio(["a.m4a", "b.m4a", "c.m4a"], embedder=emb)

    # 3 audios com batch 2 -> 2 chamadas ao processor/modelo
    assert [len(c["audios"]) for c in proc.chamadas] == [2, 1]
    assert all(c["sampling_rate"] == 48_000 for c in proc.chamadas)
    assert all(c["return_tensors"] == "pt" for c in proc.chamadas)
    # audio lido ja reamostrado para a taxa do feature extractor, em mono
    assert [c["path"] for c in fake_libs] == ["a.m4a", "b.m4a", "c.m4a"]
    assert all(c["sr"] == 48_000 and c["mono"] is True for c in fake_libs)
    # `is_longer` (exigido pelos checkpoints com fusao) chega ao modelo, no device certo
    assert "is_longer" in model.audio_batches[0]
    assert model.audio_batches[0]["input_features"].movido_para == "cpu"
    # saida [N, dim] float32 L2-normalizada, direcao preservada: [0,5,12,0]/13
    assert isinstance(vecs, np.ndarray) and vecs.dtype == np.float32
    assert vecs.shape == (3, 4)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert np.allclose(vecs[0], [0.0, 5 / 13, 12 / 13, 0.0])


def test_embed_texts_usa_o_encoder_de_texto(fake_libs):
    model, proc = _Model(), _Proc()
    emb = _bare(model, proc, batch_size=2)

    vecs = clap_embed.embed_texts(["metas de saude", "orcamento", "meio ambiente"],
                                  embedder=emb)

    assert [c["text"] for c in proc.chamadas] == [["metas de saude", "orcamento"],
                                                  ["meio ambiente"]]
    assert all(c["padding"] is True for c in proc.chamadas)
    # consulta longa demais e CORTADA, nao estoura o limite posicional do text tower
    assert all(c["truncation"] is True for c in proc.chamadas)
    assert not fake_libs                      # texto nao abre arquivo de audio
    assert not model.audio_batches            # nem passa pelo encoder de audio
    assert vecs.shape == (3, 4) and vecs.dtype == np.float32
    assert np.allclose(vecs[0], [0.6, 0.8, 0.0, 0.0])


def test_audio_longo_avisa_que_sera_truncado(fake_libs, caplog):
    """Clipe de 90 s > janela de ~10 s do extractor: a perda tem de aparecer no log."""
    sys.modules["librosa"].segundos = 90.0
    emb = _bare(_Model(), _Proc(), batch_size=2)

    with caplog.at_level(logging.WARNING, logger="benchmark.models.clap_embed"):
        vecs = clap_embed.embed_audio(["sessao_longa.m4a"], embedder=emb)

    avisos = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(avisos) == 1
    assert "sessao_longa.m4a" in avisos[0] and "90.0 s" in avisos[0] and "10 s" in avisos[0]
    # o aviso nao interrompe nada: o vetor sai normalmente (truncado pelo extractor)
    assert vecs.shape == (1, 4)


def test_audio_dentro_da_janela_nao_avisa(fake_libs, caplog):
    emb = _bare(_Model(), _Proc(), batch_size=2)
    with caplog.at_level(logging.WARNING, logger="benchmark.models.clap_embed"):
        clap_embed.embed_audio(["curto.m4a"], embedder=emb)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_texto_e_audio_sao_comparaveis(fake_libs):
    """Mesmo espaco: o produto interno texto x audio e um cosseno valido."""
    emb = _bare(_Model(), _Proc(), batch_size=4)
    q = clap_embed.embed_texts(["sessao sobre saude"], embedder=emb)
    d = clap_embed.embed_audio(["a.m4a", "b.m4a"], embedder=emb)
    scores = q @ d.T
    assert scores.shape == (1, 2)
    assert np.all(np.abs(scores) <= 1.0 + 1e-6)


def test_entrada_vazia_devolve_matriz_com_a_dim_do_espaco(fake_libs):
    emb = _bare(_Model(), _Proc())
    assert clap_embed.embed_audio([], embedder=emb).shape == (0, clap_embed.CLAP_DIM)
    assert clap_embed.embed_texts([], embedder=emb).shape == (0, clap_embed.CLAP_DIM)


def test_get_embedder_cacheia_por_configuracao(monkeypatch):
    construidos: list[tuple] = []

    class _Fake:
        def __init__(self, model_id, device="cuda", batch_size=8):
            construidos.append((model_id, device, batch_size))

        def embed_audio(self, paths):
            return np.ones((len(paths), 2), dtype="float32") / np.sqrt(2)

    monkeypatch.setattr(clap_embed, "_EMBEDDERS", {})
    monkeypatch.setattr(clap_embed, "ClapEmbedder", _Fake)

    a = clap_embed.get_embedder(device="cpu")
    b = clap_embed.get_embedder(device="cpu")
    assert a is b
    assert construidos == [(clap_embed.DEFAULT_CLAP_ID, "cpu", 8)]

    # sem `embedder`, as funcoes de modulo passam pelo mesmo cache
    out = clap_embed.embed_audio(["a.m4a"], device="cpu")
    assert out.shape == (1, 2) and len(construidos) == 1

    clap_embed.get_embedder(device="cpu", batch_size=4)      # outra chave -> outro embedder
    assert len(construidos) == 2
