"""API de inferencia do Qwen3-VL-Embedding fora do benchmark (`embed_images`/`embed_texts`).

Modelo e processor sao dublados, e `torch`/`PIL` entram por `sys.modules`: o teste roda
offline, sem baixar pesos e sem exigir as libs de GPU instaladas.
"""

import sys
import types

import numpy as np
import pytest

from benchmark.models import qwen3vl_embed as mod
from benchmark.models.qwen3vl_embed import (
    DEFAULT_MODEL_ID,
    Qwen3VLEmbedRetriever,
    embed_images,
    embed_texts,
)


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_libs(monkeypatch):
    """`torch.no_grad` e `PIL.Image.open` minimos, so o que os _embed_* usam."""
    torch = types.ModuleType("torch")
    torch.no_grad = _NoGrad
    monkeypatch.setitem(sys.modules, "torch", torch)

    opened: list[str] = []

    class _Img:
        def __init__(self, path):
            self.path = path

        def convert(self, mode):
            return self

        def close(self):
            pass

    image = types.ModuleType("PIL.Image")
    image.open = lambda p: (opened.append(str(p)), _Img(p))[1]
    pil = types.ModuleType("PIL")
    pil.Image = image
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image)
    return opened


class _ModelComEncode:
    """Duble do checkpoint custom-code: expoe `encode_query`/`encode_document`."""

    def __init__(self, dim=4):
        self.dim = dim
        self.query_batches: list[list] = []
        self.doc_batches: list[list] = []

    def encode_query(self, texts):
        self.query_batches.append(list(texts))
        # vetores propositalmente NAO normalizados (norma 5 e 13)
        return np.array([[3.0, 4.0, 0.0, 0.0]] * len(texts), dtype="float32")[:, : self.dim]

    def encode_document(self, images):
        self.doc_batches.append(list(images))
        return [np.array([0.0, 5.0, 12.0, 0.0], dtype="float32")[: self.dim] for _ in images]


def _bare(model, proc=None, batch_size=2, dim=None) -> Qwen3VLEmbedRetriever:
    """Instancia sem __init__ (nao carrega pesos)."""
    r = Qwen3VLEmbedRetriever.__new__(Qwen3VLEmbedRetriever)
    r.name = "qwen3vl:test"
    r.model_id = "test/qwen3vl"
    r.device = "cpu"
    r.batch_size = batch_size
    r.dim = dim
    r._model = model
    r._proc = proc
    return r


def _paths(tmp_path, n):
    out = []
    for i in range(n):
        p = tmp_path / f"pagina_p{i:02d}.png"
        p.write_bytes(b"")
        out.append(str(p))
    return out


def test_embed_images_shape_e_norma_l2(tmp_path, fake_libs):
    model = _ModelComEncode()
    r = _bare(model, batch_size=2)
    paths = _paths(tmp_path, 3)

    vecs = embed_images(paths, embedder=r)

    assert isinstance(vecs, np.ndarray) and vecs.dtype == np.float32
    assert vecs.shape == (3, 4)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    # direcao preservada: [0,5,12,0] / 13
    assert np.allclose(vecs[0], [0.0, 5 / 13, 12 / 13, 0.0])
    # batelado por self.batch_size (3 imagens, batch 2 -> 2 chamadas)
    assert [len(b) for b in model.doc_batches] == [2, 1]
    assert fake_libs == paths            # abriu exatamente as imagens pedidas


def test_embed_texts_shape_e_norma_l2(fake_libs):
    model = _ModelComEncode()
    r = _bare(model, batch_size=2)

    vecs = embed_texts(["saude publica", "meio ambiente", "educacao"], embedder=r)

    assert vecs.shape == (3, 4) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert np.allclose(vecs[0], [0.6, 0.8, 0.0, 0.0])
    assert model.query_batches == [["saude publica", "meio ambiente"], ["educacao"]]


def test_texto_e_imagem_no_mesmo_espaco(tmp_path, fake_libs):
    """Os dois lados saem com a mesma dimensao -> produto interno e comparavel."""
    r = _bare(_ModelComEncode(), batch_size=4)
    q = embed_texts(["orcamento"], embedder=r)
    d = embed_images(_paths(tmp_path, 2), embedder=r)
    scores = q @ d.T
    assert scores.shape == (1, 2)
    assert np.all(scores <= 1.0 + 1e-6)


def test_dim_trunca_e_renormaliza(fake_libs):
    r = _bare(_ModelComEncode(), batch_size=2, dim=2)
    vecs = embed_texts(["x"], embedder=r)
    assert vecs.shape == (1, 2)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


class _ModelGenerico:
    """Duble sem API de encode: forca o caminho processor + forward + pooling."""

    device = "cpu"

    def __init__(self):
        self.inputs_vistos: list[dict] = []

    def __call__(self, **inputs):
        self.inputs_vistos.append(inputs)
        n = len(inputs["pixel_values"].rows)
        return types.SimpleNamespace(
            embeddings=np.array([[1.0, 1.0, 1.0, 1.0]] * n, dtype="float32"))


class _Tensor:
    def __init__(self, rows):
        self.rows = rows

    def to(self, _device):
        return self


class _Processor:
    def __init__(self):
        self.chamadas: list[dict] = []

    def __call__(self, **kwargs):
        self.chamadas.append(kwargs)
        items = kwargs.get("images") or kwargs.get("text")
        return {"pixel_values": _Tensor(list(items))}


def test_caminho_generico_usa_processor_e_normaliza(tmp_path, fake_libs):
    model, proc = _ModelGenerico(), _Processor()
    r = _bare(model, proc=proc, batch_size=4)

    vecs = embed_images(_paths(tmp_path, 2), embedder=r)

    assert vecs.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert np.allclose(vecs, 0.5)                       # [1,1,1,1]/2
    assert proc.chamadas and "images" in proc.chamadas[0]


def test_default_e_o_2b():
    assert DEFAULT_MODEL_ID == "Qwen/Qwen3-VL-Embedding-2B"


def test_get_embedder_cacheia_por_configuracao(monkeypatch):
    construidos: list[tuple] = []

    class _Fake:
        def __init__(self, model_id, device="cuda", dtype="bfloat16", batch_size=8, dim=None):
            construidos.append((model_id, device, dtype, batch_size, dim))
            self.model_id = model_id

        def embed_images(self, paths):
            return np.ones((len(paths), 2), dtype="float32") / np.sqrt(2)

    monkeypatch.setattr(mod, "_EMBEDDERS", {})
    monkeypatch.setattr(mod, "Qwen3VLEmbedRetriever", _Fake)

    a = mod.get_embedder(device="cpu")
    b = mod.get_embedder(device="cpu")
    assert a is b
    assert construidos == [(DEFAULT_MODEL_ID, "cpu", "bfloat16", 8, None)]

    # sem `embedder`, as funcoes de modulo passam pelo mesmo cache
    out = embed_images(["a.png"], device="cpu")
    assert out.shape == (1, 2) and len(construidos) == 1

    mod.get_embedder(device="cpu", model_id="Qwen/Qwen3-VL-Embedding-8B")
    assert len(construidos) == 2
