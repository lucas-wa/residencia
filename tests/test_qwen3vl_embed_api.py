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
    DEFAULT_INSTRUCTION,
    DEFAULT_MODEL_ID,
    MAX_LENGTH,
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


def _bare(model, proc=None, batch_size=2, dim=None,
          instruction=DEFAULT_INSTRUCTION) -> Qwen3VLEmbedRetriever:
    """Instancia sem __init__ (nao carrega pesos)."""
    r = Qwen3VLEmbedRetriever.__new__(Qwen3VLEmbedRetriever)
    r.name = "qwen3vl:test"
    r.model_id = "test/qwen3vl"
    r.device = "cpu"
    r.batch_size = batch_size
    r.dim = dim
    r.instruction = instruction
    r.max_length = MAX_LENGTH
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


# -- caminho real: chat template -> input_ids (+ pixel_values) -> pooling do ultimo token --
#
# O checkpoint nao expoe metodo de encode (config.json sem auto_map), entao este e o
# caminho que a rodada de verdade percorre. O duble do modelo exige `input_ids`: sem
# eles o Qwen3VLModel estoura "You must specify exactly one of input_ids or
# inputs_embeds" -- o bug que o wrapper tinha ao mandar `processor(images=...)` cru.


class _Tensor:
    """Tensor minimo: so `.to(device)` e `.sum(dim=...)`, que e o que o wrapper usa."""

    def __init__(self, data):
        self.data = np.asarray(data)
        self.movido_para = None

    def to(self, device):
        self.movido_para = device
        return self

    def sum(self, dim=0):
        return self.data.sum(axis=dim)


class _Processor:
    """Processor do Qwen3-VL dublado: chat template + batch com input_ids."""

    def __init__(self, n_tokens=3):
        self.conversas: list[list[dict]] = []
        self.chamadas: list[dict] = []
        self.n_tokens = n_tokens

    def apply_chat_template(self, conversation, add_generation_prompt=False, tokenize=True):
        assert add_generation_prompt is True and tokenize is False
        self.conversas.append(conversation)
        n_img = sum(1 for msg in conversation for c in msg["content"] if c["type"] == "image")
        return "<|im_start|>" + "<|image_pad|>" * n_img + "<|im_start|>assistant\n"

    def __call__(self, **kwargs):
        self.chamadas.append(kwargs)
        n = len(kwargs["text"])
        out = {"input_ids": _Tensor([[1] * self.n_tokens] * n),
               "attention_mask": _Tensor([[1] * self.n_tokens] * n)}
        if kwargs.get("images"):
            out["pixel_values"] = _Tensor(np.zeros((len(kwargs["images"]), 2)))
        return out


class _ModelGenerico:
    """Duble do Qwen3VLModel: exige `input_ids` e devolve `last_hidden_state`."""

    device = "cpu"

    def __init__(self):
        self.inputs_vistos: list[dict] = []

    def __call__(self, **inputs):
        if "input_ids" not in inputs:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        self.inputs_vistos.append(inputs)
        n, n_tokens = inputs["input_ids"].data.shape
        hidden = np.zeros((n, n_tokens, 4), dtype="float32")
        hidden[:, -1, :] = 1.0          # so o ULTIMO token carrega o vetor [1,1,1,1]
        return types.SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture
def fake_vision(monkeypatch):
    """`qwen_vl_utils.process_vision_info` dublado; devolve as conversas que recebeu."""
    conversas: list = []

    def process_vision_info(convs, image_patch_size=None):
        conversas.append((convs, image_patch_size))
        imgs = [c["image"] for conv in convs for msg in conv for c in msg["content"]
                if c["type"] == "image"]
        return (imgs or None, None)

    qvu = types.ModuleType("qwen_vl_utils")
    qvu.process_vision_info = process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qvu)
    return conversas


def test_imagem_monta_input_ids_e_pixel_values(tmp_path, fake_libs, fake_vision):
    model, proc = _ModelGenerico(), _Processor()
    r = _bare(model, proc=proc, batch_size=4)

    vecs = embed_images(_paths(tmp_path, 2), embedder=r)

    # o batch entregue ao modelo tem input_ids (do chat template) E pixel_values
    batch = model.inputs_vistos[0]
    assert "input_ids" in batch and "pixel_values" in batch and "attention_mask" in batch
    assert batch["input_ids"].movido_para == "cpu"
    # conversa: system com a instrucao + user com a imagem
    conv = proc.conversas[0]
    assert [m["role"] for m in conv] == ["system", "user"]
    assert conv[0]["content"][0]["text"] == DEFAULT_INSTRUCTION
    assert conv[1]["content"][0]["type"] == "image"
    # imagens passam pelo process_vision_info (patch 16) -> processor sem resize proprio
    assert fake_vision[0][1] == mod.IMAGE_PATCH_SIZE
    assert proc.chamadas[0]["do_resize"] is False
    assert proc.chamadas[0]["max_length"] == MAX_LENGTH
    # saida [N, dim] L2-normalizada
    assert vecs.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert np.allclose(vecs, 0.5)                       # [1,1,1,1]/2


def test_texto_monta_input_ids_sem_pixel_values(fake_libs, fake_vision):
    model, proc = _ModelGenerico(), _Processor()
    r = _bare(model, proc=proc, batch_size=4)

    vecs = embed_texts(["metas de saude", "orcamento"], embedder=r)

    batch = model.inputs_vistos[0]
    assert "input_ids" in batch and "pixel_values" not in batch
    assert "images" not in proc.chamadas[0]
    assert not fake_vision                              # sem imagem, sem qwen_vl_utils
    conv = proc.conversas[0]
    assert conv[1]["content"] == [{"type": "text", "text": "metas de saude"}]
    assert vecs.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_sem_qwen_vl_utils_manda_as_pil_cruas(tmp_path, fake_libs, monkeypatch):
    """Sem o helper oficial o processor recebe as imagens e faz o resize sozinho."""
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", None)   # import -> ImportError
    model, proc = _ModelGenerico(), _Processor()
    r = _bare(model, proc=proc, batch_size=4)

    vecs = embed_images(_paths(tmp_path, 2), embedder=r)

    assert len(proc.chamadas[0]["images"]) == 2
    assert "do_resize" not in proc.chamadas[0]
    assert "pixel_values" in model.inputs_vistos[0]
    assert vecs.shape == (2, 4)


def test_pooling_pega_o_ultimo_token_nao_pad():
    hidden = np.arange(2 * 3 * 2, dtype="float32").reshape(2, 3, 2)
    mask = _Tensor([[1, 1, 0], [1, 1, 1]])              # padding a direita
    pooled = Qwen3VLEmbedRetriever._pool(hidden, mask)
    assert np.allclose(pooled, [hidden[0, 1], hidden[1, 2]])


def test_instrucao_customizada_vai_no_system_prompt(fake_libs, fake_vision):
    proc = _Processor()
    r = _bare(_ModelGenerico(), proc=proc, batch_size=4,
              instruction="Retrieve relevant documents for the query")
    embed_texts(["x"], embedder=r)
    # ponto final acrescentado como no modeling de referencia
    assert proc.conversas[0][0]["content"][0]["text"] == (
        "Retrieve relevant documents for the query.")


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
