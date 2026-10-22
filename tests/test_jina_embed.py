"""Contrato do wrapper jina-embeddings-v4 (denso), com modelo/processor falsos."""

import numpy as np
import pytest

from benchmark.common.config import ExperimentConfig
from benchmark.common.types import Doc, Query
from benchmark.models.factory import build_retriever, is_multi_vector
from benchmark.models.jina_embed import JinaV4Retriever, _call_encode


class _FakeModel:
    """Expoe a API do card: `encode_text` / `encode_image` com os kwargs do jina-v4."""

    def __init__(self, dim=4):
        self.dim = dim
        self.calls: list[dict] = []

    def encode_text(self, texts, task=None, max_length=32768, batch_size=8,
                    return_multivector=False, return_numpy=False, truncate_dim=None,
                    prompt_name=None):
        self.calls.append({"fn": "encode_text", "items": list(texts), "task": task,
                           "prompt_name": prompt_name, "batch_size": batch_size,
                           "return_multivector": return_multivector,
                           "truncate_dim": truncate_dim})
        # o modelo devolve uma lista de vetores [dim], um por item
        return [np.full(self.dim, 2.0, dtype="float32") for _ in texts]

    def encode_image(self, images, task=None, batch_size=8, return_multivector=False,
                     return_numpy=False, truncate_dim=None, max_pixels=None):
        self.calls.append({"fn": "encode_image", "items": list(images), "task": task,
                           "batch_size": batch_size,
                           "return_multivector": return_multivector,
                           "truncate_dim": truncate_dim})
        return [np.full(self.dim, 3.0, dtype="float32") for _ in images]


def _bare(dim=None, model=None, batch_size=2) -> JinaV4Retriever:
    """Instancia sem __init__ (nao baixa pesos, nao importa torch/transformers)."""
    r = JinaV4Retriever.__new__(JinaV4Retriever)
    r.name = "jinav4:test"
    r.model_id = "jinaai/jina-embeddings-v4"
    r.batch_size = batch_size
    r.dim = dim
    r._model = model
    return r


# -- contrato do Retriever --------------------------------------------------------


def test_contract_denso(tmp_path):
    r = _bare()
    r._embed_queries = lambda ts: np.full((len(ts), 4), 2.0, dtype="float32")
    r._embed_images = lambda ps: np.full((len(ps), 4), 3.0, dtype="float32")

    qe = r.encode_queries([Query(id="q", text="meio ambiente")])
    de = r.encode_corpus([Doc(id="d", image_path=str(tmp_path / "x.png")),
                          Doc(id="d2", image_path=str(tmp_path / "y.png"))])

    assert isinstance(qe, np.ndarray) and qe.shape == (1, 4)
    assert isinstance(de, np.ndarray) and de.shape == (2, 4)
    assert qe.dtype == np.float32 and de.dtype == np.float32
    # saida L2-normalizada nos dois lados -> score e cosine
    assert np.allclose(np.linalg.norm(qe, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(de, axis=1), 1.0)
    assert r.score(qe, de).shape == (1, 2)


def test_score_e_produto_interno():
    r = _bare()
    q = np.array([[1.0, 0.0]], dtype="float32")
    d = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    assert np.allclose(r.score(q, d), [[1.0, 0.0]])


def test_query_usa_texto_corpus_usa_imagem():
    r = _bare()
    seen: dict[str, list] = {}

    def fake_queries(texts):
        seen["q"] = list(texts)
        return np.ones((len(texts), 2), dtype="float32")

    def fake_images(paths):
        seen["d"] = list(paths)
        return np.ones((len(paths), 2), dtype="float32")

    r._embed_queries = fake_queries
    r._embed_images = fake_images
    r.encode_queries([Query(id="q1", text="orcamento")])
    r.encode_corpus([Doc(id="d1", image_path="pagina.png", text="ignorado")])
    assert seen["q"] == ["orcamento"]
    assert seen["d"] == ["pagina.png"]


def test_dim_trunca_e_renormaliza():
    r = _bare(dim=2)
    r._embed_queries = lambda ts: np.array([[3.0, 4.0, 10.0, 10.0]], dtype="float32")
    qe = r.encode_queries([Query(id="q", text="x")])
    assert qe.shape == (1, 2)
    assert np.allclose(qe, [[0.6, 0.8]])


def test_stack_aceita_lista_de_vetores():
    out = JinaV4Retriever._stack([np.array([3.0, 4.0]), np.array([0.0, 2.0])])
    assert out.shape == (2, 2) and out.dtype == np.float32


# -- caminho real de encode, com o modelo falso -----------------------------------


def test_embed_queries_chama_encode_text_com_task_retrieval():
    model = _FakeModel()
    r = _bare(model=model)
    emb = r._embed_queries(["politica ambiental", "saude"])

    assert emb.shape == (2, 4) and emb.dtype == np.float32
    assert [c["fn"] for c in model.calls] == ["encode_text"]
    call = model.calls[0]
    assert call["items"] == ["politica ambiental", "saude"]
    assert call["task"] == "retrieval"
    assert call["prompt_name"] == "query"
    assert call["return_multivector"] is False      # esta versao e so densa


def test_embed_images_passa_os_caminhos_para_encode_image():
    model = _FakeModel()
    r = _bare(model=model)
    emb = r._embed_images(["a.png", "b.png"])

    assert emb.shape == (2, 4)
    call = model.calls[0]
    assert call["fn"] == "encode_image"
    # `encode_image` aceita caminho local e faz o convert("RGB") por dentro: nada de PIL
    assert call["items"] == ["a.png", "b.png"]
    assert call["task"] == "retrieval"
    assert call["return_multivector"] is False


def test_encode_queries_fim_a_fim_com_o_modelo_falso():
    r = _bare(model=_FakeModel(dim=4))
    qe = r.encode_queries([Query(id="q1", text="meta fiscal")])
    assert qe.shape == (1, 4)
    assert np.allclose(np.linalg.norm(qe, axis=1), 1.0)


def test_batches_respeitam_o_batch_size():
    model = _FakeModel()
    r = _bare(model=model, batch_size=2)
    emb = r._embed_queries(["a", "b", "c"])
    assert emb.shape == (3, 4)
    assert [c["items"] for c in model.calls] == [["a", "b"], ["c"]]
    assert [c["batch_size"] for c in model.calls] == [2, 1]


def test_dim_vai_como_truncate_dim_para_o_modelo():
    model = _FakeModel()
    r = _bare(dim=128, model=model)
    r._embed_queries(["x"])
    assert model.calls[0]["truncate_dim"] == 128


def test_lote_vazio_nao_quebra():
    r = _bare(model=_FakeModel())
    assert r._embed_queries([]).shape == (0, 1)
    assert r._embed_images([]).shape == (0, 1)


# -- resolucao defensiva da API ---------------------------------------------------


def test_call_encode_omite_kwargs_nao_suportados():
    def encode_text(texts, task=None):          # assinatura estreita, sem prompt_name
        return [(t, task) for t in texts]

    out = _call_encode(encode_text, ["a"], task="retrieval", prompt_name="query",
                       truncate_dim=None, batch_size=1)
    assert out == [("a", "retrieval")]


def test_nome_alternativo_de_encode_e_aceito():
    class _Renomeado:
        def encode_texts(self, texts, task=None):
            return [np.ones(3, dtype="float32") for _ in texts]

    r = _bare(model=_Renomeado())
    assert r._embed_queries(["a", "b"]).shape == (2, 3)


def test_pool_faz_mean_pooling_mascarado():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(JinaV4Retriever._pool(hidden, mask),
                          torch.tensor([[2.0, 2.0]]))


# -- factory / config -------------------------------------------------------------


def test_factory_resolve_o_wrapper_jinav4(monkeypatch):
    cfg = ExperimentConfig(name="v3_jinav4", model_id="jinaai/jina-embeddings-v4",
                           model_kind="jinav4", dataset="vidore/vidore_v3_hr",
                           batch_size=4)
    assert is_multi_vector(cfg) is False          # denso: dense_search, nao MaxSim

    seen = {}

    class _Stub:
        def __init__(self, model_id, device="cuda", dtype="bfloat16", batch_size=8):
            seen.update(model_id=model_id, device=device, dtype=dtype,
                        batch_size=batch_size)

    monkeypatch.setattr("benchmark.models.jina_embed.JinaV4Retriever", _Stub)
    assert isinstance(build_retriever(cfg), _Stub)
    assert seen == {"model_id": "jinaai/jina-embeddings-v4", "device": "cuda",
                    "dtype": "bfloat16", "batch_size": 4}
