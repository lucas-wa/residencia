import numpy as np

from benchmark.common.types import Doc, Query
from benchmark.models.qwen3vl_embed import Qwen3VLEmbedRetriever


def _bare(dim=None) -> Qwen3VLEmbedRetriever:
    """Instancia sem __init__ (nao baixa pesos, nao importa torch/transformers)."""
    r = Qwen3VLEmbedRetriever.__new__(Qwen3VLEmbedRetriever)
    r.name = "qwen3vl:test"
    r.batch_size = 2
    r.dim = dim
    return r


def test_contract_denso(tmp_path):
    r = _bare()
    r._embed_queries = lambda ts: np.full((len(ts), 4), 2.0, dtype="float32")
    r._embed_images = lambda ps: np.full((len(ps), 4), 3.0, dtype="float32")

    qe = r.encode_queries([Query(id="q", text="meio ambiente")])
    de = r.encode_corpus([Doc(id="d", image_path=str(tmp_path / "x.png")),
                          Doc(id="d2", image_path=str(tmp_path / "y.png"))])

    assert isinstance(qe, np.ndarray) and qe.shape == (1, 4)
    assert isinstance(de, np.ndarray) and de.shape == (2, 4)
    # saida L2-normalizada nos dois lados
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


def test_postprocess_normaliza_lista_de_vetores():
    out = Qwen3VLEmbedRetriever._stack([np.array([3.0, 4.0]), np.array([0.0, 2.0])])
    assert out.shape == (2, 2) and out.dtype == np.float32
