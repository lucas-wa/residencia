import numpy as np

from benchmark.common.types import Doc, Query
from benchmark.models.colembed import ColEmbedRetriever


def _bare() -> ColEmbedRetriever:
    """Instancia sem __init__ (nao baixa pesos, nao importa torch/transformers)."""
    r = ColEmbedRetriever.__new__(ColEmbedRetriever)
    r.name = "colembed:test"
    r.batch_size = 2
    return r


def test_contract_multi_vector(tmp_path):
    r = _bare()
    r._embed_queries = lambda qs: [np.ones((3, 8), dtype="float32") for _ in qs]
    r._embed_images = lambda ps: [np.ones((5, 8), dtype="float32") for _ in ps]

    qe = r.encode_queries([Query(id="q", text="saude"), Query(id="q2", text="educacao")])
    de = r.encode_corpus([Doc(id="d", image_path=str(tmp_path / "p.png"))])

    assert isinstance(qe, list) and len(qe) == 2
    assert all(isinstance(x, np.ndarray) and x.shape == (3, 8) for x in qe)
    assert isinstance(de, list) and len(de) == 1
    assert de[0].shape == (5, 8)


def test_score_nao_e_usado():
    r = _bare()
    try:
        r.score(None, None)
    except NotImplementedError:
        return
    raise AssertionError("score deveria levantar NotImplementedError (runner usa maxsim)")


def test_encode_corpus_passa_os_image_paths():
    r = _bare()
    seen: list[list[str]] = []

    def fake_images(paths):
        seen.append(list(paths))
        return [np.zeros((2, 4), dtype="float32") for _ in paths]

    r._embed_images = fake_images
    r.encode_corpus([Doc(id="d1", image_path="a.png"), Doc(id="d2", image_path="b.png")])
    assert seen == [["a.png", "b.png"]]


def test_to_arrays_lista_de_tensores():
    out = ColEmbedRetriever._to_arrays([np.ones((4, 3)), np.ones((7, 3))])
    assert [a.shape for a in out] == [(4, 3), (7, 3)]
    assert all(a.dtype == np.float32 for a in out)


def test_to_arrays_batch_3d_com_mascara():
    batch = np.arange(2 * 4 * 3, dtype="float32").reshape(2, 4, 3)
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])
    out = ColEmbedRetriever._to_arrays(batch, attention_mask=mask)
    assert [a.shape for a in out] == [(2, 3), (3, 3)]
    assert np.allclose(out[0], batch[0][:2])


def test_to_arrays_batch_3d_sem_mascara():
    batch = np.zeros((3, 5, 2), dtype="float32")
    out = ColEmbedRetriever._to_arrays(batch)
    assert len(out) == 3 and out[0].shape == (5, 2)
