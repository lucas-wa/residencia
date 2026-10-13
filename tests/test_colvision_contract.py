import numpy as np

from benchmark.common.types import Doc, Query
from benchmark.models.colvision import ColVisionRetriever


def test_contract(tmp_path):
    r = ColVisionRetriever.__new__(ColVisionRetriever)
    r.name = "colqwen2:test"
    r.batch_size = 2
    r._embed_queries = lambda qs: [np.ones((3, 8), dtype="float32") for _ in qs]
    r._embed_images = lambda ps: [np.ones((5, 8), dtype="float32") for _ in ps]
    qe = r.encode_queries([Query(id="q", text="saude")])
    de = r.encode_corpus([Doc(id="d", image_path=str(tmp_path / "p.png"))])
    assert isinstance(qe, list) and qe[0].shape == (3, 8)
    assert isinstance(de, list) and de[0].shape == (5, 8)
