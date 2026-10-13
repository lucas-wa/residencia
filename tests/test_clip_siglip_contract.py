import numpy as np

from benchmark.common.types import Doc, Query
from benchmark.models.clip_siglip import ClipSiglipRetriever


def test_contract(monkeypatch, tmp_path):
    r = ClipSiglipRetriever.__new__(ClipSiglipRetriever)
    r.name = "clip:test"
    r._embed_text = lambda ts: np.ones((len(ts), 4), dtype="float32")
    r._embed_image = lambda ps: np.ones((len(ps), 4), dtype="float32")
    qe = r.encode_queries([Query(id="q", text="um gato")])
    de = r.encode_corpus([Doc(id="d", image_path=str(tmp_path / "x.png"))])
    assert r.score(qe, de).shape == (1, 1)
