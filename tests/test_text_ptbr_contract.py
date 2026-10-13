import numpy as np

from benchmark.common.types import Doc, Query
from benchmark.models.text_ptbr import TextRetriever


class _FakeST:
    def encode(self, texts, **kw):
        return np.ones((len(texts), 8), dtype="float32")


def test_contract_with_fake_encoder(monkeypatch):
    r = TextRetriever.__new__(TextRetriever)      # sem __init__ (sem download)
    r.name = "text:test"
    r.model = _FakeST()
    r.query_prefix = "query: "
    r.doc_prefix = "passage: "
    qe = r.encode_queries([Query(id="q", text="oi")])
    de = r.encode_corpus([Doc(id="d", text="tchau")])
    s = r.score(qe, de)
    assert qe.shape == (1, 8) and s.shape == (1, 1)
