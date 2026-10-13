import numpy as np

from benchmark.common.types import Doc, Query
from benchmark.models.dummy import DummyRetriever


def test_dummy_scores_shape_and_selfmatch():
    r = DummyRetriever(dim=16)
    docs = [Doc(id="a", text="imposto de renda"), Doc(id="b", text="meio ambiente")]
    qs = [Query(id="q1", text="imposto de renda")]
    qe = r.encode_queries(qs)
    de = r.encode_corpus(docs)
    s = r.score(qe, de)
    assert s.shape == (1, 2)
    assert np.argmax(s[0]) == 0           # query casa com doc 'a'
