import numpy as np

from benchmark.common.index import dense_search, maxsim_search


def test_dense_search_matches_manual():
    d = np.eye(4, dtype="float32")              # 4 docs ortonormais
    q = np.array([[0.9, 0.1, 0, 0]], dtype="float32")
    assert dense_search(q, d, k=2)[0][0] == 0   # mais perto do doc 0


def test_dense_k_truncates():
    d = np.eye(3, dtype="float32")
    q = np.ones((1, 3), dtype="float32")
    assert len(dense_search(q, d, k=10)[0]) == 3


def test_maxsim_selfmatch():
    # doc 0 contem um token identico ao token da query
    q = [np.array([[1.0, 0, 0]], dtype="float32")]
    docs = [np.array([[1.0, 0, 0], [0, 1, 0]], dtype="float32"),
            np.array([[0, 0, 1.0]], dtype="float32")]
    assert maxsim_search(q, docs, k=2)[0][0] == 0
