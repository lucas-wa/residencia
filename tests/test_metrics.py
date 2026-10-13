import math

from benchmark.common.metrics import aggregate, mrr, ndcg_at_k, recall_at_k


def test_recall_basic():
    assert recall_at_k(["a", "b", "c"], {"b"}, 5) == 1.0
    assert recall_at_k(["a", "b", "c"], {"z"}, 5) == 0.0          # relevante fora do corpus
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 1) == 0.5     # so 1 dos 2 no top-1


def test_recall_empty_relevant_is_zero_not_nan():
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_k_larger_than_ranking_no_error():
    assert recall_at_k(["a", "b"], {"b"}, 10) == 1.0


def test_ndcg_perfect_and_order():
    assert ndcg_at_k(["a", "b"], {"a": 1.0, "b": 1.0}, 2) == 1.0
    # relevante na 2a posicao: DCG=1/log2(3), IDCG=1 -> ~0.6309
    assert math.isclose(ndcg_at_k(["x", "a"], {"a": 1.0}, 2), 1 / math.log2(3), rel_tol=1e-6)


def test_ndcg_relevant_absent_is_zero():
    assert ndcg_at_k(["x", "y"], {"a": 1.0}, 2) == 0.0


def test_mrr():
    assert mrr(["x", "a"], {"a"}) == 0.5
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_aggregate_mean():
    out = aggregate([{"recall@5": 1.0}, {"recall@5": 0.0}])
    assert out["recall@5"] == 0.5
