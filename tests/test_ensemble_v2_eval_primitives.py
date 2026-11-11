import math

from ensemble.eval.metrics import mrr, ndcg_at_k, recall_at_k
from ensemble.eval.rrf import rrf_fuse


def test_rrf_soma_contribuicoes_e_dedup():
    out = rrf_fuse([["a", "b"], ["c", "a"]], k=60)
    ids = [i for i, _ in out]
    assert ids.count("a") == 1
    assert dict(out)["a"] == 1 / 61 + 1 / 62


def test_rrf_ordena_desc_e_tie_deterministico():
    out = rrf_fuse([["a", "b"], ["c", "d"]], k=60)
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)
    assert [i for i, _ in out] == ["a", "c", "b", "d"]
    assert out == rrf_fuse([["a", "b"], ["c", "d"]], k=60)


def test_rrf_listas_vazias():
    assert rrf_fuse([[], []]) == []


def test_ndcg_valor_conhecido():
    assert math.isclose(ndcg_at_k(["x", "r", "y"], {"r"}, 10), 1 / math.log2(3))


def test_mrr_primeiro_relevante():
    assert mrr(["x", "r", "y"], {"r"}) == 0.5


def test_recall_at_k():
    assert recall_at_k(["r1", "x", "r2"], {"r1", "r2"}, 2) == 0.5


def test_metricas_sem_relevante_retorna_zero():
    r = ["a", "b"]
    assert ndcg_at_k(r, set(), 10) == 0.0
    assert mrr(r, set()) == 0.0
    assert recall_at_k(r, set(), 10) == 0.0
