import csv
from pathlib import Path

from benchmark.common.runner import evaluate, write_results_csv
from benchmark.common.types import Doc, Query
from benchmark.models.dummy import DummyRetriever


def _fixture():
    docs = [Doc(id="a", text="imposto de renda"),
            Doc(id="b", text="meio ambiente"),
            Doc(id="c", text="educacao basica")]
    queries = [Query(id="q1", text="imposto de renda"),
               Query(id="q2", text="educacao basica")]
    qrels = {"q1": {"a": 1.0}, "q2": {"c": 1.0, "zzz": 1.0}}  # 'zzz' orfao
    return queries, docs, qrels


def test_evaluate_perfect_dummy():
    q, d, qr = _fixture()
    m = evaluate(DummyRetriever(), q, d, qr, k_values=(1, 5))
    assert m["recall@1"] == 1.0
    assert m["ndcg@5"] == 1.0
    assert m["mrr"] == 1.0


def test_write_csv(tmp_path: Path):
    out = tmp_path / "r.csv"
    write_results_csv(out, {"name": "x", "model_id": "dummy"}, {"recall@5": 0.5})
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["name"] == "x"
    assert rows[0]["recall@5"] == "0.5"
