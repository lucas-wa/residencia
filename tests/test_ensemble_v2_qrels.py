from pathlib import Path

import pytest

from ensemble.eval.qrels import load_qrels, load_queries, validate_qrels

FIX = Path(__file__).parent / "fixtures" / "ensemble_v2"


def test_load_queries_preserva_acentos():
    q = load_queries(FIX / "queries.tsv")
    assert q["qid1"] == "orçamento da saúde"
    assert q["q2"] == "gastos com educação básica"


def test_load_qrels_binario():
    qrels = load_qrels(FIX / "qrels.txt")
    assert qrels["q1"] == {"itemA"}


def test_validate_item_inexistente_levanta():
    with pytest.raises(ValueError, match="itemZ"):
        validate_qrels({"q1": {"itemZ"}}, {"q1": "x"}, {"itemA"})


def test_query_sem_relevante_e_permitida():
    qrels = load_qrels(FIX / "qrels.txt")
    assert qrels["q2"] == set()
    validate_qrels(qrels, {"q1": "a", "q2": "b"}, {"itemA", "itemB", "itemC"})
