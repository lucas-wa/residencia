"""Valida so o FORMATO do gabarito proposto (nao o corpus real, que e gated)."""
from pathlib import Path

from ensemble.eval.qrels import load_qrels, load_queries, validate_qrels

EVAL = Path(__file__).resolve().parent.parent / "ensemble" / "eval"
QUERIES = EVAL / "queries_orcamento_v2.txt"
QRELS = EVAL / "qrels_orcamento.tsv"


def test_queries_parseiam_com_acentos():
    q = load_queries(QUERIES)
    assert 15 <= len(q) <= 30  # 16 queries base + 8 de imagem (q17-q24) = 24
    assert all(t.strip() for t in q.values())
    assert any("ç" in t or "ã" in t or "ú" in t for t in q.values())


def test_qrels_parseiam_e_validam_contra_corpus_fixture():
    q = load_queries(QUERIES)
    qrels = load_qrels(QRELS)
    assert qrels
    # corpus fixture = os item_ids placeholder usados no proprio arquivo
    fixture_corpus = set()
    for line in QRELS.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            fixture_corpus.add(line.split()[2])
    validate_qrels(qrels, q, fixture_corpus)
