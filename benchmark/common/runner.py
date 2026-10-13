"""Evaluation loop: encode -> search -> metrics -> CSV row."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from benchmark.common.index import dense_search, maxsim_search
from benchmark.common.metrics import aggregate, mrr, ndcg_at_k, recall_at_k
from benchmark.common.types import Doc, Query


def evaluate(retriever, queries: list[Query], docs: list[Doc],
             qrels: dict[str, dict[str, float]], k_values=(1, 5, 10),
             multi_vector: bool = False) -> dict[str, float]:
    """Run `retriever` over (queries, docs) and return the averaged metrics.

    Orphan qrels -- relevant doc ids that are not in `docs` -- are dropped before
    scoring, so a corpus subset never penalises the model for documents it was
    never shown. Queries left without any relevant doc score 0.0 (never NaN).
    """
    q_emb = retriever.encode_queries(queries)
    d_emb = retriever.encode_corpus(docs)
    maxk = max(k_values)
    if multi_vector:
        ranked = maxsim_search(q_emb, d_emb, k=max(maxk, len(docs)))
    else:
        ranked = dense_search(np.asarray(q_emb), np.asarray(d_emb), k=max(maxk, len(docs)))
    doc_ids = [d.id for d in docs]
    known_ids = set(doc_ids)
    per_query = []
    for qi, q in enumerate(queries):
        ranked_ids = [doc_ids[i] for i in ranked[qi]]
        rel = {d: s for d, s in qrels.get(q.id, {}).items() if d in known_ids}
        rel_set = set(rel.keys())
        row = {}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(ranked_ids, rel_set, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, rel, k)
        row["mrr"] = mrr(ranked_ids, rel_set)
        per_query.append(row)
    return aggregate(per_query)


def write_results_csv(path, config_dict: dict, metrics: dict) -> None:
    """Append one row (config + metrics) to `path`, writing the header if new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**config_dict, **metrics}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
