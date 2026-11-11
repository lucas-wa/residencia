"""Metricas de recuperacao com relevancia binaria."""
from __future__ import annotations

import math


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(ranked_ids[:k])
        if item in relevant_ids
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for i, item in enumerate(ranked_ids, start=1):
        if item in relevant_ids:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    hits = sum(1 for item in set(ranked_ids[:k]) if item in relevant_ids)
    return hits / len(relevant_ids)
