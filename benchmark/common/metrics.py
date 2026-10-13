"""Pure-python IR metrics (CPU only, no model/array dependencies).

All functions degrade gracefully on the edge cases listed in the plan's Review Focus:
- a relevant doc missing from the ranking (or from the corpus) simply scores 0;
- a query with no relevant doc returns 0.0 instead of NaN;
- `k` larger than the ranking truncates without IndexError.
"""

from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    topk = ranked_ids[:k]
    hits = sum(1 for d in topk if d in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(ranked_ids: list[str], relevant: dict[str, float], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(ranked_ids[:k]):
        gain = relevant.get(d, 0.0)
        if gain:
            dcg += gain / math.log2(i + 2)
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for i, d in enumerate(ranked_ids):
        if d in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
