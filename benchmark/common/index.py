"""Search backends: dense single-vector (faiss or numpy) and MaxSim late interaction.

`dense_search` prefers faiss (gpu or cpu wheel) and falls back to a plain numpy
inner-product ranking that yields the same top-k. Ties are broken by document order
(stable sort) so the numpy backend stays deterministic.
"""

from __future__ import annotations

import numpy as np


def dense_search(q_emb: np.ndarray, d_emb: np.ndarray, k: int) -> list[list[int]]:
    """Top-k document indices per query, by inner product (higher = more similar)."""
    k = min(k, d_emb.shape[0])
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(d_emb.shape[1])
        index.add(np.ascontiguousarray(d_emb, dtype="float32"))
        _, idx = index.search(np.ascontiguousarray(q_emb, dtype="float32"), k)
        return idx.tolist()
    except Exception:
        sims = q_emb @ d_emb.T
        return np.argsort(-sims, axis=1, kind="stable")[:, :k].tolist()


def maxsim_search(q_mv: list[np.ndarray], d_mv: list[np.ndarray], k: int) -> list[list[int]]:
    """Late interaction: score(q, d) = sum_i max_j (q_i . d_j), one array per item."""
    k = min(k, len(d_mv))
    out = []
    for q in q_mv:
        scores = []
        for d in d_mv:
            sim = q @ d.T                      # [nq_tok, nd_tok]
            scores.append(float(sim.max(axis=1).sum()))
        order = np.argsort(-np.array(scores), kind="stable")[:k]
        out.append(order.tolist())
    return out
