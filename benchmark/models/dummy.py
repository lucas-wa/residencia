from __future__ import annotations

import hashlib

import numpy as np

from benchmark.common.types import Doc, Query


class DummyRetriever:
    """Deterministic hash-based retriever: identical texts get identical embeddings.

    Used to exercise the runner/index on CPU without downloading any model.
    """

    name = "dummy"

    def __init__(self, dim: int = 16):
        self.dim = dim

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            h = hashlib.sha256((t or "").encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            v = rng.standard_normal(self.dim)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.vstack(vecs).astype("float32")

    def encode_queries(self, queries: list[Query]) -> np.ndarray:
        return self._embed([q.text for q in queries])

    def encode_corpus(self, docs: list[Doc]) -> np.ndarray:
        return self._embed([d.text for d in docs])

    def score(self, q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
        return q_emb @ d_emb.T
