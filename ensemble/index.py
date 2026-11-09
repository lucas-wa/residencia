"""Indice textual unico em memoria: descritores embutidos uma vez, busca por cosine."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ensemble.records import PointerRecord

Embedder = Callable[[list[str]], np.ndarray]


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0.0, 1.0, norms)


@dataclass
class EnsembleIndex:
    records: list[PointerRecord]
    matrix: np.ndarray  # [n, d], linhas L2-normalizadas

    def search(
        self, query: str, k: int = 10, *, embedder: Embedder
    ) -> list[tuple[PointerRecord, float]]:
        if not query or not query.strip():
            raise ValueError("query vazia")
        q = _l2_normalize(embedder([query]))[0]
        scores = self.matrix @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.records[i], float(scores[i])) for i in order]


def build_index(records: list[PointerRecord], *, embedder: Embedder) -> EnsembleIndex:
    if not records:
        raise ValueError("corpus vazio")
    matrix = _l2_normalize(embedder([r.descriptor for r in records]))
    return EnsembleIndex(records=list(records), matrix=matrix)
