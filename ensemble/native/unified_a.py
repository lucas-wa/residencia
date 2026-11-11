"""Baseline A do ensemble v2: retriever unificado (Qwen3-VL) num unico espaco cosine em memoria."""
from __future__ import annotations

import numpy as np

from ensemble.native.specialist import Embedder, _l2_normalize
from ensemble.records import PointerRecord


def a_repr(record: PointerRecord) -> str:
    """Representacao-A: meta["a_repr"] -> meta["path"] -> descriptor."""
    return record.meta.get("a_repr") or record.meta.get("path") or record.descriptor


class UnifiedRetrieverA:
    def __init__(
        self,
        records: list[PointerRecord],
        *,
        embed_items: Embedder,
        embed_query: Embedder,
    ) -> None:
        self._embed_query = embed_query
        self._ids = [r.id for r in records]
        if self._ids:
            self._matrix = _l2_normalize(embed_items([a_repr(r) for r in records]))
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Ranking global (todas as modalidades); o driver filtra por eixo."""
        if not self._ids:
            return []
        q = _l2_normalize(self._embed_query([query]))[0]
        scores = self._matrix @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[i], float(scores[i])) for i in order]
