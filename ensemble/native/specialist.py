"""Especialistas do ensemble v2: protocolo + especialista denso (cosine em memoria)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from ensemble.records import PointerRecord

Embedder = Callable[[list[str]], np.ndarray]


@runtime_checkable
class Specialist(Protocol):
    modality: str

    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...


def item_repr(record: PointerRecord) -> str:
    """String enviada a embed_items: texto -> descriptor; demais -> meta["path"]."""
    if record.modality == "texto":
        return record.descriptor
    try:
        return record.meta["path"]
    except KeyError:
        raise KeyError(f"item {record.id} ({record.modality}) sem meta['path']") from None


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0.0, 1.0, norms)


class DenseSpecialist:
    def __init__(
        self,
        modality: str,
        records: list[PointerRecord],
        *,
        embed_items: Embedder,
        embed_query: Embedder,
    ) -> None:
        self.modality = modality
        self._embed_query = embed_query
        self._ids = [r.id for r in records if r.modality == modality]
        if self._ids:
            items = [item_repr(r) for r in records if r.modality == modality]
            self._matrix = _l2_normalize(embed_items(items))
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = _l2_normalize(self._embed_query([query]))[0]
        scores = self._matrix @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[i], float(scores[i])) for i in order]
