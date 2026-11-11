"""Especialistas do ensemble v2: protocolo + especialista denso (cosine em memoria)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from ensemble.records import PointerRecord

log = logging.getLogger(__name__)

Embedder = Callable[[list[str]], np.ndarray]


@runtime_checkable
class Specialist(Protocol):
    modality: str

    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...


def item_repr(record: PointerRecord) -> str:
    """String enviada a embed_items, por modalidade:

    - texto -> descriptor
    - imagem, audio -> meta["path"] (binario por item)
    - video -> meta["a_repr"] (keyframe POR JANELA): meta["path"] e o video
      inteiro, compartilhado por todas as janelas, logo nao discrimina janelas;
      o keyframe e o artefato discriminativo. Sem a_repr, cai para meta["path"]
      com um warning (essa janela nao sera discriminativa, mas nao quebra).
    """
    if record.modality == "texto":
        return record.descriptor
    if record.modality == "video":
        a_repr = record.meta.get("a_repr")
        if a_repr:
            return a_repr
        log.warning(
            "item %s (video) sem meta['a_repr']: usando meta['path'] "
            "(janela nao discriminativa)",
            record.id,
        )
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
        repr_fn: Callable[[PointerRecord], str] = item_repr,
    ) -> None:
        self.modality = modality
        self._embed_query = embed_query
        self._repr_fn = repr_fn
        self._ids = [r.id for r in records if r.modality == modality]
        if self._ids:
            items = [self._repr_fn(r) for r in records if r.modality == modality]
            matrix = _l2_normalize(embed_items(items))
            if matrix.shape[0] != len(self._ids):
                raise ValueError(
                    f"{len(self._ids)} itens para {matrix.shape[0]} embeddings"
                )
            self._matrix = matrix
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = _l2_normalize(self._embed_query([query]))[0]
        scores = self._matrix @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[i], float(scores[i])) for i in order]
