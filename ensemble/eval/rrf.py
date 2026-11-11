"""Reciprocal Rank Fusion de listas ranqueadas."""
from __future__ import annotations


def rrf_fuse(ranked_lists: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Funde listas de item_ids (ordem de rank, 1-based). Deduplica somando 1/(k+rank).

    Desempate deterministico: ordem da primeira aparicao.
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    # sorted e estavel: empates mantem a ordem de insercao do dict
    return sorted(scores.items(), key=lambda kv: -kv[1])
