"""Baseline A do ensemble v2: retriever unificado (Qwen3-VL) num unico espaco cosine em memoria.

O A e MULTIMODAL: o Qwen3-VL-Embedding poe imagem e texto no MESMO espaco, entao cada item
entra pela rota da sua modalidade (ver `a_route`) -- imagem como imagem, texto como texto.
A query e sempre texto (recuperacao cross-modal texto->imagem no mesmo espaco).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ensemble.native.specialist import Embedder, _l2_normalize
from ensemble.records import PointerRecord

# O embedder de ITENS do A nao recebe strings cruas, e sim pares `(rota, valor)` -- a
# rota diz por qual lado do Qwen3-VL o valor entra (`embed_images`/`embed_texts`).
ItemEmbedderA = Callable[[list[tuple[str, str]]], np.ndarray]

# Modalidades cujo `a_repr` e um CAMINHO de imagem (pagina PNG do documento, foto,
# keyframe da janela de video): entram pela rota de imagem.
A_IMAGE_MODALITIES: frozenset[str] = frozenset({"documento", "imagem", "video"})


def a_repr(record: PointerRecord) -> str:
    """Representacao-A: meta["a_repr"] -> meta["path"] -> descriptor."""
    return record.meta.get("a_repr") or record.meta.get("path") or record.descriptor


def a_route(record: PointerRecord) -> str:
    """Rota de embedding do item: `"image"` ou `"text"`.

    `documento`/`imagem`/`video` -> `"image"` (o `a_repr` e o caminho do arquivo);
    `texto`/`audio` -> `"text"` (o `a_repr` e o texto real: artigo ou transcricao ASR).
    """
    return "image" if record.modality in A_IMAGE_MODALITIES else "text"


class UnifiedRetrieverA:
    def __init__(
        self,
        records: list[PointerRecord],
        *,
        embed_items: ItemEmbedderA,
        embed_query: Embedder,
    ) -> None:
        self._embed_query = embed_query
        self._ids = [r.id for r in records]
        if self._ids:
            # `embed_items` recebe pares (rota, valor) e devolve `[n, dim]` na ordem dos
            # records; quem roteia e o embedder (ver `build_unified_a_embedder`).
            itens = [(a_route(r), a_repr(r)) for r in records]
            self._matrix = _l2_normalize(embed_items(itens))
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
