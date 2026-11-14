"""Nucleo da API de consulta: cosine top-k contra a matriz de itens ja embeddada.

Diferenca para `ensemble.native.unified_a.UnifiedRetrieverA`: la o retriever EMBEDDA os
itens no __init__ (precisa de GPU e dos binarios do corpus); aqui a matriz chega PRONTA,
de `api.index_io.load_index` -- o servico HTTP so embedda a query. A metrica e a mesma
(`_l2_normalize` + `@` + argsort estavel), reusada de `ensemble.native.specialist`, para
o ranking servido ser identico ao medido no benchmark.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ensemble.native.specialist import _l2_normalize
from ensemble.records import PointerRecord

# A query e sempre TEXTO (recuperacao cross-modal texto->item no espaco unico do A).
QueryEmbedder = Callable[[list[str]], np.ndarray]


class ConsultaRetriever:
    """Top-k cosine de uma query de texto contra os itens do indice.

    `item_matrix` e `[n, dim]` na ORDEM dos `records` (linha `i` = embedding-A do record
    `i`); vem L2-normalizada do `build_index`, mas e normalizada de novo aqui -- custa
    nada e e a mesma garantia que o `UnifiedRetrieverA` da.
    """

    def __init__(self, records: Sequence[PointerRecord], item_matrix: np.ndarray, *,
                 embed_query: QueryEmbedder) -> None:
        records = list(records)
        matrix = np.asarray(item_matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"item_matrix precisa ser [n, dim]; recebi shape {matrix.shape}")
        if matrix.shape[0] != len(records):
            raise ValueError(
                f"{len(records)} records para {matrix.shape[0]} linhas de item_matrix "
                "(a linha i tem de ser o embedding do record i)")

        self._embed_query = embed_query
        self._records = records
        self._matrix = _l2_normalize(matrix) if records else matrix

    def __len__(self) -> int:
        return len(self._records)

    def search(self, query: str, k: int) -> list[dict]:
        """Ranking global (todos os eixos), no formato do contrato HTTP.

        Devolve ate `k` dicts `{item_id, eixo, score, pointer_uri, descriptor}`, ordenados
        por score decrescente. `eixo` e a `modality` do record. Indice vazio -> `[]`.
        """
        if not self._records:
            return []
        q = _l2_normalize(self._embed_query([query]))[0]
        scores = self._matrix @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [self._item(int(i), float(scores[i])) for i in order]

    def _item(self, i: int, score: float) -> dict:
        r = self._records[i]
        return {
            "item_id": r.id,
            "eixo": r.modality,
            "score": score,
            "pointer_uri": r.pointer_uri,
            "descriptor": r.descriptor,
        }
