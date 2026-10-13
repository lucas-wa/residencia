from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from benchmark.common.types import Doc, Query


@runtime_checkable
class Retriever(Protocol):
    """Common interface for every model wrapper.

    Dense single-vector retrievers return `np.ndarray` [n, dim]; multi-vector
    late-interaction retrievers return `list[np.ndarray]` (one [n_tokens, dim] per item)
    and are evaluated through `runner.evaluate(..., multi_vector=True)`.
    """

    name: str

    def encode_queries(self, queries: list[Query]) -> Any: ...

    def encode_corpus(self, docs: list[Doc]) -> Any: ...

    def score(self, q_emb: Any, d_emb: Any) -> Any: ...
