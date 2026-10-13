"""pt-BR text retriever (BGE-M3 / multilingual-e5 / Serafim) via sentence-transformers."""

from __future__ import annotations

import numpy as np

from benchmark.common.types import Doc, Query


class TextRetriever:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16",
                 query_prefix: str = "", doc_prefix: str = "", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # import tardio

        self.name = f"text:{model_id}"
        self.model = SentenceTransformer(model_id, device=device)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.batch_size = batch_size

    def _encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")

    def encode_queries(self, queries: list[Query]) -> np.ndarray:
        return self._encode([self.query_prefix + (q.text or "") for q in queries])

    def encode_corpus(self, docs: list[Doc]) -> np.ndarray:
        return self._encode([self.doc_prefix + (d.text or "") for d in docs])

    def score(self, q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
        return q_emb @ d_emb.T
