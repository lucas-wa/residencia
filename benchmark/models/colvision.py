"""ColPali / ColQwen2 late-interaction retriever (multi-vector, one array per item).

`encode_queries` / `encode_corpus` return `list[np.ndarray]` of shape [n_tokens, dim];
ranking is done by `benchmark.common.index.maxsim_search` through
`runner.evaluate(..., multi_vector=True)`.

`_embed_queries` / `_embed_images` stay instance methods on purpose: the contract test
replaces them to exercise the wrapper without downloading weights.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from benchmark.common.types import Doc, Query


class ColVisionRetriever:
    def __init__(self, model_id: str, kind: Literal["colpali", "colqwen2"],
                 device: str = "cuda", dtype: str = "bfloat16", batch_size: int = 4):
        import torch
        # TODO confirmar em runtime: os nomes das classes mudam entre releases do
        # colpali-engine (0.3.x expoe ColPali/ColPaliProcessor/ColQwen2/ColQwen2Processor;
        # releases mais novas renomeiam para ColIdefics*/ColQwen2_5*). Conferir
        # `dir(colpali_engine.models)` antes da primeira rodada e ajustar o mapa abaixo.
        from colpali_engine.models import (
            ColPali,
            ColPaliProcessor,
            ColQwen2,
            ColQwen2Processor,
        )

        self.name = f"{kind}:{model_id}"
        self.kind = kind
        self.device = device
        self.batch_size = batch_size
        td = getattr(torch, dtype)
        model_cls, proc_cls = (
            (ColPali, ColPaliProcessor) if kind == "colpali" else (ColQwen2, ColQwen2Processor)
        )
        try:
            model = model_cls.from_pretrained(
                model_id, torch_dtype=td, device_map=device,
                attn_implementation="flash_attention_2",
            )
        except Exception:  # flash-attn ausente -> fallback obrigatorio
            model = model_cls.from_pretrained(
                model_id, torch_dtype=td, device_map=device,
                attn_implementation="eager",
            )
        self._model = model.eval()
        self._proc = proc_cls.from_pretrained(model_id)

    # -- embedding --------------------------------------------------------------

    @staticmethod
    def _unbatch(embeddings, attention_mask=None) -> list[np.ndarray]:
        """Split a [B, n_tokens, dim] tensor into per-item arrays, dropping padding."""
        out = []
        arr = embeddings.to("cpu").float().numpy()
        mask = attention_mask.to("cpu").numpy() if attention_mask is not None else None
        for i in range(arr.shape[0]):
            item = arr[i]
            if mask is not None and mask.ndim == 2 and mask.shape[1] == item.shape[0]:
                item = item[mask[i].astype(bool)]
            out.append(np.ascontiguousarray(item, dtype="float32"))
        return out

    def _embed_queries(self, queries: list[Query]) -> list[np.ndarray]:
        import torch

        texts = [q.text or "" for q in queries]
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = self._proc.process_queries(texts[start:start + self.batch_size])
            batch = {k: v.to(self._model.device) for k, v in batch.items()}
            with torch.no_grad():
                embeddings = self._model(**batch)
            out.extend(self._unbatch(embeddings, batch.get("attention_mask")))
        return out

    def _embed_images(self, paths: list[str]) -> list[np.ndarray]:
        import torch
        from PIL import Image

        out: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[start:start + self.batch_size]]
            batch = self._proc.process_images(images)
            batch = {k: v.to(self._model.device) for k, v in batch.items()}
            with torch.no_grad():
                embeddings = self._model(**batch)
            out.extend(self._unbatch(embeddings, batch.get("attention_mask")))
            for im in images:
                im.close()
        return out

    # -- Retriever protocol -----------------------------------------------------

    def encode_queries(self, queries: list[Query]) -> list[np.ndarray]:
        return self._embed_queries(queries)

    def encode_corpus(self, docs: list[Doc]) -> list[np.ndarray]:
        return self._embed_images([d.image_path or "" for d in docs])

    def score(self, q_emb, d_emb):                 # nao usado: runner usa maxsim_search
        raise NotImplementedError("use runner.evaluate(..., multi_vector=True)")
