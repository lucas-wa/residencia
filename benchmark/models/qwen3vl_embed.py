"""Qwen3-VL-Embedding -- retriever DENSO (um vetor por item), licenca Apache-2.0.

Query = texto; documento = imagem da pagina (`Doc.image_path`). Os dois lados saem
L2-normalizados, entao `score` e um produto interno -- `runner.evaluate(...,
multi_vector=False)` indexa com FAISS/numpy.

Checkpoint: `Qwen/Qwen3-VL-Embedding-8B` (ha 2B/4B na mesma familia).
Repo de referencia: github.com/QwenLM/Qwen3-VL-Embedding.

`dim` ativa Matryoshka (MRL): o vetor e truncado para `dim` e renormalizado. `None`
mantem a dimensao nativa do checkpoint.

`_embed_queries` / `_embed_images` sao metodos de instancia de proposito: o teste de
contrato os substitui para exercitar o wrapper sem baixar pesos.
"""

from __future__ import annotations

import numpy as np

from benchmark.common.types import Doc, Query
from benchmark.models._remote_api import call_batch, first_method

# TODO confirmar em runtime: a API oficial de encode do Qwen3-VL-Embedding vem do custom
# code do checkpoint (o repo QwenLM expoe algo como `encode_query`/`encode_document`;
# versoes empacotadas via sentence-transformers usam `encode`). Tentamos os nomes
# conhecidos em ordem e, se nenhum existir, caimos no caminho generico
# processor + `model(**inputs)` com pooling do ultimo token (padrao dos embedders Qwen).
_QUERY_FNS = ("encode_query", "embed_query", "get_query_embeddings", "encode_queries")
_IMAGE_FNS = ("encode_document", "embed_document", "get_doc_embeddings", "encode_images",
              "encode_corpus")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-9)


class Qwen3VLEmbedRetriever:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16",
                 batch_size: int = 8, dim: int | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.name = f"qwen3vl:{model_id}"
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.dim = dim
        self.torch_dtype = getattr(torch, dtype)
        try:
            model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=self.torch_dtype,
                device_map=device, attn_implementation="flash_attention_2",
            )
        except Exception:  # flash-attn ausente -> fallback obrigatorio
            model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=self.torch_dtype,
                device_map=device, attn_implementation="eager",
            )
        self._model = model.eval()
        self._proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # -- normalizacao da saida ---------------------------------------------------

    @staticmethod
    def _as_numpy(x) -> np.ndarray:
        """Tensor (torch) ou array -> np.float32, sem exigir torch importado."""
        if hasattr(x, "detach"):
            x = x.detach().to("cpu").float().numpy()
        return np.asarray(x, dtype="float32")

    @classmethod
    def _stack(cls, out) -> np.ndarray:
        """Saida de um batch -> [n, dim] float32 (aceita lista de vetores ou matriz)."""
        if isinstance(out, (list, tuple)):
            rows = [np.atleast_2d(cls._as_numpy(t)) for t in out]
            return np.ascontiguousarray(np.vstack(rows), dtype="float32")
        return np.ascontiguousarray(np.atleast_2d(cls._as_numpy(out)), dtype="float32")

    def _postprocess(self, emb: np.ndarray) -> np.ndarray:
        """Trunca para `self.dim` (MRL) e L2-normaliza."""
        emb = np.atleast_2d(np.asarray(emb, dtype="float32"))
        if self.dim is not None:
            emb = emb[:, : self.dim]
        return _l2_normalize(emb)

    # -- embedding ---------------------------------------------------------------

    @staticmethod
    def _pool(hidden, attention_mask=None):
        """Pooling do ultimo token nao-pad (padrao dos embedders decoder-only Qwen)."""
        if attention_mask is None:
            return hidden[:, -1]
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[range(hidden.shape[0]), lengths]

    def _forward_generic(self, inputs) -> np.ndarray:
        """Caminho generico: processor + forward + pooling, quando o card nao expoe API."""
        import torch

        inputs = {k: v.to(self._model.device) for k, v in dict(inputs).items()}
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        emb = getattr(out, "embeddings", None)
        if emb is None:
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out.hidden_states[-1]
            emb = self._pool(hidden, inputs.get("attention_mask"))
        return self._stack(emb)

    def _embed_queries(self, texts: list[str]) -> np.ndarray:
        import torch

        fn = first_method(self._model, _QUERY_FNS)
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            if fn is not None:
                with torch.no_grad():
                    out.append(self._stack(call_batch(fn, batch)))
            else:
                inputs = self._proc(text=batch, return_tensors="pt", padding=True,
                                    truncation=True)
                out.append(self._forward_generic(inputs))
        return np.vstack(out) if out else np.zeros((0, 1), dtype="float32")

    def _embed_images(self, paths: list[str]) -> np.ndarray:
        import torch
        from PIL import Image

        fn = first_method(self._model, _IMAGE_FNS)
        out: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[start:start + self.batch_size]]
            try:
                if fn is not None:
                    with torch.no_grad():
                        out.append(self._stack(call_batch(fn, images)))
                else:
                    # TODO confirmar em runtime: o processor do Qwen3-VL costuma exigir o
                    # chat template (`apply_chat_template` + `qwen_vl_utils.process_vision_info`)
                    # em vez de `images=`. Conferir o card antes da primeira rodada.
                    inputs = self._proc(images=images, return_tensors="pt")
                    out.append(self._forward_generic(inputs))
            finally:
                for im in images:
                    im.close()
        return np.vstack(out) if out else np.zeros((0, 1), dtype="float32")

    # -- Retriever protocol ------------------------------------------------------

    def encode_queries(self, queries: list[Query]) -> np.ndarray:
        return self._postprocess(self._embed_queries([q.text or "" for q in queries]))

    def encode_corpus(self, docs: list[Doc]) -> np.ndarray:
        return self._postprocess(self._embed_images([d.image_path or "" for d in docs]))

    def score(self, q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
        return q_emb @ d_emb.T
