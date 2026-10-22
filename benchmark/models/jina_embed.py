"""jina-embeddings-v4 -- retriever DENSO (um vetor por item), modo `retrieval`.

LICENCA (confirmado no card do HF em 2026-10-22): o repo saiu por engano como
CC-BY-NC-4.0 e o card corrige para a *Qwen Research License* (o modelo deriva do
Qwen2.5-VL-3B). As duas leituras sao NAO comerciais -> uso de PESQUISA (residencia),
NUNCA producao. Reconferir o card antes de qualquer uso oficial.

Query = texto; documento = imagem da pagina (`Doc.image_path`). Os dois lados saem
L2-normalizados, entao `score` e um produto interno (cosine) -- `runner.evaluate(...,
multi_vector=False)` indexa com `benchmark.common.index.dense_search`.

Checkpoint: `jinaai/jina-embeddings-v4` (custom code no repo do Hub -> `trust_remote_code=True`).

TODO multi-vetor: o jina-v4 tambem faz "late interaction" -- `encode_text`/`encode_image`
aceitam `return_multivector=True` e devolvem um tensor [n_tokens, dim] por item. Esta
primeira versao fica so no caminho denso; o modo multi-vetor pede `model_kind` proprio
(entrando em `factory._MULTI`, ranqueado por MaxSim) e nao cabe neste wrapper como esta.

`_embed_queries` / `_embed_images` sao metodos de instancia de proposito: o teste de
contrato os substitui para exercitar o wrapper sem baixar pesos.
"""

from __future__ import annotations

import inspect

import numpy as np

from benchmark.common.types import Doc, Query
from benchmark.models._remote_api import first_method

# API confirmada no card/modeling do checkpoint:
#   encode_text(texts, task=, max_length=, batch_size=, return_multivector=,
#               return_numpy=, truncate_dim=, prompt_name=)
#   encode_image(images, task=, batch_size=, return_multivector=, return_numpy=,
#                truncate_dim=, max_pixels=)
# Os nomes alternativos abaixo sao amortecedor para renomeacoes entre releases do custom
# code (mesmo espirito de `colembed.py`); `_call_encode` so passa os kwargs que a
# assinatura resolvida realmente aceita.
_TEXT_FNS = ("encode_text", "encode_texts", "encode_query", "embed_text")
_IMAGE_FNS = ("encode_image", "encode_images", "embed_image")
_PROC_TEXT_FNS = ("process_texts", "process_queries")
_PROC_IMAGE_FNS = ("process_images", "process_passages")

# `task` do jina-v4: "retrieval" | "text-matching" | "code". Aqui e sempre recuperacao.
_TASK = "retrieval"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-9)


def _call_encode(fn, items: list, **candidates):
    """Chama `fn(items, ...)` passando so os kwargs aceitos pela assinatura.

    `encode_text` e `encode_image` fazem o proprio batching, `eval()`, `no_grad()` e o
    `.to(device)` dentro de `_process_batches` -- por isso o wrapper nao precisa de torch
    neste caminho (o que tambem o deixa testavel sem baixar pesos).
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):   # builtins/C-ext sem assinatura introspectavel
        return fn(items)
    kwargs = {k: v for k, v in candidates.items() if k in params}
    return fn(items, **kwargs)


class JinaV4Retriever:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16",
                 batch_size: int = 8, dim: int | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.name = f"jinav4:{model_id}"
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        # `dim` ativa Matryoshka (MRL): 128/256/512/1024/2048 no card. `None` mantem a
        # dimensao nativa do checkpoint.
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
        """Saida de um batch -> [n, dim] float32.

        `encode_text`/`encode_image` devolvem uma lista de tensores [dim] (um por item);
        o caminho generico devolve a matriz [n, dim] de uma vez.
        """
        if isinstance(out, (list, tuple)):
            rows = [np.atleast_2d(cls._as_numpy(t)) for t in out]
            return np.ascontiguousarray(np.vstack(rows), dtype="float32")
        return np.ascontiguousarray(np.atleast_2d(cls._as_numpy(out)), dtype="float32")

    def _postprocess(self, emb: np.ndarray) -> np.ndarray:
        """Trunca para `self.dim` (MRL) e L2-normaliza.

        O proprio modelo ja devolve o vetor normalizado; renormalizamos porque o
        truncamento MRL quebra a norma e porque o caminho generico nao garante nada.
        """
        emb = np.atleast_2d(np.asarray(emb, dtype="float32"))
        if self.dim is not None:
            emb = emb[:, : self.dim]
        return _l2_normalize(emb)

    # -- embedding ---------------------------------------------------------------

    @staticmethod
    def _pool(hidden, attention_mask=None):
        """Mean pooling mascarado -- o que o `get_single_vector_embeddings` do jina faz."""
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def _forward_generic(self, inputs) -> np.ndarray:
        """Caminho generico: processor + forward + pooling, se a API de encode sumir."""
        import torch

        inputs = {k: v.to(self._model.device) for k, v in dict(inputs).items()}
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        # O forward do jina-v4 devolve `JinaEmbeddingsV4ModelOutput(vlm_last_hidden_states,
        # single_vec_emb, multi_vec_emb)`; so o denso interessa aqui.
        emb = getattr(out, "single_vec_emb", None)
        if emb is None:
            emb = getattr(out, "embeddings", None)
        if emb is None:
            hidden = getattr(out, "vlm_last_hidden_states", None)
            if hidden is None:
                hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out.hidden_states[-1]
            emb = self._pool(hidden, inputs.get("attention_mask"))
        return self._stack(emb)

    def _process(self, proc_fns: tuple[str, ...], items: list, text: bool):
        proc_fn = first_method(self._proc, proc_fns)
        if proc_fn is not None:
            return proc_fn(items)
        kwargs = {"text": items} if text else {"images": items}
        return self._proc(return_tensors="pt", padding=True, **kwargs)

    def _embed_queries(self, texts: list[str]) -> np.ndarray:
        fn = first_method(self._model, _TEXT_FNS)
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            if fn is not None:
                out.append(self._stack(_call_encode(
                    fn, batch, task=_TASK, prompt_name="query",
                    return_multivector=False, truncate_dim=self.dim,
                    batch_size=max(len(batch), 1),
                )))
            else:
                out.append(self._forward_generic(
                    self._process(_PROC_TEXT_FNS, batch, text=True)))
        return np.vstack(out) if out else np.zeros((0, 1), dtype="float32")

    def _embed_images(self, paths: list[str]) -> np.ndarray:
        fn = first_method(self._model, _IMAGE_FNS)
        out: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            batch = paths[start:start + self.batch_size]
            if fn is not None:
                # `encode_image` aceita caminho local, URL ou PIL.Image e faz o
                # `convert("RGB")` por dentro -- mandamos os caminhos crus.
                out.append(self._stack(_call_encode(
                    fn, batch, task=_TASK, return_multivector=False,
                    truncate_dim=self.dim, batch_size=max(len(batch), 1),
                )))
            else:
                from PIL import Image

                images = [Image.open(p).convert("RGB") for p in batch]
                try:
                    out.append(self._forward_generic(
                        self._process(_PROC_IMAGE_FNS, images, text=False)))
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
