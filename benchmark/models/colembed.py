"""NVIDIA Nemotron ColEmbed V2 -- late interaction (multi-vector, one array per item).

Mesmo contrato do `colvision.ColVisionRetriever`: `encode_queries` / `encode_corpus`
devolvem `list[np.ndarray]` [n_tokens, dim] e o ranqueamento sai do
`benchmark.common.index.maxsim_search`, via `runner.evaluate(..., multi_vector=True)`.

Checkpoints (familia `nemotron-colembed-vl-*-v2`, licenca NAO comercial -- uso academico):
  3B: nvidia/llama-nemotron-colembed-vl-3b-v2
  4B: nvidia/nemotron-colembed-vl-4b-v2
  8B: nvidia/nemotron-colembed-vl-8b-v2   # TODO confirmar em runtime: o id do 8B e o
      unico da familia que nao foi conferido no Hub. Se o `from_pretrained` estourar
      404, listar a org (`huggingface_hub.list_models(author="nvidia", search=
      "colembed")`) e corrigir o `model_id` do YAML -- o wrapper nao muda.

O modelo e custom code (base NVIDIA Eagle2 / Qwen3-VL), logo `trust_remote_code=True`.

`_embed_queries` / `_embed_images` sao metodos de instancia de proposito: o teste de
contrato os substitui para exercitar o wrapper sem baixar pesos.
"""

from __future__ import annotations

import numpy as np

from benchmark.common.types import Doc, Query
from benchmark.models._remote_api import call_batch, first_method

# TODO confirmar em runtime: a API de encode do ColEmbed V2 mora no custom code do
# checkpoint e o card a documenta como `forward_queries`/`forward_passages` (recebem a
# lista crua de textos/imagens e fazem o proprio batching). Releases anteriores expunham
# `get_query_embeddings`/`get_doc_embeddings`. Tentamos os nomes conhecidos em ordem e,
# se nenhum existir, caimos no caminho generico processor + `model(**batch)`.
_QUERY_FNS = ("forward_queries", "get_query_embeddings", "embed_queries", "encode_queries")
_PASSAGE_FNS = ("forward_passages", "get_doc_embeddings", "embed_passages", "forward_images",
                "encode_passages")
_PROC_QUERY_FNS = ("process_queries", "process_texts")
_PROC_IMAGE_FNS = ("process_images", "process_passages")


class ColEmbedRetriever:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16",
                 batch_size: int = 4):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.name = f"colembed:{model_id}"
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        td = getattr(torch, dtype)
        try:
            model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=td, device_map=device,
                attn_implementation="flash_attention_2",
            )
        except Exception:  # flash-attn ausente -> fallback obrigatorio
            model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=td, device_map=device,
                attn_implementation="eager",
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
    def _to_arrays(cls, out, attention_mask=None) -> list[np.ndarray]:
        """[B, n_tokens, dim] (ou lista de [n_tokens, dim]) -> um array por item."""
        if isinstance(out, (list, tuple)):
            return [np.ascontiguousarray(cls._as_numpy(t)) for t in out]
        arr = cls._as_numpy(out)
        if arr.ndim == 2:                      # item unico, ja sem dimensao de batch
            return [np.ascontiguousarray(arr)]
        mask = cls._as_numpy(attention_mask) if attention_mask is not None else None
        items: list[np.ndarray] = []
        for i in range(arr.shape[0]):
            item = arr[i]
            if mask is not None and mask.ndim == 2 and mask.shape[1] == item.shape[0]:
                item = item[mask[i].astype(bool)]   # descarta padding
            items.append(np.ascontiguousarray(item, dtype="float32"))
        return items

    # -- embedding ---------------------------------------------------------------

    def _run_batch(self, items: list, model_fns: tuple[str, ...], proc_fns: tuple[str, ...]):
        """Um batch pela API do modelo, ou pelo caminho generico processor + forward."""
        import torch

        fn = first_method(self._model, model_fns)
        with torch.no_grad():
            if fn is not None:
                return self._to_arrays(call_batch(fn, items))
            proc_fn = first_method(self._proc, proc_fns)
            if proc_fn is None:
                raise AttributeError(
                    f"{self.model_id}: nenhuma API de encode conhecida "
                    f"(modelo: {model_fns}; processor: {proc_fns}) -- conferir o card"
                )
            batch = proc_fn(items)
            batch = {k: v.to(self._model.device) for k, v in dict(batch).items()}
            out = self._model(**batch)
            # TODO confirmar em runtime: o forward do ColEmbed devolve o tensor
            # multi-vetor direto em algumas versoes e um ModelOutput em outras.
            emb = getattr(out, "embeddings", None)
            if emb is None:
                emb = getattr(out, "last_hidden_state", out)
            return self._to_arrays(emb, batch.get("attention_mask"))

    def _embed_queries(self, queries: list[Query]) -> list[np.ndarray]:
        texts = [q.text or "" for q in queries]
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._run_batch(texts[start:start + self.batch_size],
                                       _QUERY_FNS, _PROC_QUERY_FNS))
        return out

    def _embed_images(self, paths: list[str]) -> list[np.ndarray]:
        from PIL import Image

        out: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[start:start + self.batch_size]]
            try:
                out.extend(self._run_batch(images, _PASSAGE_FNS, _PROC_IMAGE_FNS))
            finally:
                for im in images:
                    im.close()
        return out

    # -- Retriever protocol ------------------------------------------------------

    def encode_queries(self, queries: list[Query]) -> list[np.ndarray]:
        return self._embed_queries(queries)

    def encode_corpus(self, docs: list[Doc]) -> list[np.ndarray]:
        return self._embed_images([d.image_path or "" for d in docs])

    def score(self, q_emb, d_emb):                 # nao usado: runner usa maxsim_search
        raise NotImplementedError("use runner.evaluate(..., multi_vector=True)")
