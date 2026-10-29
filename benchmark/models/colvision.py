"""ColPali / ColQwen2 / ColQwen2.5 late-interaction retriever (multi-vector).

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


def resolve_classes(kind: str):
    """(model_cls, processor_cls) do colpali-engine para `kind`.

    Funcao de modulo de proposito: o teste injeta um `colpali_engine.models` falso em
    `sys.modules` e cobre o dispatch sem baixar pesos nem instalar o pacote.
    """
    # Nomes CONFERIDOS no wheel do colpali-engine (nao sao simetricos -- o processor do
    # 2.5 leva um underscore a mais): `colpali_engine/models/__init__.py` exporta
    #     from .qwen2   import BiQwen2,   BiQwen2Processor,    ColQwen2,   ColQwen2Processor
    #     from .qwen2_5 import BiQwen2_5, BiQwen2_5_Processor, ColQwen2_5, ColQwen2_5_Processor
    # `ColQwen2_5`/`ColQwen2_5_Processor` existem desde a 0.3.9 (verificado nas wheels
    # 0.3.8=ausente, 0.3.9/0.3.10/0.3.13=presente), ou seja, em TODA a faixa que o
    # requirements-slurm.txt permite (`colpali-engine>=0.3.10,<0.4`). O card do
    # `vidore/colqwen2.5-v0.2` usa exatamente este import.
    if kind == "colpali":
        from colpali_engine.models import ColPali, ColPaliProcessor

        return ColPali, ColPaliProcessor
    if kind == "colqwen2.5":
        # Sem fallback para ColQwen2: as arquiteturas sao DIFERENTES (ColQwen2 estende
        # Qwen2VLModel, ColQwen2_5 estende Qwen2_5_VLModel). Carregar o checkpoint 2.5
        # pela classe 2 nao degrada -- quebra, e quebra tarde e de forma opaca
        # ("size mismatch for bias: ... torch.Size([2048]) vs torch.Size([1280])",
        # depois de um aviso facil de ignorar "model of type qwen2_5_vl to instantiate a
        # model of type qwen2_vl"). Falhar aqui, com a causa nomeada, custa menos.
        try:
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        except ImportError as exc:  # noqa: TRY003
            raise ImportError(
                "colpali-engine sem ColQwen2_5/ColQwen2_5_Processor (precisa de >=0.3.9; "
                "o requirements-slurm.txt ja pede >=0.3.10). O checkpoint ColQwen2.5 NAO "
                "carrega pela classe ColQwen2 -- arquiteturas distintas (qwen2_5_vl vs "
                "qwen2_vl). Atualize o colpali-engine."
            ) from exc

        return ColQwen2_5, ColQwen2_5_Processor
    if kind != "colqwen2":
        raise ValueError(f"kind sem wrapper no colpali-engine: {kind}")

    from colpali_engine.models import ColQwen2, ColQwen2Processor

    return ColQwen2, ColQwen2Processor


class ColVisionRetriever:
    def __init__(self, model_id: str, kind: Literal["colpali", "colqwen2", "colqwen2.5"],
                 device: str = "cuda", dtype: str = "bfloat16", batch_size: int = 4):
        import torch

        self.name = f"{kind}:{model_id}"
        self.kind = kind
        self.device = device
        self.batch_size = batch_size
        td = getattr(torch, dtype)
        model_cls, proc_cls = resolve_classes(kind)
        # flash-attn fica FORA do requirements de proposito (ver requirements-slurm.txt):
        # sem o pacote, pedir "flash_attention_2" so rende um ImportError que o except
        # engole. Checar antes evita a PRIMEIRA from_pretrained inteira e, sobretudo,
        # evita que esse traceback descartado apareca na frente do erro que importa.
        from importlib.util import find_spec

        attn = "flash_attention_2" if find_spec("flash_attn") else "eager"
        try:
            model = model_cls.from_pretrained(
                model_id, torch_dtype=td, device_map=device,
                attn_implementation=attn,
            )
        except Exception:  # flash-attn presente mas inutilizavel -> fallback obrigatorio
            if attn == "eager":
                raise
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
