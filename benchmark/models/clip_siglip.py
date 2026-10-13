"""CLIP / SigLIP2 image-text retriever.

Queries are text, corpus items are page images (`Doc.image_path`). Both sides are
L2-normalised, so `score` is a plain inner product.

Two backends:
- "hf": `transformers.AutoModel` + `AutoProcessor` (`get_text_features` /
  `get_image_features`), used for `openai/clip-*` and `google/siglip2-*`;
- "open_clip": `open_clip.create_model_and_transforms` (`encode_text` / `encode_image`).

`_embed_text` / `_embed_image` stay instance methods on purpose: the contract test
replaces them to exercise the wrapper without downloading weights.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from benchmark.common.types import Doc, Query


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-9)


class ClipSiglipRetriever:
    def __init__(self, model_id: str, backend: Literal["open_clip", "hf"] = "hf",
                 device: str = "cuda", dtype: str = "bfloat16", batch_size: int = 16):
        import torch

        self.name = f"clip:{model_id}"
        self.model_id = model_id
        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        self.torch_dtype = getattr(torch, dtype)
        # SigLIP tokenizers expect fixed-length padding; CLIP is happy with dynamic.
        self.is_siglip = "siglip" in model_id.lower()

        if backend == "hf":
            from transformers import AutoModel, AutoProcessor

            try:
                model = AutoModel.from_pretrained(
                    model_id, torch_dtype=self.torch_dtype,
                    attn_implementation="flash_attention_2",
                )
            except Exception:  # flash-attn ausente (Windows, sem compilador, ...)
                model = AutoModel.from_pretrained(
                    model_id, torch_dtype=self.torch_dtype, attn_implementation="eager",
                )
            self._model = model.to(device).eval()
            self._proc = AutoProcessor.from_pretrained(model_id)
        else:
            import open_clip

            # TODO confirmar em runtime: open_clip separa arquitetura e checkpoint
            # (`create_model_and_transforms("ViT-L-14", pretrained="openai")`). Aceitamos
            # "arch:pretrained"; sem ":" o checkpoint padrao da arquitetura e usado.
            arch, _, pretrained = model_id.partition(":")
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained or None, device=device,
            )
            self._model = self._model.eval()
            self._tok = open_clip.get_tokenizer(arch)

    # -- embedding --------------------------------------------------------------

    def _autocast(self):
        import torch

        if str(self.device).startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=self.torch_dtype)
        return torch.autocast(device_type="cpu", dtype=self.torch_dtype, enabled=False)

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        import torch

        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            with torch.no_grad(), self._autocast():
                if self.backend == "hf":
                    inputs = self._proc(
                        text=batch, return_tensors="pt", truncation=True,
                        padding="max_length" if self.is_siglip else True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    feats = self._model.get_text_features(**inputs)
                else:
                    tokens = self._tok(batch).to(self.device)
                    feats = self._model.encode_text(tokens)
            out.append(feats.float().cpu().numpy())
        return _l2_normalize(np.vstack(out))

    def _embed_image(self, paths: list[str]) -> np.ndarray:
        import torch
        from PIL import Image

        out = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start:start + self.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            with torch.no_grad(), self._autocast():
                if self.backend == "hf":
                    inputs = self._proc(images=images, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    feats = self._model.get_image_features(**inputs)
                else:
                    pixel_values = torch.stack([self._preprocess(im) for im in images])
                    feats = self._model.encode_image(pixel_values.to(self.device))
            for im in images:
                im.close()
            out.append(feats.float().cpu().numpy())
        return _l2_normalize(np.vstack(out))

    # -- Retriever protocol -----------------------------------------------------

    def encode_queries(self, queries: list[Query]) -> np.ndarray:
        return self._embed_text([q.text or "" for q in queries])

    def encode_corpus(self, docs: list[Doc]) -> np.ndarray:
        return self._embed_image([d.image_path or "" for d in docs])

    def score(self, q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
        return q_emb @ d_emb.T
