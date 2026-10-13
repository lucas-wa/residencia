from __future__ import annotations

from benchmark.common.config import ExperimentConfig

_MULTI = {"colpali", "colqwen2"}


def is_multi_vector(cfg: ExperimentConfig) -> bool:
    """True for late-interaction models, which the runner ranks with MaxSim."""
    return cfg.model_kind in _MULTI


def build_retriever(cfg: ExperimentConfig):
    """Instantiate the retriever for `cfg` (imports the heavy deps lazily)."""
    if cfg.model_kind == "text":
        from benchmark.models.text_ptbr import TextRetriever

        return TextRetriever(cfg.model_id, device=cfg.device, dtype=cfg.dtype,
                             batch_size=cfg.batch_size)
    if cfg.model_kind in {"clip", "siglip"}:
        from benchmark.models.clip_siglip import ClipSiglipRetriever

        return ClipSiglipRetriever(cfg.model_id, device=cfg.device, dtype=cfg.dtype,
                                   batch_size=cfg.batch_size)
    if cfg.model_kind in _MULTI:
        from benchmark.models.colvision import ColVisionRetriever

        return ColVisionRetriever(cfg.model_id, kind=cfg.model_kind,
                                  device=cfg.device, dtype=cfg.dtype,
                                  batch_size=cfg.batch_size)
    raise ValueError(f"model_kind desconhecido: {cfg.model_kind}")
