from __future__ import annotations

from benchmark.common.config import ExperimentConfig

# Late interaction (ranqueado por MaxSim em `benchmark.common.index.maxsim_search`).
# `qwen3vl` e `jinav4` NAO entram aqui: os dois sao densos (um vetor por item). O
# jina-v4 tambem sabe fazer multi-vetor, mas o wrapper atual so expoe o caminho denso.
_MULTI = {"colpali", "colqwen2", "colqwen2.5", "colembed"}
# Subconjunto de _MULTI atendido pelo wrapper do colpali-engine.
_COLVISION = {"colpali", "colqwen2", "colqwen2.5"}


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
    if cfg.model_kind in _COLVISION:
        from benchmark.models.colvision import ColVisionRetriever

        return ColVisionRetriever(cfg.model_id, kind=cfg.model_kind,
                                  device=cfg.device, dtype=cfg.dtype,
                                  batch_size=cfg.batch_size)
    if cfg.model_kind == "colembed":
        from benchmark.models.colembed import ColEmbedRetriever

        return ColEmbedRetriever(cfg.model_id, device=cfg.device, dtype=cfg.dtype,
                                 batch_size=cfg.batch_size)
    if cfg.model_kind == "qwen3vl":
        from benchmark.models.qwen3vl_embed import Qwen3VLEmbedRetriever

        return Qwen3VLEmbedRetriever(cfg.model_id, device=cfg.device, dtype=cfg.dtype,
                                     batch_size=cfg.batch_size)
    if cfg.model_kind == "jinav4":
        from benchmark.models.jina_embed import JinaV4Retriever

        return JinaV4Retriever(cfg.model_id, device=cfg.device, dtype=cfg.dtype,
                               batch_size=cfg.batch_size)
    raise ValueError(f"model_kind desconhecido: {cfg.model_kind}")
