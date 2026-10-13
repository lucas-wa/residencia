"""CLI: evaluate one model on a ViDoRe v2 subset and append the metrics to a CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.common.config import ExperimentConfig
from benchmark.common.datasets import load_vidore_v2
from benchmark.common.runner import evaluate, write_results_csv
from benchmark.models.factory import build_retriever, is_multi_vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = ExperimentConfig.from_yaml(a.config)
    queries, docs, qrels = load_vidore_v2(cfg.dataset, cache_dir=Path("data/vidore_cache"))
    if a.limit:
        queries = queries[: a.limit]
    r = build_retriever(cfg)
    m = evaluate(r, queries, docs, qrels, k_values=cfg.k_values, multi_vector=is_multi_vector(cfg))
    write_results_csv("results/vidore.csv", cfg.to_dict(), m)
    print(cfg.name, m)


if __name__ == "__main__":
    main()
