"""CLI: evaluate one model on the pt-BR GovExe planos benchmark (local jsonl/tsv)."""

from __future__ import annotations

import argparse

from benchmark.common.config import ExperimentConfig
from benchmark.common.datasets import load_local
from benchmark.common.runner import evaluate, write_results_csv
from benchmark.models.factory import build_retriever, is_multi_vector


def run(config_path: str, data_dir: str, out_csv: str, limit: int | None) -> dict:
    cfg = ExperimentConfig.from_yaml(config_path)
    queries, docs, qrels = load_local(data_dir)
    if limit:
        queries = queries[:limit]
    r = build_retriever(cfg)
    m = evaluate(r, queries, docs, qrels, k_values=cfg.k_values, multi_vector=is_multi_vector(cfg))
    write_results_csv(out_csv, cfg.to_dict(), m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/govexe_ptbr.csv")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print(run(a.config, a.data_dir, a.out, a.limit))


if __name__ == "__main__":
    main()
