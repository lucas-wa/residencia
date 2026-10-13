"""Dataset loaders.

`load_local` reads the jsonl/tsv layout produced by `data/govexe/build_planos_benchmark.py`.
`load_vidore_v2` pulls a ViDoRe v2 dataset from the HuggingFace `vidore` collection and
materialises every corpus page as a PNG under `cache_dir`, so the retrievers only ever
see file paths.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from benchmark.common.types import Doc, Query

# Column-name aliases: ViDoRe v2 subsets are BEIR-flavoured but not perfectly uniform.
# TODO confirmar em runtime: rodar `datasets.get_dataset_config_names(<subset>)` e
# inspecionar `.column_names` do subset real antes da primeira rodada na 4090; os nomes
# abaixo cobrem as variantes conhecidas (`query-id`/`query_id`, `corpus-id`/`docid`).
_QUERY_ID_KEYS = ("query-id", "query_id", "queryid", "qid", "id")
_QUERY_TEXT_KEYS = ("query", "text", "question")
_DOC_ID_KEYS = ("corpus-id", "corpus_id", "docid", "doc_id", "image_filename", "id")
_DOC_TEXT_KEYS = ("text", "passage", "content")
_IMAGE_KEYS = ("image", "page_image", "img")
_SCORE_KEYS = ("score", "relevance", "label", "rel")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def load_local(root: str | Path):
    """Read `queries.jsonl`, `corpus.jsonl` and `qrels.tsv` from `root`."""
    root = Path(root)
    queries = [
        Query(**json.loads(line))
        for line in (root / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    docs = [
        Doc(**json.loads(line))
        for line in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qrels: dict[str, dict[str, float]] = {}
    for line in (root / "qrels.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qid, docid, score = line.split("\t")
        qrels.setdefault(qid, {})[docid] = float(score)
    return queries, docs, qrels


def _pick(row, keys):
    for k in keys:
        value = row.get(k) if hasattr(row, "get") else None
        if value is not None:
            return value
    return None


def load_vidore_v2(name: str, cache_dir: str | Path | None = None, split: str = "test"):
    """Load a ViDoRe v2 dataset (HF collection `vidore`) as (queries, docs, qrels).

    Corpus images are written to `cache_dir/<dataset>/<doc_id>.png` and referenced by
    `Doc.image_path`, keeping the retrievers free of HF/PIL objects.
    """
    from datasets import load_dataset  # import tardio: evita torch/datasets nos testes

    cache = Path(cache_dir) if cache_dir is not None else Path("data/vidore_cache")
    img_dir = cache / _SAFE.sub("_", name)
    img_dir.mkdir(parents=True, exist_ok=True)

    def _subset(config: str):
        # TODO confirmar em runtime: alguns subsets ViDoRe v2 expoem `queries`/`corpus`/
        # `qrels` como *configs* (load_dataset(name, config)), outros como *splits*
        # (load_dataset(name, split=config)). Tentamos config primeiro, split depois.
        try:
            return load_dataset(name, config, split=split)
        except Exception:
            return load_dataset(name, split=config)

    queries: list[Query] = []
    for i, row in enumerate(_subset("queries")):
        qid = _pick(row, _QUERY_ID_KEYS)
        queries.append(Query(id=str(qid if qid is not None else i),
                             text=_pick(row, _QUERY_TEXT_KEYS)))

    docs: list[Doc] = []
    for i, row in enumerate(_subset("corpus")):
        did = _pick(row, _DOC_ID_KEYS)
        did = str(did if did is not None else i)
        image = _pick(row, _IMAGE_KEYS)
        image_path = None
        if image is not None:
            target = img_dir / f"{_SAFE.sub('_', did)}.png"
            if not target.exists():
                image.save(str(target))
            image_path = str(target)
        docs.append(Doc(id=did, text=_pick(row, _DOC_TEXT_KEYS), image_path=image_path))

    qrels: dict[str, dict[str, float]] = {}
    for row in _subset("qrels"):
        qid = _pick(row, _QUERY_ID_KEYS)
        did = _pick(row, _DOC_ID_KEYS)
        if qid is None or did is None:
            continue
        score = _pick(row, _SCORE_KEYS)
        score = 1.0 if score is None else float(score)
        if score <= 0:
            continue
        qrels.setdefault(str(qid), {})[str(did)] = score

    return queries, docs, qrels
