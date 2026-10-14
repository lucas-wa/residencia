"""Dataset loaders.

`load_local` reads the jsonl/tsv layout produced by `data/govexe/build_planos_benchmark.py`.
`load_vidore_v2` pulls a ViDoRe v2 dataset from the HuggingFace `vidore` collection and
materialises every corpus page as a PNG under `cache_dir`, so the retrievers only ever
see file paths. `load_vidore_v3` does the same for the ViDoRe V3 datasets
(`vidore/vidore_v3_*`), which are BEIR-like too but spell the subsets/columns with more
variation (MTEB flavour: `_id`/`text`, subsets `query`/`documents`/`relevance`).
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
_QUERY_ID_KEYS = ("query-id", "query_id", "queryid", "qid", "_id", "id")
_QUERY_TEXT_KEYS = ("query", "text", "question")
_DOC_ID_KEYS = ("corpus-id", "corpus_id", "docid", "doc_id", "image_filename", "_id", "id")
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


def _load_beir_like(name: str, cache_dir, split: str, subsets: dict[str, tuple[str, ...]]):
    """Load a BEIR-flavoured HF dataset (queries + corpus + qrels) as plain dataclasses.

    `subsets` maps each role to the candidate subset names to try, in order. Corpus
    images are written to `cache_dir/<dataset>/<doc_id>.png` and referenced by
    `Doc.image_path`, keeping the retrievers free of HF/PIL objects.
    """
    from datasets import load_dataset  # import tardio: evita torch/datasets nos testes

    cache = Path(cache_dir) if cache_dir is not None else Path("data/vidore_cache")
    img_dir = cache / _SAFE.sub("_", name)
    img_dir.mkdir(parents=True, exist_ok=True)

    def _subset(role: str):
        # TODO confirmar em runtime: alguns subsets ViDoRe expoem `queries`/`corpus`/
        # `qrels` como *configs* (load_dataset(name, config)), outros como *splits*
        # (load_dataset(name, split=config)). Tentamos config primeiro, split depois, e
        # repetimos para cada apelido conhecido do papel.
        errors = []
        for candidate in subsets[role]:
            for attempt in (lambda c=candidate: load_dataset(name, c, split=split),
                            lambda c=candidate: load_dataset(name, split=c)):
                try:
                    return attempt()
                except Exception as exc:  # noqa: PERF203 - tentativa e erro proposital
                    errors.append(f"{candidate}: {exc}")
        raise ValueError(
            f"{name}: nao achei o subset de '{role}' (tentei {subsets[role]}): "
            + " | ".join(errors)
        )

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


# Nomes de subset por papel. v2 e uniforme; V3 (inclusive as variantes `_mteb_format`)
# varia mais, entao aceitamos apelidos.
_V2_SUBSETS = {"queries": ("queries",), "corpus": ("corpus",), "qrels": ("qrels",)}
# TODO confirmar em runtime: rodar `datasets.get_dataset_config_names("vidore/vidore_v3_hr")`
# e inspecionar `.column_names` de cada subset antes da primeira rodada no H100. Os
# apelidos abaixo cobrem o que se ve no Hub (BEIR: queries/corpus/qrels; MTEB:
# query/documents/relevance ou `default`/`test`).
_V3_SUBSETS = {
    "queries": ("queries", "query", "test_queries"),
    "corpus": ("corpus", "documents", "docs", "images"),
    "qrels": ("qrels", "relevance", "qrel", "relevant_docs"),
}


def load_vidore_v2(name: str, cache_dir: str | Path | None = None, split: str = "test"):
    """Load a ViDoRe v2 dataset (HF collection `vidore`) as (queries, docs, qrels)."""
    return _load_beir_like(name, cache_dir, split, _V2_SUBSETS)


def load_vidore_v3(name: str, cache_dir: str | Path | None = None, split: str = "test"):
    """Load a ViDoRe V3 dataset (`vidore/vidore_v3_*`) as (queries, docs, qrels).

    Mesmo contrato do v2 -- as paginas do corpus viram PNGs em `cache_dir` e os
    retrievers so veem caminhos de arquivo. A diferenca esta nos apelidos de subset
    (`_V3_SUBSETS`) e nas colunas em formato MTEB (`_id`/`text`), ja cobertas pelos
    mapas `_*_KEYS` no topo do modulo.
    """
    return _load_beir_like(name, cache_dir, split, _V3_SUBSETS)
