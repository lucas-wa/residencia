"""Loaders de queries e qrels (gabarito) para a avaliacao do ensemble v2."""
from __future__ import annotations

from pathlib import Path


def load_queries(path) -> dict[str, str]:
    """Le `qid<TAB>texto` (UTF-8). Linhas vazias sao ignoradas."""
    queries: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qid, sep, text = line.partition("\t")
        if not sep:
            raise ValueError(f"linha de query sem TAB: {line!r}")
        queries[qid] = text
    return queries


def load_qrels(path) -> dict[str, set[str]]:
    """Le formato TREC `qid 0 item_id rel` (rel em {0,1}); so rel=1 entra no set.

    Query sem nenhum rel=1 aparece como chave com set vazio.
    """
    qrels: dict[str, set[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"linha de qrels invalida (esperado 4 colunas): {line!r}")
        qid, _, item_id, rel = parts
        if rel not in ("0", "1"):
            raise ValueError(f"rel invalido {rel!r} para {qid}/{item_id}")
        bucket = qrels.setdefault(qid, set())
        if rel == "1":
            bucket.add(item_id)
    return qrels


def validate_qrels(qrels: dict[str, set[str]], queries: dict[str, str],
                   corpus_ids: set[str]) -> None:
    """Levanta ValueError nomeando qid fora de `queries` ou item_id fora do corpus."""
    for qid, items in qrels.items():
        if qid not in queries:
            raise ValueError(f"qid desconhecido nas qrels: {qid}")
        for item_id in sorted(items):
            if item_id not in corpus_ids:
                raise ValueError(f"item_id {item_id} (qid {qid}) nao existe no corpus")
