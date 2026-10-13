import json
import sys
import types
from pathlib import Path

from benchmark.common.datasets import load_local, load_vidore_v2


def test_load_local(tmp_path: Path):
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"id": "q1", "text": "x"}) + "\n", encoding="utf-8")
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"id": "d1", "image_path": "p.png"}) + "\n", encoding="utf-8")
    (tmp_path / "qrels.tsv").write_text("q1\td1\t1\n", encoding="utf-8")
    qs, docs, qrels = load_local(tmp_path)
    assert qs[0].id == "q1" and qs[0].text == "x"
    assert docs[0].image_path == "p.png"
    assert qrels["q1"]["d1"] == 1.0


class _FakeImage:
    """Stand-in for the PIL image carried by the HF corpus split."""

    def __init__(self, tag: str):
        self.tag = tag

    def save(self, path, *args, **kwargs):
        Path(path).write_text(self.tag, encoding="utf-8")


def _install_fake_datasets(monkeypatch, subsets: dict[str, list[dict]]):
    """Inject a minimal fake `datasets` module (no network, package not installed)."""
    calls: list[tuple] = []

    def load_dataset(name, config=None, split=None, **kwargs):
        calls.append((name, config, split))
        if config not in subsets:
            raise ValueError(f"unknown config {config!r}")
        return subsets[config]

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


def test_load_vidore_v2_maps_queries_docs_qrels(tmp_path: Path, monkeypatch):
    subsets = {
        "queries": [{"query-id": "q1", "query": "qual o lucro?"}],
        "corpus": [{"corpus-id": "d1", "image": _FakeImage("page-1")},
                   {"corpus-id": "d2", "image": _FakeImage("page-2")}],
        "qrels": [{"query-id": "q1", "corpus-id": "d1", "score": 1},
                  {"query-id": "q1", "corpus-id": "d2", "score": 0}],  # nao relevante
    }
    calls = _install_fake_datasets(monkeypatch, subsets)

    queries, docs, qrels = load_vidore_v2("vidore/fake_v2", cache_dir=tmp_path)

    assert [q.id for q in queries] == ["q1"]
    assert queries[0].text == "qual o lucro?"
    assert [d.id for d in docs] == ["d1", "d2"]
    # imagens materializadas em disco dentro do cache_dir
    for d in docs:
        assert d.image_path is not None
        p = Path(d.image_path)
        assert p.exists() and str(tmp_path) in str(p.resolve())
    assert Path(docs[0].image_path).read_text(encoding="utf-8") == "page-1"
    # score 0 nao vira qrel
    assert qrels == {"q1": {"d1": 1.0}}
    # os tres subsets foram pedidos ao HF
    assert {c[1] for c in calls} == {"queries", "corpus", "qrels"}


def test_load_vidore_v2_accepts_alternative_column_names(tmp_path: Path, monkeypatch):
    subsets = {
        "queries": [{"query_id": "q9", "text": "meio ambiente"}],
        "corpus": [{"docid": "p9", "image": _FakeImage("x"), "text": "pagina 9"}],
        "qrels": [{"query_id": "q9", "docid": "p9", "relevance": 2}],
    }
    _install_fake_datasets(monkeypatch, subsets)

    queries, docs, qrels = load_vidore_v2("vidore/fake_v2", cache_dir=tmp_path)

    assert queries[0].id == "q9" and queries[0].text == "meio ambiente"
    assert docs[0].id == "p9" and docs[0].text == "pagina 9"
    assert qrels["q9"]["p9"] == 2.0
