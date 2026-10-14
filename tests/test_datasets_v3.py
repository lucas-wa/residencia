"""Loader do ViDoRe V3 (sem rede: `datasets.load_dataset` e substituido)."""

import sys
import types
from pathlib import Path

from benchmark.common.datasets import load_vidore_v3


class _FakeImage:
    def __init__(self, tag: str):
        self.tag = tag

    def save(self, path, *args, **kwargs):
        Path(path).write_text(self.tag, encoding="utf-8")


def _install_fake_datasets(monkeypatch, subsets: dict[str, list[dict]],
                           splits_only: bool = False):
    calls: list[tuple] = []

    def load_dataset(name, config=None, split=None, **kwargs):
        calls.append((name, config, split))
        if splits_only and config is not None:
            raise ValueError("este dataset nao tem configs")
        key = config if config is not None else split
        if key not in subsets:
            raise ValueError(f"unknown subset {key!r}")
        return subsets[key]

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


def test_load_vidore_v3_mapeia_queries_docs_qrels(tmp_path: Path, monkeypatch):
    subsets = {
        "queries": [{"query-id": "q1", "query": "qual a meta do plano?"},
                    {"query-id": "q2", "query": "quanto custa?"}],
        "corpus": [{"corpus-id": "d1", "image": _FakeImage("pagina-1")},
                   {"corpus-id": "d2", "image": _FakeImage("pagina-2")}],
        "qrels": [{"query-id": "q1", "corpus-id": "d1", "score": 1},
                  {"query-id": "q2", "corpus-id": "d2", "score": 2},
                  {"query-id": "q2", "corpus-id": "d1", "score": 0}],
    }
    _install_fake_datasets(monkeypatch, subsets)

    queries, docs, qrels = load_vidore_v3("vidore/vidore_v3_hr", cache_dir=tmp_path)

    assert [q.id for q in queries] == ["q1", "q2"]
    assert queries[0].text == "qual a meta do plano?"
    assert [d.id for d in docs] == ["d1", "d2"]
    for d in docs:
        p = Path(d.image_path)
        assert p.exists() and str(tmp_path) in str(p.resolve())
    assert Path(docs[1].image_path).read_text(encoding="utf-8") == "pagina-2"
    assert qrels == {"q1": {"d1": 1.0}, "q2": {"d2": 2.0}}


def test_load_vidore_v3_formato_mteb(tmp_path: Path, monkeypatch):
    """Variante `_mteb_format`: colunas `_id`/`text` e nomes de subset alternativos."""
    subsets = {
        "query": [{"_id": "q7", "text": "emissoes de carbono"}],
        "documents": [{"_id": "p7", "page_image": _FakeImage("z"), "text": "pagina 7"}],
        "relevance": [{"query-id": "q7", "corpus-id": "p7", "relevance": 1}],
    }
    _install_fake_datasets(monkeypatch, subsets)

    queries, docs, qrels = load_vidore_v3("vidore/vidore_v3_hr_mteb_format", cache_dir=tmp_path)

    assert queries[0].id == "q7" and queries[0].text == "emissoes de carbono"
    assert docs[0].id == "p7" and docs[0].text == "pagina 7"
    assert docs[0].image_path is not None
    assert qrels == {"q7": {"p7": 1.0}}


def test_load_vidore_v3_cai_para_split_quando_config_falha(tmp_path: Path, monkeypatch):
    """Alguns subsets expoem queries/corpus/qrels como SPLITS, nao como configs."""
    subsets = {
        "queries": [{"query_id": "q1", "query": "x"}],
        "corpus": [{"docid": "d1", "image": _FakeImage("y")}],
        "qrels": [{"query_id": "q1", "docid": "d1", "score": 1}],
    }
    calls = _install_fake_datasets(monkeypatch, subsets, splits_only=True)

    queries, docs, qrels = load_vidore_v3("vidore/vidore_v3_hr", cache_dir=tmp_path)

    assert len(queries) == 1 and len(docs) == 1 and qrels == {"q1": {"d1": 1.0}}
    # o caminho por SPLIT foi exercitado (config falhou antes)
    assert ("vidore/vidore_v3_hr", None, "queries") in calls
