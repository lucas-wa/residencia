import json
from pathlib import Path


def test_smoke_with_dummy(tmp_path: Path, monkeypatch):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "queries.jsonl").write_text(
        json.dumps({"id": "BR001_destaque", "text": "saude"}) + "\n", encoding="utf-8")
    (d / "corpus.jsonl").write_text(
        json.dumps({"id": "BR001_p00", "text": "saude publica"}) + "\n", encoding="utf-8")
    (d / "qrels.tsv").write_text("BR001_destaque\tBR001_p00\t1\n", encoding="utf-8")
    from benchmark import run_govexe_ptbr as mod
    from benchmark.models.dummy import DummyRetriever
    monkeypatch.setattr(mod, "build_retriever", lambda cfg: DummyRetriever())
    monkeypatch.setattr(mod, "is_multi_vector", lambda cfg: False)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "name: smoke\nmodel_id: dummy\nmodel_kind: text\ndataset: govexe\ndevice: cpu\n",
        encoding="utf-8")
    out = tmp_path / "out.csv"
    mod.run(str(cfg), str(d), str(out), limit=None)
    assert out.exists()
