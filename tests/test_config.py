from pathlib import Path

from benchmark.common.config import ExperimentConfig


def test_from_yaml_roundtrip(tmp_path: Path):
    p = tmp_path / "exp.yaml"
    p.write_text(
        "name: smoke\nmodel_id: dummy\nmodel_kind: text\ndataset: synthetic\n"
        "batch_size: 4\ndevice: cpu\n",
        encoding="utf-8",
    )
    cfg = ExperimentConfig.from_yaml(p)
    assert cfg.name == "smoke"
    assert cfg.model_kind == "text"
    assert cfg.batch_size == 4
    assert cfg.device == "cpu"
    assert cfg.k_values == (1, 5, 10)      # default
    assert cfg.seed == 42                   # default
    assert cfg.to_dict()["model_id"] == "dummy"
