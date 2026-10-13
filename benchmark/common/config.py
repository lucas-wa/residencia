from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml

ModelKind = Literal["clip", "siglip", "colpali", "colqwen2", "text"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Single benchmark run configuration (one YAML file per experiment)."""

    name: str
    model_id: str
    model_kind: ModelKind
    dataset: str
    dtype: str = "bfloat16"
    batch_size: int = 8
    device: str = "cuda"
    k_values: tuple[int, ...] = (1, 5, 10)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if "k_values" in data:
            data["k_values"] = tuple(data["k_values"])
        return cls(**data)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["k_values"] = list(self.k_values)
        return d
