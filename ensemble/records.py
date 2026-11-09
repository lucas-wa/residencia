"""PointerRecord: descritor textual + ponteiro (URI) para o binario original."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, get_args

Modality = Literal["documento", "imagem", "audio", "texto", "video"]
MODALITIES: tuple[str, ...] = get_args(Modality)


@dataclass(frozen=True)
class PointerRecord:
    id: str
    modality: Modality
    descriptor: str
    pointer_uri: str
    source: str
    meta: dict = field(default_factory=dict)


def validate_record(r: PointerRecord) -> None:
    if r.modality not in MODALITIES:
        raise ValueError(f"modality invalida: {r.modality!r} (esperado um de {MODALITIES})")
    if not r.descriptor or not r.descriptor.strip():
        raise ValueError("descriptor vazio")
    if not r.pointer_uri or not r.pointer_uri.strip():
        raise ValueError("pointer_uri vazio")


def save_records(records: list[PointerRecord], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def load_records(path: str | Path) -> list[PointerRecord]:
    out: list[PointerRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(PointerRecord(**json.loads(line)))
    return out
