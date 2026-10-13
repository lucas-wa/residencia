from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    id: str
    text: str | None = None
    image_path: str | None = None


@dataclass(frozen=True)
class Doc:
    id: str
    text: str | None = None
    image_path: str | None = None
