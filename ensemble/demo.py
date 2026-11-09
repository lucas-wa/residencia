"""Runner de demonstracao cross-modal: ranking intercalado anotado por modalidade."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ensemble.index import Embedder, EnsembleIndex
from ensemble.records import PointerRecord

MODALIDADES = frozenset({"documento", "imagem", "audio", "texto", "video"})
_TRECHO_MAX = 80


@dataclass
class DemoResult:
    query: str
    hits: list[tuple[PointerRecord, float]]
    modalidades_presentes: set[str]
    modalidades_ausentes: set[str]
    # uri -> resolve? (vazio quando nenhum resolver foi injetado)
    resolvidos: dict[str, bool]


def run_demo(
    index: EnsembleIndex,
    queries: list[str],
    *,
    embedder: Embedder,
    k: int = 10,
    resolver: Callable[[str], bool] | None = None,
) -> list[DemoResult]:
    results = []
    for q in queries:
        hits = index.search(q, k, embedder=embedder)
        presentes = {r.modality for r, _ in hits}
        resolvidos = (
            {r.pointer_uri: bool(resolver(r.pointer_uri)) for r, _ in hits}
            if resolver is not None
            else {}
        )
        results.append(
            DemoResult(
                query=q,
                hits=hits,
                modalidades_presentes=presentes,
                modalidades_ausentes=set(MODALIDADES) - presentes,
                resolvidos=resolvidos,
            )
        )
    return results


def format_cli(result: DemoResult) -> str:
    lines = [f"Consulta: {result.query}"]
    for rec, score in result.hits:
        trecho = rec.descriptor
        if len(trecho) > _TRECHO_MAX:
            trecho = trecho[: _TRECHO_MAX - 1] + "…"
        line = f"[{rec.modality.upper()}] {trecho} → {rec.pointer_uri} ({score:.3f})"
        if not result.resolvidos.get(rec.pointer_uri, True):
            line += " [ponteiro indisponível]"
        lines.append(line)
    ausentes = ", ".join(sorted(result.modalidades_ausentes)) or "nenhuma"
    lines.append(f"Modalidades ausentes: {ausentes}")
    return "\n".join(lines)
