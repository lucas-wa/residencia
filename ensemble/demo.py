"""Runner de demonstracao cross-modal: ranking intercalado anotado por modalidade."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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
    # melhor hit de cada modalidade presente (intercalado), ordenado por score desc
    destaques_por_modalidade: list[tuple[PointerRecord, float]] = field(
        default_factory=list
    )


def _destaques_por_modalidade(
    ranking: list[tuple[PointerRecord, float]],
) -> list[tuple[PointerRecord, float]]:
    """O melhor hit (maior score) de CADA modalidade, a partir de um ranking desc."""
    melhores: dict[str, tuple[PointerRecord, float]] = {}
    for rec, score in ranking:  # ranking ja vem por score decrescente
        if rec.modality not in melhores:
            melhores[rec.modality] = (rec, score)
    return sorted(melhores.values(), key=lambda t: t[1], reverse=True)


def run_demo(
    index: EnsembleIndex,
    queries: list[str],
    *,
    embedder: Embedder,
    k: int = 10,
    resolver: Callable[[str], bool] | None = None,
) -> list[DemoResult]:
    results = []
    # modalidades a nivel de corpus (nao do top-k por query): "ex. video falhou na captura"
    presentes = set(index.modalidades)
    ausentes = set(MODALIDADES) - presentes
    for q in queries:
        hits = index.search(q, k, embedder=embedder)
        # rankeia o corpus inteiro p/ extrair o melhor de cada modalidade (intercalado)
        ranking_total = index.search(q, len(index.records), embedder=embedder)
        destaques = _destaques_por_modalidade(ranking_total)
        # resolve todo uri que format_cli pode imprimir (top-k U destaques), senao um
        # ponteiro morto numa modalidade fora do top-k ficaria sem marca de indisponivel
        resolvidos = (
            {
                r.pointer_uri: bool(resolver(r.pointer_uri))
                for r, _ in (*hits, *destaques)
            }
            if resolver is not None
            else {}
        )
        results.append(
            DemoResult(
                query=q,
                hits=hits,
                modalidades_presentes=presentes,
                modalidades_ausentes=ausentes,
                resolvidos=resolvidos,
                destaques_por_modalidade=destaques,
            )
        )
    return results


def _format_hit(result: DemoResult, rec: PointerRecord, score: float) -> str:
    trecho = rec.descriptor
    if len(trecho) > _TRECHO_MAX:
        trecho = trecho[: _TRECHO_MAX - 1] + "…"
    line = f"[{rec.modality.upper()}] {trecho} → {rec.pointer_uri} ({score:.3f})"
    if not result.resolvidos.get(rec.pointer_uri, True):
        line += " [ponteiro indisponível]"
    return line


def format_cli(result: DemoResult) -> str:
    lines = [f"Consulta: {result.query}"]
    lines.append("Melhor de cada modalidade:")
    for rec, score in result.destaques_por_modalidade:
        lines.append(_format_hit(result, rec, score))
    lines.append("Ranking global (top-k):")
    for rec, score in result.hits:
        lines.append(_format_hit(result, rec, score))
    presentes = ", ".join(sorted(result.modalidades_presentes)) or "nenhuma"
    ausentes = ", ".join(sorted(result.modalidades_ausentes)) or "nenhuma"
    lines.append(f"Modalidades presentes (corpus): {presentes}")
    lines.append(f"Modalidades ausentes (corpus): {ausentes}")
    return "\n".join(lines)
