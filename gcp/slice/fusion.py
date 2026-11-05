"""Fusao RRF (Reciprocal Rank Fusion) de rankings vindos de espacos vetoriais distintos.

O problema: cada espaco (Qwen3-VL para pagina/foto/texto, CLAP para audio, ...) tem sua
propria escala de similaridade -- somar ou comparar os scores crus entre espacos nao
significa nada. O RRF resolve descartando o score e ficando so com a **posicao**: cada
lista contribui `1/(c + posicao_0based + 1)` para o item, e o score final e a soma das
contribuicoes. Assim um item bem colocado em varias listas vence um item otimo numa so.

A constante `c` (default 60, o valor do artigo original de Cormack et al., 2009) amortece
o topo: com `c = 60` a diferenca entre 1o e 2o lugar (1/61 vs 1/62) e pequena, o que evita
que um unico espaco domine a fusao.

Python puro, sem numpy -- a entrada e uma lista de `item_id` por espaco, nao vetores.

Uso:
    from gcp.slice.fusion import rrf_fuse
    fundido = rrf_fuse([ids_qwen3vl, ids_clap])   # [(item_id, score), ...] score desc
"""

from __future__ import annotations

__all__ = ["DEFAULT_C", "rrf_fuse"]

# Cormack et al. (2009), "Reciprocal Rank Fusion outperforms Condorcet and individual
# Rank Learning Methods".
DEFAULT_C = 60


def rrf_fuse(rankings: list[list[str]], *, c: int = DEFAULT_C) -> list[tuple[str, float]]:
    """Funde rankings de `item_id` (melhor primeiro) num unico ranking por RRF.

    Cada lista de `rankings` e um ranking independente -- tipicamente o top-k de um espaco
    vetorial. Listas vazias e de tamanhos diferentes sao aceitas (uma lista vazia apenas
    nao contribui). Devolve `(item_id, score)` ordenado por score decrescente, com
    desempate estavel pelo `item_id` -- a mesma entrada da sempre a mesma saida.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        vistos: set[str] = set()
        for posicao, item_id in enumerate(ranking):
            # um id repetido DENTRO da mesma lista vale so pela melhor posicao: a soma e
            # sobre listas, nao sobre ocorrencias.
            if item_id in vistos:
                continue
            vistos.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (c + posicao + 1)
    return sorted(scores.items(), key=lambda par: (-par[1], par[0]))
