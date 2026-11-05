"""Fusao RRF (`gcp/slice/fusion.py`): combina rankings de espacos vetoriais distintos.

Python puro, sem numpy: os testes exercitam a aritmetica exata do score
(`1/(c + posicao_0based + 1)`) e a ordenacao final.
"""

from gcp.slice.fusion import rrf_fuse


def test_item_em_varias_listas_soma_contribuicoes():
    fused = rrf_fuse([["a", "b"], ["a", "c"]], c=60)
    d = dict(fused)
    assert d["a"] == (1/61 + 1/61)           # topo em ambas
    assert d["b"] == 1/62 and d["c"] == 1/62
    assert fused[0][0] == "a"                 # a vence


def test_listas_de_tamanhos_diferentes_e_vazia():
    fused = rrf_fuse([["a"], [], ["b", "a"]], c=60)
    d = dict(fused)
    assert d["a"] == (1/61 + 1/62) and d["b"] == 1/61
    assert [i for i, _ in fused] == ["a", "b"]
