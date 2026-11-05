"""Consulta multi-espaco (`gcp.slice.query.search_architecture`): top-k por espaco + RRF.

O que se testa aqui e o *encanamento*, nao o modelo: stores e embedders sao injetados, e
os fakes devolvem rankings conhecidos por espaco. Assim da para afirmar a ordem final
contra o `rrf_fuse` puro -- sem GPU, sem Cloud SQL e sem baixar checkpoint nenhum.
"""

import numpy as np
import pytest

from gcp.slice.embed_job import Hit
from gcp.slice.fusion import rrf_fuse
from gcp.slice.query import search_architecture


class FakeStore:
    """`VectorStore` de mentira: devolve sempre o mesmo ranking e anota como foi chamado."""

    def __init__(self, ids: list[str]):
        self.ids = ids
        self.calls: list[dict] = []

    def search(self, vec, k=5, modality=None) -> list[Hit]:
        self.calls.append({"vec": np.asarray(vec), "k": k, "modality": modality})
        # score decrescente so para parecer um indice de verdade: o RRF olha a POSICAO.
        return [Hit(iid, "text", f"gs://x/{iid}", 1.0 - 0.1 * pos)
                for pos, iid in enumerate(self.ids[:k])]


def fake_embedder(vec: np.ndarray, registro: list[list[str]] | None = None):
    """Embedder de texto injetavel: `list[str] -> np.ndarray`, com log opcional das chamadas."""
    def embed(texts: list[str]) -> np.ndarray:
        if registro is not None:
            registro.append(list(texts))
        return np.repeat(np.atleast_2d(vec), len(texts), axis=0)
    return embed


def test_funde_dois_espacos_por_rrf():
    stores = {"qwen3vl": FakeStore(["p1", "p2"]), "clap": FakeStore(["aud1", "p1"])}
    embedders = {"qwen3vl": fake_embedder(np.ones(4, dtype="float32")),
                 "clap": fake_embedder(np.ones(2, dtype="float32"))}

    fundido = search_architecture("A+audio", "sessao sobre saneamento",
                                  stores=stores, embedders=embedders)

    assert fundido == rrf_fuse([["p1", "p2"], ["aud1", "p1"]])
    # p1 aparece nos DOIS espacos -> vence aud1, que lidera so um
    assert [i for i, _ in fundido] == ["p1", "aud1", "p2"]


def test_arquitetura_A_usa_so_um_espaco():
    qwen, clap = FakeStore(["p1", "p2", "p3"]), FakeStore(["aud1"])
    stores = {"qwen3vl": qwen, "clap": clap}
    embedders = {"qwen3vl": fake_embedder(np.ones(4, dtype="float32")),
                 "clap": fake_embedder(np.ones(2, dtype="float32"))}

    fundido = search_architecture("A", "metas para a saude publica",
                                  stores=stores, embedders=embedders)

    assert [i for i, _ in fundido] == ["p1", "p2", "p3"]   # a ordem do unico espaco
    assert clap.calls == []                                # o espaco de fora nem e tocado


def test_cada_espaco_recebe_o_seu_embedder_e_o_k():
    chamadas_qwen: list[list[str]] = []
    chamadas_clap: list[list[str]] = []
    stores = {"qwen3vl": FakeStore(["p1"]), "clap": FakeStore(["aud1"])}
    embedders = {
        "qwen3vl": fake_embedder(np.full(4, 0.5, dtype="float32"), chamadas_qwen),
        "clap": fake_embedder(np.full(2, 0.25, dtype="float32"), chamadas_clap),
    }

    search_architecture("A+audio", "educacao", k=3, stores=stores, embedders=embedders)

    assert chamadas_qwen == [["educacao"]] and chamadas_clap == [["educacao"]]
    # cada store recebe o vetor DO SEU espaco (dimensoes diferentes) e o mesmo k
    assert stores["qwen3vl"].calls[0]["vec"].shape == (4,)
    assert stores["clap"].calls[0]["vec"].shape == (2,)
    assert [s.calls[0]["k"] for s in stores.values()] == [3, 3]


def test_arquitetura_desconhecida_falha_na_hora():
    with pytest.raises(ValueError, match="arquitetura desconhecida"):
        search_architecture("Z", "x", stores={}, embedders={})
