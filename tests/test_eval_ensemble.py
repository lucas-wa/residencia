"""Hook de avaliacao antes/depois no benchmark do ENSEMBLE (v4) -- tudo OFFLINE.

`unified_ndcg_at10` recebe os embedders por injecao (os mesmos callables que o
`UnifiedRetrieverA` recebe no driver), entao um duble deterministico de bag-of-words
exercita a metrica inteira sem GPU e sem baixar o Qwen3-VL.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ensemble.records import PointerRecord
from finetune import eval_ensemble as E

VOCAB = ["orcamento", "saude", "educacao", "estrada"]


def _bow(texts: list[str]) -> np.ndarray:
    """Bag-of-words one-hot: o item cujo `a_repr` casa com a query e o mais similar."""
    return np.array([[float(w in t.lower().split()) for w in VOCAB] for t in texts],
                    dtype=np.float32)


def _bow_itens(pares: list[tuple[str, str]]) -> np.ndarray:
    """`embed_items` do A: recebe `(rota, valor)` (o A e multimodal); o duble embute o valor."""
    return _bow([valor for _, valor in pares])


def _bow_desalinhado(texts: list[str]) -> np.ndarray:
    """Query com 1.0 no item ERRADO e 0.9 no certo: o relevante cai para o 2o lugar."""
    saida = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for i, linha in enumerate(_bow(texts)):
        certo = int(np.argmax(linha))
        saida[i, (certo + 1) % len(VOCAB)] = 1.0
        saida[i, certo] = 0.9
    return saida


def _recs() -> list[PointerRecord]:
    """Um item por eixo; `a_repr` e o token que a query procura."""
    return [
        PointerRecord("doc1", "documento", "d", "file://doc1", "t", {"a_repr": "orcamento"}),
        PointerRecord("img1", "imagem", "d", "file://img1", "t", {"a_repr": "saude"}),
        PointerRecord("aud1", "audio", "d", "file://aud1", "t", {"a_repr": "educacao"}),
        PointerRecord("vid1", "video", "d", "file://vid1", "t", {"a_repr": "estrada"}),
    ]


QUERIES = {"q1": "orcamento", "q2": "saude"}
QRELS = {"q1": {"doc1"}, "q2": {"img1"}}

# nDCG@10 de um unico relevante colocado em 2o lugar: (1/log2(3)) / 1.
NDCG_SEGUNDO = 1 / math.log2(3)


def test_unified_ndcg_at10_perfeito_e_um():
    valor = E.unified_ndcg_at10(_recs(), QUERIES, QRELS,
                                embed_items=_bow_itens, embed_query=_bow)

    assert valor == pytest.approx(1.0)


def test_unified_ndcg_at10_cai_quando_o_relevante_e_segundo():
    valor = E.unified_ndcg_at10(_recs(), QUERIES, QRELS,
                                embed_items=_bow_itens, embed_query=_bow_desalinhado)

    assert valor == pytest.approx(NDCG_SEGUNDO, rel=1e-6)
    assert 0.62 < valor < 0.64                      # ~0.63, nao 0 e nao 1


def test_unified_ndcg_at10_so_conta_as_queries_julgadas():
    """Criterio do `run_eval.evaluate` (linha `conjunto`): so entra quem tem relevante."""
    queries = {**QUERIES, "q3": "estrada", "q4": "educacao"}
    qrels = {**QRELS, "q3": set()}                  # q3 julgada vazia, q4 sem linha

    valor = E.unified_ndcg_at10(_recs(), queries, qrels,
                                embed_items=_bow_itens, embed_query=_bow)

    # media sobre 2 julgadas (q1/q2), nao sobre as 4 queries
    assert valor == pytest.approx(1.0)


def test_unified_ndcg_at10_sem_query_julgada_e_zero():
    valor = E.unified_ndcg_at10(_recs(), {"q9": "estrada"}, {},
                                embed_items=_bow_itens, embed_query=_bow)

    assert valor == 0.0


def test_unified_ndcg_at10_embute_o_corpus_uma_vez():
    chamadas: list[list[tuple[str, str]]] = []

    def contando(pares):
        chamadas.append(list(pares))
        return _bow_itens(pares)

    E.unified_ndcg_at10(_recs(), QUERIES, QRELS,
                        embed_items=contando, embed_query=_bow)

    assert len(chamadas) == 1 and len(chamadas[0]) == 4
    # contrato do A multimodal: chegam pares (rota, valor), na ordem dos records
    assert chamadas[0] == [("image", "orcamento"), ("image", "saude"),
                           ("text", "educacao"), ("image", "estrada")]


# -- wrapper antes/depois --------------------------------------------------------------


def test_run_ensemble_eval_before_after_usa_o_adaptador_so_no_depois(tmp_path, monkeypatch):
    """Antes = A sem adaptador; depois = A com `adapter_path=<dir do treino>`."""
    import ensemble.native.embedders as emb
    from finetune.train_lora import resolve_config

    construidos: list[str | None] = []

    def fake_builder(adapter_path=None):
        construidos.append(adapter_path)
        # sem adaptador o A erra o topo; com o adaptador acerta
        return _bow_itens, (_bow if adapter_path else _bow_desalinhado)

    monkeypatch.setattr(emb, "build_unified_a_embedder", fake_builder)
    adapter = tmp_path / "qwen3vl-lora"
    adapter.mkdir()
    cfg = resolve_config("qwen3vl", "lora")

    res = E.run_ensemble_eval_before_after(cfg, adapter, _recs(), QUERIES, QRELS)

    assert construidos == [None, str(adapter)]
    assert res["antes"] == pytest.approx(NDCG_SEGUNDO, rel=1e-6)
    assert res["depois"] == pytest.approx(1.0)
    assert res["delta"] == pytest.approx(res["depois"] - res["antes"])
    assert res["n_queries"] == 2


def test_load_eval_data_le_corpus_queries_e_qrels(tmp_path):
    from ensemble.records import save_records

    corpus = tmp_path / "records.jsonl"
    save_records(_recs(), corpus)
    queries = tmp_path / "queries.txt"
    queries.write_text("q1\torcamento\n", encoding="utf-8")
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("q1 0 doc1 1\nq1 0 img1 0\n", encoding="utf-8")

    records, qs, qr = E.load_eval_data(corpus, queries, qrels)

    assert [r.id for r in records] == ["doc1", "img1", "aud1", "vid1"]
    assert qs == {"q1": "orcamento"}
    assert qr == {"q1": {"doc1"}}
