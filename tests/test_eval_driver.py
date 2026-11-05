"""Driver de avaliacao (`gcp.slice.eval`): qualidade + latencia -> linha de CSV.

O retriever e INJETADO (`fn(texto) -> list[str]`), entao aqui se testa so o driver:
montagem do `relevant` a partir do qrels, chamada cronometrada, agregacao e escrita do
CSV -- sem GPU, sem Cloud SQL e sem reimplementar metrica nenhuma (quem calcula recall,
nDCG e MRR continua sendo `benchmark.common.metrics`, ja coberto por `test_metrics.py`).
"""

import csv

import pytest

from benchmark.common.types import Query
from gcp.slice import eval as eval_mod
from gcp.slice.eval import evaluate_architecture, write_comparison_csv


def test_metricas_com_retriever_conhecido():
    qrels = {"q1": {"d1"}}                      # d1 relevante
    def retrieve(q):                            # d1 na posicao 2
        return ["d2", "d1", "d3"]

    m = evaluate_architecture("A", [{"id": "q1", "text": "x"}], qrels, retrieve,
                              k_values=(1, 3))

    assert m["recall@1"] == 0.0 and m["recall@3"] == 1.0
    assert m["mrr"] == 0.5 and m["latencia_s"] >= 0.0


def test_query_sem_relevante_nao_quebra():
    m = evaluate_architecture("A", [{"id": "q9", "text": "x"}], {"q9": set()},
                              lambda q: ["d1"], k_values=(1,))

    assert m["recall@1"] == 0.0 and m["ndcg@1"] == 0.0   # 0.0, nao NaN


def test_aceita_a_forma_real_do_load_local():
    """`load_local` devolve `Query` (dataclass) e qrels `{qid: {docid: score}}`."""
    queries = [Query(id="q1", text="metas para a saude"), Query(id="q2", text="saneamento")]
    qrels = {"q1": {"d1": 1.0}, "q2": {"d7": 1.0}}
    retrieve = {"metas para a saude": ["d1", "dx"], "saneamento": ["dx", "d7"]}.__getitem__

    m = evaluate_architecture("A+audio", queries, qrels, retrieve, k_values=(1, 5))

    assert m["recall@5"] == 1.0          # os dois relevantes aparecem no top-5
    assert m["recall@1"] == 0.5          # so o da q1 esta no top-1
    assert m["mrr"] == pytest.approx((1.0 + 0.5) / 2)


def test_query_fora_do_qrels_vale_zero():
    m = evaluate_architecture("A", [{"id": "orfa", "text": "x"}], {}, lambda q: ["d1"],
                              k_values=(1,))

    assert m["recall@1"] == 0.0 and m["ndcg@1"] == 0.0 and m["mrr"] == 0.0


def test_julgamento_zero_nao_e_relevante():
    """Score <= 0 no qrels e julgamento NEGATIVO: nao entra no denominador do recall."""
    qrels = {"q1": {"d1": 1.0, "d2": 0.0}}

    m = evaluate_architecture("A", [{"id": "q1", "text": "x"}], qrels, lambda q: ["d1"],
                              k_values=(1,))

    assert m["recall@1"] == 1.0          # 1/1, e nao 1/2


def test_retriever_recebe_o_texto_e_so_uma_vez_por_query():
    chamadas = []

    def retrieve(texto):
        chamadas.append(texto)
        return ["d1"]

    evaluate_architecture("A", [{"id": "q1", "text": "educacao"},
                                {"id": "q2", "text": "saude"}],
                          {"q1": {"d1"}}, retrieve, k_values=(1,))

    assert chamadas == ["educacao", "saude"]


def test_latencia_e_a_media_do_tempo_de_cada_retrieve(monkeypatch):
    """Cronometra so o `retrieve` -- e a media por query que vai para o CSV."""
    relogio = iter([0.0, 1.0, 10.0, 13.0])       # q1 levou 1s, q2 levou 3s
    monkeypatch.setattr(eval_mod.time, "perf_counter", lambda: next(relogio))

    m = evaluate_architecture("A", [{"id": "q1", "text": "a"}, {"id": "q2", "text": "b"}],
                              {}, lambda q: ["d1"], k_values=(1,))

    assert m["latencia_s"] == pytest.approx(2.0)


def test_resultado_traz_a_arquitetura_e_o_n_de_queries():
    m = evaluate_architecture("A+audio", [{"id": "q1", "text": "x"}], {"q1": {"d1"}},
                              lambda q: ["d1"], k_values=(1,))

    assert m["arquitetura"] == "A+audio" and m["n_queries"] == 1


def test_query_sem_texto_falha_na_hora():
    """`Query.text` e opcional no dataclass; um `retrieve(None)` so falharia la na frente."""
    with pytest.raises(ValueError, match="q5"):
        evaluate_architecture("A", [Query(id="q5", image_path="p.png")], {},
                              lambda q: ["d1"], k_values=(1,))


def test_conjunto_vazio_nao_quebra():
    m = evaluate_architecture("A", [], {}, lambda q: ["d1"], k_values=(1,))

    assert m["n_queries"] == 0 and m["arquitetura"] == "A"


def test_csv_uma_linha_por_arquitetura_e_tarefa(tmp_path):
    destino = tmp_path / "saida" / "comparacao.csv"     # diretorio ainda nao existe
    linhas = [
        {"arquitetura": "A", "tarefa": "planos", "recall@5": 0.5, "latencia_s": 0.1},
        {"arquitetura": "A+audio", "tarefa": "planos", "recall@5": 0.75, "latencia_s": 0.2},
    ]

    write_comparison_csv(destino, linhas)

    lido = list(csv.DictReader(destino.read_text(encoding="utf-8").splitlines()))
    assert [(r["arquitetura"], r["tarefa"]) for r in lido] == [("A", "planos"),
                                                               ("A+audio", "planos")]
    assert lido[1]["recall@5"] == "0.75"


def test_csv_une_as_colunas_de_linhas_diferentes(tmp_path):
    """Arquiteturas avaliadas com `k_values` diferentes nao podem derrubar a escrita."""
    destino = tmp_path / "c.csv"

    write_comparison_csv(destino, [{"arquitetura": "A", "recall@1": 1.0},
                                   {"arquitetura": "B", "recall@10": 0.5}])

    lido = list(csv.DictReader(destino.read_text(encoding="utf-8").splitlines()))
    assert lido[0]["recall@10"] == "" and lido[1]["recall@1"] == ""


def test_csv_sem_linhas_nao_quebra(tmp_path):
    destino = tmp_path / "vazio.csv"

    write_comparison_csv(destino, [])

    assert destino.exists() and destino.read_text(encoding="utf-8") == ""
