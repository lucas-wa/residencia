"""Driver de avaliacao (`gcp.slice.eval`): qualidade + latencia -> linha de CSV.

O retriever e INJETADO (`fn(texto) -> list[str]`), entao aqui se testa so o driver:
montagem do `relevant` a partir do qrels, chamada cronometrada, agregacao e escrita do
CSV -- sem GPU, sem Cloud SQL e sem reimplementar metrica nenhuma (quem calcula recall,
nDCG e MRR continua sendo `benchmark.common.metrics`, ja coberto por `test_metrics.py`).
"""

import csv

import numpy as np
import pytest

from benchmark.common.types import Doc, Query
from gcp.slice import eval as eval_mod
from gcp.slice.embed_job import Hit
from gcp.slice.eval import (
    corpus_modalities,
    evaluate_architecture,
    spaces_for_corpus,
    write_comparison_csv,
)
from gcp.slice.query import search_architecture


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


# -- modalidade do corpus -> espacos fundidos ---------------------------------------------
#
# A tarefa e definida pelo CORPUS, e so os espacos que indexam aquela modalidade entram na
# fusao. E o que mantem `A` e `A+audio` IDENTICAS na tarefa doc-visual (spec secao 5).

PAGINAS = [Doc(id="110002543953_p00", image_path="pages/110002543953_p00.png"),
           Doc(id="110002543953_p01", image_path="pages/110002543953_p01.png")]
# Formato do mini-set de audio (spec secao 3.4): `id` + `audio_path`. O id vem de
# `collector.default_item_id` -- `<modalidade>:<hash da uri de origem>`.
CLIPES = [{"id": "audio_asr:552e77e1bc4d355a", "audio_path": "audio_eval/audio/a.wav"},
          {"id": "audio_asr:2aba99074d83898c", "audio_path": "audio_eval/audio/b.wav"}]


def test_corpus_de_paginas_e_doc_visual():
    assert corpus_modalities(PAGINAS) == {"doc_image"}


def test_corpus_de_clipes_cobra_as_duas_trilhas_de_audio():
    """O mesmo clipe, mesmo `item_id`: `audio_asr` no Qwen3-VL e `audio_native` no CLAP."""
    assert corpus_modalities(CLIPES) == {"audio_asr", "audio_native"}


def test_clipe_reconhecido_pela_extensao_quando_o_id_nao_diz():
    assert corpus_modalities([{"id": "sessao-1", "uri": "gs://b/sessao.MP3"}]) == {
        "audio_asr", "audio_native"}


def test_corpus_sem_pista_e_indecidivel():
    assert corpus_modalities([{"id": "x"}]) == set()


def test_doc_visual_funde_so_o_qwen3vl_nas_duas_arquiteturas():
    assert spaces_for_corpus("A", PAGINAS) == {"qwen3vl"}
    assert spaces_for_corpus("A+audio", PAGINAS) == {"qwen3vl"}   # o CLAP fica de fora


def test_audio_funde_os_espacos_capazes_de_cada_arquitetura():
    assert spaces_for_corpus("A", CLIPES) == {"qwen3vl"}           # so a trilha ASR
    assert spaces_for_corpus("A+audio", CLIPES) == {"qwen3vl", "clap"}


def test_corpus_indecidivel_funde_todos_os_espacos():
    """`None` = o comportamento antigo (todos os espacos da arquitetura)."""
    assert spaces_for_corpus("A+audio", [{"id": "x"}]) is None


class FakeStore:
    """`VectorStore` de mentira: ranking fixo, com registro das chamadas."""

    def __init__(self, ids):
        self.ids, self.calls = ids, []

    def search(self, vec, k=5, modality=None):
        self.calls.append({"k": k})
        return [Hit(iid, "text", f"gs://x/{iid}", 1.0 - 0.1 * pos)
                for pos, iid in enumerate(self.ids[:k])]


def _ranking(arch, docs, stores):
    """O mesmo encanamento do `main()`: corpus -> espacos da tarefa -> fusao."""
    embedders = {"qwen3vl": lambda ts: np.ones((len(ts), 4), dtype="float32"),
                 "clap": lambda ts: np.ones((len(ts), 2), dtype="float32")}
    return [i for i, _ in search_architecture(
        arch, "metas para a saude publica", stores=stores, embedders=embedders,
        only_spaces=spaces_for_corpus(arch, docs))]


def test_doc_visual_em_Aaudio_nao_toca_o_clap_e_iguala_A():
    """Spec secao 5: na tarefa doc-visual, `A` e `A+audio` sao IDENTICAS."""
    paginas, clipes = ["p1", "p2", "p3"], ["audio_asr:552e77e1bc4d355a"]
    stores_a = {"qwen3vl": FakeStore(paginas), "clap": FakeStore(clipes)}
    stores_aa = {"qwen3vl": FakeStore(paginas), "clap": FakeStore(clipes)}

    assert _ranking("A+audio", PAGINAS, stores_aa) == _ranking("A", PAGINAS, stores_a)
    assert stores_aa["clap"].calls == []        # o CLAP nem e consultado


def test_tarefa_de_audio_em_Aaudio_funde_os_dois_espacos():
    clipes = ["audio_asr:552e77e1bc4d355a", "audio_asr:2aba99074d83898c"]
    stores = {"qwen3vl": FakeStore(clipes[::-1]), "clap": FakeStore(clipes)}

    ranking = _ranking("A+audio", CLIPES, stores)

    assert len(stores["clap"].calls) == 1 and len(stores["qwen3vl"].calls) == 1
    assert sorted(ranking) == sorted(clipes)


def test_arquitetura_sem_espaco_para_o_corpus_falha_na_hora(monkeypatch):
    """0.0 em tudo pareceria metrica: melhor parar antes de medir o nada.

    Hoje nenhuma arquitetura do registro cai nesse caso (A ja cobre audio por ASR), entao
    o que se exercita aqui e a guarda -- ela protege o dia em que o registro crescer.
    """
    monkeypatch.setattr(eval_mod, "spaces_for_modalities", lambda *_a, **_k: [])

    with pytest.raises(ValueError, match="nao indexa nada do corpus"):
        spaces_for_corpus("A", CLIPES, tarefa="audio_eval")
