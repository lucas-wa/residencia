"""Arquitetura B (eixo documento): roteamento late-interaction + ranqueamento MaxSim.

Tudo aqui roda SEM GPU e SEM Cloud SQL: o `ColVisionRetriever` e substituido por um duble
que devolve matrizes [n_tokens, dim] conhecidas, no mesmo padrao dos fakes de
`test_eval_driver.py` / `test_gcp_slice.py`. O que se prova e:

  (a) `--arch B` roteia para o caminho late-interaction e NAO toca o pgvector;
  (b) com um ranqueador de ordem conhecida, `main()` grava a linha `B,ptbr_planos,...`
      com as metricas certas, ao lado da linha de `A`;
  (c) o registro trata `B` sem espaco pgvector sem quebrar `A`/`A+audio`.
"""

import csv
import json

import numpy as np
import pytest

from benchmark.common.types import Doc
from gcp.slice import eval as eval_mod
from gcp.slice.doc_specialist import (
    DocSpecialistRanker,
    build_doc_retriever,
    resolve_image_path,
)
from gcp.slice.query import search_architecture
from gcp.slice.spaces import (
    SPECIALISTS,
    architecture_spaces,
    architecture_specialists,
    specialists_for_modalities,
    uses_pgvector,
)

ESPECIALISTA = SPECIALISTS["colqwen25_doc"]


# -- (c) registro: B sem espaco pgvector, A/A+audio intactas ------------------------------


def test_B_nao_tem_espaco_pgvector_e_A_continua_tendo():
    assert architecture_spaces("B") == []
    assert uses_pgvector("B") is False
    assert uses_pgvector("A") is True and uses_pgvector("A+audio") is True


def test_B_tem_o_especialista_do_eixo_documento():
    assert [e.name for e in architecture_specialists("B")] == ["colqwen25_doc"]
    assert ESPECIALISTA.kind == "colqwen2.5"
    assert ESPECIALISTA.model_id == "vidore/colqwen2.5-v0.2"


def test_A_e_Aaudio_nao_tem_especialista():
    assert architecture_specialists("A") == []
    assert architecture_specialists("A+audio") == []


def test_especialista_por_modalidade_sai_do_registro():
    assert [e.name for e in specialists_for_modalities("B", {"doc_image"})] \
        == ["colqwen25_doc"]
    # o eixo documento nao ranqueia audio: B nao tem o que dizer sobre aquele corpus
    assert specialists_for_modalities("B", {"audio_asr", "audio_native"}) == []


def test_arquitetura_desconhecida_falha_nos_dois_registros():
    with pytest.raises(ValueError, match="desconhecida"):
        architecture_specialists("Z")
    with pytest.raises(ValueError, match="desconhecida"):
        architecture_spaces("Z")


def test_consulta_pgvector_recusa_arquitetura_sem_espaco():
    """`B` por `search_architecture` devolveria `[]` em silencio: tem de ser erro."""
    with pytest.raises(ValueError, match="nao tem espaco pgvector"):
        search_architecture("B", "metas", stores={}, embedders={})


# -- duble do retriever late-interaction ---------------------------------------------------


class FakeColRetriever:
    """Duble de `ColVisionRetriever`: vetores 1-hot, MaxSim com ordem CONHECIDA.

    Cada pagina vira UM token one-hot na sua posicao; a consulta vira um token one-hot por
    pagina desejada, com peso decrescente. MaxSim entao soma, para cada token da consulta,
    o peso da pagina que ele aponta -- ou seja, o ranking sai exatamente na ordem que o
    teste pediu, e os pesos de `encode_corpus`/`encode_queries` continuam sendo o que o
    `maxsim_search` de verdade consome.
    """

    def __init__(self, doc_ids: list[str], ordem: dict[str, list[str]]):
        self.doc_ids = doc_ids
        self.ordem = ordem
        self.corpus_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def encode_corpus(self, docs):
        self.corpus_calls.append([d.image_path for d in docs])
        n = len(self.doc_ids)
        return [np.eye(n, dtype="float32")[self.doc_ids.index(d.id)][None, :]
                for d in docs]

    def encode_queries(self, queries):
        n = len(self.doc_ids)
        saida = []
        for q in queries:
            self.query_calls.append(q.text)
            desejada = self.ordem[q.text]
            tokens = np.zeros((len(desejada), n), dtype="float32")
            for posicao, item_id in enumerate(desejada):
                tokens[posicao, self.doc_ids.index(item_id)] = float(len(desejada) - posicao)
            saida.append(tokens)
        return saida


def _corpus(tmp_path, ids):
    """Corpus minimo em disco: `<tmp>/pages/<id>.png` + os `Doc` que o apontam."""
    pages = tmp_path / "pages"
    pages.mkdir(exist_ok=True)
    docs = []
    for item_id in ids:
        arquivo = pages / f"{item_id}.png"
        arquivo.write_bytes(b"")
        docs.append(Doc(id=item_id, image_path=str(arquivo)))
    return docs


IDS = ["p00", "p01", "p02"]


# -- ranqueamento MaxSim -------------------------------------------------------------------


def test_ranking_sai_na_ordem_do_maxsim_e_com_os_ids_do_corpus(tmp_path):
    docs = _corpus(tmp_path, IDS)
    fake = FakeColRetriever(IDS, {"saude": ["p02", "p00", "p01"]})

    retrieve = build_doc_retriever(docs, specialist=ESPECIALISTA, retriever=fake)

    assert retrieve("saude") == ["p02", "p00", "p01"]
    assert fake.query_calls == ["saude"]


def test_corpus_e_embutido_uma_vez_so(tmp_path):
    """Indexar e custo de indice, nao de query: nao pode entrar no cronometro por query."""
    docs = _corpus(tmp_path, IDS)
    fake = FakeColRetriever(IDS, {"a": IDS, "b": IDS[::-1]})

    retrieve = build_doc_retriever(docs, specialist=ESPECIALISTA, retriever=fake)
    retrieve("a")
    retrieve("b")

    assert len(fake.corpus_calls) == 1 and len(fake.corpus_calls[0]) == 3


def test_ranking_cobre_o_corpus_inteiro_e_nao_um_top_k(tmp_path):
    """Cortar antes zeraria um `recall@10` sem economizar nada com 93 paginas."""
    docs = _corpus(tmp_path, IDS)
    fake = FakeColRetriever(IDS, {"x": ["p01"]})

    ranking = build_doc_retriever(docs, specialist=ESPECIALISTA, retriever=fake)("x")

    assert ranking[0] == "p01" and sorted(ranking) == sorted(IDS)


def test_corpus_vazio_falha():
    with pytest.raises(ValueError, match="corpus vazio"):
        build_doc_retriever([], specialist=ESPECIALISTA, retriever=object())


def test_item_sem_pagina_falha_nomeando_o_id():
    with pytest.raises(ValueError, match="p09"):
        build_doc_retriever([Doc(id="p09", text="so texto")], specialist=ESPECIALISTA,
                            retriever=object())


def test_ids_e_embeddings_de_tamanhos_diferentes_falham():
    with pytest.raises(ValueError, match="2 ids para 1 embeddings"):
        DocSpecialistRanker(["a", "b"], [np.zeros((1, 2), dtype="float32")], object())


# -- resolucao do caminho da pagina --------------------------------------------------------


def test_caminho_do_corpus_vem_com_barra_invertida_do_windows(tmp_path):
    """O `corpus.jsonl` real grava `data\\govexe\\...\\p00.png`: no Linux isso e um nome."""
    _corpus(tmp_path, ["p00"])                      # cria <tmp>/pages/p00.png

    achado = resolve_image_path("pages\\p00.png", data_dir=tmp_path)

    assert achado == tmp_path / "pages" / "p00.png"


def test_pages_dir_tem_precedencia_e_casa_pelo_basename(tmp_path):
    """Na VM a pagina vem do Bronze, nao do repo -- o objeto preserva o nome do arquivo."""
    bronze = tmp_path / "bronze"
    bronze.mkdir()
    (bronze / "p00.png").write_bytes(b"")

    achado = resolve_image_path("data/govexe/ptbr_planos/pages/p00.png", pages_dir=bronze)

    assert achado == bronze / "p00.png"


def test_pagina_ausente_falha_apontando_o_pages_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="--pages-dir"):
        resolve_image_path("pages/sumiu.png", data_dir=tmp_path)


# -- (a)+(b) roteamento do `main()` e a linha do CSV ---------------------------------------


def _escrever_conjunto(tmp_path, *, relevante: str):
    """Um `data-dir` no formato de `load_local` (o mesmo de `data/govexe/ptbr_planos`)."""
    dados = tmp_path / "ptbr_planos"
    dados.mkdir()
    pages = dados / "pages"
    pages.mkdir()
    linhas = []
    for item_id in IDS:
        (pages / f"{item_id}.png").write_bytes(b"")
        # caminho no formato do corpus REAL: relativo e com barra invertida do Windows
        linhas.append(json.dumps({"id": item_id,
                                  "image_path": f"pages\\{item_id}.png"}))
    (dados / "corpus.jsonl").write_text("\n".join(linhas), encoding="utf-8")
    (dados / "queries.jsonl").write_text(
        json.dumps({"id": "q1", "text": "saude"}), encoding="utf-8")
    (dados / "qrels.tsv").write_text(f"q1\t{relevante}\t1\n", encoding="utf-8")
    return dados


def test_main_arch_B_grava_a_linha_com_as_metricas_do_ranker_fake(tmp_path, monkeypatch):
    dados = _escrever_conjunto(tmp_path, relevante="p01")
    destino = tmp_path / "results" / "comparacao.csv"
    fake = FakeColRetriever(IDS, {"saude": ["p02", "p01", "p00"]})   # relevante na 2a posicao

    # Nenhum store pgvector pode ser aberto no caminho de B.
    monkeypatch.setattr(eval_mod, "_stores_e_embedders",
                        lambda *a, **k: pytest.fail("B nao pode abrir store pgvector"))

    rc = eval_mod.main(["--arch", "B", "--data-dir", str(dados), "--out", str(destino),
                        "--k-values", "1", "3"], retriever=fake)

    assert rc == 0
    linha = list(csv.DictReader(destino.read_text(encoding="utf-8").splitlines()))[0]
    assert linha["arquitetura"] == "B" and linha["tarefa"] == "ptbr_planos"
    assert float(linha["recall@1"]) == 0.0 and float(linha["recall@3"]) == 1.0
    assert float(linha["mrr"]) == 0.5            # relevante na posicao 2
    assert int(linha["n_queries"]) == 1
    assert fake.query_calls == ["saude"]         # a consulta passou pelo especialista


def test_main_arch_B_acrescenta_a_linha_sem_apagar_a_de_A(tmp_path):
    """O CSV e comparativo: a linha de B entra ao lado da de A, mesma tarefa."""
    dados = _escrever_conjunto(tmp_path, relevante="p00")
    destino = tmp_path / "comparacao.csv"
    destino.write_text("arquitetura,tarefa,recall@1\nA,ptbr_planos,0.25\n", encoding="utf-8")

    eval_mod.main(["--arch", "B", "--data-dir", str(dados), "--out", str(destino),
                   "--k-values", "1"],
                  retriever=FakeColRetriever(IDS, {"saude": ["p00", "p01", "p02"]}))

    lido = list(csv.DictReader(destino.read_text(encoding="utf-8").splitlines()))
    assert [(r["arquitetura"], r["tarefa"]) for r in lido] == [("A", "ptbr_planos"),
                                                               ("B", "ptbr_planos")]
    assert float(lido[1]["recall@1"]) == 1.0


def test_main_arch_B_reexecutado_nao_duplica_a_propria_linha(tmp_path):
    dados = _escrever_conjunto(tmp_path, relevante="p00")
    destino = tmp_path / "comparacao.csv"
    for _ in range(2):
        eval_mod.main(["--arch", "B", "--data-dir", str(dados), "--out", str(destino),
                       "--k-values", "1"],
                      retriever=FakeColRetriever(IDS, {"saude": IDS}))

    lido = list(csv.DictReader(destino.read_text(encoding="utf-8").splitlines()))
    assert [r["arquitetura"] for r in lido] == ["B"]


def test_main_arch_B_usa_as_paginas_do_pages_dir(tmp_path):
    """Na VM o corpus vem do repo e as paginas do Bronze -- e `--pages-dir` as liga."""
    dados = _escrever_conjunto(tmp_path, relevante="p00")
    bronze = tmp_path / "bronze"
    bronze.mkdir()
    for item_id in IDS:
        (bronze / f"{item_id}.png").write_bytes(b"x")
    fake = FakeColRetriever(IDS, {"saude": IDS})

    eval_mod.main(["--arch", "B", "--data-dir", str(dados), "--out",
                   str(tmp_path / "c.csv"), "--pages-dir", str(bronze)], retriever=fake)

    assert all(str(bronze) in caminho for caminho in fake.corpus_calls[0])


def test_pages_dir_e_recusado_nas_arquiteturas_pgvector(tmp_path):
    dados = _escrever_conjunto(tmp_path, relevante="p00")
    with pytest.raises(SystemExit):
        eval_mod.main(["--arch", "A", "--data-dir", str(dados), "--out",
                       str(tmp_path / "c.csv"), "--pages-dir", str(tmp_path)])


# -- tarefa x especialista: quem ranqueia o que sai do registro ----------------------------


CLIPES = [{"id": "audio_asr:552e77e1bc4d355a", "audio_path": "a.wav"}]
PAGINAS = [Doc(id="p00", image_path="pages/p00.png")]


def test_especialista_da_tarefa_doc_visual_e_o_eixo_documento():
    assert eval_mod.specialist_for_corpus("B", PAGINAS, tarefa="ptbr_planos").name \
        == "colqwen25_doc"


def test_B_recusa_um_corpus_de_audio_em_vez_de_medir_zero():
    with pytest.raises(ValueError, match="nao ranqueia nada do corpus"):
        eval_mod.specialist_for_corpus("B", CLIPES, tarefa="audio_eval")


def test_B_recusa_corpus_de_modalidade_indecidivel():
    with pytest.raises(ValueError, match="modalidade do corpus"):
        eval_mod.specialist_for_corpus("B", [{"id": "x"}])


def test_dois_especialistas_para_a_mesma_tarefa_param_a_rodada(monkeypatch):
    """Fundir bracos do ensemble ainda nao existe: escolher um em silencio seria pior."""
    monkeypatch.setattr(eval_mod, "specialists_for_modalities",
                        lambda *a, **k: [ESPECIALISTA, ESPECIALISTA])

    with pytest.raises(ValueError, match="fusao entre bracos"):
        eval_mod.specialist_for_corpus("B", PAGINAS)
