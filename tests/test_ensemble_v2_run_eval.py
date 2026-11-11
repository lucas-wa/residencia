"""Driver de avaliacao do ensemble v2: RRF + metricas por eixo/conjunto + CSV + main.

Especialistas/retrievers sao dubles (sem GPU, sem download): o `evaluate` e puro e
injetavel, e o `main` recebe `builders` dubles. Os valores de A sao conferidos com
calculo a mao (posicoes escolhidas a dedo; DCG/IDCG escritos aqui, nao via a
implementacao)."""
from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from ensemble.eval.rrf import rrf_fuse
from ensemble.eval.run_eval import (
    AXES,
    CSV_HEADER,
    _build_systems,
    ensemble_search,
    evaluate,
    main,
)
from ensemble.records import PointerRecord, save_records

METRIC_KEYS = [
    "n_queries", "recall@1", "nDCG@1", "recall@5", "nDCG@5",
    "recall@10", "nDCG@10", "MRR", "latencia",
]

# Header REAL de results/comparacao.csv (literal independente para o teste conferir).
COMPARACAO_HEADER = [
    "arquitetura", "tarefa", "n_queries",
    "recall@1", "ndcg@1", "recall@5", "ndcg@5", "recall@10", "ndcg@10",
    "mrr", "latencia_s",
]


class Ranker:
    """Duble de Specialist/retriever: ranking conhecido por texto de query, com scores
    estritamente decrescentes. Registra as chamadas para as poder espiar."""

    def __init__(self, rankings: dict[str, list[str]], modality: str = "x"):
        self.modality = modality
        self._r = rankings
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        self.calls.append((query, k))
        ids = self._r.get(query, [])
        return [(i, float(len(ids) - p)) for p, i in enumerate(ids)][:k]


def _rec(id_: str, modality: str, path: str | None = None) -> PointerRecord:
    meta = {"path": path} if path else {}
    return PointerRecord(id=id_, modality=modality, descriptor=f"desc {id_}",
                         pointer_uri=f"file://{id_}", source="t", meta=meta)


# --------------------------------------------------------------------------- #
# test 1: o ensemble e o rrf_fuse das 5 listas dos especialistas
# --------------------------------------------------------------------------- #
def test_ensemble_usa_rrf():
    q = "orçamento"
    listas = {
        "documento": ["d1", "d2"],
        "imagem": ["i1"],
        "audio": ["a1", "a2"],
        "texto": ["t1"],
        "video": ["v1"],
    }
    specs = {mod: Ranker({q: ids}, modality=mod) for mod, ids in listas.items()}

    fused = ensemble_search(specs, q, 10)

    esperado = rrf_fuse([listas[m] for m in specs], k=60)
    assert fused == esperado
    # cada especialista foi consultado exatamente uma vez, com o mesmo (query, k)
    for sp in specs.values():
        assert sp.calls == [(q, 10)]
    # os scores sao do RRF (1/(60+rank)), nao os dos especialistas
    assert fused[0][1] == pytest.approx(1.0 / 61)


# --------------------------------------------------------------------------- #
# test 2: B e n/a fora de documento; numeros em documento + conjunto
# --------------------------------------------------------------------------- #
def test_B_na_fora_de_documento():
    corpus = [
        _rec("d1", "documento"), _rec("i1", "imagem"), _rec("a1", "audio"),
        _rec("t1", "texto"), _rec("v1", "video"),
    ]
    queries = {"q1": "orçamento"}
    qrels = {"q1": {"d1", "i1"}}
    specs = {
        "documento": Ranker({"orçamento": ["d1"]}, "documento"),
        "imagem": Ranker({"orçamento": ["i1"]}, "imagem"),
        "audio": Ranker({"orçamento": ["a1"]}, "audio"),
        "texto": Ranker({"orçamento": ["t1"]}, "texto"),
        "video": Ranker({"orçamento": ["v1"]}, "video"),
    }
    retriever_a = Ranker({"orçamento": ["i1", "d1", "a1", "t1", "v1"]})
    retriever_b = Ranker({"orçamento": ["d1"]}, "documento")

    rows = evaluate(corpus, queries, qrels, specialists=specs,
                    retriever_a=retriever_a, retriever_b=retriever_b)

    b = {r["eixo"]: r for r in rows if r["sistema"] == "B"}
    # n/a (None) em todos os eixos nao-documento, inclusive latencia
    for eixo in ("imagem", "audio", "texto", "video"):
        for m in METRIC_KEYS:
            assert b[eixo][m] is None, f"B/{eixo}/{m} deveria ser None"
    # numeros reais em documento e conjunto
    for eixo in ("documento", "conjunto"):
        for m in METRIC_KEYS:
            assert b[eixo][m] is not None, f"B/{eixo}/{m} nao deveria ser None"
    assert b["documento"]["recall@1"] == 1.0          # d1 no topo, relevante-doc {d1}
    assert b["conjunto"]["recall@1"] == pytest.approx(0.5)  # d1 topo, relevante {d1,i1}


# --------------------------------------------------------------------------- #
# test 3: metricas por eixo e conjunto conferem com calculo a mao (sistema A)
# --------------------------------------------------------------------------- #
def test_metricas_por_eixo_e_conjunto():
    corpus = [
        _rec("d1", "documento"), _rec("d2", "documento"),
        _rec("i1", "imagem"), _rec("i2", "imagem"),
        _rec("a1", "audio"), _rec("t1", "texto"), _rec("v1", "video"),
    ]
    queries = {"q1": "qa", "q2": "qb"}
    qrels = {"q1": {"d1", "i1"}, "q2": {"d2", "a1"}}

    retriever_a = Ranker({
        "qa": ["i1", "i2", "d2", "d1", "a1", "t1", "v1"],
        "qb": ["d2", "a1", "d1", "i1", "i2", "t1", "v1"],
    })
    # especialistas/B triviais: so precisam nao quebrar o laco
    specs = {
        "documento": Ranker({"qa": ["d1", "d2"], "qb": ["d2", "d1"]}, "documento"),
        "imagem": Ranker({"qa": ["i1", "i2"], "qb": ["i1", "i2"]}, "imagem"),
        "audio": Ranker({"qa": ["a1"], "qb": ["a1"]}, "audio"),
        "texto": Ranker({"qa": ["t1"], "qb": ["t1"]}, "texto"),
        "video": Ranker({"qa": ["v1"], "qb": ["v1"]}, "video"),
    }
    retriever_b = Ranker({"qa": ["d1", "d2"], "qb": ["d2", "d1"]}, "documento")

    rows = evaluate(corpus, queries, qrels, specialists=specs,
                    retriever_a=retriever_a, retriever_b=retriever_b)
    a = {r["eixo"]: r for r in rows if r["sistema"] == "A"}
    e = {r["eixo"]: r for r in rows if r["sistema"] == "ensemble"}

    l3 = 1.0 / math.log2(3)   # ganho descontado na posicao 2 (indice 1)
    l5 = 1.0 / math.log2(5)   # ganho descontado na posicao 4 (indice 3)

    # --- sistema A (ranking global filtrado por eixo), media so sobre queries julgadas ---
    # documento (julgadas q1,q2): q1 doc=[d2,d1] rel={d1} (d1 no rank2); q2 doc=[d2,d1]
    # rel={d2} (perfeito).
    assert a["documento"]["n_queries"] == 2
    assert a["documento"]["recall@1"] == pytest.approx(0.5)
    assert a["documento"]["nDCG@1"] == pytest.approx(0.5)
    assert a["documento"]["recall@5"] == pytest.approx(1.0)
    assert a["documento"]["nDCG@5"] == pytest.approx((l3 + 1.0) / 2)
    assert a["documento"]["recall@10"] == pytest.approx(1.0)
    assert a["documento"]["nDCG@10"] == pytest.approx((l3 + 1.0) / 2)
    assert a["documento"]["MRR"] == pytest.approx(0.75)

    # imagem: so q1 e julgada (q2 nao tem relevante-imagem) e e perfeita -> media 1.0, n=1
    assert a["imagem"]["n_queries"] == 1
    for m in ("recall@1", "nDCG@1", "recall@5", "nDCG@5", "recall@10", "nDCG@10", "MRR"):
        assert a["imagem"][m] == pytest.approx(1.0), f"imagem/{m}"
    # audio: so q2 e julgada e e perfeita -> media 1.0, n=1
    assert a["audio"]["n_queries"] == 1
    for m in ("recall@1", "nDCG@1", "recall@5", "nDCG@5", "recall@10", "nDCG@10", "MRR"):
        assert a["audio"][m] == pytest.approx(1.0), f"audio/{m}"
    # texto e video: nenhuma query julgada nesses eixos -> celula n/a (None), nao 0.0
    for eixo in ("texto", "video"):
        for m in METRIC_KEYS:
            assert a[eixo][m] is None, f"A/{eixo}/{m} deveria ser None (nao julgado)"

    # conjunto (julgadas q1,q2): q1 rel={d1,i1} (i1 topo, d1 no indice3); q2 rel={d2,a1}
    # (perfeito nos 2).
    assert a["conjunto"]["n_queries"] == 2
    assert a["conjunto"]["recall@1"] == pytest.approx(0.5)
    assert a["conjunto"]["nDCG@1"] == pytest.approx(1.0)
    assert a["conjunto"]["recall@5"] == pytest.approx(1.0)
    ndcg5_q1 = (1.0 + l5) / (1.0 + l3)
    assert a["conjunto"]["nDCG@5"] == pytest.approx((ndcg5_q1 + 1.0) / 2)
    assert a["conjunto"]["recall@10"] == pytest.approx(1.0)
    assert a["conjunto"]["nDCG@10"] == pytest.approx((ndcg5_q1 + 1.0) / 2)
    assert a["conjunto"]["MRR"] == pytest.approx(1.0)

    # --- ensemble (RRF das 5 listas), calculado a mao no corpus pequeno ---
    # listas disjuntas por modalidade: os rank-1 empatam em 1/61 e desempatam pela ordem
    # de insercao (ordem AXES, depois rank). Para qa a lista fundida completa e:
    assert [i for i, _ in ensemble_search(specs, "qa", 7)] == \
        ["d1", "i1", "a1", "t1", "v1", "d2", "i2"]
    # per-eixo = a fundida RESTRITA aos ids do eixo; documento -> [d1,d2] (d1 topo, perfeito
    # nas duas queries julgadas); conjunto = a ordem fundida inteira.
    assert e["documento"]["n_queries"] == 2
    assert e["documento"]["recall@1"] == pytest.approx(1.0)
    assert e["documento"]["nDCG@5"] == pytest.approx(1.0)
    assert e["documento"]["MRR"] == pytest.approx(1.0)
    assert e["imagem"]["n_queries"] == 1 and e["imagem"]["recall@1"] == pytest.approx(1.0)
    assert e["audio"]["n_queries"] == 1 and e["audio"]["recall@1"] == pytest.approx(1.0)
    assert e["texto"]["recall@1"] is None and e["video"]["recall@1"] is None
    # conjunto ensemble: qa full=[d1,i1,...] rel={d1,i1}; qb full=[d2,i1,a1,...] rel={d2,a1}
    # (a1 no indice2 -> ganho 1/log2(4)=1/2).
    ndcg5_qb = 1.5 / (1.0 + l3)
    assert e["conjunto"]["n_queries"] == 2
    assert e["conjunto"]["recall@1"] == pytest.approx(0.5)
    assert e["conjunto"]["nDCG@1"] == pytest.approx(1.0)
    assert e["conjunto"]["nDCG@5"] == pytest.approx((1.0 + ndcg5_qb) / 2)
    assert e["conjunto"]["MRR"] == pytest.approx(1.0)

    # existe linha conjunto para todos os sistemas
    for sistema in ("ensemble", "A", "B"):
        assert any(r["sistema"] == sistema and r["eixo"] == "conjunto" for r in rows)


# --------------------------------------------------------------------------- #
# test 4: main injeta builders dubles, grava CSV com as colunas certas, retorna 0
# --------------------------------------------------------------------------- #
VOCAB = ["orçamento", "saúde", "estrada", "educação", "praia"]


def _bow(texts: list[str]) -> np.ndarray:
    return np.array(
        [[float(w in str(t).lower().split()) for w in VOCAB] for t in texts],
        dtype=np.float32,
    )


class FakeColRetriever:
    """Duble de ColVisionRetriever: 1 token por query sobre vetores one-hot de pagina;
    ranqueia as paginas na ordem em que foram embutidas (determinista, sem torch)."""

    def __init__(self) -> None:
        self._ids: list[str] = []

    def encode_corpus(self, docs):
        self._ids = [d.id for d in docs]
        n = len(docs)
        return [np.eye(n, dtype="float32")[i][None, :] for i in range(n)]

    def encode_queries(self, queries):
        n = len(self._ids)
        tok = np.array([[float(n - i) for i in range(n)]], dtype="float32")
        return [tok for _ in queries]


def test_main_grava_csv(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "d0.png").write_bytes(b"")
    (pages / "d1.png").write_bytes(b"")

    corpus = [
        _rec("d0", "documento", str(pages / "d0.png")),
        _rec("d1", "documento", str(pages / "d1.png")),
        _rec("i0", "imagem", "orçamento"),
        _rec("a0", "audio", "saúde"),
        _rec("t0", "texto"),
        _rec("v0", "video", "estrada"),
    ]
    corpus_path = tmp_path / "corpus.jsonl"
    save_records(corpus, corpus_path)

    queries_path = tmp_path / "queries.txt"
    queries_path.write_text("q1\torçamento saúde\nq2\tsaúde estrada\n", encoding="utf-8")

    qrels_path = tmp_path / "qrels.tsv"
    qrels_path.write_text(
        "q1 0 d0 1\nq1 0 i0 1\nq2 0 d1 1\nq2 0 a0 1\n", encoding="utf-8")

    out_path = tmp_path / "comparacao_v2.csv"

    def pair():
        return _bow, _bow

    builders = {
        "documento": FakeColRetriever,
        "imagem": pair,
        "audio": pair,
        "texto": pair,
        "video": pair,
        "unified_a": pair,
    }

    rc = main(
        ["--corpus", str(corpus_path), "--queries", str(queries_path),
         "--qrels", str(qrels_path), "--out", str(out_path)],
        builders=builders,
    )

    assert rc == 0
    assert out_path.is_file()
    linhas = out_path.read_text(encoding="utf-8").splitlines()
    # o cabecalho escrito bate com o header REAL de results/comparacao.csv (literal
    # independente), inclusive a coluna n_queries como 3a.
    assert linhas[0].split(",") == COMPARACAO_HEADER
    assert CSV_HEADER == COMPARACAO_HEADER
    assert linhas[0].split(",")[2] == "n_queries"
    # 3 sistemas x (5 eixos + conjunto) = 18 linhas de dados + cabecalho
    assert len(linhas) == 1 + 18
    corpo = "\n".join(linhas[1:])
    assert "conjunto" in corpo
    assert "n/a" in corpo                 # B fora de documento
    # nao baixou modelo: nenhuma lib de modelo foi importada
    assert not {"torch", "sentence_transformers"} & set(sys.modules)


# --------------------------------------------------------------------------- #
# test 5: --audio-mode escolhe entre ASR->texto (padrao) e CLAP (ablacao)
# --------------------------------------------------------------------------- #
def _audio_corpus() -> list[PointerRecord]:
    """Audio com transcricao em `descriptor` e arquivo em meta["path"] -- distintos."""
    return [
        PointerRecord(id="a1", modality="audio",
                      descriptor="orçamento da saúde", pointer_uri="file://a1",
                      source="radio", meta={"path": "clipes/a1.mp3"}),
        PointerRecord(id="a2", modality="audio",
                      descriptor="estrada", pointer_uri="file://a2",
                      source="radio", meta={"path": "clipes/a2.mp3"}),
    ]


def _spy_pair(registro: list, rotulo: str):
    """Builder duble: registra o rotulo e os itens que chegam a `embed_items`."""

    def build():
        def embed_items(texts):
            registro.append((rotulo, list(texts)))
            return _bow(texts)

        return embed_items, _bow

    return build


def _builders_com_espioes(registro: list) -> dict:
    builders = {
        "documento": FakeColRetriever,
        "imagem": _spy_pair(registro, "imagem"),
        "texto": _spy_pair(registro, "texto"),
        "video": _spy_pair(registro, "video"),
        "unified_a": _spy_pair(registro, "unified_a"),
        "audio": _spy_pair(registro, "audio-clap"),
        "audio_asr": _spy_pair(registro, "audio-asr"),
    }
    return builders


def test_audio_mode_asr_usa_builder_asr_e_a_transcricao():
    registro: list = []
    specialists, _, _ = _build_systems(
        _audio_corpus(), _builders_com_espioes(registro), audio_mode="asr")

    chamadas = [c for c in registro if c[0].startswith("audio-")]
    # o builder do eixo audio e o `audio_asr`, e os itens embutidos sao as TRANSCRICOES
    assert chamadas == [("audio-asr", ["orçamento da saúde", "estrada"])]
    # e o especialista montado ranqueia por elas
    assert specialists["audio"].search("orçamento saúde", 2)[0][0] == "a1"


def test_audio_mode_asr_e_o_padrao():
    registro: list = []
    _build_systems(_audio_corpus(), _builders_com_espioes(registro))
    assert [c for c in registro if c[0].startswith("audio-")] == [
        ("audio-asr", ["orçamento da saúde", "estrada"])]


def test_audio_mode_clap_preserva_o_comportamento_atual():
    registro: list = []
    _build_systems(_audio_corpus(), _builders_com_espioes(registro), audio_mode="clap")
    # builder `audio` (CLAP) sobre os CAMINHOS dos arquivos -- representacao padrao
    assert [c for c in registro if c[0].startswith("audio-")] == [
        ("audio-clap", ["clipes/a1.mp3", "clipes/a2.mp3"])]


def test_audio_asr_cai_para_o_builder_audio_quando_a_chave_falta():
    # builders injetados sem `audio_asr` (dubles antigos) continuam funcionando no
    # modo asr: so a representacao muda para a transcricao.
    registro: list = []
    builders = _builders_com_espioes(registro)
    del builders["audio_asr"]
    _build_systems(_audio_corpus(), builders, audio_mode="asr")
    assert [c for c in registro if c[0].startswith("audio-")] == [
        ("audio-clap", ["orçamento da saúde", "estrada"])]


def test_default_builders_registra_audio_asr():
    from ensemble.eval.run_eval import _default_builders

    builders = _default_builders()
    assert "audio_asr" in builders and "audio" in builders
    assert callable(builders["audio_asr"])
    assert not {"torch", "sentence_transformers"} & set(sys.modules)


def test_main_arquivo_faltando_retorna_nao_zero(tmp_path):
    rc = main(
        ["--corpus", str(tmp_path / "nao_existe.jsonl"),
         "--queries", str(tmp_path / "q.txt"),
         "--qrels", str(tmp_path / "qr.tsv"),
         "--out", str(tmp_path / "out.csv")],
        builders={},
    )
    assert rc != 0
