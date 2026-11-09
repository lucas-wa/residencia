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
    ensemble_search,
    evaluate,
    main,
)
from ensemble.records import PointerRecord, save_records

METRIC_KEYS = [
    "recall@1", "nDCG@1", "recall@5", "nDCG@5",
    "recall@10", "nDCG@10", "MRR", "latencia",
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

    l3 = 1.0 / math.log2(3)   # ganho descontado na posicao 2 (indice 1)
    l5 = 1.0 / math.log2(5)   # ganho descontado na posicao 4 (indice 3)

    # documento: q1 doc=[d2,d1] rel={d1} (d1 no rank2); q2 doc=[d2,d1] rel={d2} (perfeito)
    assert a["documento"]["recall@1"] == pytest.approx(0.5)
    assert a["documento"]["nDCG@1"] == pytest.approx(0.5)
    assert a["documento"]["recall@5"] == pytest.approx(1.0)
    assert a["documento"]["nDCG@5"] == pytest.approx((l3 + 1.0) / 2)
    assert a["documento"]["recall@10"] == pytest.approx(1.0)
    assert a["documento"]["nDCG@10"] == pytest.approx((l3 + 1.0) / 2)
    assert a["documento"]["MRR"] == pytest.approx(0.75)

    # imagem: q1 perfeito (i1 topo), q2 sem relevante-imagem -> 0 -> media 0.5
    for m in ("recall@1", "nDCG@1", "recall@5", "nDCG@5", "recall@10", "nDCG@10", "MRR"):
        assert a["imagem"][m] == pytest.approx(0.5), f"imagem/{m}"
    # audio: q1 sem relevante-audio -> 0, q2 perfeito -> media 0.5
    for m in ("recall@1", "nDCG@1", "recall@5", "nDCG@5", "recall@10", "nDCG@10", "MRR"):
        assert a["audio"][m] == pytest.approx(0.5), f"audio/{m}"
    # texto e video: nenhuma query tem relevante nesses eixos -> 0.0
    for eixo in ("texto", "video"):
        for m in ("recall@1", "nDCG@1", "recall@5", "nDCG@5", "recall@10", "nDCG@10", "MRR"):
            assert a[eixo][m] == pytest.approx(0.0), f"{eixo}/{m}"

    # conjunto: q1 rel={d1,i1} (i1 topo, d1 no indice3); q2 rel={d2,a1} (perfeito nos 2)
    assert "conjunto" in a
    assert a["conjunto"]["recall@1"] == pytest.approx(0.5)
    assert a["conjunto"]["nDCG@1"] == pytest.approx(1.0)
    assert a["conjunto"]["recall@5"] == pytest.approx(1.0)
    ndcg5_q1 = (1.0 + l5) / (1.0 + l3)
    assert a["conjunto"]["nDCG@5"] == pytest.approx((ndcg5_q1 + 1.0) / 2)
    assert a["conjunto"]["recall@10"] == pytest.approx(1.0)
    assert a["conjunto"]["nDCG@10"] == pytest.approx((ndcg5_q1 + 1.0) / 2)
    assert a["conjunto"]["MRR"] == pytest.approx(1.0)

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
    assert linhas[0].split(",") == CSV_HEADER
    # 3 sistemas x (5 eixos + conjunto) = 18 linhas de dados + cabecalho
    assert len(linhas) == 1 + 18
    corpo = "\n".join(linhas[1:])
    assert "conjunto" in corpo
    assert "n/a" in corpo                 # B fora de documento
    # nao baixou modelo: nenhuma lib de modelo foi importada
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
