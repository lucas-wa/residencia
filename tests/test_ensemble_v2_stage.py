"""STAGE `ensemble_v2` do slurm_run.sh: prova que o driver de eval consumido pela stage
(`ensemble.eval.run_eval:main`) grava o CSV comparativo com builders dubles, sem GPU nem
download. Reusa a fiacao da Task 7 (builders injetados): importar o driver e rodar o `main`
nao toca em torch/checkpoint -- os modelos reais so rodam no Slurm H100.

O gate REAL desta task e o shell (`bash -n slurm_run.sh` rc=0 + `ensemble_v2` no `case`);
este teste fixa o contrato Python que a stage invoca."""
from __future__ import annotations

import sys

import numpy as np

from ensemble.eval.run_eval import CSV_HEADER, main
from ensemble.records import PointerRecord, save_records

# Header REAL de results/comparacao.csv (literal independente para o teste conferir).
COMPARACAO_HEADER = [
    "arquitetura", "tarefa", "n_queries",
    "recall@1", "ndcg@1", "recall@5", "ndcg@5", "recall@10", "ndcg@10",
    "mrr", "latencia_s",
]

VOCAB = ["orçamento", "saúde", "estrada", "educação", "praia"]


def _rec(id_: str, modality: str, path: str | None = None) -> PointerRecord:
    meta = {"path": path} if path else {}
    return PointerRecord(id=id_, modality=modality, descriptor=f"desc {id_}",
                         pointer_uri=f"file://{id_}", source="t", meta=meta)


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


def test_main_grava_csv_v2(tmp_path):
    """A stage ensemble_v2 roda `run_eval:main --corpus --queries --qrels --out --log`:
    com builders dubles o `main` grava o CSV comparativo (cabecalho real + 18 linhas),
    aceita o `--log` que a stage repassa, e NAO importa nenhuma lib de modelo."""
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
    corpus_path = tmp_path / "records.jsonl"
    save_records(corpus, corpus_path)

    queries_path = tmp_path / "queries.txt"
    queries_path.write_text("q1\torçamento saúde\nq2\tsaúde estrada\n", encoding="utf-8")

    qrels_path = tmp_path / "qrels.tsv"
    qrels_path.write_text(
        "q1 0 d0 1\nq1 0 i0 1\nq2 0 d1 1\nq2 0 a0 1\n", encoding="utf-8")

    out_path = tmp_path / "comparacao_v2.csv"
    log_path = tmp_path / "ensemble_v2.txt"

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
         "--qrels", str(qrels_path), "--out", str(out_path),
         "--log", str(log_path)],
        builders=builders,
    )

    assert rc == 0
    assert out_path.is_file()
    linhas = out_path.read_text(encoding="utf-8").splitlines()
    assert linhas[0].split(",") == COMPARACAO_HEADER
    assert CSV_HEADER == COMPARACAO_HEADER
    # 3 sistemas x (5 eixos + conjunto) = 18 linhas de dados + cabecalho
    assert len(linhas) == 1 + 18
    corpo = "\n".join(linhas[1:])
    assert "conjunto" in corpo
    assert "n/a" in corpo                 # B fora de documento
    # o `--log` que a stage repassa foi aceito pelo parser (rc==0, sem erro de argumento);
    # a materializacao do arquivo depende do estado do logging root (no-op sob o pytest).
    assert rc == 0
    # nao baixou modelo: nenhuma lib de modelo foi importada
    assert not {"torch", "sentence_transformers"} & set(sys.modules)
