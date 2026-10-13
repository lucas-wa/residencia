# Benchmark de RI Multimodal + Arquitetura de Fusão (GovExe) — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar um pipeline reutilizável que avalia modelos de embeddings multimodais em RI de documento visual (ViDoRe v2 e um conjunto pt-BR derivado dos planos de governo do GovExe), além do doc de arquitetura de fusão.

**Architecture:** Núcleo testável em CPU (métricas, índice, runner, loaders) independente dos modelos; wrappers de modelo implementam uma interface `Retriever` comum (dense single-vector via FAISS/numpy; multi-vetor late-interaction via colpali-engine). CLIs finos (`run_vidore.py`, `run_govexe_ptbr.py`) orquestram loader → encode → busca → métricas → CSV.

**Tech Stack:** Python 3.11, PyTorch, transformers, sentence-transformers, open_clip_torch, colpali-engine, mteb, FAISS (gpu no WSL2 / cpu fallback), datasets (HF), pypdfium2, pandas, pyyaml, pytest.

**Spec:** `docs/superpowers/specs/2026-10-13-experimentos-ri-multimodal-govexe-design.md`

## Global Constraints

- Python **3.11**; venv isolado (NÃO reusar `.venv-dataset` do GovExe).
- Ambiente-alvo de execução GPU: **WSL2 (Linux)** na 4090 (24 GB). Código portátil: `dtype=bfloat16`, `batch_size` configurável, 1 GPU.
- Fallbacks obrigatórios: `attn_implementation="eager"` quando flash-attn ausente; backend **numpy** no índice dense quando `faiss` não instalado.
- Classe de modelos ≤ ~3B (ColQwen2-2B, ColPali-3B, CLIP/SigLIP2, encoders de texto pt-BR). **Proibido** nesta entrega: GME/VLM2Vec/MM-Embed/E5-V, treino/fine-tuning, M-BEIR global.
- **Todos os testes do plano rodam em CPU e SEM baixar modelos ou datasets grandes** (usar dummies/sintéticos/fixtures minúsculas). Execuções reais em GPU são manuais pelo aluno.
- Métricas padrão: `Recall@{1,5}`, `nDCG@{5,10}`, `MRR`. Saída: CSV em `results/` + versões de modelo/seed logadas.
- Seed fixa (`seed=42`) em qualquer amostragem.
- Idioma: código/comentários em inglês; docs de entrega em português.
- Sem segredos no repo. Sem commit automático (só quando o aluno pedir).

## Review Focus

- **qrels com doc relevante ausente do corpus** (`run_*`): métrica deve ignorar/receber 0 sem quebrar — coberto em Task 2 (nDCG/recall com relevante fora do ranking) e Task 5 (runner com qrel órfão).
- **query sem nenhum documento relevante** (divisão por zero em recall/MRR): retornar 0.0, não NaN — Task 2.
- **índice dense sem faiss instalado**: backend numpy produz o mesmo top-k — Task 4 (teste compara numpy vs ordenação manual).
- **PDF de plano corrompido/ilegível** no `build_planos_benchmark.py`: pular com aviso, não abortar o lote — Task 11.
- **`k` maior que o nº de documentos** no ranking: métrica trunca sem erro de índice — Task 2.

---

### Task 1: Scaffold do projeto + requirements + config

**Files:**
- Create: `benchmark/__init__.py`, `benchmark/common/__init__.py`, `benchmark/models/__init__.py`
- Create: `requirements.txt`, `requirements-gpu.txt`
- Create: `benchmark/common/config.py`
- Create: `pyproject.toml` (configura pytest + ruff básicos)
- Create: `.gitignore` (adiciona `results/`, `data/govexe/pages/`, `*.parquet` grandes, `.venv*`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ExperimentConfig` dataclass com campos: `name: str`, `model_id: str`, `model_kind: Literal["clip","siglip","colpali","colqwen2","text"]`, `dataset: str`, `dtype: str = "bfloat16"`, `batch_size: int = 8`, `device: str = "cuda"`, `k_values: tuple[int,...] = (1,5,10)`, `seed: int = 42`. Método `from_yaml(path) -> ExperimentConfig` e `to_dict()`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_config.py
from pathlib import Path
from benchmark.common.config import ExperimentConfig

def test_from_yaml_roundtrip(tmp_path: Path):
    p = tmp_path / "exp.yaml"
    p.write_text(
        "name: smoke\nmodel_id: dummy\nmodel_kind: text\ndataset: synthetic\n"
        "batch_size: 4\ndevice: cpu\n",
        encoding="utf-8",
    )
    cfg = ExperimentConfig.from_yaml(p)
    assert cfg.name == "smoke"
    assert cfg.model_kind == "text"
    assert cfg.batch_size == 4
    assert cfg.device == "cpu"
    assert cfg.k_values == (1, 5, 10)      # default
    assert cfg.seed == 42                   # default
    assert cfg.to_dict()["model_id"] == "dummy"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError: benchmark.common.config)

- [ ] **Step 3: Write minimal implementation**
```python
# benchmark/common/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal
import yaml

ModelKind = Literal["clip", "siglip", "colpali", "colqwen2", "text"]

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_id: str
    model_kind: ModelKind
    dataset: str
    dtype: str = "bfloat16"
    batch_size: int = 8
    device: str = "cuda"
    k_values: tuple[int, ...] = (1, 5, 10)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if "k_values" in data:
            data["k_values"] = tuple(data["k_values"])
        return cls(**data)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["k_values"] = list(self.k_values)
        return d
```
Também criar os `__init__.py` vazios, `requirements.txt` (sem versões GPU: `transformers`, `sentence-transformers`, `open_clip_torch`, `colpali-engine`, `mteb`, `datasets`, `pandas`, `pyyaml`, `tqdm`, `pypdfium2`, `pytest`, `numpy`), `requirements-gpu.txt` (`torch` CUDA, `flash-attn`, `faiss-gpu` com instruções em comentário), `pyproject.toml` (`[tool.pytest.ini_options] pythonpath = ["."]`), `.gitignore`.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add benchmark requirements.txt requirements-gpu.txt pyproject.toml .gitignore tests/test_config.py
git commit -m "chore: scaffold benchmark package + config"
```

---

### Task 2: Métricas de RI (puras, CPU)

**Files:**
- Create: `benchmark/common/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float`
  - `ndcg_at_k(ranked_ids: list[str], relevant: dict[str, float], k: int) -> float` (ganho binário quando valor=1.0)
  - `mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float`
  - `aggregate(per_query: list[dict[str, float]]) -> dict[str, float]` (média por métrica)

- [ ] **Step 1: Write the failing test**
```python
# tests/test_metrics.py
import math
from benchmark.common.metrics import recall_at_k, ndcg_at_k, mrr, aggregate

def test_recall_basic():
    assert recall_at_k(["a","b","c"], {"b"}, 5) == 1.0
    assert recall_at_k(["a","b","c"], {"z"}, 5) == 0.0          # relevante fora do corpus
    assert recall_at_k(["a","b","c"], {"a","c"}, 1) == 0.5      # só 1 dos 2 no top-1

def test_recall_empty_relevant_is_zero_not_nan():
    assert recall_at_k(["a"], set(), 5) == 0.0

def test_k_larger_than_ranking_no_error():
    assert recall_at_k(["a","b"], {"b"}, 10) == 1.0

def test_ndcg_perfect_and_order():
    assert ndcg_at_k(["a","b"], {"a":1.0,"b":1.0}, 2) == 1.0
    # relevante na 2a posicao: DCG=1/log2(3), IDCG=1 -> ~0.6309
    assert math.isclose(ndcg_at_k(["x","a"], {"a":1.0}, 2), 1/math.log2(3), rel_tol=1e-6)

def test_ndcg_relevant_absent_is_zero():
    assert ndcg_at_k(["x","y"], {"a":1.0}, 2) == 0.0

def test_mrr():
    assert mrr(["x","a"], {"a"}) == 0.5
    assert mrr(["x","y"], {"a"}) == 0.0

def test_aggregate_mean():
    out = aggregate([{"recall@5":1.0}, {"recall@5":0.0}])
    assert out["recall@5"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# benchmark/common/metrics.py
from __future__ import annotations
import math

def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    topk = ranked_ids[:k]
    hits = sum(1 for d in topk if d in relevant_ids)
    return hits / len(relevant_ids)

def ndcg_at_k(ranked_ids: list[str], relevant: dict[str, float], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(ranked_ids[:k]):
        gain = relevant.get(d, 0.0)
        if gain:
            dcg += gain / math.log2(i + 2)
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for i, d in enumerate(ranked_ids):
        if d in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add benchmark/common/metrics.py tests/test_metrics.py
git commit -m "feat: IR metrics (recall@k, nDCG@k, MRR)"
```

---

### Task 3: Tipos + interface Retriever + DummyRetriever

**Files:**
- Create: `benchmark/common/types.py`
- Create: `benchmark/models/base.py`
- Create: `benchmark/models/dummy.py`
- Test: `tests/test_dummy_retriever.py`

**Interfaces:**
- Produces:
  - `types.Query(id: str, text: str | None = None, image_path: str | None = None)`
  - `types.Doc(id: str, text: str | None = None, image_path: str | None = None)`
  - `base.Retriever` Protocol: `name: str`; `encode_queries(queries: list[Query]) -> Any`; `encode_corpus(docs: list[Doc]) -> Any`; `score(q_emb, d_emb) -> "np.ndarray"` (shape `[nq, nd]`, maior = mais similar).
  - `dummy.DummyRetriever`: embeddings determinísticos a partir de hash do texto (para testar runner/índice em CPU, sem modelos).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_dummy_retriever.py
import numpy as np
from benchmark.common.types import Query, Doc
from benchmark.models.dummy import DummyRetriever

def test_dummy_scores_shape_and_selfmatch():
    r = DummyRetriever(dim=16)
    docs = [Doc(id="a", text="imposto de renda"), Doc(id="b", text="meio ambiente")]
    qs = [Query(id="q1", text="imposto de renda")]
    qe = r.encode_queries(qs)
    de = r.encode_corpus(docs)
    s = r.score(qe, de)
    assert s.shape == (1, 2)
    assert np.argmax(s[0]) == 0           # query casa com doc 'a'
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_dummy_retriever.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# benchmark/common/types.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Query:
    id: str
    text: str | None = None
    image_path: str | None = None

@dataclass(frozen=True)
class Doc:
    id: str
    text: str | None = None
    image_path: str | None = None
```
```python
# benchmark/models/base.py
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
from benchmark.common.types import Query, Doc

@runtime_checkable
class Retriever(Protocol):
    name: str
    def encode_queries(self, queries: list[Query]) -> Any: ...
    def encode_corpus(self, docs: list[Doc]) -> Any: ...
    def score(self, q_emb: Any, d_emb: Any) -> Any: ...
```
```python
# benchmark/models/dummy.py
from __future__ import annotations
import hashlib
import numpy as np
from benchmark.common.types import Query, Doc

class DummyRetriever:
    name = "dummy"
    def __init__(self, dim: int = 16):
        self.dim = dim

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            h = hashlib.sha256((t or "").encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            v = rng.standard_normal(self.dim)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.vstack(vecs).astype("float32")

    def encode_queries(self, queries: list[Query]) -> np.ndarray:
        return self._embed([q.text for q in queries])

    def encode_corpus(self, docs: list[Doc]) -> np.ndarray:
        return self._embed([d.text for d in docs])

    def score(self, q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
        return q_emb @ d_emb.T
```

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_dummy_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add benchmark/common/types.py benchmark/models/base.py benchmark/models/dummy.py tests/test_dummy_retriever.py
git commit -m "feat: Retriever protocol + types + DummyRetriever"
```

---

### Task 4: Índice de busca (dense numpy/faiss + MaxSim late-interaction)

**Files:**
- Create: `benchmark/common/index.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: nada (opera sobre arrays).
- Produces:
  - `dense_search(q_emb: np.ndarray, d_emb: np.ndarray, k: int) -> list[list[int]]` — top-k índices por query, backend faiss se disponível senão numpy (produto interno). Resultado idêntico ao numpy.
  - `maxsim_search(q_mv: list[np.ndarray], d_mv: list[np.ndarray], k: int) -> list[list[int]]` — late interaction: score(q,d)=Σ_i max_j (q_i · d_j); multi-vetor por item.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_index.py
import numpy as np
from benchmark.common.index import dense_search, maxsim_search

def test_dense_search_matches_manual():
    d = np.eye(4, dtype="float32")              # 4 docs ortonormais
    q = np.array([[0.9,0.1,0,0]], dtype="float32")
    assert dense_search(q, d, k=2)[0][0] == 0   # mais perto do doc 0

def test_dense_k_truncates():
    d = np.eye(3, dtype="float32")
    q = np.ones((1,3), dtype="float32")
    assert len(dense_search(q, d, k=10)[0]) == 3

def test_maxsim_selfmatch():
    # doc 0 contem um token identico ao token da query
    q = [np.array([[1.0,0,0]], dtype="float32")]
    docs = [np.array([[1.0,0,0],[0,1,0]], dtype="float32"),
            np.array([[0,0,1.0]], dtype="float32")]
    assert maxsim_search(q, docs, k=2)[0][0] == 0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_index.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# benchmark/common/index.py
from __future__ import annotations
import numpy as np

def dense_search(q_emb: np.ndarray, d_emb: np.ndarray, k: int) -> list[list[int]]:
    k = min(k, d_emb.shape[0])
    try:
        import faiss  # type: ignore
        index = faiss.IndexFlatIP(d_emb.shape[1])
        index.add(np.ascontiguousarray(d_emb))
        _, idx = index.search(np.ascontiguousarray(q_emb), k)
        return idx.tolist()
    except Exception:
        sims = q_emb @ d_emb.T
        return np.argsort(-sims, axis=1)[:, :k].tolist()

def maxsim_search(q_mv: list[np.ndarray], d_mv: list[np.ndarray], k: int) -> list[list[int]]:
    k = min(k, len(d_mv))
    out = []
    for q in q_mv:
        scores = []
        for d in d_mv:
            sim = q @ d.T                      # [nq_tok, nd_tok]
            scores.append(float(sim.max(axis=1).sum()))
        order = np.argsort(-np.array(scores))[:k]
        out.append(order.tolist())
    return out
```

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_index.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add benchmark/common/index.py tests/test_index.py
git commit -m "feat: dense (faiss/numpy) + MaxSim search"
```

---

### Task 5: Runner de avaliação + escrita de CSV

**Files:**
- Create: `benchmark/common/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `Retriever` (Task 3), `dense_search` (Task 4), métricas (Task 2), `Query`/`Doc` (Task 3).
- Produces:
  - `evaluate(retriever, queries: list[Query], docs: list[Doc], qrels: dict[str, dict[str, float]], k_values=(1,5,10), multi_vector: bool=False) -> dict[str,float]`
  - `write_results_csv(path, config_dict: dict, metrics: dict) -> None` (uma linha por experimento; cria cabeçalho se novo).
  - Qrel órfão (doc não presente no corpus) é ignorado sem erro.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_runner.py
from pathlib import Path
import csv
from benchmark.common.types import Query, Doc
from benchmark.models.dummy import DummyRetriever
from benchmark.common.runner import evaluate, write_results_csv

def _fixture():
    docs = [Doc(id="a", text="imposto de renda"),
            Doc(id="b", text="meio ambiente"),
            Doc(id="c", text="educacao basica")]
    queries = [Query(id="q1", text="imposto de renda"),
               Query(id="q2", text="educacao basica")]
    qrels = {"q1": {"a": 1.0}, "q2": {"c": 1.0, "zzz": 1.0}}  # 'zzz' orfao
    return queries, docs, qrels

def test_evaluate_perfect_dummy():
    q, d, qr = _fixture()
    m = evaluate(DummyRetriever(), q, d, qr, k_values=(1,5))
    assert m["recall@1"] == 1.0
    assert m["ndcg@5"] == 1.0
    assert m["mrr"] == 1.0

def test_write_csv(tmp_path: Path):
    out = tmp_path / "r.csv"
    write_results_csv(out, {"name":"x","model_id":"dummy"}, {"recall@5":0.5})
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["name"] == "x"
    assert rows[0]["recall@5"] == "0.5"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# benchmark/common/runner.py
from __future__ import annotations
import csv
from pathlib import Path
from benchmark.common.types import Query, Doc
from benchmark.common.index import dense_search, maxsim_search
from benchmark.common.metrics import recall_at_k, ndcg_at_k, mrr, aggregate

def evaluate(retriever, queries: list[Query], docs: list[Doc],
             qrels: dict[str, dict[str, float]], k_values=(1,5,10),
             multi_vector: bool = False) -> dict[str, float]:
    q_emb = retriever.encode_queries(queries)
    d_emb = retriever.encode_corpus(docs)
    maxk = max(k_values)
    if multi_vector:
        ranked = maxsim_search(q_emb, d_emb, k=max(maxk, len(docs)))
    else:
        import numpy as np
        ranked = dense_search(np.asarray(q_emb), np.asarray(d_emb), k=max(maxk, len(docs)))
    doc_ids = [d.id for d in docs]
    per_query = []
    for qi, q in enumerate(queries):
        ranked_ids = [doc_ids[i] for i in ranked[qi]]
        rel = qrels.get(q.id, {})
        rel_set = set(rel.keys())
        row = {}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(ranked_ids, rel_set, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, rel, k)
        row["mrr"] = mrr(ranked_ids, rel_set)
        per_query.append(row)
    return aggregate(per_query)

def write_results_csv(path, config_dict: dict, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**config_dict, **metrics}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add benchmark/common/runner.py tests/test_runner.py
git commit -m "feat: evaluation runner + CSV writer"
```

---

### Task 6: Loaders de dataset (ViDoRe v2 + loader genérico local)

**Files:**
- Create: `benchmark/common/datasets.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: `Query`/`Doc` (Task 3).
- Produces:
  - `load_local(root: Path) -> tuple[list[Query], list[Doc], dict[str,dict[str,float]]]` — lê `queries.jsonl` (`{"id","text"?,"image_path"?}`), `corpus.jsonl` (idem com `id`,`image_path`/`text`), `qrels.tsv` (`qid\tdocid\tscore`). É o formato que o conjunto pt-BR vai gerar.
  - `load_vidore_v2(name: str, cache_dir: Path | None = None) -> tuple[...]` — carrega um dataset ViDoRe v2 do HF (coleção `vidore`), mapeando para `Query`/`Doc(image_path=...)` após salvar as imagens em cache. **Sem download nos testes** (testar só `load_local`; `load_vidore_v2` ganha teste de parsing com um objeto HF falso/monkeypatch).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_datasets.py
import json
from pathlib import Path
from benchmark.common.datasets import load_local

def test_load_local(tmp_path: Path):
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"id":"q1","text":"x"})+"\n", encoding="utf-8")
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"id":"d1","image_path":"p.png"})+"\n", encoding="utf-8")
    (tmp_path / "qrels.tsv").write_text("q1\td1\t1\n", encoding="utf-8")
    qs, docs, qrels = load_local(tmp_path)
    assert qs[0].id == "q1" and qs[0].text == "x"
    assert docs[0].image_path == "p.png"
    assert qrels["q1"]["d1"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_datasets.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# benchmark/common/datasets.py
from __future__ import annotations
import json
from pathlib import Path
from benchmark.common.types import Query, Doc

def load_local(root: str | Path):
    root = Path(root)
    queries = [Query(**json.loads(l)) for l in (root/"queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = [Doc(**json.loads(l)) for l in (root/"corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    qrels: dict[str, dict[str, float]] = {}
    for line in (root/"qrels.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qid, docid, score = line.split("\t")
        qrels.setdefault(qid, {})[docid] = float(score)
    return queries, docs, qrels

def load_vidore_v2(name: str, cache_dir: str | Path | None = None):
    """Carrega um dataset ViDoRe v2 da coleção `vidore` no HF.
    Estrutura confirmada na implementacao (datasets HF: splits de queries, corpus
    com coluna de imagem, e qrels). Salva imagens em cache_dir e retorna paths."""
    from datasets import load_dataset  # import tardio
    raise NotImplementedError("preencher conforme schema do dataset ViDoRe v2 (ver NOTE na spec §5)")
```
> NOTE p/ implementer: confirmar o nome/estrutura exatos dos datasets ViDoRe v2 na coleção HF `vidore` **em runtime** (há variação entre subconjuntos). Implementar `load_vidore_v2` lendo `queries`, `corpus` (coluna de imagem → salvar PNG em `cache_dir`, setar `image_path`) e `qrels`. Adicionar um teste com `monkeypatch` de `load_dataset` devolvendo um objeto minúsculo, validando o mapeamento para `Query`/`Doc`/qrels (sem rede).

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_datasets.py -v`
Expected: PASS (teste de `load_local`; adicionar o teste monkeypatch de `load_vidore_v2`)

- [ ] **Step 5: Commit**
```bash
git add benchmark/common/datasets.py tests/test_datasets.py
git commit -m "feat: dataset loaders (local jsonl/tsv + vidore v2)"
```

---

### Task 7: Wrapper de texto pt-BR (sentence-transformers)

**Files:**
- Create: `benchmark/models/text_ptbr.py`
- Test: `tests/test_text_ptbr_contract.py`

**Interfaces:**
- Consumes: `Retriever` protocol, `Query`/`Doc`.
- Produces: `TextRetriever(model_id: str, device: str="cuda", dtype: str="bfloat16", query_prefix: str="", doc_prefix: str="")` implementando `Retriever`. `encode_*` retornam `np.ndarray` L2-normalizado; `score` = produto interno. Prefixos para modelos estilo E5 (`"query: "`/`"passage: "`).

- [ ] **Step 1: Write the failing test** (contrato, sem baixar modelo — mocka o encoder interno)
```python
# tests/test_text_ptbr_contract.py
import numpy as np
from benchmark.common.types import Query, Doc
from benchmark.models.text_ptbr import TextRetriever

class _FakeST:
    def encode(self, texts, **kw):
        return np.ones((len(texts), 8), dtype="float32")

def test_contract_with_fake_encoder(monkeypatch):
    r = TextRetriever.__new__(TextRetriever)      # sem __init__ (sem download)
    r.name = "text:test"; r.model = _FakeST()
    r.query_prefix = "query: "; r.doc_prefix = "passage: "
    qe = r.encode_queries([Query(id="q", text="oi")])
    de = r.encode_corpus([Doc(id="d", text="tchau")])
    s = r.score(qe, de)
    assert qe.shape == (1,8) and s.shape == (1,1)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_text_ptbr_contract.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# benchmark/models/text_ptbr.py
from __future__ import annotations
import numpy as np
from benchmark.common.types import Query, Doc

class TextRetriever:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16",
                 query_prefix: str = "", doc_prefix: str = ""):
        from sentence_transformers import SentenceTransformer
        self.name = f"text:{model_id}"
        self.model = SentenceTransformer(model_id, device=device)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def _encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")

    def encode_queries(self, queries: list[Query]) -> np.ndarray:
        return self._encode([self.query_prefix + (q.text or "") for q in queries])

    def encode_corpus(self, docs: list[Doc]) -> np.ndarray:
        return self._encode([self.doc_prefix + (d.text or "") for d in docs])

    def score(self, q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
        return q_emb @ d_emb.T
```

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_text_ptbr_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add benchmark/models/text_ptbr.py tests/test_text_ptbr_contract.py
git commit -m "feat: pt-BR text retriever wrapper (sentence-transformers)"
```

---

### Task 8: Wrapper CLIP/SigLIP2 (imagem↔texto)

**Files:**
- Create: `benchmark/models/clip_siglip.py`
- Test: `tests/test_clip_siglip_contract.py`

**Interfaces:**
- Produces: `ClipSiglipRetriever(model_id: str, backend: Literal["open_clip","hf"]="hf", device="cuda", dtype="bfloat16")` implementando `Retriever`. `encode_queries` embeda **texto**; `encode_corpus` embeda **imagem** (`Doc.image_path`). Saída `np.ndarray` normalizado; `score`=IP.

- [ ] **Step 1: Write the failing test** (contrato com modelo falso)
```python
# tests/test_clip_siglip_contract.py
import numpy as np
from benchmark.common.types import Query, Doc
from benchmark.models.clip_siglip import ClipSiglipRetriever

def test_contract(monkeypatch, tmp_path):
    r = ClipSiglipRetriever.__new__(ClipSiglipRetriever)
    r.name = "clip:test"
    r._embed_text = lambda ts: np.ones((len(ts),4), dtype="float32")
    r._embed_image = lambda ps: np.ones((len(ps),4), dtype="float32")
    qe = r.encode_queries([Query(id="q", text="um gato")])
    de = r.encode_corpus([Doc(id="d", image_path=str(tmp_path/"x.png"))])
    assert r.score(qe, de).shape == (1,1)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_clip_siglip_contract.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# benchmark/models/clip_siglip.py
from __future__ import annotations
from typing import Literal
import numpy as np
from benchmark.common.types import Query, Doc

class ClipSiglipRetriever:
    def __init__(self, model_id: str, backend: Literal["open_clip","hf"]="hf",
                 device: str = "cuda", dtype: str = "bfloat16"):
        self.name = f"clip:{model_id}"
        self.backend = backend
        self.device = device
        if backend == "hf":
            from transformers import AutoModel, AutoProcessor
            self._model = AutoModel.from_pretrained(model_id).to(device).eval()
            self._proc = AutoProcessor.from_pretrained(model_id)
        else:
            import open_clip
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(model_id, device=device)
            self._tok = open_clip.get_tokenizer(model_id)
    # _embed_text / _embed_image: implementar p/ cada backend, retornando np.ndarray L2-normalizado.
    # (ver NOTE) — manter como metodos para o teste de contrato poder substitui-los.
    def _embed_text(self, texts: list[str]) -> np.ndarray: ...
    def _embed_image(self, paths: list[str]) -> np.ndarray: ...
    def encode_queries(self, queries: list[Query]) -> np.ndarray:
        return self._embed_text([q.text or "" for q in queries])
    def encode_corpus(self, docs: list[Doc]) -> np.ndarray:
        return self._embed_image([d.image_path or "" for d in docs])
    def score(self, q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
        return q_emb @ d_emb.T
```
> NOTE p/ implementer: implementar `_embed_text`/`_embed_image` nos dois backends (HF `AutoModel.get_text_features/get_image_features` para CLIP; SigLIP2 via `AutoModel` + `AutoProcessor`; open_clip via `encode_text`/`encode_image`), com `torch.no_grad()`, `autocast` no dtype, PIL para abrir imagem, e L2-normalização. O teste de contrato substitui esses dois métodos, então eles devem permanecer como métodos de instância.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_clip_siglip_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add benchmark/models/clip_siglip.py tests/test_clip_siglip_contract.py
git commit -m "feat: CLIP/SigLIP2 image-text retriever wrapper"
```

---

### Task 9: Wrapper ColPali/ColQwen2 (late interaction)

**Files:**
- Create: `benchmark/models/colvision.py`
- Test: `tests/test_colvision_contract.py`

**Interfaces:**
- Produces: `ColVisionRetriever(model_id: str, kind: Literal["colpali","colqwen2"], device="cuda", dtype="bfloat16", batch_size=4)` implementando `Retriever` **multi-vetor**: `encode_queries`/`encode_corpus` retornam `list[np.ndarray]` (um array `[n_tokens, dim]` por item); `score` usa MaxSim (reuso de `maxsim_search` no runner com `multi_vector=True`, então `score` pode delegar/ficar não usado). `encode_corpus` processa `Doc.image_path` (página-imagem).

- [ ] **Step 1: Write the failing test** (contrato com processor/model falsos)
```python
# tests/test_colvision_contract.py
import numpy as np
from benchmark.common.types import Query, Doc
from benchmark.models.colvision import ColVisionRetriever

def test_contract(tmp_path):
    r = ColVisionRetriever.__new__(ColVisionRetriever)
    r.name = "colqwen2:test"; r.batch_size = 2
    r._embed_queries = lambda qs: [np.ones((3,8),dtype="float32") for _ in qs]
    r._embed_images = lambda ps: [np.ones((5,8),dtype="float32") for _ in ps]
    qe = r.encode_queries([Query(id="q", text="saude")])
    de = r.encode_corpus([Doc(id="d", image_path=str(tmp_path/"p.png"))])
    assert isinstance(qe, list) and qe[0].shape == (3,8)
    assert isinstance(de, list) and de[0].shape == (5,8)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_colvision_contract.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# benchmark/models/colvision.py
from __future__ import annotations
from typing import Literal
import numpy as np
from benchmark.common.types import Query, Doc

class ColVisionRetriever:
    def __init__(self, model_id: str, kind: Literal["colpali","colqwen2"],
                 device: str = "cuda", dtype: str = "bfloat16", batch_size: int = 4):
        self.name = f"{kind}:{model_id}"
        self.kind = kind
        self.device = device
        self.batch_size = batch_size
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
        td = getattr(torch, dtype)
        if kind == "colpali":
            self._model = ColPali.from_pretrained(model_id, torch_dtype=td, device_map=device).eval()
            self._proc = ColPaliProcessor.from_pretrained(model_id)
        else:
            self._model = ColQwen2.from_pretrained(model_id, torch_dtype=td, device_map=device).eval()
            self._proc = ColQwen2Processor.from_pretrained(model_id)
    # _embed_queries(list[Query]) / _embed_images(list[str]) -> list[np.ndarray]  (ver NOTE)
    def _embed_queries(self, queries: list[Query]) -> list[np.ndarray]: ...
    def _embed_images(self, paths: list[str]) -> list[np.ndarray]: ...
    def encode_queries(self, queries: list[Query]) -> list[np.ndarray]:
        return self._embed_queries(queries)
    def encode_corpus(self, docs: list[Doc]) -> list[np.ndarray]:
        return self._embed_images([d.image_path or "" for d in docs])
    def score(self, q_emb, d_emb):                 # nao usado: runner usa maxsim_search
        raise NotImplementedError("use runner.evaluate(..., multi_vector=True)")
```
> NOTE p/ implementer: implementar `_embed_queries`/`_embed_images` com `colpali_engine` (processar em lotes de `batch_size`, `torch.no_grad()`, mover para CPU, `float()` → `np.ndarray` por item). Fixar versões compatíveis de `colpali-engine`+`transformers` no `requirements.txt` (confirmar em runtime). Fallback `attn_implementation="eager"` se flash-attn faltar. Métodos permanecem de instância para o teste de contrato substituí-los.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_colvision_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add benchmark/models/colvision.py tests/test_colvision_contract.py
git commit -m "feat: ColPali/ColQwen2 late-interaction retriever wrapper"
```

---

### Task 10: Fábrica de modelos + CLI `run_vidore.py`

**Files:**
- Create: `benchmark/models/factory.py`
- Create: `benchmark/run_vidore.py`
- Create: `benchmark/configs/vidore_colqwen2.yaml`, `benchmark/configs/vidore_colpali.yaml`
- Test: `tests/test_factory.py`

**Interfaces:**
- Consumes: todos os wrappers + `ExperimentConfig`.
- Produces:
  - `factory.build_retriever(cfg: ExperimentConfig) -> Retriever` (switch por `model_kind`; `model_kind in {"colpali","colqwen2"}` → multi-vetor).
  - `factory.is_multi_vector(cfg) -> bool`.
  - `run_vidore.py`: CLI `--config path.yaml [--limit N]` → carrega ViDoRe v2, builda modelo, `evaluate`, escreve `results/vidore.csv`. `--limit` amostra p/ smoke.

- [ ] **Step 1: Write the failing test** (factory escolhe multi-vetor certo; sem instanciar modelo real)
```python
# tests/test_factory.py
from benchmark.common.config import ExperimentConfig
from benchmark.models.factory import is_multi_vector

def test_is_multi_vector():
    base = dict(model_id="x", dataset="d")
    assert is_multi_vector(ExperimentConfig(name="a", model_kind="colqwen2", **base)) is True
    assert is_multi_vector(ExperimentConfig(name="b", model_kind="clip", **base)) is False
    assert is_multi_vector(ExperimentConfig(name="c", model_kind="text", **base)) is False
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_factory.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# benchmark/models/factory.py
from __future__ import annotations
from benchmark.common.config import ExperimentConfig

_MULTI = {"colpali", "colqwen2"}

def is_multi_vector(cfg: ExperimentConfig) -> bool:
    return cfg.model_kind in _MULTI

def build_retriever(cfg: ExperimentConfig):
    if cfg.model_kind == "text":
        from benchmark.models.text_ptbr import TextRetriever
        return TextRetriever(cfg.model_id, device=cfg.device, dtype=cfg.dtype)
    if cfg.model_kind in {"clip", "siglip"}:
        from benchmark.models.clip_siglip import ClipSiglipRetriever
        return ClipSiglipRetriever(cfg.model_id, device=cfg.device, dtype=cfg.dtype)
    if cfg.model_kind in _MULTI:
        from benchmark.models.colvision import ColVisionRetriever
        return ColVisionRetriever(cfg.model_id, kind=cfg.model_kind,
                                  device=cfg.device, dtype=cfg.dtype, batch_size=cfg.batch_size)
    raise ValueError(f"model_kind desconhecido: {cfg.model_kind}")
```
```python
# benchmark/run_vidore.py
from __future__ import annotations
import argparse
from pathlib import Path
from benchmark.common.config import ExperimentConfig
from benchmark.common.datasets import load_vidore_v2
from benchmark.common.runner import evaluate, write_results_csv
from benchmark.models.factory import build_retriever, is_multi_vector

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = ExperimentConfig.from_yaml(a.config)
    queries, docs, qrels = load_vidore_v2(cfg.dataset, cache_dir=Path("data/vidore_cache"))
    if a.limit:
        queries = queries[: a.limit]
    r = build_retriever(cfg)
    m = evaluate(r, queries, docs, qrels, k_values=cfg.k_values, multi_vector=is_multi_vector(cfg))
    write_results_csv("results/vidore.csv", cfg.to_dict(), m)
    print(cfg.name, m)

if __name__ == "__main__":
    main()
```
Criar os dois YAMLs (`vidore_colqwen2.yaml`: `model_kind: colqwen2`, `model_id: vidore/colqwen2-v1.0`, `dataset: <subset ViDoRe v2 — confirmar>`; `vidore_colpali.yaml`: `model_kind: colpali`, `model_id: vidore/colpali-v1.3`).

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_factory.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add benchmark/models/factory.py benchmark/run_vidore.py benchmark/configs/*.yaml tests/test_factory.py
git commit -m "feat: model factory + run_vidore CLI"
```

---

### Task 11: Construtor do benchmark pt-BR a partir dos planos GovExe

**Files:**
- Create: `data/govexe/build_planos_benchmark.py`
- Test: `tests/test_build_planos.py`

**Interfaces:**
- Consumes: `datasets.load_dataset` (HF) OU parquet local `D:\projects\govexe\dataset\planos\train.parquet`; `pypdfium2` p/ renderizar PDF→PNG.
- Produces: escreve em `data/govexe/ptbr_planos/`: `corpus.jsonl` (uma linha por página: `{"id":"<tse_seq>_p<NN>","image_path":"pages/<...>.png"}`), `queries.jsonl` (de `destaque`/`eixos`/`resumo`), `qrels.tsv` (query do plano → todas as páginas daquele plano). Função `build(planos_rows, out_dir, pdf_fetcher, renderer) -> None` desacoplada de IO p/ teste.
- PDF corrompido/ilegível → `logging.warning` e pular aquele plano, sem abortar.

- [ ] **Step 1: Write the failing test** (lógica de montagem com renderer/fetcher falsos — sem rede/PDF real)
```python
# tests/test_build_planos.py
import json
from pathlib import Path
from data.govexe.build_planos_benchmark import build

def test_build_makes_queries_qrels(tmp_path: Path):
    rows = [{
        "tse_sequencial": "BR001",
        "resumo": "Plano de saude e educacao",
        "destaque": "Foco em saude publica",
        "eixos": json.dumps([{"tema":"saude","titulo":"Saude para todos","propostas":"mais UBS"}]),
        "pdf_url": "http://x/plano.pdf",
    }, {
        "tse_sequencial": "BR002", "resumo": "", "destaque": "", "eixos": "[]",
        "pdf_url": "http://x/corrompido.pdf",
    }]
    def fetcher(url):            # BR002 "corrompe"
        return b"%PDF-ok" if "corrompido" not in url else b"lixo"
    def renderer(pdf_bytes, out_dir, stem):     # 2 paginas se ok; erro se lixo
        if pdf_bytes == b"lixo":
            raise ValueError("PDF ilegivel")
        paths = []
        for i in range(2):
            p = Path(out_dir) / f"{stem}_p{i:02d}.png"; p.write_bytes(b"png")
            paths.append(str(p))
        return paths
    build(rows, tmp_path, fetcher, renderer)
    corpus = [json.loads(l) for l in (tmp_path/"corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = [json.loads(l) for l in (tmp_path/"queries.jsonl").read_text(encoding="utf-8").splitlines()]
    qrels = (tmp_path/"qrels.tsv").read_text(encoding="utf-8").splitlines()
    assert len(corpus) == 2                       # so BR001 (2 paginas); BR002 pulado
    assert any(q["text"] == "Foco em saude publica" for q in queries)
    assert all(line.split("\t")[1].startswith("BR001") for line in qrels)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_build_planos.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# data/govexe/build_planos_benchmark.py
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

def build(rows, out_dir, pdf_fetcher: Callable[[str], bytes],
          renderer: Callable[[bytes, str, str], list[str]]) -> None:
    out = Path(out_dir); pages_dir = out / "pages"; pages_dir.mkdir(parents=True, exist_ok=True)
    corpus, queries, qrels = [], [], []
    for row in rows:
        seq = str(row["tse_sequencial"])
        try:
            pdf = pdf_fetcher(row["pdf_url"])
            page_paths = renderer(pdf, str(pages_dir), seq)
        except Exception as e:                       # PDF corrompido/ilegivel
            log.warning("plano %s pulado: %s", seq, e); continue
        page_ids = []
        for pth in page_paths:
            pid = Path(pth).stem
            corpus.append({"id": pid, "image_path": str(pth)}); page_ids.append(pid)
        # queries derivadas dos alvos ja existentes
        def add_q(suffix, text):
            if text and text.strip():
                queries.append({"id": f"{seq}_{suffix}", "text": text.strip()})
                for pid in page_ids:
                    qrels.append(f"{seq}_{suffix}\t{pid}\t1")
        add_q("destaque", row.get("destaque"))
        add_q("resumo", row.get("resumo"))
        try:
            for i, eixo in enumerate(json.loads(row.get("eixos") or "[]")):
                add_q(f"eixo{i}", f'{eixo.get("titulo","")} {eixo.get("propostas","")}')
        except json.JSONDecodeError:
            log.warning("eixos invalidos em %s", seq)
    (out/"corpus.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in corpus), encoding="utf-8")
    (out/"queries.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in queries), encoding="utf-8")
    (out/"qrels.tsv").write_text("\n".join(qrels), encoding="utf-8")

def main():   # wiring real (parquet + pypdfium2) — ver NOTE
    raise NotImplementedError
```
> NOTE p/ implementer: no `main()`, carregar os planos (parquet local do GovExe ou `load_dataset("GovExe/gov-br-linguagem-simples","planos")`), implementar `pdf_fetcher` (HTTP GET de `pdf_url`, timeout, headers) e `renderer` (pypdfium2: abrir bytes, rasterizar a ~150 dpi → PNG). Fixar seed; logar contagens finais.

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_build_planos.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add data/govexe/build_planos_benchmark.py tests/test_build_planos.py
git commit -m "feat: build pt-BR planos visual-doc benchmark"
```

---

### Task 12: CLI `run_govexe_ptbr.py` + README de execução

**Files:**
- Create: `benchmark/run_govexe_ptbr.py`
- Create: `benchmark/configs/govexe_colqwen2.yaml`
- Create: `benchmark/README.md`
- Test: `tests/test_run_govexe_smoke.py`

**Interfaces:**
- Consumes: `load_local` (Task 6), `build_retriever`/`is_multi_vector` (Task 10), `evaluate`/`write_results_csv` (Task 5).
- Produces: `run_govexe_ptbr.py` CLI `--config --data-dir [--limit]` → `load_local(data_dir)` → eval → `results/govexe_ptbr.csv`. Smoke test roda o CLI **com DummyRetriever** sobre fixture local (monkeypatch de `build_retriever`).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_run_govexe_smoke.py
import json, subprocess, sys
from pathlib import Path

def test_smoke_with_dummy(tmp_path: Path, monkeypatch):
    d = tmp_path / "ds"; d.mkdir()
    (d/"queries.jsonl").write_text(json.dumps({"id":"BR001_destaque","text":"saude"})+"\n", encoding="utf-8")
    (d/"corpus.jsonl").write_text(json.dumps({"id":"BR001_p00","text":"saude publica"})+"\n", encoding="utf-8")
    (d/"qrels.tsv").write_text("BR001_destaque\tBR001_p00\t1\n", encoding="utf-8")
    from benchmark import run_govexe_ptbr as mod
    from benchmark.models.dummy import DummyRetriever
    monkeypatch.setattr(mod, "build_retriever", lambda cfg: DummyRetriever())
    monkeypatch.setattr(mod, "is_multi_vector", lambda cfg: False)
    cfg = tmp_path/"c.yaml"
    cfg.write_text("name: smoke\nmodel_id: dummy\nmodel_kind: text\ndataset: govexe\ndevice: cpu\n", encoding="utf-8")
    out = tmp_path/"out.csv"
    mod.run(str(cfg), str(d), str(out), limit=None)
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python -m pytest tests/test_run_govexe_smoke.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
```python
# benchmark/run_govexe_ptbr.py
from __future__ import annotations
import argparse
from benchmark.common.config import ExperimentConfig
from benchmark.common.datasets import load_local
from benchmark.common.runner import evaluate, write_results_csv
from benchmark.models.factory import build_retriever, is_multi_vector

def run(config_path: str, data_dir: str, out_csv: str, limit: int | None) -> dict:
    cfg = ExperimentConfig.from_yaml(config_path)
    queries, docs, qrels = load_local(data_dir)
    if limit:
        queries = queries[:limit]
    r = build_retriever(cfg)
    m = evaluate(r, queries, docs, qrels, k_values=cfg.k_values, multi_vector=is_multi_vector(cfg))
    write_results_csv(out_csv, cfg.to_dict(), m)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/govexe_ptbr.csv")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print(run(a.config, a.data_dir, a.out, a.limit))

if __name__ == "__main__":
    main()
```
`benchmark/README.md`: passos de execução no **WSL2** — criar venv, `pip install -r requirements.txt` + `requirements-gpu.txt`, `python -m data.govexe.build_planos_benchmark`, `python -m benchmark.run_vidore --config ... --limit 50`, `python -m benchmark.run_govexe_ptbr --config ... --data-dir data/govexe/ptbr_planos`, onde ver resultados (`results/*.csv`). Documentar fallbacks (faiss-cpu, eager).

- [ ] **Step 4: Run test to verify it passes**
Run: `python -m pytest tests/test_run_govexe_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add benchmark/run_govexe_ptbr.py benchmark/configs/govexe_colqwen2.yaml benchmark/README.md tests/test_run_govexe_smoke.py
git commit -m "feat: run_govexe_ptbr CLI + execution README"
```

---

### Task 13: Suíte completa — rodar todos os testes

**Files:** nenhum (gate de verificação).

- [ ] **Step 1: Rodar a suíte inteira**
Run: `python -m pytest -q`
Expected: todos PASS, em CPU, sem rede.

- [ ] **Step 2: Checar que nenhum teste baixa modelo/dataset**
Inspecionar que os testes de wrapper usam substituição de método/`__new__` e que loaders usam fixtures locais.

- [ ] **Step 3: Commit (se houver ajustes)**
```bash
git add -A && git commit -m "test: green suite (cpu, offline)"
```

---

## Entregáveis fora deste plano (autoria do orquestrador, não do implementer)

- `fusao/arquitetura.md` — doc de arquitetura de fusão + diagrama Mermaid (§4 da spec). Documento, não código.
- `docs/fontes-recentes-semana6.md` — tabela de fontes (§7 da spec), com verificação de URLs, separando "estabelecido" de "preprint a verificar".
- Reconciliação da seção de experimentos no `README.md` da residência (após os primeiros resultados).
