# Multi-espaço + RRF + CLAP (A+áudio, S9 #4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir a infraestrutura de recuperação multi-espaço com fusão RRF e o embedder CLAP, habilitando indexar e avaliar as arquiteturas **A** (unificada) e **A+áudio** (unificada + CLAP nativo) ponta a ponta.

**Architecture:** Cada "espaço" (modelo) grava numa tabela pgvector própria (dim/HNSW independentes); a consulta embeda a query em cada espaço da arquitetura, coleta top-k de cada um e funde por **Reciprocal Rank Fusion** — que opera sobre *rankings*, não vetores, então espaços heterogêneos convivem. CLAP entra como embedder de áudio nativo (`embed_audio`/`embed_texts`), paralelo ao caminho ASR→texto de A.

**Tech Stack:** Python 3.10+, numpy, psycopg (pgvector), transformers (`ClapModel`/`ClapProcessor`), pytest. Reusa `benchmark/common/metrics.py` e `benchmark/common/datasets.py`.

**Spec:** `docs/superpowers/specs/2026-11-05-arquiteturas-multimodais-A-Aaudio-B-design.md`

## Global Constraints

- Dimensão indexada **≤ 2000** (teto HNSW do pgvector); default **1024** (corte MRL validado). Reusar guarda em `embed_job.ensure_schema`.
- Embeddings **L2-normalizados** (busca por cosseno/produto interno).
- RRF com constante **c = 60**.
- `item_id` **estável entre tabelas** (o mesmo item no espaço do seu modelo).
- **Crédito preservado** por item (pointer auditável).
- Modelos **open-weight** por default (soberania do dado governamental).
- CLAP default: `laion/larger_clap_general` ⚠ confirmar id/licença no model card.
- Modalidade do áudio nativo (CLAP) = `audio_native` (distinta de `audio_asr` do pointer).

## Review Focus

- **Id presente em vários espaços** → RRF soma as contribuições do mesmo id, não descarta duplicata (Task 1).
- **Espaço devolve < k ou 0 resultados** → RRF e driver não quebram; posições ausentes não contribuem (Task 1, Task 5).
- **Áudio curto/ruidoso/não-fala no CLAP** → `embed_audio` retorna vetor finito e L2≈1, sem NaN (Task 3).
- **Query sem documento relevante nos qrels** → métricas retornam 0.0, não NaN (Task 6; já coberto em `metrics.py`).
- **Dim declarada do espaço ≠ dim de saída do modelo** → falha rápida com recado acionável, não grava lixo (Task 4).

---

### Task 1: Fusão RRF

**Files:**
- Create: `gcp/slice/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Produces: `rrf_fuse(rankings: list[list[str]], *, c: int = 60) -> list[tuple[str, float]]` — recebe várias listas de `item_id` já ordenadas (melhor primeiro), devolve `(item_id, score)` ordenado por score desc; `score = Σ_listas 1/(c + posição_0based + 1)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fusion.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: FAIL (ImportError: `gcp.slice.fusion`).

- [ ] **Step 3: Implement `rrf_fuse(rankings, *, c=60)` in `gcp/slice/fusion.py`**

Python puro, sem numpy. Acumula score por id num dict percorrendo cada lista com `enumerate`; ordena por score desc, desempate estável por id.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fusion.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add gcp/slice/fusion.py tests/test_fusion.py
git commit -m "feat(gcp/slice): fusão RRF de rankings heterogêneos"
```

---

### Task 2: Registro de espaços e arquiteturas

**Files:**
- Create: `gcp/slice/spaces.py`
- Test: `tests/test_spaces.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Space(name: str, table: str, model_id: str, kind: str, dim: int, modalities: tuple[str, ...])` — `kind ∈ {"qwen3vl", "clap"}` (este plano); `modalities` = as modalidades que este espaço indexa.
  - `SPACES: dict[str, Space]` com `"qwen3vl"` (table `govexe_qwen3vl`, dim 1024, modalities `("doc_image","image","text","audio_asr")`) e `"clap"` (table `govexe_clap`, dim 512, modalities `("audio_native",)`). ⚠ dim do CLAP a confirmar no model card.
  - `ARCHITECTURES: dict[str, tuple[str, ...]]` = `{"A": ("qwen3vl",), "A+audio": ("qwen3vl", "clap")}`.
  - `architecture_spaces(name: str) -> list[Space]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spaces.py
from gcp.slice.spaces import architecture_spaces, SPACES

def test_A_tem_um_espaco_e_Aaudio_dois():
    assert [s.name for s in architecture_spaces("A")] == ["qwen3vl"]
    assert [s.name for s in architecture_spaces("A+audio")] == ["qwen3vl", "clap"]

def test_dim_dentro_do_teto_hnsw_e_tabelas_distintas():
    tabs = {s.table for s in SPACES.values()}
    assert len(tabs) == len(SPACES)                      # 1 tabela por espaço
    assert all(s.dim <= 2000 for s in SPACES.values())

def test_clap_so_indexa_audio_native():
    assert SPACES["clap"].modalities == ("audio_native",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spaces.py -v` → FAIL (ImportError).

- [ ] **Step 3: Implement `spaces.py`**

Declarativo, sem imports pesados (o builder do embedder é lazy, em Task 4). `architecture_spaces` levanta `ValueError` para nome desconhecido.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_spaces.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add gcp/slice/spaces.py tests/test_spaces.py
git commit -m "feat(gcp/slice): registro de espaços e arquiteturas (A, A+áudio)"
```

---

### Task 3: Embedder CLAP

**Files:**
- Create: `benchmark/models/clap_embed.py`
- Test: `tests/test_clap_embed.py`

**Interfaces:**
- Produces (espelha a API de módulo de `qwen3vl_embed.py`):
  - `embed_audio(paths: list[str], *, model_id=DEFAULT_CLAP_ID, device="cuda", batch_size=8) -> np.ndarray` — `(n, dim)` float32, **L2-normalizado**.
  - `embed_texts(texts: list[str], *, model_id=DEFAULT_CLAP_ID, device="cuda", batch_size=8) -> np.ndarray` — mesma forma/espaço.
  - `get_embedder(...)` com cache por chave `(model_id, device, batch_size)`.
  - `DEFAULT_CLAP_ID = "laion/larger_clap_general"`.
- Consumes: transformers `ClapModel`, `ClapProcessor` (áudio reamostrado a 48 kHz, o padrão do processor).

- [ ] **Step 1: Write the failing test** (contrato, sem baixar o modelo — padrão de `tests/test_qwen3vl_contract.py`)

```python
# tests/test_clap_embed.py  — monkeypatcha o backbone; valida contrato, não pesos
import numpy as np
from benchmark.models import clap_embed

def test_embed_audio_normalizado_e_finito(monkeypatch):
    emb = clap_embed.ClapEmbedder.__new__(clap_embed.ClapEmbedder)
    monkeypatch.setattr(emb, "_audio_vectors", lambda paths: np.random.randn(len(paths), 512))
    out = emb._postprocess(emb._audio_vectors(["a.wav", "b.wav"]))
    assert out.shape == (2, 512)
    assert np.isfinite(out).all()
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

def test_texto_e_audio_no_mesmo_dim(monkeypatch):
    emb = clap_embed.ClapEmbedder.__new__(clap_embed.ClapEmbedder)
    monkeypatch.setattr(emb, "_audio_vectors", lambda p: np.random.randn(len(p), 512))
    monkeypatch.setattr(emb, "_text_vectors", lambda t: np.random.randn(len(t), 512))
    assert emb._postprocess(emb._audio_vectors(["a.wav"])).shape[1] == \
           emb._postprocess(emb._text_vectors(["metas de saúde"])).shape[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clap_embed.py -v` → FAIL (ImportError).

- [ ] **Step 3: Implement `clap_embed.py`**

Classe `ClapEmbedder` com `_audio_vectors`/`_text_vectors` (chamam `ClapModel.get_audio_features`/`get_text_features`) e `_postprocess` (float32 + L2-normalize). Funções de módulo `embed_audio`/`embed_texts`/`get_embedder` finas sobre a classe, como em `qwen3vl_embed`. Carregamento real do modelo isolado em `_load` (não exercitado no teste de contrato).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_clap_embed.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmark/models/clap_embed.py tests/test_clap_embed.py
git commit -m "feat(models): embedder CLAP (áudio nativo + texto, mesmo espaço)"
```

---

### Task 4: Roteamento por espaço no embed_job

**Files:**
- Modify: `gcp/slice/embed_job.py`
- Test: `tests/test_gcp_slice.py` (acrescenta casos)

**Interfaces:**
- Consumes: `SPACES`/`Space` (Task 2); `clap_embed.embed_audio` (Task 3); `qwen3vl_embed.embed_images/embed_texts` (existe).
- Produces: flag `--space <nome>` (default `qwen3vl`, retrocompatível) que fixa `table` = `SPACES[space].table` e seleciona o embedder por `kind`. Em `prepare_record`/`embed_records`, itens de modalidade `audio_native` vão por `embed_audio`; `doc_image`/`image` por `embed_images`; `text`/`audio_asr` por `embed_texts`. Guarda: se `vetor.shape[1] != SPACES[space].dim`, `raise ValueError` acionável antes do upsert.

- [ ] **Step 1: Write the failing test** (store/embedder fakes, no padrão já existente no arquivo)

```python
def test_space_clap_roteia_audio_native_para_embed_audio(monkeypatch, fake_store):
    # record modality=audio_native; embed_audio fake retorna (1, 512)
    # espera: grava na tabela govexe_clap e NÃO chama embed_texts/embed_images
    ...

def test_dim_divergente_falha_rapido(monkeypatch, fake_store):
    # embedder fake devolve dim 256 num espaço declarado dim 512
    import pytest
    with pytest.raises(ValueError, match="dim"):
        ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gcp_slice.py -k "space_clap or dim_divergente" -v` → FAIL.

- [ ] **Step 3: Implement o roteamento em `embed_job.py`**

Mapear `kind`→módulo de embedder num dict lazy; escolher a função por modalidade; validar dim contra `SPACES[space].dim`. Não quebrar o caminho default (`--space qwen3vl`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gcp_slice.py -v` → PASS (inclui os casos antigos).

- [ ] **Step 5: Commit**

```bash
git add gcp/slice/embed_job.py tests/test_gcp_slice.py
git commit -m "feat(gcp/slice): embed_job roteia por --space (qwen3vl, clap)"
```

---

### Task 5: Consulta multi-espaço com RRF

**Files:**
- Modify: `gcp/slice/query.py`
- Test: `tests/test_query_rrf.py`

**Interfaces:**
- Consumes: `architecture_spaces` (Task 2); `rrf_fuse` (Task 1); o `search(vec, k, modality)` do `VectorStore` (existe).
- Produces: `search_architecture(arch: str, query_text: str, *, k: int = 10, stores: dict[str, VectorStore], embedders: dict[str, Callable[[list[str]], np.ndarray]]) -> list[tuple[str, float]]` — para cada espaço da arquitetura, embeda `query_text` com o embedder de texto do espaço, pega top-k do store do espaço, e funde por RRF. `stores`/`embedders` injetados (produção monta por `SPACES`; teste injeta fakes).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_query_rrf.py — stores fakes devolvem rankings conhecidos por espaço
def test_funde_dois_espacos_por_rrf():
    # qwen3vl -> ["p1","p2"] ; clap -> ["aud1","p1"]
    # espera ordem = rrf_fuse([["p1","p2"],["aud1","p1"]])
    ...
def test_arquitetura_A_usa_so_um_espaco():
    # arch="A" consulta só qwen3vl; resultado = ranking desse espaço
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_query_rrf.py -v` → FAIL.

- [ ] **Step 3: Implement `search_architecture` em `query.py`**

Itera `architecture_spaces(arch)`, chama o embedder de texto injetado e o `search` do store, coleta os ids ordenados por espaço, delega a `rrf_fuse`. Preserva o modo single-space atual do CLI.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_query_rrf.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add gcp/slice/query.py tests/test_query_rrf.py
git commit -m "feat(gcp/slice): consulta multi-espaço fundida por RRF"
```

---

### Task 6: Driver de avaliação (qualidade + eficiência → CSV)

**Files:**
- Create: `gcp/slice/eval.py`
- Test: `tests/test_eval_driver.py`

**Interfaces:**
- Consumes: `benchmark.common.datasets.load_local` (lê `corpus.jsonl`/`queries.jsonl`/`qrels.tsv`); `benchmark.common.metrics` (`recall_at_k`, `ndcg_at_k`, `mrr`, `aggregate`); um retriever `fn(query_text) -> list[str]` (injetado). Em produção o `main()` adapta `search_architecture` (Task 5, devolve `list[tuple[id, score]]`) para `list[str]` com `retrieve = lambda q: [i for i, _ in search_architecture(...)]`.
- Produces: `evaluate_architecture(arch, queries, qrels, retrieve, *, k_values=(1,5,10)) -> dict` (recall@k, ndcg@k, mrr agregados + `latencia_s` médio por query); `write_comparison_csv(path, rows: list[dict])` — uma linha por (arquitetura, tarefa).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_driver.py
def test_metricas_com_retriever_conhecido():
    qrels = {"q1": {"d1"}}                      # d1 relevante
    retrieve = lambda q: ["d2", "d1", "d3"]     # d1 na posição 2
    m = evaluate_architecture("A", [{"id":"q1","text":"x"}], qrels, retrieve, k_values=(1,3))
    assert m["recall@1"] == 0.0 and m["recall@3"] == 1.0
    assert m["mrr"] == 0.5 and m["latencia_s"] >= 0.0

def test_query_sem_relevante_nao_quebra():
    m = evaluate_architecture("A", [{"id":"q9","text":"x"}], {"q9": set()},
                              lambda q: ["d1"], k_values=(1,))
    assert m["recall@1"] == 0.0 and m["ndcg@1"] == 0.0   # 0.0, não NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_driver.py -v` → FAIL (ImportError).

- [ ] **Step 3: Implement `eval.py`**

`evaluate_architecture` cronometra cada `retrieve`, calcula as métricas por query via `metrics.py` e agrega com `aggregate`; `qrels` binário → `relevant_ids`/`relevant{id:1.0}`. `write_comparison_csv` usa `csv.DictWriter`. `main()` com `--arch`, `--data-dir`, `--out` fica fino (sem teste unitário; exercitado no run GCP).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_driver.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add gcp/slice/eval.py tests/test_eval_driver.py
git commit -m "feat(gcp/slice): driver de avaliação (qualidade + latência → CSV)"
```

---

### Task 7: Orquestração GCP com teardown garantido

**Files:**
- Create: `gcp/slice/run_slice.sh`
- Test: verificação manual (bash não entra no pytest; lógica mínima)

**Interfaces:**
- Produces: script parametrizado por `ARCH` (`A`/`A+audio`) e `ZONE` (lista de fallback começando por `northamerica-northeast2-a`): `provisiona (L4+Cloud SQL) → bootstrap → indexa os espaços de ARCH (embed_job --space …) → eval.py → emite metrics → teardown`. Teardown via `trap 'teardown' EXIT` — roda em QUALQUER saída (sucesso, erro, interrupção).

- [ ] **Step 1: Escrever `run_slice.sh`** com a função `teardown` (delete VM + delete Cloud SQL) registrada em `trap ... EXIT`, e um `--dry-run` que imprime cada etapa sem executar gcloud. Consolida os scripts validados de scratchpad (`ingest.sh`, `audio.sh`, `bootstrap.sh`).

- [ ] **Step 2: Verificar sintaxe e o trap**

Run: `bash -n gcp/slice/run_slice.sh` (sintaxe) e `bash gcp/slice/run_slice.sh --dry-run --arch A+audio` injetando uma falha simulada na etapa de indexação.
Expected: a saída mostra `teardown` executando MESMO após a falha injetada.

- [ ] **Step 3: Commit**

```bash
git add gcp/slice/run_slice.sh
git commit -m "feat(gcp/slice): orquestração de run com teardown garantido (trap)"
```

---

## Fases de execução (fora do TDD — gate humano/orçamento)

Não são tarefas de código; seguem o plano acima.

- **Mini-set de áudio (gate humano):** capturar ~15 clipes via `audio_collector`, escrever `data/govexe/audio_eval/{corpus,queries,qrels}`; **Lucas julga a relevância**. Sem isso, a tarefa áudio de `eval.py` não tem o que medir.
- **Runs pagos na GCP:** `run_slice.sh --arch A` e `--arch A+audio` → coletar os CSVs de comparação (doc-visual já mensurável; áudio depende do mini-set).
- **Plano 2 (#3, ensemble B):** reusa Tasks 1-7; acrescenta wrappers BGE-M3/SigLIP 2 e a ponte ColQwen2.5↔RRF (late-interaction fora do pgvector). Spec §3.2/§8.
