# Ensemble v2 — especialistas nativos + RRF + métricas — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Medir se o **ensemble de especialistas nativos** (fundido por RRF) recupera melhor que o unificado **A** e o especialista de documento **B**, sobre o corpus de Orçamento, com nDCG/MRR/recall por eixo e no conjunto.

**Architecture:** Cada modalidade tem um **especialista nativo** com índice **em memória** próprio (cosine para denso; MaxSim para documento). Para uma query de texto, cada especialista ranqueia os itens do seu eixo; as 5 listas são fundidas por **RRF uniforme (k=60)**. **A** (Qwen3-VL unificado, cosine em memória) e **B** (ColQwen2.5, MaxSim — já existe) rodam na mesma base. Um driver de eval computa as métricas por (sistema, eixo) + conjunto e grava um CSV espelhando `results/comparacao.csv`. Roda no Slurm H100 (nova STAGE). Modelos e rede isolados em wrappers finos de import tardio; a lógica (RRF, métricas, índices, driver) é testada por dublês/fixtures.

**Tech Stack:** Python 3.10, numpy. Reusa `ensemble/` do v1 (`records.py`, `index.py` cosine), `gcp/slice/doc_specialist.py` (ColQwen2.5 MaxSim), o harness `benchmark/` e `slurm_run.sh`. Modelos: `BAAI/bge-m3` (texto), `google/siglip2-so400m-patch14-384` (imagem), `laion/larger_clap_general` (áudio), `WeMM-Embedding-2B` (vídeo), `vidore/colqwen2.5-v0.2` (doc), Qwen3-VL-Embedding (A).

**Spec:** `docs/superpowers/specs/2026-11-09-ensemble-v2-native-rrf-design.md` (ler junto; argumenta de `fusao/multimodal.md §3` e `fusao/ensemble.md §3`).

> **Dependência de código:** reusa o pacote `ensemble/` do v1 (PR #3, branch `feat/ensemble-v1-pointer`, ainda não mergeado). A branch do v2 deve sair de `feat/ensemble-v1-pointer` (ou do `main` após o merge do #3). Decisão de setup confirmada com o Lucas antes de executar.

## Global Constraints

- **Roda no Slurm H100** (`slurm_run.sh`, nova STAGE `ensemble_v2`); **nunca** local, **nunca** GCP para runs de modelo. Captura = stage Slurm (o nó tem internet).
- **Índices em memória** por especialista (cosine denso / MaxSim doc). **Sem pgvector, sem DB** — inclusive **A** reimplementado em memória (cosine sobre embeddings Qwen3-VL), não pgvector.
- **Tema = Orçamento**; **mesmo corpus do v1** (~20 unidades/eixo), binários retidos. **pt-BR** nas queries.
- **RRF uniforme, k=60**, pesos iguais. **Sem rerank, sem pesos** (v2 futuro).
- **Relevância = topical, binária**; unidade de recuperação = **página**(doc) / **janela**(vídeo) / **arquivo**(imagem/áudio/texto); qrels **item-level** com **dedup à unidade**.
- **Pontuação dos sistemas:** ensemble = RRF dos 5; **B = n/a** fora de documento (não zero); **A** = ranking global filtrado ao eixo, com áudio via ASR→texto e vídeo via keyframes.
- **Métricas** espelham `results/comparacao.csv`: `recall@1, nDCG@1, recall@5, nDCG@5, recall@10, nDCG@10, MRR, latência`, por (sistema, eixo) + linha **conjunto**.
- **Modelos/rede isolados** em wrappers finos de import tardio; lógica testada por dublês/fixtures (padrão `test_doc_specialist.py`), nunca por GPU/rede no teste.
- **Commits** terminam com os trailers de atribuição (`Co-Authored-By: Claude Opus 4.8 …` + `Claude-Session: …`); **push/PR só com OK do Lucas**.

## Review Focus

- **Query sem itens relevantes nas qrels** → métricas devolvem 0.0, não NaN nem crash (Task 1 testa).
- **Eixo com 0 itens no corpus** → o especialista constrói vazio e `search` devolve `[]`; o driver segue (Task 3, Task 7 testam).
- **Item na lista de >1 sistema / duplicado numa lista** (A indexa todos; dedup) → RRF e métricas **deduplicam à unidade** (Task 1 testa).
- **qrels referenciando item inexistente no corpus** → erro claro nomeando o id (Task 2 testa).
- **B pontuado fora de documento** → sai como **n/a** no CSV (não 0, não exceção) (Task 7 testa).

## File Structure

- `ensemble/eval/__init__.py`
- `ensemble/eval/rrf.py` — RRF (Task 1)
- `ensemble/eval/metrics.py` — nDCG/MRR/recall (Task 1)
- `ensemble/eval/qrels.py` — queries + qrels schema/loader (Task 2)
- `ensemble/native/__init__.py`
- `ensemble/native/specialist.py` — `DenseSpecialist` + protocolo `Specialist` (Task 3)
- `ensemble/native/doc.py` — adapter ColQwen2.5 MaxSim → `Specialist` (Task 4)
- `ensemble/native/unified_a.py` — retriever A unificado em memória (Task 5)
- `ensemble/native/embedders.py` — wrappers de embedder nativos (import tardio) (Task 6)
- `ensemble/eval/run_eval.py` — driver + CSV + `main` (Task 7)
- `ensemble/capture/*`, `ensemble/build_corpus.py` — retenção de binários (extensão) (Task 8)
- `slurm_run.sh` — STAGE `ensemble_v2` (Task 9)
- `ensemble/eval/queries_orcamento_v2.txt`, `ensemble/eval/qrels_orcamento.tsv` — gabarito proposto (Task 10, gated)
- `tests/test_ensemble_v2_*.py`, `tests/fixtures/ensemble_v2/` — testes (dublês/fixtures, sem GPU/rede)

---

### Task 1: Primitivas de avaliação — RRF + métricas

**Files:**
- Create: `ensemble/eval/__init__.py`, `ensemble/eval/rrf.py`, `ensemble/eval/metrics.py`
- Test: `tests/test_ensemble_v2_eval_primitives.py`

**Interfaces:**
- Produces: `rrf_fuse(ranked_lists: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]` (cada lista = item_ids em ordem de rank; devolve (item_id, score) desc, **deduplicado** somando contribuições). `ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float`; `mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float`; `recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float`. Relevância **binária** (`relevant_ids: set`).

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_v2_eval_primitives.py`:
  - `test_rrf_soma_contribuicoes_e_dedup`: o mesmo item em duas listas (ranks 1 e 2) soma `1/(k+1)+1/(k+2)` e aparece **uma vez**.
  - `test_rrf_ordena_desc_e_tie_deterministico`: ordem decrescente; empates em ordem estável.
  - `test_rrf_listas_vazias`: `rrf_fuse([[], []])` → `[]`.
  - `test_ndcg_valor_conhecido`: ranking com 1 relevante no rank 2 → `nDCG@10 == 1/log2(3)` (normalizado).
  - `test_mrr_primeiro_relevante`: relevante no rank 2 → `mrr == 0.5`.
  - `test_recall_at_k`: 2 relevantes, 1 no top-k → `recall@k == 0.5`.
  - `test_metricas_sem_relevante_retorna_zero`: `relevant_ids=set()` → todas 0.0 (não NaN).
- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_ensemble_v2_eval_primitives.py -v` → FAIL.
- [ ] **Step 3: Implementar** `rrf_fuse` (acumula score por item num dict, ordena desc), `ndcg_at_k` (DCG/IDCG binário), `mrr`, `recall_at_k`. numpy opcional; cuidar da divisão por zero (sem relevante → 0.0).
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble-v2): primitivas RRF + metricas`).

---

### Task 2: qrels + queries — schema e loader

**Files:**
- Create: `ensemble/eval/qrels.py`
- Test: `tests/test_ensemble_v2_qrels.py`, `tests/fixtures/ensemble_v2/` (sample de queries + qrels)

**Interfaces:**
- Consumes: nada (opera sobre arquivos).
- Produces: `load_queries(path) -> dict[str, str]` (query_id → texto; formato `qid<TAB>texto`, UTF-8); `load_qrels(path) -> dict[str, set[str]]` (query_id → conjunto de item_ids relevantes; formato TREC `qid 0 item_id rel`, rel∈{0,1}, só rel=1 entra no set); `validate_qrels(qrels, queries, corpus_ids: set[str]) -> None` (levanta `ValueError` nomeando qid/item_id desconhecido).

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_v2_qrels.py` (fixtures salvas):
  - `test_load_queries_preserva_acentos`: `qid1<TAB>"orçamento da saúde"` volta idêntico.
  - `test_load_qrels_binario`: linhas `q1 0 itemA 1` / `q1 0 itemB 0` → `{"q1": {"itemA"}}` (itemB fora).
  - `test_validate_item_inexistente_levanta`: qrels com item_id fora de `corpus_ids` → `ValueError` nomeando o id.
  - `test_query_sem_relevante_e_permitida`: query sem nenhum rel=1 → entra como `set()` sem erro.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** os loaders (UTF-8, `ensure_ascii` N/A — texto puro) + `validate_qrels`.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble-v2): schema e loader de qrels/queries`).

---

### Task 3: Especialista denso (abstração) + índice cosine

**Files:**
- Create: `ensemble/native/__init__.py`, `ensemble/native/specialist.py`
- Test: `tests/test_ensemble_v2_specialist.py`

**Interfaces:**
- Consumes: `PointerRecord` (v1 `ensemble.records`); a mecânica cosine do v1 `ensemble.index` (reusar `build_index`/`search` ou L2-normalize+dot internamente).
- Produces: protocolo `Specialist` (tem `.modality: str` e `.search(query: str, k: int) -> list[tuple[str, float]]` devolvendo (item_id, score)); `DenseSpecialist(modality, records, *, embed_items: Callable[[list[str]], np.ndarray], embed_query: Callable[[list[str]], np.ndarray])` — filtra `records` à `modality`, embeda os itens (pelo caminho do binário / texto em `meta`/descriptor) **uma vez** com `embed_items`, e `search` embeda a query com `embed_query` e ranqueia por cosine. Corpus-da-modalidade vazio → `search` devolve `[]` (sem erro).

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_v2_specialist.py` (dublês determinísticos):
  - `test_dense_ranqueia_por_cosine`: embed_items/embed_query dublê (bag-of-words) → a query "orçamento saúde" ranqueia o item da modalidade cujo texto é "orçamento da saúde" no topo.
  - `test_dense_so_indexa_sua_modalidade`: records com 3 modalidades, `DenseSpecialist("imagem", …)` só devolve item_ids de imagem.
  - `test_dense_modalidade_vazia_retorna_lista_vazia`: nenhum item da modalidade → `search` → `[]`.
  - `test_dense_embed_items_chamado_uma_vez`: espia `embed_items`; chamado 1× na construção (não por query).
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** `DenseSpecialist` (filtra modalidade, embeda itens 1×, L2-normaliza, cosine no search). Reusar `ensemble.index` se encaixar; senão numpy direto. Definir `Specialist` como `typing.Protocol`.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble-v2): DenseSpecialist + indice cosine por modalidade`).

---

### Task 4: Especialista de documento (ColQwen2.5 MaxSim) — adapter

**Files:**
- Create: `ensemble/native/doc.py`
- Test: `tests/test_ensemble_v2_doc_specialist.py`

**Interfaces:**
- Consumes: `gcp/slice/doc_specialist.py` (ColQwen2.5 MaxSim — ranqueador `DocSpecialistRanker`/`build_doc_retriever`); protocolo `Specialist` (Task 3).
- Produces: `DocSpecialist(records, *, ranker)` implementando `Specialist` (`.modality == "documento"`): indexa as **páginas** de documento (multi-vetor em memória, MaxSim), `search(query, k)` → page item_ids ranqueados. `ranker` injetado (dublê no teste, padrão `test_doc_specialist.py`). Itens não-documento não são indexados.

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_v2_doc_specialist.py` (ranker dublê de ordem conhecida):
  - `test_doc_ranqueia_paginas`: ranker dublê → `search` devolve os item_ids das páginas na ordem do ranker.
  - `test_doc_ignora_nao_documento`: corpus misto → só páginas de documento entram.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** `DocSpecialist` reusando a mecânica MaxSim do `doc_specialist.py` (sem pgvector, em memória). `ranker` injetável.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble-v2): DocSpecialist ColQwen2.5 MaxSim adapter`).

---

### Task 5: A — retriever unificado em memória (Qwen3-VL)

**Files:**
- Create: `ensemble/native/unified_a.py`
- Test: `tests/test_ensemble_v2_unified_a.py`

**Interfaces:**
- Consumes: `PointerRecord`; a mecânica cosine (Task 3).
- Produces: `UnifiedRetrieverA(records, *, embed_items, embed_query)` — embeda **todos** os itens num **único** espaço cosine (representação-A: doc-imagem/imagem nativos; áudio via **ASR→texto**; vídeo via **keyframes**), `search(query, k)` → **ranking global** sobre todos os itens (não por eixo). A representação-A de cada item vem de `meta` (ex. `meta["a_repr"]`: caminho de imagem/keyframe ou texto ASR), montada na captura/driver. Guarda os item_ids para o driver filtrar por eixo depois.

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_v2_unified_a.py` (dublê):
  - `test_a_espaco_unico_ranqueia_cross_modal`: itens de modalidades diferentes num só espaço; a query ranqueia-os juntos (o top-k mistura modalidades).
  - `test_a_ranking_global_nao_por_eixo`: `search` devolve a lista global; nenhuma filtragem interna por modalidade.
  - `test_a_embed_items_uma_vez`: `embed_items` chamado 1× na construção.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** `UnifiedRetrieverA` (cosine em memória sobre a representação-A de todos os itens). Sem pgvector.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble-v2): retriever A unificado em memoria`).

---

### Task 6: Wrappers de embedder nativos (import tardio)

**Files:**
- Create: `ensemble/native/embedders.py`
- Test: `tests/test_ensemble_v2_embedders.py`

**Interfaces:**
- Produces: para cada eixo denso, um builder que devolve `(embed_items, embed_query)` com **import tardio dentro da função** (sem download no import/teste), cada um `Callable[[list[str]], np.ndarray]` → `[n, d]` float32:
  `build_text_embedder()` (BGE-M3 via sentence-transformers; items e query = texto), `build_image_embedder()` (SigLIP 2 `google/siglip2-so400m-patch14-384`; items = caminhos de imagem, query = texto), `build_audio_embedder()` (CLAP `laion/larger_clap_general`; items = caminhos de áudio, query = texto), `build_video_embedder()` (WeMM-Embedding-2B; items = caminhos de vídeo/keyframes, query = texto), `build_unified_a_embedder()` (Qwen3-VL; items = representação-A, query = texto). Um registry `EMBEDDER_BUILDERS: dict[str, Callable]` (modalidade → builder).

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_v2_embedders.py`:
  - `test_modulo_importa_sem_baixar_modelo`: `import ensemble.native.embedders` não dispara download (imports de torch/modelos são tardios, dentro dos builders).
  - `test_registry_cobre_os_5_builders`: `EMBEDDER_BUILDERS` tem chaves para texto/imagem/audio/video/unified_a, cada valor chamável.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** os 5 builders com import tardio (espelhar `_build_embedder` do v1 / `_build_retriever` do doc_specialist) + o registry. Checkpoints fixados nas constantes do módulo. (Os modelos reais só rodam no Slurm; aqui o teste cobre só a fiação.)
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble-v2): wrappers de embedder nativos (import tardio)`).

---

### Task 7: Driver de avaliação + CSV comparativo

**Files:**
- Create: `ensemble/eval/run_eval.py`
- Test: `tests/test_ensemble_v2_run_eval.py`

**Interfaces:**
- Consumes: Tasks 1–6 + `PointerRecord`.
- Produces: `evaluate(corpus, queries, qrels, *, specialists: dict[str, Specialist], retriever_a, retriever_b) -> list[dict]` — para cada query: cada especialista ranqueia seu eixo; **ensemble** = `rrf_fuse` das 5 listas; **A** = `retriever_a.search` (global); **B** = `retriever_b.search` (só documento). Computa métricas por (sistema, eixo) + **conjunto**: ensemble/A por eixo = ranking restrito ao eixo; conjunto = lista completa; **B = n/a** fora de documento (valor sentinela, ex. `None`). Devolve linhas `{sistema, eixo, recall@1, nDCG@1, recall@5, nDCG@5, recall@10, nDCG@10, MRR, latencia}`. `write_csv(rows, path)` espelha `results/comparacao.csv`. `main(argv=None, *, builders=None) -> int` (lê corpus/queries/qrels dos args, monta especialistas via registry da Task 6 com import tardio, grava CSV).

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_v2_run_eval.py` (especialistas/retrievers dublê, sem GPU):
  - `test_ensemble_usa_rrf`: espiando, o ranking do ensemble é o `rrf_fuse` das listas dos 5 especialistas.
  - `test_B_na_fora_de_documento`: as linhas de B nos eixos imagem/texto/audio/video têm as métricas = `None` (n/a), e B aparece em documento+conjunto com números.
  - `test_metricas_por_eixo_e_conjunto`: valores batem com cálculo à mão num corpus pequeno; existe linha `conjunto`.
  - `test_main_grava_csv` (builders dublê injetados): `main` escreve o CSV com as colunas certas e retorna 0; não baixa modelo.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** `evaluate` (laço query × sistema × eixo, RRF no ensemble, dedup à unidade, métricas da Task 1), `write_csv`, `main` (import tardio dos builders).
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble-v2): driver de avaliacao + CSV comparativo`).

---

### Task 8: Retenção de binários na captura

**Files:**
- Modify: `ensemble/capture/{text,document,image,audio,video}.py`, `ensemble/build_corpus.py`
- Test: `tests/test_ensemble_v2_capture_retencao.py`

**Interfaces:**
- Consumes: a captura do v1 (os `fetch_*`/parsers) + `PointerRecord`.
- Produces: cada captura passa a **gravar o binário baixado** num diretório de corpus e registra o caminho local em `meta["path"]` (e, para A, a representação-A em `meta["a_repr"]` quando aplicável: ASR p/ áudio, keyframe p/ vídeo). `build_corpus(destino, *, por_eixo=20, bin_dir) -> Path` salva `records.jsonl` **e** os binários sob `bin_dir`. A lógica de nomear/escrever é testável; o download real segue nos wrappers finos.

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_v2_capture_retencao.py`:
  - `test_parser_registra_path_do_binario`: dado um binário-fixture, o parser do eixo põe `meta["path"]` apontando ao arquivo salvo.
  - `test_build_corpus_salva_binarios` (fetch monkeypatchado devolvendo records com binário-dublê): `build_corpus` grava `records.jsonl` + os arquivos em `bin_dir`, e cada record tem `meta["path"]` existente.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** a retenção (salvar binário + `meta["path"]`/`meta["a_repr"]`) nos parsers e no `build_corpus`, sem quebrar os testes do v1 (campos novos em `meta`, compatível).
- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_ensemble_v2_capture_retencao.py tests/test_ensemble_*.py -v` (v1 verde também).
- [ ] **Step 5: Commit** (`feat(ensemble-v2): retencao de binarios na captura`).

---

### Task 9: STAGE `ensemble_v2` no Slurm (captura + eval)

**Files:**
- Modify: `slurm_run.sh`
- Test: `tests/test_ensemble_v2_stage.py`

**Interfaces:**
- Consumes: `build_corpus` (Task 8), `ensemble.eval.run_eval:main` (Task 7).
- Produces: nova STAGE `ensemble_v2` no `slurm_run.sh` (validação no `case`, mensagem de STAGE inválida, função `run_ensemble_v2` espelhando `run_govexe`/`run_ensemble`): roda a **captura** (internet; grava binários no bind) → `python3 -m ensemble.eval.run_eval --corpus … --queries … --qrels … --out results/comparacao_v2.csv --log logs/ensemble_v2.txt`, com `--gres=gpu:h100:1`. Parametrizável por env (ex. `ENSEMBLE_V2_CORPUS`).

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_v2_stage.py`:
  - `test_main_grava_csv_v2` (builders dublê): o `main` do driver escreve o CSV v2. (reusa a fiação da Task 7.)
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** a STAGE no `slurm_run.sh` (mirror de `run_govexe`) + a entrada de usage documentada (`STAGE=ensemble_v2 sbatch --gres=gpu:h100:1 slurm_run.sh`).
- [ ] **Step 4: Verificar** — `pytest tests/test_ensemble_v2_stage.py -v` PASS; `bash -n slurm_run.sh` rc=0; `ensemble_v2` aparece no `case`.
- [ ] **Step 5: Commit** (`feat(ensemble-v2): STAGE ensemble_v2 no slurm_run.sh`).

---

### Task 10: Gabarito — queries + qrels propostos (gated, humano no laço)

**Files:**
- Create: `ensemble/eval/queries_orcamento_v2.txt`, `ensemble/eval/qrels_orcamento.tsv`
- Test: `tests/test_ensemble_v2_gabarito_formato.py`

**Interfaces:**
- Consumes: os loaders (Task 2), o corpus real (Task 8).
- Produces: ~15-20 queries de Orçamento (expandindo `queries_orcamento.txt`) + um **gabarito de relevância binária** (TREC `qid 0 item_id 1`) proposto por mim sobre o corpus real, **para o Lucas revisar**. O teste valida só o **formato** (parseável por `load_queries`/`load_qrels`, ids batem com o corpus).

- [ ] **Step 1: Teste falhando** — `tests/test_ensemble_v2_gabarito_formato.py`: `load_queries`/`load_qrels` parseiam os arquivos sem erro e `validate_qrels` passa contra os ids do corpus (quando o corpus existir; usar fixture pequena enquanto o real não existe).
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3 (TDD/código):** garantir o formato com uma fixture pequena. **Depois (fora do TDD, gated — requer o corpus real da Task 8 + revisão do Lucas):** eu proponho as ~15-20 queries + o gabarito binário sobre o corpus real; Lucas revisa/corrige antes do run de métricas.
- [ ] **Step 4: Verificar** — teste de formato PASS.
- [ ] **Step 5: Commit** (`feat(ensemble-v2): queries + gabarito de orcamento (proposta)`).

---

## Self-Review

- **Spec coverage:** §4.1 especialistas → Tasks 3 (densos), 4 (doc), 6 (embedders). §4.2 RRF → Task 1. §4.3 A/B mesma base → Tasks 4 (B), 5 (A). §4.4 métricas/CSV → Tasks 1, 7. §4.5 corpus/qrels → Tasks 8 (binários), 2 + 10 (qrels). §4.6 semântica (relevância topical, unidade, B n/a, A global) → Tasks 1, 2, 5, 7. Captura stage + §6 fluxo → Tasks 8, 9. Non-goals (rerank, pesos) fora do plano. ✓
- **Tipos consistentes:** `Specialist` (`.modality`, `.search(query,k)->list[tuple[str,float]]`) em Tasks 3/4/7; `embed_items`/`embed_query: Callable[[list[str]], np.ndarray]` em Tasks 3/5/6; `rrf_fuse(list[list[str]])` (Task 1) alimentado pelas listas de item_ids dos especialistas no driver (Task 7); `load_qrels -> dict[str,set[str]]` (Task 2) consumido pelas métricas (Task 1) no driver (Task 7). ✓
- **Review Focus coberto:** query sem relevante (T1), eixo vazio (T3/T7), dedup (T1), qrels item inexistente (T2), B n/a fora de doc (T7). ✓
- **Proporção:** passos = assinatura + nome/asserção de teste, sem corpos. ✓
- **Dependências externas isoladas:** modelos (BGE-M3/SigLIP2/CLAP/WeMM/Qwen3-VL/ColQwen2.5) e rede (captura) em wrappers finos de import tardio; o testável é RRF/métricas/qrels/índice/driver com dublês/fixtures (padrão `test_doc_specialist.py`). ✓
