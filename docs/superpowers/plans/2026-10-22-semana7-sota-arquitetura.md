# Semana 7 — SOTA (ViDoRe V3), finetuning provisionado e 1ª arquitetura na GCP

> **Para quem for executar:** o `implementer` faz todo o código; a thread principal orquestra
> e revisa. Tarefas de *ops* (rodar `sbatch` no cluster, provisionar VM/GCP) são do Lucas —
> o plano entrega os scripts/configs e o comando exato. Marcação: **[exec]** = fazer esta
> semana · **[prov]** = deixar pronto, sem rodar (treino/ingestão real só sob ok).

**Objetivo:** avaliar o estado da arte 2026 no **ViDoRe V3 oficial**, deixar **provisionado** um
finetuning político pt-BR (LoRA/QLoRA, H100 e L4), subir a **primeira fatia da arquitetura GovExe
na GCP** fim a fim com o **modelo unificado** (Qwen3-VL-Embedding), e documentar o **ensemble**
multimodal para A/B futuro.

**Arquitetura:** o pacote `benchmark/` já tem wrappers (colvision, colembed, qwen3vl_embed),
índice MaxSim/numpy, runner e loaders v2/V3. Esta semana adiciona o wrapper **jina-v4** e configs
V3 do par de continuidade; cria o módulo de **finetuning** (colpali-engine/LoRA) e a **fatia GCP**
(coletor → GCS → embedding em VM efêmera → Cloud SQL+pgvector → consulta).

**Tech stack:** Python 3.12, PyTorch 2.9/cu129, transformers 5.x, colpali-engine, FAISS/numpy,
Singularity/Slurm (H100), GCP (Cloud Run Jobs, GCS, Cloud SQL+pgvector, VM L4 g2-standard-4), pgvector.

## Global Constraints

- **Benchmark só no ViDoRe V3 oficial** nesta entrega — **não** rodar o mini-benchmark pt-BR dos planos.
- Set do V3 (5 modelos): `vidore/colpali-v1.3` + `vidore/colqwen2-v1.0` (continuidade) · **Nemotron
  ColEmbed V2 8B** · **jina-embeddings-v4** · **ColQwen2.5**. ColPali = baseline **não-governamental**
  (licença Gemma) — registrar a ressalva, não usar como candidato de produção.
- **Licenças a confirmar no model card do HF** antes de qualquer uso oficial (vários "Apache-2.0"
  vêm só de catálogos de terceiros): Qwen3-*, Nemotron ColEmbed V2, jina-v4, SigLIP 2, WeMM.
- Finetuning: **LoRA** (H100, bf16) / **QLoRA-4bit** (L4 24 GB). Bases: **ColQwen2.5 (3B, primária)**
  e **Nemotron ColEmbed V2 8B (opção)**. **Não treinar** nesta semana.
- GCP: VM de embedding **efêmera, liga/desliga**, L4 (g2-standard-4) ou menor (T4); **teardown** ao fim;
  **prints do console** como prova. Índice = **Cloud SQL + pgvector**.
- Unificado a implementar: **Qwen3-VL-Embedding-2B** (cabe folgado em L4); 8B como opção.
- EOL LF em `.sh` (`.gitattributes`). Commits só sob pedido; nada de credenciais no repo.

## Review Focus (pontos que nenhum teste cobre sozinho)

- **jina-v4 API de encode** difere dos Col* (tem `encode_text`/`encode_image` próprios e modo
  "retrieval") → o wrapper precisa normalizar saída para o contrato `Retriever` (multi-vetor vs denso).
- **ViDoRe V3 é multilíngue e maior** (10 domínios, ~26k páginas) → custo/tempo em H100; prever `LIMIT`
  para smoke test antes da rodada cheia.
- **QLoRA em L4**: footprint real de VRAM do 8B 4-bit com páginas-imagem; validar que cabe em 24 GB.
- **Segredos GCP** (Cloud SQL): senha via Secret Manager, nunca no script nem no git.
- **Idempotência da ingestão** no slice: reprocesso não deve duplicar linhas no pgvector (upsert por hash).

---

## Workstream A — ViDoRe V3 oficial (item 2) **[exec]**

**Objetivo:** rodar os 5 modelos no ViDoRe V3 e produzir um relatório comparável ao da Semana 6.

### A1. Config V3 do par de continuidade **[implementer]**
- **Criar:** `benchmark/configs/v3_colpali.yaml`, `benchmark/configs/v3_colqwen2.yaml`
  (espelhar `v3_colqwen25.yaml`, trocando `model_id` para `vidore/colpali-v1.3` e
  `vidore/colqwen2-v1.0`; reaproveitam o wrapper `colvision`).
- **Aceite:** `python -m benchmark.run_vidore --config benchmark/configs/v3_colpali.yaml --limit 2`
  carrega e roda 2 queries em CPU/GPU sem erro de config.

### A2. Wrapper + config do jina-embeddings-v4 **[implementer, TDD]**
- **Criar:** `benchmark/models/jina_embed.py` (classe `JinaV4Retriever` seguindo o protocolo de
  `benchmark/models/base.py`; suportar modo denso e, se aplicável, multi-vetor "late"); registrar no
  `benchmark/models/factory.py`; `benchmark/configs/v3_jinav4.yaml`.
- **Teste primeiro:** `benchmark/tests/test_jina_embed.py` — mock do modelo/processor, exercita
  `encode_queries`/`encode_corpus` e o shape de saída (sem baixar pesos), como os outros wrappers.
- **Aceite:** testes passam offline; `check_env` reconhece a dependência; `--limit 2` roda.
- **Atenção:** confirmar licença jina-v4 no card (CC-BY-NC? → marcar como pesquisa, não produção).

### A3. Ajustar runner/preflight e rodar **[implementer p/ config; exec p/ rodar]**
- **Modificar:** default `VIDORE_CONFIGS` no `slurm_run.sh` para o set de 5 (colpali, colqwen2,
  colembed8b, jinav4, colqwen25); **remover** `v3_qwen3vl8b` do default do benchmark (fica p/ o slice).
- **Rodar (Lucas):** `git pull` no cluster → smoke `LIMIT=50 STAGE=vidore sbatch slurm_run.sh` →
  rodada cheia `STAGE=vidore sbatch slurm_run.sh`.
- **Aceite:** `results/vidore.csv` com uma linha por modelo (nDCG@10, Recall@k, MRR).
- **Verificação:** conferir que os 5 modelos aparecem; comparar nDCG@10 do Nemotron com o leaderboard.

### A4. Relatório + Notion **[thread principal]**
- Atualizar o artifact de resultados (ou novo) com gráficos e a tabela; página de resultados aninhada
  na Semana 7, no padrão da Semana 6 (métricas explicadas). Registrar ressalva de licença do ColPali.

---

## Workstream B — 1ª arquitetura na GCP + unificado + pipeline (itens 3, 5, 6-unificado)

### B1. Implementar o unificado Qwen3-VL-Embedding **[implementer, TDD]** **[exec]**
- **Criar/estender:** wrapper `benchmark/models/qwen3vl_embed.py` já existe para o embedding denso —
  confirmar que serve para **inferência de embedding de página-imagem e texto**; adicionar caminho
  de uso fora do benchmark (função `embed_images(paths)->np.ndarray`, `embed_texts(...)`).
- **Teste:** mock de encode; shape e normalização L2.
- **Aceite:** um script `scripts/embed_qwen3vl.py` embute N imagens e grava vetores (npy/pgvector).

### B2. Fatia GCP fim a fim **[implementer p/ scripts; exec p/ provisionar]** **[exec]**
- **Criar:** `gcp/slice/` com:
  - `collector.py` (ou reaproveitar os planos já baixados) → sobe PDFs/páginas para **GCS (Bronze)**.
  - `embed_job.py` → roda em **VM L4 efêmera (g2-standard-4)**; lê páginas do GCS, embute com
    Qwen3-VL-Embedding-2B, faz **upsert** no **Cloud SQL + pgvector** (dedup por hash).
  - `query.py` → consulta texto → top-k páginas (prova fim a fim).
  - `README-slice.md` → passo a passo: criar Cloud SQL (pgvector), VM efêmera on/off, rodar, **teardown**.
- **Segredos:** senha do Cloud SQL via **Secret Manager** (nunca no repo).
- **Aceite:** consulta retorna páginas plausíveis de um plano; **screenshots do console** (VM, Cloud SQL,
  resultado da query) salvos como prova.
- **Verificação:** rodar `query.py` duas vezes com a mesma ingestão → sem duplicatas (upsert idempotente);
  VM derrubada ao fim (sem custo residual).

### B3. Pipeline GCP (plano) + etapa de embedding **[thread principal]** **[exec]**
- **Modificar:** `fusao/coleta-gcp.md` → detalhar a **etapa de embedding no Gold** (modelo unificado
  Qwen3-VL-Embedding, VM efêmera on/off, Cloud SQL+pgvector como índice) e a nota de custo/teardown.
- **Aceite:** doc reflete a fatia implementada; diagrama/updação coerente com B2.

---

## Workstream C — Finetuning provisionado (item 4) **[prov]**

**Objetivo:** deixar **pronto para disparar** (não treinar) um finetuning político pt-BR.

### C1. Geração de dados sintéticos (consulta→página) **[implementer, TDD]**
- **Criar:** `finetune/make_pairs.py` → para cada página de plano, gera K consultas pt-BR (LLM) e grava
  pares `(query, page_image, label=1)` + negativos amostrados; cache idempotente; `--limit`/`--force`.
- **Teste:** com um fetcher/gerador mockado, valida formato dos pares e dedup.
- **Aceite:** gera um shard pequeno de exemplo; formato aceito pelo treinador (C2).

### C2. Scaffolding de treino LoRA/QLoRA **[implementer]**
- **Criar:** `finetune/train_lora.py` (colpali-engine/PEFT): LoRA bf16 (H100) e **QLoRA-4bit** (L4);
  base configurável — **ColQwen2.5 (3B, primária)** e **Nemotron ColEmbed V2 8B (opção)**; hook de
  avaliação chamando `benchmark.run_vidore` ao fim.
- **Criar:** `finetune/slurm_finetune.sh` (H100) e `finetune/run_gcp_finetune.sh` (L4/QLoRA) — mesmo
  padrão de deps/cache do `slurm_run.sh`; `DRY_RUN=1` que só monta e valida sem treinar.
- **Aceite:** `DRY_RUN=1` monta modelo+LoRA e sai; nota **go/no-go** de VRAM (8B-QLoRA cabe em L4?).
- **Verificação:** `bash -n` nos `.sh`; `train_lora.py --help` ok; **nenhum treino disparado**.

---

## Workstream D — Multimodal: ensemble (design) (item 6-ensemble) **[prov]**

**Objetivo:** documentar a abordagem de **ensemble de especialistas + pointer vector** como alternativa
ao unificado, para o A/B futuro.

### D1. Doc de design **[thread principal]**
- **Criar:** `fusao/multimodal.md` com:
  - **Unificado** (implementado em B): Qwen3-VL-Embedding — um espaço, texto+imagem+doc+vídeo.
  - **Ensemble + pointer vector:** Qwen3-Embedding/BGE-M3 (texto pt-BR) · SigLIP 2 (imagem) · ColQwen2.5
    (doc, late-interaction) · **Whisper→texto** (áudio, pointer) · Qwen3-VL/WeMM (vídeo); fusão **RRF +
    rerank two-stage**; **áudio nunca** entra em embedder unificado.
  - Opções **API** registradas (Cohere Embed 4, Voyage-3.5, Gemini, TwelveLabs) com nota de custo/licença.
  - Plano de **A/B**: mesma query-set, comparar unificado × ensemble por nDCG@k/MRR e custo/latência.
- **Aceite:** doc cobre as duas abordagens e o protocolo de comparação; sem implementar o ensemble agora.

---

## Ordem sugerida e dependências

1. **A1–A3** (configs + jina wrapper) → Lucas roda o smoke/rodada no cluster.
2. **B1** (unificado) → **B2** (slice GCP) → **B3** (doc pipeline).
3. **C1→C2** (finetuning provisionado) em paralelo a B.
4. **D1** (design ensemble) e **A4** (relatório/Notion) ao fim.

A é independente de B/C/D. B1 bloqueia B2. C é independente. D depende só da pesquisa (já feita).

## Entregáveis da semana

- `results/vidore.csv` (V3, 5 modelos) + relatório/Notion.
- Fatia GCP rodando + **screenshots** + `gcp/slice/` + `fusao/coleta-gcp.md` atualizado.
- `finetune/` pronto para disparar (sem treino) + nota go/no-go.
- `fusao/multimodal.md` (unificado × ensemble, A/B).
