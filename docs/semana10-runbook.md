# Semana 10 — Run-book de execução (do código pronto aos entregáveis)

> O **código** dos 4 subsistemas da Semana 10 está pronto, testado (565 testes) e commitado na
> branch `feat/semana10-finetune-denso`. Este run-book lista os passos para **rodar** os
> entregáveis — os que dependem de infra/credenciais/julgamento humano e por isso ficam com o
> Lucas. Design: `docs/superpowers/specs/2026-11-12-semana10-finetune-infra-api-design.md`.

Legenda: 🟢 roda local/sem credencial · 🟡 precisa H100/GCP · 🧑 precisa julgamento humano (Lucas).

---

## Plano A — corpus v4 balanceado + finetune do unificado

### A1. Construir o corpus v4 (records.jsonl) 🟡
Captura ~50 itens por eixo de fontes gov reais e **recusa salvar** se algum item de
imagem/documento/vídeo não tiver arquivo em disco (senão o `embed_images` do A quebraria):

```bash
python -m ensemble.build_corpus          # usa build_corpus() v3 (demo);  OU, para o v4:
python -c "from ensemble.build_corpus import build_corpus_v4; print(build_corpus_v4())"
# → data/govexe/ensemble_orcamento_v4/records.jsonl  (+ binários em .../bin/<eixo>/)
```
Roda onde houver rede para as fontes (Câmara/Senado/DOU/rádio); em nuvem use o fluxo [3] abaixo.

### A2. Propor e revisar queries + qrels por eixo 🧑
Regra do projeto: **eu proponho, você revisa**. Precisamos de ≥ **8 queries julgadas por eixo**
(piso `N`, `ensemble/corpus_v4.check_balance`). Saída: `queries.jsonl` + `qrels.tsv` do v4.
> Me peça para gerar a proposta de queries/qrels a partir do `records.jsonl` do v4 quando ele existir.

### A3. Validar balanceamento e cortar o split disjunto 🟢
```python
from ensemble.corpus_v4 import finalize_split, check_corpus_paths
from ensemble.records import load_records
from ensemble.eval.qrels import load_queries, load_qrels
# check_balance roda dentro de finalize_split e falha se um eixo < 8 julgadas
train_ids, eval_ids = finalize_split(load_queries("queries.jsonl"), load_qrels("qrels.tsv"))
```
`eval_ids` (~30%/eixo) é o conjunto de **avaliação** do hook antes/depois — nunca entra no treino.
> `finalize_split` devolve só os **ids** (ainda não há `save_*` para os formatos de queries/qrels).
> Materialize os arquivos do split filtrando `queries.jsonl`/`qrels.tsv` por `eval_ids` (→ os
> `EVAL_QUERIES`/`EVAL_QRELS` do passo A5) e por `train_ids` (→ a base da geração de pares A4).

### A4. Gerar os pares de treino (todos os domínios) 🟡
Queries por **LLM local** (Qwen2.5-7B-Instruct) sobre o *descriptor* de cada item, rodando no H100:
```bash
python -m finetune.gen_queries_llm --records data/govexe/ensemble_orcamento_v4/records.jsonl \
       --out finetune/data/pairs_v4.jsonl
# (offline/determinístico: --generator template não precisa de LLM)
```
Itens de imagem/documento/vídeo entram como **imagem** (positivo = arquivo) e texto/áudio como
**texto** — a mesma rota do baseline A, para o positivo treinar pelo lado em que será avaliado.

### A5. Finetune no H100 (Slurm), com avaliação antes/depois 🟡
```bash
BASE=qwen3vl \
PAIRS=finetune/data/pairs_v4.jsonl \
EVAL_CORPUS=data/govexe/ensemble_orcamento_v4/records.jsonl \
EVAL_QUERIES=.../queries_eval.txt  EVAL_QRELS=.../qrels_eval.tsv \
EPOCHS=3  OUTPUT_DIR=finetune/out/qwen3vl_v4 \
bash finetune/slurm_finetune.sh
# alternativa L4 Spot/QLoRA: bash finetune/run_gcp_finetune.sh (mesmas envs)
```
`EVAL_CORPUS` **liga** o hook denso: loga `nDCG@10 do A antes=… depois=… delta=…` no v4.
Dica: rode antes com `DRY_RUN=1` (estimativa de VRAM GO/NO-GO) e logue
`torch.cuda.max_memory_allocated()` no 1º smoke (a liberação pós-treino reduz o pico).

> **Resultado honesto:** se `depois ≤ antes`, reportamos assim (resultado negativo é resultado) +
> ablação de r/lr/dados — anti-meta do design é overfit do gabarito.

---

## Plano B — infra event-driven [3] + API de consulta [4]

### B1. Construir o índice servível (matriz-A) 🟡
`build_index` embeda os itens do corpus pelo **lado de itens do A** (rota por modalidade) e grava
`records.jsonl` + `matrix.npy` + `index_meta.json` no diretório do índice (o adaptador treinado
entra aqui):
```bash
python -m api.build_index --records data/govexe/ensemble_orcamento_v4/records.jsonl \
       --adapter finetune/out/qwen3vl_v4 --out index_v4/
# subir para GCS: gsutil -m cp -r index_v4/ gs://<bucket>/index_v4/
```
O `index_meta.json` carimba **com qual adaptador** a matriz foi feita. O serviço [4] compara esse
carimbo com o `ADAPTER_PATH` dele no startup e **falha fechado** na divergência (itens e queries em
espaços diferentes degradam o ranking em silêncio); `ALLOW_ADAPTER_MISMATCH=1` libera o caso
deliberado. Caminho inexistente num item *image-route* para o build **antes** de carregar o modelo
(`CorpusPathError`), não no meio do embed.

### B2. Subir a API de consulta (Cloud Run, min-instances=0) 🟡
```bash
# build + deploy da imagem api/Dockerfile; env no serviço:
#   API_KEY=<segredo>   INDEX_PATH=gs://<bucket>/index_v4   ADAPTER_PATH=gs://.../qwen3vl_v4
gcloud run deploy consulta-govexe --source api/ --region southamerica-east1 \
       --min-instances=0 --set-env-vars API_KEY=...,INDEX_PATH=...,ADAPTER_PATH=...
```
Contrato para o time da GovExe:
```
POST /consulta   header X-API-Key: <chave>   body {"query": "...", "k": 10}
 → {"results":[{"item_id","eixo","score","pointer_uri","descriptor"}], "latency_ms": …}
GET  /health → {"status":"ok"}
```

### B3. Ligar o ciclo event-driven + Spot (custo ocioso zero) 🟡
```bash
# (1x) imagem do Cloud Run Job, da raiz do repo:
docker build -f gcp/serving/Dockerfile -t gcr.io/$PROJECT_ID/ciclo-captura:latest . \
  && docker push gcr.io/$PROJECT_ID/ciclo-captura:latest
bash gcp/serving/create_scheduler.sh job        # cria/atualiza o Cloud Run Job
bash gcp/serving/create_scheduler.sh create     # cron diário 06:00 America/Sao_Paulo
# cada disparo: Spot VM sobe → captura/indexa → upsert do records.jsonl no CORPUS_URI (GCS) → teardown
bash gcp/serving/create_scheduler.sh delete      # desliga tudo (custo-zero garantido)
```
Ver `gcp/serving/README.md`. O índice **servido** por [4] é republicado com `api/build_index`
(passo B1) a partir dos records acumulados — upsert e rebuild são separados de propósito
(a API nunca lê um índice com n records ≠ m linhas de matriz).

---

## O que ainda falta construir (gaps conhecidos, staged)
- **A imagem do ciclo nunca foi construída de verdade**: `gcp/serving/Dockerfile` existe e é
  coerente com o que o job roda, mas o `docker build` (e o push) são passos de deploy, ainda não
  executados — as versões de pacote da base do SDK só se confirmam lá.
- **O ciclo [3] captura com `build_corpus` (v3)**, não `build_corpus_v4` (50/eixo + sanity de
  caminho) — para o ciclo usar o v4 é preciso tocar `gcp/slice/run_slice.sh`/`slice_remote.sh`.
- **Rebuild da matriz é manual** (passo B1 após cada ciclo); automatizar é trabalho futuro.
- **Hard-negatives minerados**: os negativos textuais usam in-batch; mineração de hard-negatives
  ficou fora de escopo.
- **Gerador de queries multimodal (VLM)**: hoje usa o *descriptor* (texto); um VLM que veja a
  imagem é trabalho futuro anotado em `finetune/gen_queries_llm.py`.

## Custo e teardown
GCP projeto `residencia-511517` (pago, autorizado, efêmero). Todo fluxo tem teardown por `trap`;
confira que nada ficou vivo ao fim de cada ciclo: `gcloud compute instances list`.
