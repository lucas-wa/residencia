# Infra event-driven [3] — cron → Spot L4 → índice no GCS → **nada de pé**

Subsistema [3] do desenho: trocar a execução **manual** de VM por um fluxo **acionado por
agendamento**, com **custo ocioso zero**. Entre dois ciclos não existe VM, não existe
Cloud SQL e não existe GPU parada — o que persiste é objeto no GCS (centavos) e, do lado
do serviço de consulta [4], um Cloud Run com `min-instances=0`.

```
Cloud Scheduler  06:00 America/São_Paulo  (create_scheduler.sh)
        │  POST + OAuth  →  run.googleapis.com/.../jobs/ciclo-captura:run
        ▼
Cloud Run Job (sem GPU, efêmero)
        │  bash gcp/serving/trigger_cycle.sh
        ▼
[1/3] gcp/slice/run_slice.sh --arch ensemble --phase capture  (PROVISIONING_MODEL=SPOT)
        │     Spot L4 sobe → bootstrap → captura → corpus volta → **teardown por trap**
[2/3] gcp/serving/upsert_index.py   records do ciclo ⟶ upsert em gs://…/serving/corpus
[3/3] gcloud storage cp             binários retidos ⟶ gs://…/serving/bin/bin-<ciclo>.tgz
```

## As três peças

| Arquivo | O que é |
|---|---|
| `upsert_index.py` | **núcleo testável**: `upsert_records` (merge idempotente por `id`) + IO do `records.jsonl` (local ou `gs://`). Testes: `tests/test_upsert_index.py`. |
| `trigger_cycle.sh` | **um ciclo** ponta a ponta. Não reimplementa provisão nem teardown: chama `gcp/slice/run_slice.sh`. |
| `create_scheduler.sh` | cria/remove o **cron diário** e o Cloud Run Job que ele dispara. |
| `Dockerfile` | imagem do Cloud Run Job: base do **SDK** (o ciclo chama `gcloud compute …`) + `openssh-client` + `google-cloud-storage` + o repo em `/app`. Sem GPU e sem torch — a GPU é a Spot VM. |

### Idempotência — por que o retry é seguro

`upsert_records(existentes, novos)` deduplica por `id` (o `item_id` da captura, hash
estável da URL): o record **novo** vence o antigo e a **ordem dos existentes é
preservada** — ids inéditos entram ao fim. Repetir um ciclo (preempção do Spot, disparo
duplicado do cron, retry manual) não duplica item nem embaralha o índice. A ordem não é
estética: a linha `i` de `matrix.npy` é o embedding do record `i`.

### Divisão com `api/build_index.py` — este fluxo **não embedda**

O índice que a API [4] lê é um diretório com `records.jsonl` **+** `matrix.npy` na mesma
ordem (`api/index_io.py`). O ciclo mexe só nos **records**, e num prefixo **próprio**
(`CORPUS_URI`), separado do índice servido (`INDEX_URI`). Quem escreve o índice servido é
`python -m api.build_index` — ele precisa de GPU e grava os dois arquivos **juntos**. Se o
upsert mexesse direto no prefixo servido, a API veria `n` records para `m` linhas de
matriz e recusaria a carga até o próximo rebuild (falha fechada, mas indisponibilidade).

É também por isso que o ciclo sobe os **binários** retidos: sem eles, os records dos
ciclos antigos apontariam para arquivos que morreram com a VM efêmera, e o rebuild da
matriz não teria o que embeddar.

## Run-book

### Ensaio (não toca a nuvem)

```bash
bash gcp/serving/trigger_cycle.sh --dry-run        # o ciclo inteiro, nenhum gcloud executado
bash gcp/serving/create_scheduler.sh create --dry-run
python -m pytest tests/test_upsert_index.py -q     # o núcleo
```

### Um ciclo à mão (GPU paga por ~minutos)

```bash
bash gcp/serving/trigger_cycle.sh 2>&1 | tee logs/ciclo-$(date +%F).log
```

### Habilitar o cron (uma vez)

```bash
# pré-requisitos: APIs run/cloudscheduler/compute ativas e a service account com
# compute.instanceAdmin.v1, iam.serviceAccountUser, storage.objectAdmin e run.invoker
# — a lista completa está no topo do create_scheduler.sh
# imagem do job (da RAIZ do repo; `gcloud builds submit --tag` não serve: ele exige o
# Dockerfile na raiz do contexto, que já é o do serviço [4]):
docker build -f gcp/serving/Dockerfile -t gcr.io/residencia-511517/ciclo-captura:latest . \
  && docker push gcr.io/residencia-511517/ciclo-captura:latest
bash gcp/serving/create_scheduler.sh job      # cria/atualiza o Cloud Run Job
bash gcp/serving/create_scheduler.sh create   # cria o cron 06:00 America/São_Paulo
bash gcp/serving/create_scheduler.sh run-now  # dispara um ciclo agora, pela mesma rota
```

### Conferir que **nada** ficou vivo

```bash
gcloud compute instances list --project=residencia-511517   # esperado: "Listed 0 items."
gcloud sql instances list     --project=residencia-511517   # esperado: vazio
gcloud storage ls -l gs://residencia-511517-bronze/serving/corpus/
bash gcp/serving/create_scheduler.sh status                 # cron + execuções + instâncias
```

O `run_slice.sh` já **asserta** isso ao sair: se sobrou VM, ele grita `LEAK`, imprime o
comando de remoção manual e sai com status 70 — e o `trigger_cycle.sh` morre junto
(`set -e`), antes do upsert.

### Desabilitar tudo (custo de volta a zero)

```bash
bash gcp/serving/create_scheduler.sh pause             # para o cron, preserva a config
bash gcp/serving/create_scheduler.sh delete            # REMOVE o cron
bash gcp/serving/create_scheduler.sh delete --com-job  # remove também o Cloud Run Job
```

Sem cron não há ciclo; sem ciclo não há GPU. Resta objeto no GCS — apague o prefixo com
`gcloud storage rm -r gs://…/serving/` se quiser zerar até isso.

### Republicar o índice servido (precisa de GPU — passo manual)

```bash
# 1) baixe o corpus acumulado e os binários de todos os ciclos
gcloud storage cp gs://residencia-511517-bronze/serving/corpus/records.jsonl \
    data/govexe/ciclo/records.jsonl
gcloud storage cp 'gs://residencia-511517-bronze/serving/bin/bin-*.tgz' /tmp/bins/
for t in /tmp/bins/bin-*.tgz; do tar xzf "$t" -C data/govexe/ciclo/bin; done

# 2) reconstrua records + matriz juntos (4090/H100) e publique o diretório inteiro
python -m api.build_index --records data/govexe/ciclo/records.jsonl \
    --out data/govexe/indice_api_v4 --adapter finetune/out/qwen3vl-lora
gcloud storage cp -r data/govexe/indice_api_v4/* \
    gs://residencia-511517-bronze/serving/indice_api_v4/
```

O Cloud Run de [4] lê esse prefixo no startup (`INDEX_PATH`); como ele tem
`min-instances=0`, a próxima requisição já pega o índice novo.

## Parâmetros (todos por ambiente)

`PROJECT_ID`, `REGION`, `SERVING_URI` (e os derivados `CORPUS_URI`, `BIN_URI`,
`INDEX_URI`), `CYCLE_CORPUS`, `CYCLE_BIN_DIR`, `VM`, `ZONES`, `SUBIR_BINARIOS` no
`trigger_cycle.sh`; `SCHEDULER_JOB`, `RUN_JOB`, `SCHEDULE`, `TIME_ZONE`, `SA`, `IMAGE`,
`JOB_TIMEOUT` no `create_scheduler.sh`. Os defaults são os do projeto efêmero
`residencia-511517`.

## Limites conhecidos

- **Preempção do Spot**: a L4 pode morrer no meio da captura. O ciclo falha, a VM se apaga
  sozinha (`--instance-termination-action=DELETE`) e o índice **não** é tocado — o próximo
  ciclo recomeça do zero. Não há checkpoint intermediário.
- **`--max-retries=0`** nos dois lados (Scheduler e Cloud Run Job) é deliberado: retry
  automático subiria **outra** L4 sobre uma rodada que talvez ainda esteja de pé.
- O gatilho é **cron**, não evento de fonte: as fontes são RSS/polling. Pub/Sub + Eventarc
  fica como expansão futura e não muda nada daqui para baixo — o alvo continua `jobs:run`.
