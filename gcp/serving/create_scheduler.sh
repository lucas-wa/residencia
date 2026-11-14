#!/usr/bin/env bash
# =====================================================================================
# Gatilho do fluxo event-driven [3]: Cloud Scheduler diario as 06:00 America/Sao_Paulo.
#
# OPCAO ESCOLHIDA -- Cloud Scheduler (HTTP) -> Cloud Run JOB -> gcp/serving/trigger_cycle.sh
#
#   Scheduler (cron)  --POST + OAuth-->  run.googleapis.com .../jobs/$RUN_JOB:run
#                                             |
#                                             v
#                                   Cloud Run Job (sem GPU, efemero)
#                                             |
#                                   bash gcp/serving/trigger_cycle.sh
#                                             |
#                              Spot L4 sobe, captura, e e apagada pelo trap
#
# Por que um Cloud Run JOB e nao um SERVICE com endpoint HTTP: o ciclo dura dezenas de
# minutos (provisao + bootstrap + captura) e nao devolve resposta -- e um batch, nao um
# request. Um service teria de responder em ate 60 min, manter a instancia viva o tempo
# todo e ainda precisaria de guard de concorrencia; o job roda ate terminar, tem
# `--max-retries` proprio e some depois. E ele NAO precisa de GPU: a GPU e a Spot VM que
# o `trigger_cycle.sh` sobe e derruba.
#
# Fontes do corpus sao RSS/polling: um cron diario e o gatilho certo. Pub/Sub + Eventarc
# (webhook da fonte -> topico -> job) fica como expansao futura e nao muda nada daqui
# para baixo -- o alvo continua sendo `jobs:run`.
#
# -------------------------------------------------------------------------------------
# CUSTO
#
# O scheduler em si e gratuito ate 3 jobs. O job do Cloud Run so cobra enquanto roda. A
# GPU so existe dentro do ciclo. `--max-retries=0` nos dois lados e deliberado: um retry
# automatico subiria OUTRA L4 sobre uma rodada que talvez ainda esteja de pe.
#
# -------------------------------------------------------------------------------------
# RUN-BOOK
#
#   bash gcp/serving/create_scheduler.sh --dry-run        # imprime tudo, nao toca a nuvem
#   bash gcp/serving/create_scheduler.sh job              # (1x) cria/atualiza o Cloud Run Job
#   bash gcp/serving/create_scheduler.sh create           # (1x) cria o cron diario
#   bash gcp/serving/create_scheduler.sh status           # o cron existe? quando rodou?
#   bash gcp/serving/create_scheduler.sh run-now          # dispara um ciclo AGORA (teste)
#
#   # DESABILITAR -- custo de volta a zero:
#   bash gcp/serving/create_scheduler.sh pause            # para o cron, preserva a config
#   bash gcp/serving/create_scheduler.sh delete           # REMOVE o cron (e, com
#                                                         # --com-job, tambem o Cloud Run Job)
#
# Depois de qualquer um dos dois, confira que nada ficou vivo:
#   gcloud compute instances list --project=residencia-511517     # "Listed 0 items."
#
# -------------------------------------------------------------------------------------
# PRE-REQUISITOS (uma vez, fora deste script)
#
#   1. APIs: run.googleapis.com, cloudscheduler.googleapis.com, compute.googleapis.com
#   2. Imagem do job publicada (ela precisa do repo + gcloud + python):
#          gcloud builds submit --tag "$IMAGE" .
#      O `trigger_cycle.sh` chama `gcloud compute instances create` e `gcloud compute
#      ssh`, entao a imagem tem de ter o SDK -- a base `gcr.io/google.com/cloudsdktool/
#      google-cloud-cli` com o repo e `pip install google-cloud-storage` resolve.
#   3. Service account $SA com, no projeto:
#          roles/compute.instanceAdmin.v1   (criar/apagar a Spot VM)
#          roles/iam.serviceAccountUser     (atribuir a SA da VM)
#          roles/storage.objectAdmin        (ler/escrever o indice no bucket)
#          roles/run.invoker                (o Scheduler chamar jobs:run)
# =====================================================================================

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-residencia-511517}"
REGION="${REGION:-us-central1}"

SCHEDULER_JOB="${SCHEDULER_JOB:-ciclo-captura-diario}"
RUN_JOB="${RUN_JOB:-ciclo-captura}"
SCHEDULE="${SCHEDULE:-0 6 * * *}"
TIME_ZONE="${TIME_ZONE:-America/Sao_Paulo}"
SA="${SA:-ciclo-captura@$PROJECT_ID.iam.gserviceaccount.com}"
IMAGE="${IMAGE:-gcr.io/$PROJECT_ID/ciclo-captura:latest}"

# Teto de uma execucao do ciclo (provisao + bootstrap + captura + upsert). Acima disto o
# Cloud Run mata a tarefa -- e o `trap EXIT` do run_slice ainda apaga a VM.
JOB_TIMEOUT="${JOB_TIMEOUT:-7200s}"
SERVING_URI="${SERVING_URI:-gs://$PROJECT_ID-bronze/serving}"

ACAO="${1:-create}"
COM_JOB=0
DRY_RUN=0
shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --project)  PROJECT_ID="$2"; shift 2 ;;
        --region)   REGION="$2"; shift 2 ;;
        --schedule) SCHEDULE="$2"; shift 2 ;;
        --com-job)  COM_JOB=1; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  sed -n '2,65p' "$0"; exit 0 ;;
        *) echo "flag desconhecida: $1" >&2; exit 2 ;;
    esac
done

run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] %s\n' "$*"
        return 0
    fi
    "$@"
}

# Endpoint administrativo do Cloud Run que EXECUTA o job (v2). E este URI que o cron
# chama; o `--oauth-service-account-email` e o que o autentica.
URI_DO_JOB="https://run.googleapis.com/v2/projects/$PROJECT_ID/locations/$REGION/jobs/$RUN_JOB:run"

echo "=== acao=$ACAO | projeto=$PROJECT_ID | regiao=$REGION | dry-run=$DRY_RUN ==="
echo "    cron:     '$SCHEDULE' ($TIME_ZONE)  job=$SCHEDULER_JOB"
echo "    dispara:  $URI_DO_JOB"

case "$ACAO" in

# -------------------------------------------------------------------------------------
job)
# -------------------------------------------------------------------------------------
    # O Cloud Run Job que roda UM ciclo. `--max-retries=0`: um ciclo que falhou tem de ser
    # investigado, nao repetido as cegas sobre uma L4 que pode ainda estar de pe.
    # `--parallelism=1 --tasks=1`: um ciclo e uma tarefa, sempre.
    echo "--- criando/atualizando o Cloud Run Job $RUN_JOB (imagem: $IMAGE) ---"
    # Em `--dry-run` nem a consulta de existencia roda: o ensaio nao pode depender de
    # nuvem autenticada (e a saida mostra o caminho de criacao, que e o que interessa ver).
    if [ "$DRY_RUN" != "1" ] \
       && gcloud run jobs describe "$RUN_JOB" --region="$REGION" --project="$PROJECT_ID" \
              >/dev/null 2>&1; then
        SUBCOMANDO=update
    else
        SUBCOMANDO=create
    fi
    run gcloud run jobs "$SUBCOMANDO" "$RUN_JOB" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --image="$IMAGE" \
        --service-account="$SA" \
        --max-retries=0 \
        --parallelism=1 \
        --tasks=1 \
        --task-timeout="$JOB_TIMEOUT" \
        --set-env-vars="PROJECT_ID=$PROJECT_ID,SERVING_URI=$SERVING_URI" \
        --command=bash \
        --args=gcp/serving/trigger_cycle.sh
    ;;

# -------------------------------------------------------------------------------------
create)
# -------------------------------------------------------------------------------------
    # `--attempt-deadline`: o Scheduler so espera o `jobs:run` ACEITAR a execucao (a
    # resposta e imediata; quem roda por horas e o job). `--max-retry-attempts=0` pelo
    # mesmo motivo do job: retry automatico = GPU duplicada.
    echo "--- criando o cron $SCHEDULER_JOB ---"
    run gcloud scheduler jobs create http "$SCHEDULER_JOB" \
        --project="$PROJECT_ID" \
        --location="$REGION" \
        --schedule="$SCHEDULE" \
        --time-zone="$TIME_ZONE" \
        --uri="$URI_DO_JOB" \
        --http-method=POST \
        --oauth-service-account-email="$SA" \
        --attempt-deadline=180s \
        --max-retry-attempts=0 \
        --description="Ciclo diario de captura/indexacao (Spot L4, teardown garantido)"
    echo "--- criado. Confira com: bash $0 status ---"
    ;;

# -------------------------------------------------------------------------------------
status)
# -------------------------------------------------------------------------------------
    run gcloud scheduler jobs describe "$SCHEDULER_JOB" \
        --project="$PROJECT_ID" --location="$REGION"
    echo ""
    echo "--- execucoes recentes do Cloud Run Job ---"
    run gcloud run jobs executions list --job="$RUN_JOB" \
        --project="$PROJECT_ID" --region="$REGION" --limit=5 || true
    echo ""
    echo "--- instancias vivas no projeto (o esperado e NENHUMA) ---"
    run gcloud compute instances list --project="$PROJECT_ID" || true
    ;;

# -------------------------------------------------------------------------------------
run-now)
# -------------------------------------------------------------------------------------
    echo "--- disparando o cron AGORA (mesma rota do horario) ---"
    run gcloud scheduler jobs run "$SCHEDULER_JOB" \
        --project="$PROJECT_ID" --location="$REGION"
    ;;

# -------------------------------------------------------------------------------------
pause|resume)
# -------------------------------------------------------------------------------------
    run gcloud scheduler jobs "$ACAO" "$SCHEDULER_JOB" \
        --project="$PROJECT_ID" --location="$REGION"
    echo "--- cron $SCHEDULER_JOB: $ACAO ---"
    ;;

# -------------------------------------------------------------------------------------
delete)
# -------------------------------------------------------------------------------------
    # REMOCAO -- o caminho de custo-zero garantido. Sem o cron nao ha ciclo, sem ciclo nao
    # ha GPU. O que sobra e objeto no GCS (centavos) e, se `--com-job` nao for passado, a
    # definicao do Cloud Run Job, que nao cobra parada.
    echo "--- removendo o cron $SCHEDULER_JOB ---"
    run gcloud scheduler jobs delete "$SCHEDULER_JOB" \
        --project="$PROJECT_ID" --location="$REGION" --quiet
    if [ "$COM_JOB" = "1" ]; then
        echo "--- removendo tambem o Cloud Run Job $RUN_JOB ---"
        run gcloud run jobs delete "$RUN_JOB" \
            --project="$PROJECT_ID" --region="$REGION" --quiet
    else
        echo "--- o Cloud Run Job $RUN_JOB foi PRESERVADO (parado nao cobra);" \
             "use --com-job para remove-lo tambem ---"
    fi
    echo ""
    echo "--- conferencia final: nada pode ter sobrado de pe ---"
    run gcloud compute instances list --project="$PROJECT_ID" || true
    ;;

*)
    echo "ERRO: acao desconhecida: '$ACAO'" >&2
    echo "use: job | create | status | run-now | pause | resume | delete" >&2
    exit 2 ;;
esac
