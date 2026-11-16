#!/usr/bin/env bash
# =====================================================================================
# FINETUNE QLoRA do baseline denso (Qwen3-VL) numa L4 SPOT efemera, persistindo em GCS,
# com teardown GARANTIDO. Gemeo de gcp/serving/capture_v4_spot.sh: la a VM CAPTURA o
# corpus v4, aqui a VM TREINA com o corpus que ja esta no bucket.
#
#   [1/9] preflight local (GCS_OUT, GCS_CORPUS, branch do checkout) -- nada pago ainda
#   [2/9] cria a VM L4 em SPOT, com fallback de zona            (a partir daqui ha custo)
#   [3/9] espera o SSH e envia repo + scripts
#   [4/9] bootstrap: untar do repo (o treino instala as deps sozinho; ver abaixo)
#   [5/9] stage: baixa `records.jsonl`, `bin/` e os dois arquivos de eval do $GCS_CORPUS
#   [6/9] pairs: `finetune.make_pairs` (gerador template, deterministico)
#   [7/9] smoke: 2 passos reais com a VRAM instrumentada -- o gate e OOM-ou-nao
#   [8/9] finetune: `finetune/run_gcp_finetune.sh` (nDCG@10 do A no v4 ANTES e DEPOIS)
#   [9/9] sobe o adaptador PEFT + o log do treino para $GCS_OUT -- DA VM
#   [trap] teardown: tenta o upload de novo, apaga a VM e ASSERTA que nada sobrou
#
# O QUE SOBREVIVE A RODADA E SO O QUE ESTA EM $GCS_OUT: o adaptador
# (`finetune/runs/<base>-<metodo>/`, PEFT + train_config.json) e o log com o antes/depois.
# A VM e descartavel. O treino e CURTO (corpus de 150 itens, ~450 pares template, 1 epoca
# ~ 28 passos): minutos de L4 -- por isso SPOT, onde a preempcao quase nao importa.
#
# -------------------------------------------------------------------------------------
# O QUE ESTE SCRIPT NAO FAZ (de proposito)
#
# Ele NAO reimplementa o ambiente de treino. Venv `--system-site-packages`, deps por hash
# do requirements, o guard "preferir o torch da imagem", o reparo do torchaudio e o
# `bitsandbytes --no-deps` do QLoRA ja estao em `finetune/run_gcp_finetune.sh`, exercitados
# fora da nuvem. A fase remota de treino DELEGA aquele script, so passando as env vars.
# Mexer aqui no que ele faz seria manter duas copias da mesma fiacao fragil.
#
# Tambem NAO instala ffmpeg nem faster-whisper, ao contrario do capture: o hook de
# avaliacao le os binarios JA extraidos do `bin/` que veio do bucket. O baseline A roteia
# `documento/imagem/video` para a rota de imagem com `a_repr` = CAMINHO do arquivo
# (keyframe ja e PNG em disco) e `texto/audio` para a rota de texto com `a_repr` = texto
# real (a transcricao ASR ja esta no record) -- ver `ensemble/native/unified_a.py`,
# `a_route`/`a_repr`. Nenhuma midia e reprocessada no embed, entao nenhum codec e preciso.
#
# O teardown (trap que ignora sinais, retry de cada delete, varredura de TODAS as zonas
# tentadas e a assercao final anti-leak) e o mesmo de `capture_v4_spot.sh`, que por sua vez
# o portou de `gcp/slice/run_slice.sh`. Credito la; aqui nao ha nenhuma variacao: e a parte
# que nao pode divergir entre os scripts.
#
# -------------------------------------------------------------------------------------
# RUN-BOOK
#
#   # ensaio completo, sem tocar a nuvem (nenhum gcloud e executado):
#   bash gcp/serving/finetune_v4_spot.sh --gcs-out gs://residencia-511517-corpus-v4/finetune/qwen3vl-qlora --dry-run
#
#   # rodada de verdade (L4 Spot; guarde a saida):
#   bash gcp/serving/finetune_v4_spot.sh \
#        2>&1 | tee logs/finetune-v4-$(date +%F).log
#
#   # conferir que NADA sobrou de pe (a resposta certa e "Listed 0 items."):
#   gcloud compute instances list --project=residencia-511517
#
#   # o que a rodada deixou no bucket (adaptador + log com o antes/depois):
#   gcloud storage ls -r gs://residencia-511517-corpus-v4/finetune/qwen3vl-qlora --project=residencia-511517
#
#   # trazer o adaptador para outra maquina:
#   gcloud storage rsync -r gs://residencia-511517-corpus-v4/finetune/qwen3vl-qlora/adapter \
#        finetune/runs/qwen3vl-qlora --project=residencia-511517
#
#   # provar o teardown sem esperar o treino (falha injetada):
#   FINETUNE_FAIL_AT=finetune bash gcp/serving/finetune_v4_spot.sh --gcs-out gs://x/y --dry-run
#
# PROJETO: toda chamada gcloud leva `--project=$PROJECT` explicito. Este script NUNCA roda
# `gcloud config set project`: o default do operador e dele.
# =====================================================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

PROJECT="${PROJECT:-residencia-511517}"

# ENTRADA: prefixo do corpus v4 no GCS, tal como `capture_v4_spot.sh` o deixou --
# `records.jsonl`, `bin/<eixo>/` e `eval/{queries_v4.txt,qrels_v4.tsv}`.
GCS_CORPUS="${GCS_CORPUS:-gs://residencia-511517-corpus-v4/ensemble/v4}"

# SAIDA no GCS -- PARAMETRO OBRIGATORIO (ver o preflight). Sem ele a rodada so gastaria
# GPU para jogar o adaptador fora junto com a VM.
GCS_OUT="${GCS_OUT:-}"

BRANCH="${BRANCH:-feat/semana10-finetune-denso}"   # branch esperada no checkout local

VM="${VM:-govexe-finetune-v4}"          # nome PROPRIO: nao colide com govexe-capture-v4,
                                        # govexe-embed nem govexe-ciclo
# Mesma ordem do capture: a L4 deu STOCKOUT em todas as zonas us-* na rodada do objetivo
# #2 e so northamerica-northeast2-a tinha capacidade. O pool SPOT e outro, mas a ordem
# continua valendo como aposta.
ZONES="${ZONES:-northamerica-northeast2-a us-central1-a us-central1-b us-east1-c us-east4-a us-west1-a}"

MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"
ACCELERATOR="${ACCELERATOR:-nvidia-l4}"
# 150GB: corpus v4 (~1GB com os binarios) + pesos do Qwen3-VL-2B (~5GB) + venv + cache do
# HF. Menos que os 200GB do capture, que retinha os MP4 das sessoes.
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-150GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
# SPOT e o default DESTE script: o treino dura minutos e o que ja subiu ao GCS fica.
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"

# Arvore do corpus v4 no repo da VM (relativa a raiz, como todo caminho de dados daqui):
# e para ca que a fase `stage` baixa o bucket, e e isto que `meta["path"]` dos records
# espera encontrar.
CORPUS_DIR="${CORPUS_DIR:-data/govexe/ensemble_orcamento_v4}"

# Hiperparametros do treino (repassados ao run_gcp_finetune.sh, que valida base/metodo).
BASE="${BASE:-qwen3vl}"
METHOD="${METHOD:-qlora}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEG="${MAX_NEG:-1}"
LORA_R="${LORA_R:-32}"
# 1 => desliga o gradient checkpointing (`--no-gradient-checkpointing` no train_lora).
# DEFAULT 1 AQUI, ao contrario do run_gcp_finetune.sh: com QLoRA (base 4-bit congelada) o
# bloco checkpointed REENTRANT no caminho de IMAGEM do Qwen3-VL recebe entradas sem
# requires_grad, destaca a saida do grafo e o passo morre com "element 0 of tensors does
# not require grad". O pico medido na L4 foi 3.7GB de 23GB, entao o checkpointing so
# custava tempo: desligado, o gradiente volta E o treino acelera. 0 religa o checkpointing.
NO_GRAD_CKPT="${NO_GRAD_CKPT:-1}"

DRY_RUN=0

# -------------------------------------------------------------------------------------
# Estado do teardown (o que ja foi tocado na nuvem)
# -------------------------------------------------------------------------------------
ZONE=""                 # zona onde a VM subiu de fato
ZONAS_TENTADAS=""       # toda zona em que um `create` foi tentado (o teardown varre todas)
UPLOAD_FEITO=0          # 1 depois que o adaptador chegou em $GCS_OUT
TMP_DIR=""

TEARDOWN_RETRIES="${TEARDOWN_RETRIES:-3}"
TEARDOWN_ESPERA="${TEARDOWN_ESPERA:-20}"      # segundos entre tentativas de delete
UPLOAD_TIMEOUT="${UPLOAD_TIMEOUT:-1800}"      # teto do upload do adaptador
LEAK_EXIT="${LEAK_EXIT:-70}"                  # status quando sobra recurso de pe

uso() {
    sed -n '3,/^# =====/p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --gcs-out)        GCS_OUT="$2"; shift 2 ;;
        --gcs-corpus)     GCS_CORPUS="$2"; shift 2 ;;
        --project)        PROJECT="$2"; shift 2 ;;
        --zones)          ZONES="$2"; shift 2 ;;
        --vm)             VM="$2"; shift 2 ;;
        --branch)         BRANCH="$2"; shift 2 ;;
        --boot-disk-size) BOOT_DISK_SIZE="$2"; shift 2 ;;
        --corpus-dir)     CORPUS_DIR="$2"; shift 2 ;;
        --base)           BASE="$2"; shift 2 ;;
        --method)         METHOD="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --batch-size)     BATCH_SIZE="$2"; shift 2 ;;
        --max-neg)        MAX_NEG="$2"; shift 2 ;;
        --lora-r)         LORA_R="$2"; shift 2 ;;
        --no-grad-ckpt)   NO_GRAD_CKPT=1; shift ;;
        --grad-ckpt)      NO_GRAD_CKPT=0; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        uso 0 ;;
        *) echo "flag desconhecida: $1" >&2; uso 2 ;;
    esac
done

# -------------------------------------------------------------------------------------
# Primitivas de execucao (de gcp/serving/capture_v4_spot.sh)
#
# Tudo que toca a nuvem passa por `run`/`existe`/`com_timeout`: e isso que faz o
# `--dry-run` imprimir a rodada inteira sem executar um unico gcloud.
# -------------------------------------------------------------------------------------
etapa() {
    echo ""
    echo "=== $* ==="
}

run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] %s\n' "$*"
        return 0
    fi
    "$@"
}

# Teste de existencia de recurso. Em `--dry-run` devolve 1 (NAO existe) sem consultar a
# nuvem: na conferencia do teardown isso significa "nada a vazar", que e a verdade num
# dry-run (nenhuma VM foi criada).
existe() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] %s   # (dry-run nao consulta a nuvem: segue como inexistente)\n' "$*"
        return 1
    fi
    "$@" >/dev/null 2>&1
}

# `run` com teto de tempo, para o que nao pode travar o fluxo (o upload de resgate numa VM
# meio-preemptada). Onde nao houver `timeout` (coreutils), segue sem o teto.
com_timeout() {
    local segundos="$1"; shift
    if command -v timeout >/dev/null 2>&1; then
        run timeout "$segundos" "$@"
    else
        run "$@"
    fi
}

# Precondicao de AMBIENTE que falta. Numa rodada de verdade e erro fatal; em `--dry-run` e
# so um aviso: a verificacao do script (sintaxe, fluxo, teardown) nao pode depender de ter
# nuvem autenticada na maquina de quem revisa.
precondicao_ausente() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "--- aviso (dry-run segue assim mesmo): $* ---" >&2
        return 0
    fi
    echo "ERRO: $*" >&2
    exit 2
}

# Gancho de verificacao do trap: `FINETUNE_FAIL_AT=<etapa>` faz a etapa falhar de proposito
# (etapas: provision, bootstrap, stage, finetune, upload). Serve para provar que o teardown
# roda em qualquer saida -- e so isso.
gancho_de_falha() {
    if [ "${FINETUNE_FAIL_AT:-}" = "$1" ]; then
        echo "!!! falha INJETADA na etapa '$1' (FINETUNE_FAIL_AT) !!!" >&2
        return 1
    fi
    return 0
}

remoto() {   # executa um comando DENTRO da VM
    run gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --quiet \
        --command "$1"
}

remoto_com_timeout() {   # idem, com teto de tempo (usado no upload e no resgate)
    local segundos="$1"; shift
    com_timeout "$segundos" gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
        --quiet --command "$1"
}

# -------------------------------------------------------------------------------------
# PERSISTENCIA -- o upload roda DENTRO da VM
#
# De la o adaptador vai direto para o bucket pela rede da GCP, e nao depende de a maquina
# que disparou a rodada continuar viva. A VM sobe com `--scopes=cloud-platform`, entao o
# gcloud dela ja tem credencial para escrever no bucket do projeto.
#
# Chamada DUAS vezes de proposito, como o capture faz com o corpus: no fluxo normal (antes
# do teardown) e de novo de dentro do teardown, se a primeira nao tiver acontecido. Se o
# treino nao chegou a produzir o adaptador, a fase falha limpo e o teardown segue para os
# deletes.
# -------------------------------------------------------------------------------------
subir_resultado_para_gcs() {
    echo "--- subindo o adaptador da VM para $GCS_OUT (teto: ${UPLOAD_TIMEOUT}s) ---"
    echo "    [na VM] gcloud storage rsync -r <finetune/runs/$BASE-$METHOD> $GCS_OUT/adapter --project=$PROJECT"
    echo "    [na VM] gcloud storage cp ~/finetune_train.log $GCS_OUT/finetune_train.log --project=$PROJECT"
    if remoto_com_timeout "$UPLOAD_TIMEOUT" "bash ~/finetune_remote.sh upload"; then
        UPLOAD_FEITO=1
        return 0
    fi
    return 1
}

# -------------------------------------------------------------------------------------
# TEARDOWN -- a razao de ser deste script (identico ao de capture_v4_spot.sh)
#
# Registrado em `trap ... EXIT` logo antes do primeiro recurso ser criado, entao roda no
# sucesso, no erro (`set -e`), na interrupcao e na queda do terminal.
#
#   1. `trap '' INT TERM HUP`: o teardown IGNORA sinais. Um segundo Ctrl-C no meio dele
#      nao pode matar o shell deixando a L4 de pe, cobrando.
#   2. `set +eu +o pipefail`: nem erro, nem variavel nao-setada, nem pipe quebrado podem
#      abortar a limpeza antes dos deletes.
#   3. O resgate do adaptador vem antes dos deletes (apagar a VM destroi o treino), mas com
#      TIMEOUT: um upload travado numa VM preemptada nao pode adiar a limpeza para sempre.
#   4. Todo delete tem RETRY: "operation in progress" logo apos um create interrompido e
#      falha transitoria, e engoli-la com `|| true` seria vazar recurso pago.
#   5. Conferencia final que ASSERTA: se algo sobreviveu, o script grita LEAK, imprime o
#      delete manual e sai com status != 0.
# -------------------------------------------------------------------------------------
apagar_com_retry() {
    local rotulo="$1"; shift
    local tentativa=1
    while [ "$tentativa" -le "$TEARDOWN_RETRIES" ]; do
        if run "$@"; then
            return 0
        fi
        if [ "$tentativa" -lt "$TEARDOWN_RETRIES" ]; then
            echo "--- $rotulo: tentativa $tentativa falhou (operacao em andamento?);" \
                 "nova tentativa em ${TEARDOWN_ESPERA}s ---" >&2
            run sleep "$TEARDOWN_ESPERA"
        fi
        tentativa=$((tentativa + 1))
    done
    echo "--- $rotulo: FALHOU em $TEARDOWN_RETRIES tentativas ---" >&2
    return 1
}

teardown() {
    local rc=$?
    # Ordem importa: desligar `set -e`/`-u` e ignorar sinais ANTES de qualquer comando.
    set +e +u +o pipefail
    trap - EXIT
    trap '' INT TERM HUP

    etapa "TEARDOWN (saida com status $rc) -- apagando a VM $VM"
    echo "--- a partir daqui INT/TERM/HUP sao IGNORADOS: a limpeza nao se interrompe ---"

    # Ultima chance de salvar o adaptador: se a rodada morreu depois de treinar mas antes
    # de subir, ele ainda esta na VM que estamos prestes a apagar. Se o treino nem chegou
    # la, a fase falha limpo e seguimos para os deletes.
    if [ "${UPLOAD_FEITO:-0}" = "0" ] && [ -n "${ZONE:-}" ]; then
        echo "--- tentativa final de subir o adaptador para $GCS_OUT ---"
        subir_resultado_para_gcs || echo "--- o resgate do adaptador falhou ou estourou o tempo; seguindo para os deletes ---" >&2
    fi

    local zona
    for zona in ${ZONAS_TENTADAS:-}; do
        echo "--- apagando a VM $VM em $zona (se existir) ---"
        apagar_com_retry "delete da VM em $zona" \
            gcloud compute instances delete "$VM" --zone="$zona" \
                --project="$PROJECT" --quiet
    done
    if [ -z "${ZONAS_TENTADAS:-}" ]; then
        echo "--- nenhuma VM foi criada nesta rodada ---"
    fi

    # -- conferencia que ASSERTA (criterio de aceite: nao pode sobrar nada) --------------
    # Pequena espera para o estado da API assentar e nao gerar LEAK falso logo apos um
    # delete que ja retornou sucesso.
    run sleep 10
    echo "--- conferencia final: nada pode ter sobrado ---"
    local vazou=0
    for zona in ${ZONAS_TENTADAS:-}; do
        if existe gcloud compute instances describe "$VM" --zone="$zona" \
                  --project="$PROJECT"; then
            vazou=1
            echo "!!! LEAK: a VM $VM AINDA EXISTE em $zona -- ela CONTINUA COBRANDO !!!" >&2
            echo "    apague a mao:" >&2
            echo "    gcloud compute instances delete $VM --zone=$zona --project=$PROJECT --quiet" >&2
        fi
    done

    if [ -n "${TMP_DIR:-}" ] && [ -d "${TMP_DIR:-}" ]; then
        rm -rf "$TMP_DIR"
    fi

    if [ "$vazou" = "1" ]; then
        echo "!!! TEARDOWN INCOMPLETO: ha recurso pago de pe (veja os LEAK acima) !!!" >&2
        if [ "$rc" = "0" ]; then rc="$LEAK_EXIT"; fi
    else
        echo "--- conferencia ok: nenhuma VM sobrou ---"
    fi

    if [ "${UPLOAD_FEITO:-0}" = "1" ]; then
        echo "--- o adaptador esta em $GCS_OUT (e so isso sobreviveu a rodada) ---"
    else
        echo "--- ATENCAO: o adaptador NAO chegou a $GCS_OUT; a rodada nao deixou nada ---" >&2
    fi

    echo "=== teardown concluido (status final: $rc) ==="
    exit "$rc"
}

# -------------------------------------------------------------------------------------
# Script remoto (roda DENTRO da VM)
#
# Gerado aqui e enviado por scp, como no capture: a configuracao viaja num
# `finetune_env.sh` separado para o corpo do script nao virar um emaranhado de aspas.
# -------------------------------------------------------------------------------------
gerar_scripts_remotos() {
    TMP_DIR="$(mktemp -d)"

    cat > "$TMP_DIR/finetune_env.sh" <<ENV
# Gerado por gcp/serving/finetune_v4_spot.sh -- configuracao do finetune v4.
PROJECT='$PROJECT'
GCS_CORPUS='$GCS_CORPUS'
GCS_OUT='$GCS_OUT'
CORPUS_DIR='$CORPUS_DIR'
BASE='$BASE'
METHOD='$METHOD'
EPOCHS='$EPOCHS'
BATCH_SIZE='$BATCH_SIZE'
MAX_NEG='$MAX_NEG'
LORA_R='$LORA_R'
NO_GRAD_CKPT='$NO_GRAD_CKPT'
BRANCH='$BRANCH'
ENV

    cat > "$TMP_DIR/finetune_remote.sh" <<'REMOTE'
#!/usr/bin/env bash
# Gerado por gcp/serving/finetune_v4_spot.sh -- roda DENTRO da VM L4. Fase em $1.
set -euo pipefail
FASE="${1:?uso: finetune_remote.sh <bootstrap|stage|pairs|smoke|finetune|upload>}"
# shellcheck disable=SC1091
source "$HOME/finetune_env.sh"
echo "=== [vm $(hostname)] fase=$FASE $(date) ==="

REPO_DIR="$HOME/residencia"
RECORDS_REL="$CORPUS_DIR/records.jsonl"
BIN_REL="$CORPUS_DIR/bin"
QUERIES_REL="ensemble/eval/queries_v4.txt"
QRELS_REL="ensemble/eval/qrels_v4.tsv"
PAIRS_REL="finetune/data/pairs.jsonl"
# Onde `train_lora.py` grava quando `--output-dir` nao e passado (`DEFAULT_OUT` =
# "finetune/runs", `out_path` = "<DEFAULT_OUT>/<base>-<metodo>"): e esta a arvore que a
# fase `upload` sobe.
OUT_REL="finetune/runs/${BASE}-${METHOD}"
TRAIN_LOG="$HOME/finetune_train.log"

case "$FASE" in

# =====================================================================================
bootstrap)
# =====================================================================================
    # Guard por SENTINELA DE CODIGO (e nao por `[ -d "$REPO_DIR" ]`), como no capture:
    # outra etapa pode PRE-CRIAR `~/residencia` e o teste de diretorio daria falso. A
    # sentinela aqui e o proprio script que a fase `finetune` vai invocar.
    if [ ! -f "$REPO_DIR/finetune/run_gcp_finetune.sh" ]; then
        mkdir -p "$REPO_DIR"
        tar xzf "$HOME/repo.tgz" -C "$REPO_DIR"
    fi
    cd "$REPO_DIR"
    [ -f "$REPO_DIR/requirements-slurm.txt" ] \
        || { echo "ERRO: requirements-slurm.txt ausente -- o tarball chegou incompleto" >&2; exit 1; }

    mkdir -p "$HOME/.cache/huggingface"

    # SEM ffmpeg/whisper de proposito: o hook de avaliacao le os binarios JA extraidos do
    # `bin/` (keyframe de video e um PNG em disco; a transcricao do audio ja esta no
    # record). Ver o cabecalho do script local.
    echo "--- deps do treino: a cargo de finetune/run_gcp_finetune.sh (venv, torch, bitsandbytes) ---"

    # GPU: informativo aqui (o driver pode ainda estar instalando no primeiro boot). O
    # gate de verdade e a fase `smoke`, que so passa se um passo real couber na VRAM.
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        || echo "--- sem GPU visivel ainda (driver instalando?); o smoke e quem decide ---"
    echo "=== bootstrap concluido $(date) ==="
    ;;

# =====================================================================================
stage)
# =====================================================================================
    # O corpus v4 vem do bucket, nao do tarball. A arvore tem de ser reconstruida TAL E
    # QUAL: `meta["path"]` dos records aponta para `$CORPUS_DIR/bin/<eixo>/<arquivo>`.
    cd "$REPO_DIR"
    [ -n "$GCS_CORPUS" ] || { echo "ERRO: GCS_CORPUS vazio na VM" >&2; exit 2; }
    mkdir -p "$CORPUS_DIR" ensemble/eval

    echo "=== [stage] $GCS_CORPUS -> $CORPUS_DIR + ensemble/eval ==="
    gcloud storage rsync -r "$GCS_CORPUS/bin" "$BIN_REL" --project="$PROJECT"
    gcloud storage cp "$GCS_CORPUS/records.jsonl" "$RECORDS_REL" --project="$PROJECT"
    # Nomes CANONICOS no repo: e com estes que a fase `finetune` chama o hook de eval.
    gcloud storage cp "$GCS_CORPUS/eval/queries_v4.txt" "$QUERIES_REL" --project="$PROJECT"
    gcloud storage cp "$GCS_CORPUS/eval/qrels_v4.tsv" "$QRELS_REL" --project="$PROJECT"

    # Falhar ALTO aqui: sem records ou sem binarios o treino apontaria para o vazio e so
    # descobriria isso depois de carregar o modelo (GPU paga a toa).
    [ -s "$RECORDS_REL" ] \
        || { echo "ERRO: $RECORDS_REL vazio ou ausente apos o stage" >&2; exit 1; }
    [ -d "$BIN_REL" ] \
        || { echo "ERRO: $BIN_REL ausente apos o stage -- os records apontam para o vazio" >&2; exit 1; }
    [ -s "$QUERIES_REL" ] && [ -s "$QRELS_REL" ] \
        || { echo "ERRO: queries/qrels do v4 ausentes -- o antes/depois ficaria mudo" >&2; exit 1; }

    echo "--- records: $(wc -l < "$RECORDS_REL") linhas (esperado: 150) ---"
    du -sh "$BIN_REL" || true
    for eixo in documento imagem audio video; do
        if [ -d "$BIN_REL/$eixo" ]; then
            echo "    $eixo: $(find "$BIN_REL/$eixo" -type f | wc -l) arquivos"
        fi
    done
    echo "--- eval: $(wc -l < "$QUERIES_REL") queries | $(wc -l < "$QRELS_REL") julgamentos ---"
    echo "=== stage concluido $(date) ==="
    ;;

# =====================================================================================
pairs)
# =====================================================================================
    # `make_pairs` (gerador `template`, o default) importa so numpy + stdlib via
    # `ensemble.native.unified_a`/`ensemble.records`: roda no python do SISTEMA da DLVM,
    # antes do venv do treino existir. Deterministico e idempotente.
    cd "$REPO_DIR"
    python3 -c "import numpy" 2>/dev/null || python3 -m pip install --user -q numpy
    echo "=== [pairs] $RECORDS_REL -> $PAIRS_REL (gerador template) ==="
    PYTHONPATH="$REPO_DIR" python3 -m finetune.make_pairs \
        --records "$RECORDS_REL" --out "$PAIRS_REL"
    [ -s "$PAIRS_REL" ] || { echo "ERRO: $PAIRS_REL vazio" >&2; exit 1; }
    echo "--- pares gerados: $(wc -l < "$PAIRS_REL") ---"
    echo "=== pairs concluido $(date) ==="
    ;;

# =====================================================================================
smoke)
# =====================================================================================
    # O gate e OOM-ou-nao: 2 passos REAIS com o batch REAL ja atingem o pico de VRAM (mais
    # passos nao o aumentam). Sem eval aqui -- o antes/depois e da fase `finetune`.
    cd "$REPO_DIR"
    POLL_CSV="$HOME/smoke_vram.csv"
    rm -f "$POLL_CSV"
    POLL_PID=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 \
            > "$POLL_CSV" 2>/dev/null &
        POLL_PID=$!
    fi

    echo "=== [smoke] 2 passos (BATCH_SIZE=1 MAX_NEG=1) com a VRAM instrumentada ==="
    set +e
    DRY_RUN=0 MAX_STEPS=2 BATCH_SIZE=1 MAX_NEG=1 NO_EVAL=1 \
        BASE="$BASE" METHOD="$METHOD" LORA_R="$LORA_R" \
        NO_GRAD_CKPT="$NO_GRAD_CKPT" \
        HF_CACHE_DIR="$HOME/.cache/huggingface" \
        bash finetune/run_gcp_finetune.sh
    RC=$?
    set -e

    if [ -n "$POLL_PID" ]; then
        kill "$POLL_PID" 2>/dev/null || true
        wait "$POLL_PID" 2>/dev/null || true
    fi
    PICO="$(sort -n "$POLL_CSV" 2>/dev/null | tail -n 1)"
    if [ -n "${PICO:-}" ]; then
        echo "--- PICO de memory.used observado: ${PICO} MiB (L4: 23034 MiB;" \
             "folga = $((23034 - PICO)) MiB) ---"
    else
        echo "--- sem leitura de VRAM (nvidia-smi indisponivel): seguindo pelo status do treino ---"
    fi

    if [ "$RC" != "0" ]; then
        echo "ERRO: o smoke de 2 passos falhou (status $RC) -- OOM provavel." >&2
        echo "    Mitigacao, nesta ordem (ver a nota de VRAM no cabecalho de" >&2
        echo "    finetune/run_gcp_finetune.sh): 1) limitar 'max_pixels' do processor do" >&2
        echo "    Qwen3-VL, que e o que domina as ativacoes de pagina inteira;" >&2
        echo "    2) so entao reduzir BATCH_SIZE/MAX_NEG (ja estao em 1/1 aqui)." >&2
        echo "    NADA sobe para o GCS: a rodada para antes de pagar o treino inteiro." >&2
        exit "$RC"
    fi
    echo "=== smoke concluido $(date) ==="
    ;;

# =====================================================================================
finetune)
# =====================================================================================
    # O treino de verdade. `EVAL_CORPUS` e quem LIGA o hook do ensemble: o run_gcp/
    # train_lora imprime o nDCG@10 do baseline A no v4 ANTES e DEPOIS do adaptador -- o
    # entregavel cientifico da semana. Por isso o log inteiro vai para $TRAIN_LOG e sobe
    # junto com o adaptador.
    cd "$REPO_DIR"
    [ -s "$PAIRS_REL" ] || { echo "ERRO: $PAIRS_REL ausente -- rode a fase pairs antes" >&2; exit 1; }
    echo "=== [finetune] BASE=$BASE METHOD=$METHOD EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE MAX_NEG=$MAX_NEG LORA_R=$LORA_R ==="
    DRY_RUN=0 EPOCHS="$EPOCHS" BASE="$BASE" METHOD="$METHOD" \
        BATCH_SIZE="$BATCH_SIZE" MAX_NEG="$MAX_NEG" LORA_R="$LORA_R" \
        NO_GRAD_CKPT="$NO_GRAD_CKPT" \
        HF_CACHE_DIR="$HOME/.cache/huggingface" \
        EVAL_CORPUS="$RECORDS_REL" \
        EVAL_QUERIES="$QUERIES_REL" EVAL_QRELS="$QRELS_REL" \
        bash finetune/run_gcp_finetune.sh 2>&1 | tee "$TRAIN_LOG"
    [ -d "$OUT_REL" ] \
        || { echo "ERRO: $OUT_REL nao existe -- o treino nao produziu adaptador" >&2; exit 1; }
    echo "--- adaptador em $OUT_REL ---"
    ls -la "$OUT_REL"
    echo "--- antes/depois no log ---"
    grep -i "ndcg" "$TRAIN_LOG" || echo "    (nenhuma linha de nDCG no log: confira o hook de eval)"
    echo "=== finetune concluido $(date) ==="
    ;;

# =====================================================================================
upload)
# =====================================================================================
    # O que sobrevive a rodada: o adaptador PEFT (+ train_config.json) e o log do treino.
    # `rsync -r` para a arvore do adaptador (retomavel, o que importa quando a primeira
    # tentativa morreu por preempcao) e `cp` para o log (um arquivo).
    cd "$REPO_DIR"
    [ -n "$GCS_OUT" ] || { echo "ERRO: GCS_OUT vazio na VM" >&2; exit 2; }
    if [ ! -d "$OUT_REL" ]; then
        echo "ERRO: $OUT_REL nao existe -- o treino nao terminou; nada a subir" >&2
        exit 1
    fi
    echo "=== [upload] $OUT_REL + $TRAIN_LOG -> $GCS_OUT ==="
    gcloud storage rsync -r "$OUT_REL" "$GCS_OUT/adapter" --project="$PROJECT"
    if [ -f "$TRAIN_LOG" ]; then
        gcloud storage cp "$TRAIN_LOG" "$GCS_OUT/finetune_train.log" --project="$PROJECT"
    else
        echo "--- aviso: $TRAIN_LOG ausente; so o adaptador sobe ---" >&2
    fi
    echo "--- conferencia do que ficou no bucket ---"
    gcloud storage ls -r "$GCS_OUT" --project="$PROJECT"
    echo "=== upload concluido $(date) ==="
    ;;

*)
    echo "ERRO: fase desconhecida: $FASE" >&2
    exit 2
    ;;
esac
REMOTE
    chmod +x "$TMP_DIR/finetune_remote.sh"
}

# -------------------------------------------------------------------------------------
# Provisionamento (fallback de zona e flags de SPOT: de gcp/serving/capture_v4_spot.sh)
# -------------------------------------------------------------------------------------
provisionar_vm() {
    etapa "[2/9] VM L4 ($MACHINE_TYPE + $ACCELERATOR, disco $BOOT_DISK_SIZE) com fallback de zona"
    echo "--- zonas na ordem: $ZONES ---"
    # `--instance-termination-action=DELETE` e o que importa para o teardown: numa
    # preempcao a VM se APAGA em vez de ficar STOPPED cobrando disco -- e o trap de saida
    # ainda tenta o delete depois, que vira no-op.
    local spot_flags=()
    if [ "$PROVISIONING_MODEL" = "SPOT" ]; then
        echo "--- provisionamento SPOT (o treino dura minutos; a preempcao quase nao pesa) ---"
        spot_flags=(--provisioning-model=SPOT --instance-termination-action=DELETE)
    fi
    local zona
    for zona in $ZONES; do
        echo "--- tentando $zona ---"
        ZONAS_TENTADAS="$ZONAS_TENTADAS $zona"
        if run gcloud compute instances create "$VM" \
                --project="$PROJECT" \
                --zone="$zona" \
                --machine-type="$MACHINE_TYPE" \
                --accelerator="type=$ACCELERATOR,count=1" \
                --maintenance-policy=TERMINATE \
                ${spot_flags[@]+"${spot_flags[@]}"} \
                --image-family="$IMAGE_FAMILY" \
                --image-project="$IMAGE_PROJECT" \
                --boot-disk-size="$BOOT_DISK_SIZE" \
                --boot-disk-type=pd-balanced \
                --metadata="install-nvidia-driver=True" \
                --scopes=https://www.googleapis.com/auth/cloud-platform; then
            ZONE="$zona"
            echo "--- VM $VM de pe em $ZONE ---"
            return 0
        fi
        echo "--- $zona falhou (STOCKOUT de L4?): proxima zona ---" >&2
    done
    echo "ERRO: nenhuma zona da lista tinha capacidade de L4 em $PROVISIONING_MODEL: $ZONES" >&2
    return 1
}

esperar_ssh() {
    etapa "[3a/9] esperando o SSH da VM responder"
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] gcloud compute ssh %s --zone=%s --project=%s --command "true"  (ate 20x)\n' \
            "$VM" "$ZONE" "$PROJECT"
        return 0
    fi
    local i
    for i in $(seq 1 20); do
        if gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --quiet \
               --command "true" >/dev/null 2>&1; then
            echo "--- ssh ok na tentativa $i ---"
            return 0
        fi
        sleep 15
    done
    echo "ERRO: a VM nao respondeu ao SSH" >&2
    return 1
}

enviar_repo() {
    etapa "[3b/9] enviando repo e scripts para a VM"
    # Arvore de trabalho (e nao `git archive`): o que vale e o checkout local, ja conferido
    # contra $BRANCH no preflight -- `finetune/run_gcp_finetune.sh` e o eixo documento do
    # v4 so existem nesta branch, que nem esta pushed. O corpus v4 fica FORA do tarball: a
    # fase `stage` o baixa do bucket, e mandar uma copia local so gastaria link e correria
    # o risco de sobrescrever o corpus bom.
    run tar czf "$TMP_DIR/repo.tgz" -C "$REPO_DIR" \
        --exclude=.git --exclude='.venv*' --exclude=__pycache__ \
        --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.benchmarks \
        --exclude=results --exclude=logs --exclude=.cache \
        --exclude='data/govexe/ptbr_planos/pages' \
        --exclude='data/govexe/ptbr_planos/pdf_cache' \
        --exclude="$CORPUS_DIR" \
        --exclude=data/vidore_cache .

    # No Windows o `gcloud compute scp` delega ao `pscp.exe`, que NAO aceita `~` como
    # destino remoto nem varios arquivos de origem de uma vez: um arquivo por chamada, com
    # destino RELATIVO (o pscp resolve a partir do home de login).
    run gcloud compute scp "$TMP_DIR/repo.tgz" "$VM:repo.tgz" \
        --zone="$ZONE" --project="$PROJECT" --quiet
    run gcloud compute scp "$TMP_DIR/finetune_remote.sh" "$VM:finetune_remote.sh" \
        --zone="$ZONE" --project="$PROJECT" --quiet
    run gcloud compute scp "$TMP_DIR/finetune_env.sh" "$VM:finetune_env.sh" \
        --zone="$ZONE" --project="$PROJECT" --quiet
}

# -------------------------------------------------------------------------------------
# [1/9] Preflight (local, antes de criar qualquer recurso pago)
# -------------------------------------------------------------------------------------
etapa "[1/9] finetune QLoRA do v4 | projeto=$PROJECT | vm=$VM ($PROVISIONING_MODEL) | dry-run=$DRY_RUN"
echo "    corpus de entrada (GCS): $GCS_CORPUS"
echo "    saida no GCS:            ${GCS_OUT:-<VAZIO>}"
echo "    treino: base=$BASE metodo=$METHOD epochs=$EPOCHS batch=$BATCH_SIZE max_neg=$MAX_NEG lora_r=$LORA_R no_grad_ckpt=$NO_GRAD_CKPT"
echo "    disco=$BOOT_DISK_SIZE  maquina=$MACHINE_TYPE+$ACCELERATOR"
echo "    corpus na VM:   $CORPUS_DIR (records.jsonl + bin/)"
echo "    adaptador na VM: finetune/runs/$BASE-$METHOD"
echo "    branch esperada no checkout: ${BRANCH:-<checagem desligada>}"

# GCS_OUT e OBRIGATORIO: sem destino a rodada pagaria GPU para jogar o adaptador fora junto
# com a VM. Este erro e fatal mesmo em --dry-run -- ele nao depende da nuvem, e um ensaio
# sem destino ensaiaria a rodada errada.
if [ -z "$GCS_OUT" ]; then
    echo "ERRO: GCS_OUT e obrigatorio -- sem ele o adaptador morre junto com a VM." >&2
    echo "    use: --gcs-out gs://<bucket>/<prefixo>/   (ou GCS_OUT=gs://...)" >&2
    exit 2
fi
case "$GCS_OUT" in
    gs://*) ;;
    *) echo "ERRO: GCS_OUT tem de ser um URI gs:// (recebido: '$GCS_OUT')" >&2; exit 2 ;;
esac
case "$GCS_CORPUS" in
    gs://*) ;;
    *) echo "ERRO: GCS_CORPUS tem de ser um URI gs:// (recebido: '$GCS_CORPUS')" >&2; exit 2 ;;
esac
GCS_OUT="${GCS_OUT%/}"            # normaliza: os filhos sao montados com "$GCS_OUT/..."
GCS_CORPUS="${GCS_CORPUS%/}"

# A saida nao pode cair DENTRO do corpus de entrada: o `rsync -r` do adaptador escreveria
# por cima de `records.jsonl`/`bin/` e corromperia o corpus que custou a rodada de captura.
if [ "$GCS_OUT" = "$GCS_CORPUS" ] || [ "${GCS_OUT#"$GCS_CORPUS"/}" != "$GCS_OUT" ]; then
    echo "ERRO: GCS_OUT ('$GCS_OUT') esta dentro do corpus de entrada ('$GCS_CORPUS')." >&2
    echo "    o upload do adaptador sobrescreveria records.jsonl/bin/: use outro prefixo." >&2
    exit 2
fi

case "$PROVISIONING_MODEL" in
    STANDARD|SPOT) ;;
    *) echo "ERRO: PROVISIONING_MODEL invalido: '$PROVISIONING_MODEL' (use STANDARD ou SPOT)" >&2
       exit 2 ;;
esac

# Base/metodo: a validacao de verdade e do run_gcp_finetune.sh, mas errar aqui custa
# barato e errar la custa uma VM de pe.
case "$BASE" in colqwen2.5|colembed8b|qwen3vl) ;;
    *) echo "ERRO: BASE invalida: '$BASE' (use colqwen2.5|colembed8b|qwen3vl)" >&2; exit 2 ;;
esac
case "$METHOD" in lora|qlora) ;;
    *) echo "ERRO: METHOD invalido: '$METHOD' (use lora|qlora)" >&2; exit 2 ;;
esac

# Hiperparametros numericos: eles viajam no `finetune_env.sh` e viram env vars na VM.
for _par in EPOCHS BATCH_SIZE MAX_NEG LORA_R; do
    eval "_val=\$$_par"
    case "$_val" in
        ''|*[!0-9]*) echo "ERRO: $_par tem de ser inteiro >= 0 (recebido: '$_val')" >&2; exit 2 ;;
    esac
done
[ "$EPOCHS" -gt 0 ] || { echo "ERRO: EPOCHS tem de ser > 0" >&2; exit 2; }
[ "$BATCH_SIZE" -gt 0 ] || { echo "ERRO: BATCH_SIZE tem de ser > 0" >&2; exit 2; }
[ "$LORA_R" -gt 0 ] || { echo "ERRO: LORA_R tem de ser > 0" >&2; exit 2; }

# Toggle, nao numero: `run_gcp_finetune.sh` so compara com "1", entao qualquer outro valor
# (um "true", um "2") religaria o checkpointing EM SILENCIO -- e o treino morreria de novo
# com o grad-detach do QLoRA. Melhor barrar aqui, antes da VM.
case "$NO_GRAD_CKPT" in
    0|1) ;;
    *) echo "ERRO: NO_GRAD_CKPT tem de ser 0 ou 1 (recebido: '$NO_GRAD_CKPT')" >&2
       echo "    1 = checkpointing DESLIGADO (default: evita o grad-detach do QLoRA no Qwen3-VL)" >&2
       exit 2 ;;
esac

# Caminhos de dados sao RELATIVOS a raiz do repo: eles viajam no tarball e sao relidos na
# VM como "$REPO_DIR/<caminho>". Um absoluto viraria concatenacao sem sentido.
case "$CORPUS_DIR" in
    /*|[A-Za-z]:[\\/]*|*:*)
        echo "ERRO: CORPUS_DIR tem de ser relativo a raiz do repo ($REPO_DIR): '$CORPUS_DIR'" >&2
        exit 2 ;;
esac

if ! command -v gcloud >/dev/null 2>&1; then
    precondicao_ausente "gcloud nao esta no PATH"
fi

# A VM recebe o CHECKOUT LOCAL (tarball da arvore de trabalho, sem .git): a branch que
# chega la e a que esta aqui. Conferir e barato e evita treinar com o codigo errado --
# `finetune/run_gcp_finetune.sh` e o hook de eval do v4 so existem nesta branch, que nem
# esta pushed. `--branch ''` desliga.
if [ -n "$BRANCH" ]; then
    BRANCH_LOCAL="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<sem git>')"
    if [ "$BRANCH_LOCAL" != "$BRANCH" ]; then
        precondicao_ausente \
            "o checkout local esta em '$BRANCH_LOCAL', mas o finetune v4 espera '$BRANCH'." \
            "Faca checkout da branch (ou passe --branch '$BRANCH_LOCAL' se for proposital)."
    else
        echo "    checkout local: $BRANCH_LOCAL (confere)"
    fi
fi

# A fase remota de treino DELEGA a este script: se ele nao estiver no checkout, o tarball
# chega sem ele e a VM so descobriria isso depois de paga.
if [ ! -f "$REPO_DIR/finetune/run_gcp_finetune.sh" ]; then
    echo "ERRO: finetune/run_gcp_finetune.sh ausente em $REPO_DIR -- a fase de treino o invoca." >&2
    exit 2
fi

gerar_scripts_remotos

# -------------------------------------------------------------------------------------
# A partir daqui existe recurso pago em jogo: o teardown passa a ser obrigatorio.
#
# INT/TERM/HUP viram uma saida normal para que o EXIT dispare de forma confiavel (bash nem
# sempre roda o trap de EXIT quando o shell morre por sinal nao tratado). HUP cobre o caso
# do terminal fechado no meio do treino.
# -------------------------------------------------------------------------------------
trap 'teardown' EXIT
trap 'echo ""; echo "--- interrompido (SIGINT) ---" >&2; exit 130' INT
trap 'echo ""; echo "--- terminado (SIGTERM) ---" >&2; exit 143' TERM
trap 'echo ""; echo "--- terminal caiu (SIGHUP) ---" >&2; exit 129' HUP

gancho_de_falha provision
provisionar_vm
esperar_ssh
enviar_repo

etapa "[4/9] bootstrap da VM (untar do repo; as deps ficam com o run_gcp_finetune.sh)"
gancho_de_falha bootstrap
remoto "bash ~/finetune_remote.sh bootstrap"

etapa "[5/9] stage: baixando o corpus v4 de $GCS_CORPUS para a VM"
echo "--- records.jsonl + bin/<eixo>/ + eval/{queries_v4.txt,qrels_v4.tsv} ---"
gancho_de_falha stage
remoto "bash ~/finetune_remote.sh stage"

etapa "[6/9] pares de treino (finetune.make_pairs, gerador template)"
remoto "bash ~/finetune_remote.sh pairs"

etapa "[7/9] smoke de VRAM: 2 passos reais na L4 (gate de OOM antes do treino inteiro)"
remoto "bash ~/finetune_remote.sh smoke"

etapa "[8/9] finetune QLoRA na VM (o log traz o nDCG@10 do A no v4 ANTES e DEPOIS)"
gancho_de_falha finetune
remoto "bash ~/finetune_remote.sh finetune"

etapa "[9/9] persistindo o adaptador em $GCS_OUT (ANTES do teardown)"
gancho_de_falha upload
subir_resultado_para_gcs

etapa "finetune v4 concluido -- o teardown vem a seguir"
echo "    adaptador: $GCS_OUT/adapter/"
echo "    log do treino (antes/depois): $GCS_OUT/finetune_train.log"
exit 0
