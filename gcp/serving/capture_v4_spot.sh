#!/usr/bin/env bash
# =====================================================================================
# Captura do CORPUS V4 numa L4 SPOT efemera, persistindo em GCS, com teardown GARANTIDO.
#
#   [1/6] preflight local (GCS_DEST, POR_EIXO, branch do checkout) -- nada pago ainda
#   [2/6] cria a VM L4 em SPOT, com fallback de zona            (a partir daqui ha custo)
#   [3/6] espera o SSH e envia repo + scripts
#   [4/6] bootstrap: deps do repo + faster-whisper, ffmpeg e pypdfium2
#   [5/6] captura: `ensemble.build_corpus.build_corpus_v4(por_eixo=$POR_EIXO)`
#   [6/6] sobe `records.jsonl` E os binarios (`bin/<eixo>/`) para $GCS_DEST -- DA VM
#   [trap] teardown: tenta o upload de novo, apaga a VM e ASSERTA que nada sobrou
#
# O QUE SOBREVIVE A RODADA E SO O QUE ESTA EM $GCS_DEST. A VM e descartavel; o corpus v4
# (records + binarios, ~GB com os videos de sessao) fica no bucket e outra maquina --
# o fine-tune denso da Semana 10, o rebuild do indice servido -- o puxa de la.
#
# -------------------------------------------------------------------------------------
# POR QUE UM SCRIPT PROPRIO (e nao mais um modo do gcp/serving/trigger_cycle.sh)
#
# O trigger_cycle (Task B7) delega a rodada paga a `run_slice.sh --arch ensemble --phase
# capture`, e aquela fase roda `python3 -m ensemble.build_corpus`, que e a captura V3
# (20/eixo, PLOA unico, sem `check_corpus_paths`). Fazer o ciclo rodar o v4 exigiria mexer
# na fase `capture` do script remoto do run_slice -- justamente o caminho que a rodada v3
# ja exercitou e que nao pode regredir. Alem disso os dois contratos de saida sao
# diferentes: o ciclo faz UPSERT dos records num indice acumulado e guarda os binarios
# como um tar por ciclo; aqui o destino e um espelho FIEL da arvore capturada
# (`records.jsonl` + `bin/<eixo>/` um objeto por arquivo), porque `meta["path"]` dos
# records aponta para esses arquivos e quem baixar o prefixo tem de reconstruir a arvore
# tal e qual.
#
# O que NAO foi reinventado e a parte que nao pode falhar: o `trap EXIT` que ignora
# sinais, o retry de cada delete, a varredura de TODAS as zonas tentadas e a asserção
# final anti-leak vieram de `gcp/slice/run_slice.sh` (secao "TEARDOWN -- a razao de ser
# deste script"), assim como o fallback de zona do create, o `remoto()` e a forma do
# bootstrap. Credito la; aqui so o que muda: um unico recurso (nao ha Cloud SQL), captura
# v4 e upload para GCS feito DE DENTRO da VM.
#
# -------------------------------------------------------------------------------------
# RUN-BOOK
#
#   # ensaio completo, sem tocar a nuvem (nenhum gcloud e executado):
#   bash gcp/serving/capture_v4_spot.sh --gcs-dest gs://residencia-511517-bronze/ensemble/v4/ --dry-run
#
#   # rodada de verdade (L4 Spot paga por ~1-2h; guarde a saida):
#   bash gcp/serving/capture_v4_spot.sh --gcs-dest gs://residencia-511517-bronze/ensemble/v4/ \
#        2>&1 | tee logs/captura-v4-$(date +%F).log
#
#   # conferir que NADA sobrou de pe (a resposta certa e "Listed 0 items."):
#   gcloud compute instances list --project=residencia-511517
#
#   # o que a rodada deixou no bucket:
#   gcloud storage ls -r gs://residencia-511517-bronze/ensemble/v4/ --project=residencia-511517
#
#   # trazer o corpus para outra maquina (reconstroi a arvore que os records esperam):
#   gcloud storage rsync -r gs://residencia-511517-bronze/ensemble/v4/bin \
#        data/govexe/ensemble_orcamento_v4/bin --project=residencia-511517
#   gcloud storage cp gs://residencia-511517-bronze/ensemble/v4/records.jsonl \
#        data/govexe/ensemble_orcamento_v4/records.jsonl --project=residencia-511517
#
#   # provar o teardown sem esperar a captura (falha injetada):
#   CAPTURE_FAIL_AT=capture bash gcp/serving/capture_v4_spot.sh --gcs-dest gs://... --dry-run
#
# PROJETO: toda chamada gcloud leva `--project=$PROJECT` explicito. Este script NUNCA roda
# `gcloud config set project`: o default do operador e dele.
# =====================================================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

PROJECT="${PROJECT:-residencia-511517}"

# Destino no GCS -- PARAMETRO OBRIGATORIO (ver o preflight). Sem ele a rodada so gastaria
# GPU para jogar o corpus fora junto com a VM.
GCS_DEST="${GCS_DEST:-}"

POR_EIXO="${POR_EIXO:-50}"              # itens por eixo do v4 (POR_EIXO_V4 do build_corpus)
BRANCH="${BRANCH:-feat/semana10-finetune-denso}"   # branch esperada no checkout local

VM="${VM:-govexe-capture-v4}"           # nome PROPRIO: nao colide com govexe-embed (fatia
                                        # manual) nem com govexe-ciclo (trigger_cycle)
# Mesma ordem do run_slice: a L4 deu STOCKOUT em todas as zonas us-* na rodada do objetivo
# #2 e so northamerica-northeast2-a tinha capacidade. O pool SPOT e outro, mas a ordem
# continua valendo como aposta.
ZONES="${ZONES:-northamerica-northeast2-a us-central1-a us-central1-b us-east1-c us-east4-a us-west1-a}"

MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"
ACCELERATOR="${ACCELERATOR:-nvidia-l4}"
# 200GB nao e folga: a captura retem os MP4 das sessoes do eixo video, e o disco tambem
# guarda os pesos do Whisper large-v3 e do cache do HF.
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
# SPOT e o default DESTE script (no run_slice o default e STANDARD): a captura e
# interrompivel -- se a VM for preemptada, o que ja subiu para o GCS fica e a rodada
# recomeca mais barata.
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"

# Arvore do corpus v4 no repo (relativa a raiz, como todo caminho de dados daqui). O
# default espelha `ensemble.build_corpus.DESTINO_V4`; quem o resolve de verdade e a VM,
# que le o caminho impresso pelo proprio `build_corpus_v4` (ver a fase `capture` remota).
CORPUS_DIR="${CORPUS_DIR:-data/govexe/ensemble_orcamento_v4}"
CORPUS_REL="$CORPUS_DIR/records.jsonl"
BIN_REL="$CORPUS_DIR/bin"

DRY_RUN=0

# -------------------------------------------------------------------------------------
# Estado do teardown (o que ja foi tocado na nuvem)
# -------------------------------------------------------------------------------------
ZONE=""                 # zona onde a VM subiu de fato
ZONAS_TENTADAS=""       # toda zona em que um `create` foi tentado (o teardown varre todas)
UPLOAD_FEITO=0          # 1 depois que o corpus chegou em $GCS_DEST
TMP_DIR=""

TEARDOWN_RETRIES="${TEARDOWN_RETRIES:-3}"
TEARDOWN_ESPERA="${TEARDOWN_ESPERA:-20}"      # segundos entre tentativas de delete
UPLOAD_TIMEOUT="${UPLOAD_TIMEOUT:-1800}"      # teto do upload (os videos sao grandes)
LEAK_EXIT="${LEAK_EXIT:-70}"                  # status quando sobra recurso de pe

uso() {
    sed -n '2,64p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --gcs-dest)       GCS_DEST="$2"; shift 2 ;;
        --project)        PROJECT="$2"; shift 2 ;;
        --por-eixo)       POR_EIXO="$2"; shift 2 ;;
        --zones)          ZONES="$2"; shift 2 ;;
        --vm)             VM="$2"; shift 2 ;;
        --branch)         BRANCH="$2"; shift 2 ;;
        --boot-disk-size) BOOT_DISK_SIZE="$2"; shift 2 ;;
        --corpus-dir)     CORPUS_DIR="$2"; CORPUS_REL="$2/records.jsonl"; BIN_REL="$2/bin"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        uso 0 ;;
        *) echo "flag desconhecida: $1" >&2; uso 2 ;;
    esac
done

# -------------------------------------------------------------------------------------
# Primitivas de execucao (de gcp/slice/run_slice.sh)
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

# Gancho de verificacao do trap: `CAPTURE_FAIL_AT=<etapa>` faz a etapa falhar de proposito
# (etapas: provision, bootstrap, capture, upload). Serve para provar que o teardown roda em
# qualquer saida -- e so isso.
gancho_de_falha() {
    if [ "${CAPTURE_FAIL_AT:-}" = "$1" ]; then
        echo "!!! falha INJETADA na etapa '$1' (CAPTURE_FAIL_AT) !!!" >&2
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
# De la o corpus vai direto para o bucket pela rede da GCP: nao passa pelo link do
# operador (os binarios do eixo video sao GB) e nao depende de a maquina que disparou a
# rodada continuar viva. A VM sobe com `--scopes=cloud-platform`, entao o gcloud dela ja
# tem credencial para escrever no bucket do projeto.
#
# Chamada DUAS vezes de proposito, como o run_slice faz com o CSV: no fluxo normal (antes
# do teardown) e de novo de dentro do teardown, se a primeira nao tiver acontecido.
# -------------------------------------------------------------------------------------
subir_para_gcs() {
    echo "--- subindo o corpus v4 da VM para $GCS_DEST (teto: ${UPLOAD_TIMEOUT}s) ---"
    echo "    [na VM] gcloud storage cp <records.jsonl> $GCS_DEST/records.jsonl --project=$PROJECT"
    echo "    [na VM] gcloud storage rsync -r <bin/> $GCS_DEST/bin --project=$PROJECT"
    if remoto_com_timeout "$UPLOAD_TIMEOUT" "bash ~/capture_remote.sh upload"; then
        UPLOAD_FEITO=1
        return 0
    fi
    return 1
}

# -------------------------------------------------------------------------------------
# TEARDOWN -- a razao de ser deste script (portado de gcp/slice/run_slice.sh)
#
# Registrado em `trap ... EXIT` logo antes do primeiro recurso ser criado, entao roda no
# sucesso, no erro (`set -e`), na interrupcao e na queda do terminal.
#
#   1. `trap '' INT TERM HUP`: o teardown IGNORA sinais. Um segundo Ctrl-C no meio dele
#      nao pode matar o shell deixando a L4 de pe, cobrando.
#   2. `set +eu +o pipefail`: nem erro, nem variavel nao-setada, nem pipe quebrado podem
#      abortar a limpeza antes dos deletes.
#   3. O resgate do corpus vem antes dos deletes (apagar a VM destroi o corpus), mas com
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

    # Ultima chance de salvar o corpus: se a rodada morreu depois de capturar mas antes de
    # subir, ele ainda esta na VM que estamos prestes a apagar.
    if [ "${UPLOAD_FEITO:-0}" = "0" ] && [ -n "${ZONE:-}" ]; then
        echo "--- tentativa final de subir o corpus para $GCS_DEST ---"
        subir_para_gcs || echo "--- o resgate do corpus falhou ou estourou o tempo; seguindo para os deletes ---" >&2
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
        echo "--- o corpus v4 esta em $GCS_DEST (e so isso sobreviveu a rodada) ---"
    else
        echo "--- ATENCAO: o corpus NAO chegou a $GCS_DEST; a rodada nao deixou nada ---" >&2
    fi

    echo "=== teardown concluido (status final: $rc) ==="
    exit "$rc"
}

# -------------------------------------------------------------------------------------
# Script remoto (roda DENTRO da VM)
#
# Gerado aqui e enviado por scp, como no run_slice: a configuracao viaja num
# `capture_env.sh` separado para o corpo do script nao virar um emaranhado de aspas.
# -------------------------------------------------------------------------------------
gerar_scripts_remotos() {
    TMP_DIR="$(mktemp -d)"

    cat > "$TMP_DIR/capture_env.sh" <<ENV
# Gerado por gcp/serving/capture_v4_spot.sh -- configuracao da captura v4.
PROJECT='$PROJECT'
GCS_DEST='$GCS_DEST'
POR_EIXO='$POR_EIXO'
CORPUS_REL='$CORPUS_REL'
BIN_REL='$BIN_REL'
BRANCH='$BRANCH'
ENV

    cat > "$TMP_DIR/capture_remote.sh" <<'REMOTE'
#!/usr/bin/env bash
# Gerado por gcp/serving/capture_v4_spot.sh -- roda DENTRO da VM L4. Fase em $1.
set -euo pipefail
FASE="${1:?uso: capture_remote.sh <bootstrap|capture|upload>}"
# shellcheck disable=SC1091
source "$HOME/capture_env.sh"
echo "=== [vm $(hostname)] fase=$FASE $(date) ==="

REPO_DIR="$HOME/residencia"
# Onde a fase `capture` anota o caminho que o proprio `build_corpus_v4` devolveu -- e esse
# o caminho que a fase `upload` sobe. Assim nao ha como o destino do script e o do Python
# divergirem em silencio (o default do modulo e `DESTINO_V4`).
DESTINO_ANOTADO="$HOME/corpus_v4_destino.txt"

ambiente() {
    cd "$REPO_DIR"
    if [ -f "$REPO_DIR/.venv-gcp/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$REPO_DIR/.venv-gcp/bin/activate"
    fi
    export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
    export HF_HOME="$HOME/.cache/huggingface"
    export HF_XET_HIGH_PERFORMANCE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

# Caminho do corpus capturado: o que a fase `capture` anotou, ou o esperado pelo driver.
destino_do_corpus() {
    if [ -s "$DESTINO_ANOTADO" ]; then
        tail -n 1 "$DESTINO_ANOTADO"
    else
        echo "$CORPUS_REL"
    fi
}

case "$FASE" in

# =====================================================================================
bootstrap)
# =====================================================================================
    # Guard por SENTINELA DE CODIGO (e nao por `[ -d "$REPO_DIR" ]`), como no run_slice:
    # outra etapa pode PRE-CRIAR `~/residencia` e o teste de diretorio daria falso.
    if [ ! -f "$REPO_DIR/requirements-slurm.txt" ]; then
        mkdir -p "$REPO_DIR"
        tar xzf "$HOME/repo.tgz" -C "$REPO_DIR"
    fi
    cd "$REPO_DIR"
    export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
    export HF_HOME="$HOME/.cache/huggingface"
    export HF_XET_HIGH_PERFORMANCE=1
    mkdir -p "$HF_HOME"

    # ffmpeg: o eixo VIDEO extrai keyframes com ele (`ensemble/capture/video.py`:
    # `_ffmpeg_keyframe`) e o decode de audio do faster-whisper tambem o quer. A imagem
    # DLVM nao traz.
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "--- instalando ffmpeg ---"
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    fi

    # python3.10-venv: a imagem DLVM tambem nao traz, e sem ele o `python3 -m venv` morre
    # em "ensurepip is not available". Passo PROPRIO para nao ser pulado quando o ffmpeg
    # ja existe.
    if ! dpkg -s python3.10-venv >/dev/null 2>&1; then
        echo "--- instalando python3.10-venv ---"
        sudo apt-get update -qq && sudo apt-get install -y -qq python3.10-venv \
            || echo "--- python3.10-venv falhou; o venv pode cair no fallback ---"
    fi

    # venv --system-site-packages: ENXERGA o torch/CUDA da imagem (reinstalar torch por
    # pip quebraria o casamento com o driver) e isola so o resto.
    VENV_DIR="$REPO_DIR/.venv-gcp"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv --system-site-packages "$VENV_DIR" \
            || echo "--- venv falhou; seguindo no python do sistema ---"
    fi
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        DEPS_MARKER_DIR="$VENV_DIR"
    else
        DEPS_MARKER_DIR="$HOME/.local"
    fi
    echo "--- python: $(command -v python3) ($(python3 --version 2>&1)) ---"

    REQ_HASH="$(sha256sum requirements-slurm.txt | cut -d' ' -f1)"
    DEPS_MARKER="$DEPS_MARKER_DIR/.residencia_deps_${REQ_HASH}_capturav4"
    if [ -f "$DEPS_MARKER" ]; then
        echo "--- deps ja instaladas ($DEPS_MARKER) ---"
    else
        mkdir -p "$DEPS_MARKER_DIR"
        echo "--- instalando deps (requirements-slurm.txt) ---"
        python3 -m pip install --quiet -r requirements-slurm.txt
        # faster-whisper (ASR do eixo audio/video) e pypdfium2 (render das paginas do eixo
        # documento) JA ESTAO no requirements-slurm.txt; a linha abaixo e um piso
        # explicito -- se alguem os tirar de la, a captura v4 continua de pe. Nenhum dos
        # dois arrasta torch (ver o cabecalho do requirements), entao o pin e seguro.
        echo "--- deps da captura: faster-whisper, pypdfium2, yt-dlp, gcs ---"
        python3 -m pip install --quiet "faster-whisper>=1.0" "pypdfium2>=4.30" \
            yt-dlp google-cloud-storage
        touch "$DEPS_MARKER"
    fi

    # Guard "preferir o torch da imagem": qualquer dep que declare torch instala um por
    # cima, SEM as libs CUDA do no, e ele sombra o torch bom (run_slice, mesma secao).
    if [ -n "${VIRTUAL_ENV:-}" ]; then TORCH_ROOT="$VIRTUAL_ENV"; else TORCH_ROOT="$HOME/.local"; fi
    LOCAL_TORCH=""
    for _f in "$TORCH_ROOT"/lib/python*/site-packages/torch/__init__.py; do
        [ -f "$_f" ] && LOCAL_TORCH="$_f"
    done
    if [ -n "$LOCAL_TORCH" ]; then
        echo "--- torch em $LOCAL_TORCH: removendo para o torch da imagem prevalecer ---"
        for _sp in "$TORCH_ROOT"/lib/python*/site-packages; do
            [ -d "$_sp" ] || continue
            rm -rf "$_sp"/torch "$_sp"/torch-* "$_sp"/torchvision* \
                   "$_sp"/torchaudio* "$_sp"/nvidia* "$_sp"/functorch* || true
        done
    else
        echo "--- sem torch no venv: o torch da imagem prevalece ---"
    fi

    echo "=== validacao de imports + GPU (antes da GPU ficar paga a toa) ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    # A fiacao INTEIRA da captura v4 e conferida aqui: um `faster_whisper` ausente ou um
    # `check_corpus_paths` que nao importa descobertos no meio da captura custariam a
    # rodada paga inteira.
    python3 - <<'PY'
import bs4                                              # noqa: F401  (html do eixo imagem)
import pypdfium2                                        # (render das paginas do documento)
from PIL import Image                                   # noqa: F401  (imagem)
from faster_whisper import WhisperModel                 # noqa: F401  (audio/video, ASR)
print("pypdfium2", pypdfium2.V_PYPDFIUM2, "| bs4", bs4.__version__)
from ensemble.build_corpus import DESTINO_V4, DOCUMENTO_V4, POR_EIXO_V4, build_corpus_v4
from ensemble.corpus_v4 import check_corpus_paths       # noqa: F401  (sanity de caminho)
print("destino v4 (default do modulo):", DESTINO_V4)
print("eixo documento do v4:", len(DOCUMENTO_V4), "PDFs legislativos")
print("por_eixo default:", POR_EIXO_V4, "| build_corpus_v4 ok")
PY
    command -v ffmpeg >/dev/null 2>&1 \
        || { echo "ERRO: ffmpeg ausente -- o eixo video nao extrai keyframe" >&2; exit 1; }
    ffmpeg -version | head -n 1
    echo "=== bootstrap concluido $(date) ==="
    ;;

# =====================================================================================
capture)
# =====================================================================================
    # Captura do corpus v4 (5 eixos, PRECISA DE INTERNET -- a VM tem saida). O destino e
    # RELATIVO ao repo na VM de proposito: `build_corpus` grava `meta["path"]` com o
    # caminho que recebeu, e um caminho relativo continua valido depois do round-trip pelo
    # GCS (um absoluto `/home/<user>/...` nao continuaria).
    #
    # `check_corpus_paths` roda DENTRO do `build_corpus_v4`, antes de escrever o JSONL:
    # item image-route sem arquivo em disco reprova o corpus inteiro aqui, e nao no meio
    # do `embed_images` do baseline A.
    ambiente
    echo "=== [capture] corpus v4 (por_eixo=$POR_EIXO) -> $CORPUS_REL ==="
    python3 -c "from ensemble.build_corpus import build_corpus_v4; print(build_corpus_v4(por_eixo=$POR_EIXO))" \
        | tee "$DESTINO_ANOTADO"
    DESTINO="$(destino_do_corpus)"
    BIN_DIR="$(dirname "$DESTINO")/bin"
    echo "--- corpus: $DESTINO | binarios: $BIN_DIR ---"
    wc -l "$DESTINO"
    du -sh "$BIN_DIR" || true
    for eixo in documento imagem audio video; do
        if [ -d "$BIN_DIR/$eixo" ]; then
            echo "    $eixo: $(find "$BIN_DIR/$eixo" -type f | wc -l) arquivos"
        fi
    done
    echo "=== captura concluida $(date) ==="
    ;;

# =====================================================================================
upload)
# =====================================================================================
    # O que sobrevive a rodada. `cp` para o records (um arquivo) e `rsync -r` para os
    # binarios (arvore de muitos arquivos, alguns grandes): o rsync e retomavel, o que
    # importa quando a primeira tentativa morreu por preempcao no meio.
    cd "$REPO_DIR"
    [ -n "$GCS_DEST" ] || { echo "ERRO: GCS_DEST vazio na VM" >&2; exit 2; }
    DESTINO="$(destino_do_corpus)"
    BIN_DIR="$(dirname "$DESTINO")/bin"
    if [ ! -f "$DESTINO" ]; then
        echo "ERRO: $DESTINO nao existe -- a captura nao terminou; nada a subir" >&2
        exit 1
    fi
    echo "=== [upload] $DESTINO + $BIN_DIR -> $GCS_DEST ==="
    gcloud storage cp "$DESTINO" "$GCS_DEST/records.jsonl" --project="$PROJECT"
    if [ -d "$BIN_DIR" ]; then
        gcloud storage rsync -r "$BIN_DIR" "$GCS_DEST/bin" --project="$PROJECT"
    else
        echo "ERRO: $BIN_DIR nao existe -- os records apontariam para o vazio" >&2
        exit 1
    fi
    echo "--- conferencia do que ficou no bucket ---"
    gcloud storage ls -l "$GCS_DEST/records.jsonl" --project="$PROJECT"
    gcloud storage du -s "$GCS_DEST/bin" --project="$PROJECT" || true
    echo "=== upload concluido $(date) ==="
    ;;

*)
    echo "ERRO: fase desconhecida: $FASE" >&2
    exit 2
    ;;
esac
REMOTE
    chmod +x "$TMP_DIR/capture_remote.sh"
}

# -------------------------------------------------------------------------------------
# Provisionamento (fallback de zona e flags de SPOT: de gcp/slice/run_slice.sh)
# -------------------------------------------------------------------------------------
provisionar_vm() {
    etapa "[2/6] VM L4 ($MACHINE_TYPE + $ACCELERATOR, disco $BOOT_DISK_SIZE) com fallback de zona"
    echo "--- zonas na ordem: $ZONES ---"
    # `--instance-termination-action=DELETE` e o que importa para o teardown: numa
    # preempcao a VM se APAGA em vez de ficar STOPPED cobrando disco -- e o trap de saida
    # ainda tenta o delete depois, que vira no-op.
    local spot_flags=()
    if [ "$PROVISIONING_MODEL" = "SPOT" ]; then
        echo "--- provisionamento SPOT (sujeito a preempcao; o upload parcial fica no GCS) ---"
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
    etapa "[3a/6] esperando o SSH da VM responder"
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
    etapa "[3b/6] enviando repo e scripts para a VM"
    # Arvore de trabalho (e nao `git archive`): o que vale e o checkout local, ja conferido
    # contra $BRANCH no preflight. O corpus v4 fica FORA do tarball -- ele nasce na VM, e
    # mandar um corpus velho so sobrescreveria o novo (alem do tamanho).
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
    run gcloud compute scp "$TMP_DIR/capture_remote.sh" "$VM:capture_remote.sh" \
        --zone="$ZONE" --project="$PROJECT" --quiet
    run gcloud compute scp "$TMP_DIR/capture_env.sh" "$VM:capture_env.sh" \
        --zone="$ZONE" --project="$PROJECT" --quiet
}

# -------------------------------------------------------------------------------------
# [1/6] Preflight (local, antes de criar qualquer recurso pago)
# -------------------------------------------------------------------------------------
etapa "[1/6] captura do corpus v4 | projeto=$PROJECT | vm=$VM ($PROVISIONING_MODEL) | dry-run=$DRY_RUN"
echo "    destino no GCS: ${GCS_DEST:-<VAZIO>}"
echo "    por_eixo=$POR_EIXO  disco=$BOOT_DISK_SIZE  maquina=$MACHINE_TYPE+$ACCELERATOR"
echo "    corpus na VM:   $CORPUS_REL (binarios: $BIN_REL)"
echo "    branch esperada no checkout: ${BRANCH:-<checagem desligada>}"

# GCS_DEST e OBRIGATORIO: sem destino a rodada pagaria GPU para jogar o corpus fora junto
# com a VM. Este erro e fatal mesmo em --dry-run -- ele nao depende da nuvem, e um ensaio
# sem destino ensaiaria a rodada errada.
if [ -z "$GCS_DEST" ]; then
    echo "ERRO: GCS_DEST e obrigatorio -- sem ele o corpus morre junto com a VM." >&2
    echo "    use: --gcs-dest gs://<bucket>/<prefixo>/   (ou GCS_DEST=gs://...)" >&2
    exit 2
fi
case "$GCS_DEST" in
    gs://*) ;;
    *) echo "ERRO: GCS_DEST tem de ser um URI gs:// (recebido: '$GCS_DEST')" >&2; exit 2 ;;
esac
GCS_DEST="${GCS_DEST%/}"          # normaliza: os filhos sao montados com "$GCS_DEST/..."

# POR_EIXO vai INTERPOLADO no `python3 -c` que roda na VM: so digitos passam daqui.
case "$POR_EIXO" in
    ''|*[!0-9]*) echo "ERRO: POR_EIXO tem de ser inteiro positivo (recebido: '$POR_EIXO')" >&2; exit 2 ;;
esac
[ "$POR_EIXO" -gt 0 ] || { echo "ERRO: POR_EIXO tem de ser > 0" >&2; exit 2; }

case "$PROVISIONING_MODEL" in
    STANDARD|SPOT) ;;
    *) echo "ERRO: PROVISIONING_MODEL invalido: '$PROVISIONING_MODEL' (use STANDARD ou SPOT)" >&2
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
# chega la e a que esta aqui. Conferir e barato e evita capturar com o codigo errado --
# o eixo documento do v4 (DOCUMENTO_V4) so existe nesta branch. `--branch ''` desliga.
if [ -n "$BRANCH" ]; then
    BRANCH_LOCAL="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<sem git>')"
    if [ "$BRANCH_LOCAL" != "$BRANCH" ]; then
        precondicao_ausente \
            "o checkout local esta em '$BRANCH_LOCAL', mas a captura v4 espera '$BRANCH'." \
            "Faca checkout da branch (ou passe --branch '$BRANCH_LOCAL' se for proposital)."
    else
        echo "    checkout local: $BRANCH_LOCAL (confere)"
    fi
fi

gerar_scripts_remotos

# -------------------------------------------------------------------------------------
# A partir daqui existe recurso pago em jogo: o teardown passa a ser obrigatorio.
#
# INT/TERM/HUP viram uma saida normal para que o EXIT dispare de forma confiavel (bash nem
# sempre roda o trap de EXIT quando o shell morre por sinal nao tratado). HUP cobre o caso
# do terminal fechado no meio de uma captura longa.
# -------------------------------------------------------------------------------------
trap 'teardown' EXIT
trap 'echo ""; echo "--- interrompido (SIGINT) ---" >&2; exit 130' INT
trap 'echo ""; echo "--- terminado (SIGTERM) ---" >&2; exit 143' TERM
trap 'echo ""; echo "--- terminal caiu (SIGHUP) ---" >&2; exit 129' HUP

gancho_de_falha provision
provisionar_vm
esperar_ssh
enviar_repo

etapa "[4/6] bootstrap da VM (deps do repo + faster-whisper, ffmpeg, pypdfium2)"
gancho_de_falha bootstrap
remoto "bash ~/capture_remote.sh bootstrap"

etapa "[5/6] captura do corpus v4 na VM (rede: RSS, PDFs legislativos, podcasts, sessoes)"
echo "--- por_eixo=$POR_EIXO; o audio/video transcreve com Whisper large-v3 (faster-whisper) ---"
gancho_de_falha capture
remoto "bash ~/capture_remote.sh capture"

etapa "[6/6] persistindo o corpus v4 em $GCS_DEST (ANTES do teardown)"
gancho_de_falha upload
subir_para_gcs

etapa "captura v4 concluida -- o teardown vem a seguir"
echo "    records: $GCS_DEST/records.jsonl"
echo "    binarios: $GCS_DEST/bin/<eixo>/"
exit 0
