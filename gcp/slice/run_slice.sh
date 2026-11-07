#!/usr/bin/env bash
# =====================================================================================
# Orquestracao de UMA rodada paga da fatia GovExe na GCP, com TEARDOWN GARANTIDO.
#
#   provisiona (Cloud SQL + VM L4) -> bootstrap -> indexa os espacos da ARQUITETURA
#   -> avalia (gcp/slice/eval.py) -> traz o CSV -> TEARDOWN (sempre)
#
# Consolida num fluxo unico e versionado os tres scripts descartaveis que funcionaram na
# rodada manual do objetivo #2 (bootstrap da VM, ingestao das paginas, fase de audio). A
# fraqueza daquela rodada era o teardown MANUAL: aqui ele esta num `trap ... EXIT`, entao
# roda no sucesso, no erro, na interrupcao (Ctrl-C) e na queda do terminal (SIGHUP) -- e,
# uma vez comecado, IGNORA sinais, faz retry de cada delete e ASSERTA no fim que nada
# sobrou (se sobrou, grita LEAK e sai com status 70).
#
# -------------------------------------------------------------------------------------
# ATENCAO -- ESTE SCRIPT E DONO DO CICLO DE VIDA DOS RECURSOS
#
# Ao sair, ele APAGA a VM ($VM) e a instancia Cloud SQL ($INSTANCE) -- mesmo que ja
# existissem antes dele. O indice pgvector vive na instancia e morre com ela; o que tem de
# sobreviver a rodada e o CSV de metricas, trazido para `results/` ANTES do teardown (e,
# por garantia, tentado de novo DENTRO do teardown). NAO aponte este script para uma
# instancia Cloud SQL que voce queira manter.
#
# O bucket Bronze e o secret da senha NAO sao tocados: sao insumos, nao recursos da rodada.
# -------------------------------------------------------------------------------------
# USO
#
#   bash gcp/slice/run_slice.sh --arch A
#   bash gcp/slice/run_slice.sh --arch A+audio \
#        --audio-records data/govexe/audio_eval/records.jsonl \
#        --audio-dir     data/govexe/audio_eval/audio
#   bash gcp/slice/run_slice.sh --arch A --dry-run      # imprime tudo, nao toca a nuvem
#
#   Numa rodada paga, guarde a saida:
#       bash gcp/slice/run_slice.sh --arch A+audio 2>&1 | tee logs/run_slice.log
#
# VERIFICACAO (sem gastar nuvem)
#
#   bash -n gcp/slice/run_slice.sh                              # sintaxe
#   SLICE_FAIL_AT=index bash gcp/slice/run_slice.sh --dry-run --arch A+audio
#       -> a etapa de indexacao falha DE PROPOSITO e a saida mostra o teardown rodando
#          assim mesmo. `SLICE_FAIL_AT` existe so para provar o trap (etapas: provision,
#          bootstrap, index, eval); nao tem uso em producao.
#
#   Em `--dry-run`, precondicao de AMBIENTE que falta (gcloud fora do PATH, mini-set de
#   audio ainda nao capturado) vira AVISO e a rodada segue: a verificacao do script nao
#   pode depender de ter nuvem autenticada e dado local na maquina de quem revisa.
#
# A arquitetura (`--arch`) define os ESPACOS a indexar -- o registro e
# `gcp/slice/spaces.py`, e a VM confere a lista deste script contra ele antes de indexar:
#
#   A         -> qwen3vl              (pagina, foto, texto e audio-via-ASR num indice so)
#   A+audio   -> qwen3vl + clap       (o segundo indexa a forma de onda NATIVA)
# =====================================================================================

set -euo pipefail

# -------------------------------------------------------------------------------------
# Parametros (sobrescreviveis por variavel de ambiente ou por flag)
#
# Os defaults sao os valores da rodada do objetivo #2, que funcionou de ponta a ponta.
# -------------------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

PROJECT_ID="${PROJECT_ID:-residencia-511517}"
REGION="${REGION:-us-central1}"             # regiao da instancia Cloud SQL
BUCKET="${BUCKET:-gs://residencia-511517-bronze}"
INSTANCE="${INSTANCE:-govexe-pg}"
DB="${DB:-govexe}"
DB_USER="${DB_USER:-postgres}"
SECRET="${SECRET:-govexe-pg-password}"
VM="${VM:-govexe-embed}"

# Zonas na ordem de tentativa. A L4 (g2-standard-4) deu STOCKOUT em TODAS as zonas us-*
# na rodada do objetivo #2 -- so northamerica-northeast2-a tinha capacidade, por isso ela
# vem primeiro. Zona sem familia g2 simplesmente falha no create e o laco segue para a
# proxima; editar a lista e esperado.
ZONES="${ZONES:-northamerica-northeast2-a us-central1-a us-central1-b us-east1-c us-east4-a us-west1-a}"

MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"
ACCELERATOR="${ACCELERATOR:-nvidia-l4}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"
# Imagem Deep Learning usada na rodada que funcionou (torch 2.9 + CUDA 12.9 + driver 580).
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

# Insumos da indexacao.
SOURCE="${SOURCE:-tse_planos}"              # particao `fonte=` do Bronze
DT="${DT:-$(date +%F)}"                     # particao `dt=` do Bronze
MIXED_RECORDS="${MIXED_RECORDS:-data/govexe/mixed_records.bronze.jsonl}"
AUDIO_RECORDS="${AUDIO_RECORDS:-data/govexe/audio_eval/records.jsonl}"
AUDIO_DIR="${AUDIO_DIR:-data/govexe/audio_eval/audio}"
BATCH_SIZE="${BATCH_SIZE:-2}"

# Conjuntos de avaliacao (um `eval.py` por conjunto; o CSV acumula as linhas).
EVAL_DIRS="${EVAL_DIRS:-data/govexe/ptbr_planos}"
EVAL_AUDIO_DIR="${EVAL_AUDIO_DIR:-data/govexe/audio_eval}"
OUT_CSV="${OUT_CSV:-results/comparacao.csv}"

ARCH="${ARCH:-A}"
DRY_RUN=0

# -------------------------------------------------------------------------------------
# Estado do teardown (o que ja foi tocado na nuvem)
# -------------------------------------------------------------------------------------
ZONE=""                 # zona onde a VM subiu de fato
ZONAS_TENTADAS=""       # toda zona em que um `create` foi tentado (o teardown varre todas)
SQL_TOCADA=0            # 1 depois que o provisionamento do Cloud SQL comecou
RESULTADOS_COLETADOS=0

# -------------------------------------------------------------------------------------
# Flags
# -------------------------------------------------------------------------------------
uso() {
    sed -n '2,54p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --arch)           ARCH="$2"; shift 2 ;;
        --zones)          ZONES="$2"; shift 2 ;;
        --project)        PROJECT_ID="$2"; shift 2 ;;
        --bucket)         BUCKET="$2"; shift 2 ;;
        --dt)             DT="$2"; shift 2 ;;
        --audio-records)  AUDIO_RECORDS="$2"; shift 2 ;;
        --audio-dir)      AUDIO_DIR="$2"; shift 2 ;;
        --eval-dirs)      EVAL_DIRS="$2"; shift 2 ;;
        --out)            OUT_CSV="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        uso 0 ;;
        *) echo "flag desconhecida: $1" >&2; uso 2 ;;
    esac
done

# A lista de espacos de cada arquitetura esta aqui em bash E em `gcp/slice/spaces.py`. A
# fonte de verdade e o registro Python; para a copia nao divergir em silencio, a VM
# confere as duas listas antes de indexar (fase `index`, no script remoto).
case "$ARCH" in
    A)        SPACES="qwen3vl" ;;
    A+audio)  SPACES="qwen3vl clap" ;;
    *) echo "ERRO: arquitetura desconhecida: '$ARCH' (use A ou A+audio)" >&2; exit 2 ;;
esac

# -------------------------------------------------------------------------------------
# Primitivas de execucao
#
# Tudo que toca a nuvem passa por `run`/`existe`/`run_secreto`: e isso que faz o
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

# Como `run`, mas o log mostra `$1` no lugar do comando -- para a linha que carrega a
# senha do Cloud SQL, que nao pode aparecer em log nenhum.
run_secreto() {
    local rotulo="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] %s\n' "$rotulo"
        return 0
    fi
    printf '    %s\n' "$rotulo"
    "$@"
}

# Teste de existencia de recurso. Em `--dry-run` devolve 1 (NAO existe) sem consultar a
# nuvem: no provisionamento isso imprime o caminho de criacao (o que interessa ver) e na
# conferencia do teardown significa "nada a vazar", que e a verdade num dry-run.
existe() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] %s   # (dry-run nao consulta a nuvem: segue como inexistente)\n' "$*"
        return 1
    fi
    "$@" >/dev/null 2>&1
}

# `run` com teto de tempo, para o que nao pode travar o fluxo (o scp de resgate do CSV
# numa VM meio-bootada). Onde nao houver `timeout` (coreutils), segue sem o teto -- melhor
# sem limite que sem resgate.
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
# nuvem autenticada e mini-set de audio na maquina de quem revisa.
precondicao_ausente() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "--- aviso (dry-run segue assim mesmo): $* ---" >&2
        return 0
    fi
    echo "ERRO: $*" >&2
    exit 2
}

# Gancho de verificacao do trap: `SLICE_FAIL_AT=<etapa>` faz a etapa falhar de proposito.
# Serve para provar que o teardown roda em qualquer saida -- e so isso.
gancho_de_falha() {
    if [ "${SLICE_FAIL_AT:-}" = "$1" ]; then
        echo "!!! falha INJETADA na etapa '$1' (SLICE_FAIL_AT) !!!" >&2
        return 1
    fi
    return 0
}

senha_do_secret() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] gcloud secrets versions access latest --secret=$SECRET" >&2
        printf '<senha-do-secret-manager>'
        return 0
    fi
    gcloud secrets versions access latest --secret="$SECRET" --project="$PROJECT_ID"
}

remoto() {   # executa um comando DENTRO da VM
    run gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT_ID" --quiet \
        --command "$1"
}

# -------------------------------------------------------------------------------------
# TEARDOWN -- a razao de ser deste script
#
# Registrado em `trap ... EXIT` logo antes do primeiro recurso ser criado, entao roda no
# sucesso, no erro (`set -e`), na interrupcao e na queda do terminal (os traps de
# INT/TERM/HUP convertem o sinal numa saida normal, que e o que dispara o EXIT de forma
# confiavel).
#
# Regras de robustez -- a limpeza nao pode ser derrotada por nada:
#
#   1. `trap '' INT TERM HUP`: o teardown IGNORA sinais. Um segundo Ctrl-C no meio dele
#      nao pode matar o shell deixando L4 e Cloud SQL de pe, cobrando.
#   2. `set +eu +o pipefail`: nem erro, nem variavel nao-setada, nem pipe quebrado podem
#      abortar a limpeza antes dos deletes.
#   3. O resgate do CSV vem antes dos deletes (apagar a VM destroi o arquivo), mas com
#      TIMEOUT: um scp travado numa VM meio-bootada nao pode adiar a limpeza para sempre.
#   4. Todo delete tem RETRY: "operation in progress" logo apos um create interrompido e
#      falha transitoria, e engoli-la com `|| true` seria vazar recurso pago.
#   5. Conferencia final que ASSERTA: se algo sobreviveu, o script grita LEAK, imprime os
#      comandos de delete manuais e sai com status != 0.
# -------------------------------------------------------------------------------------
TEARDOWN_RETRIES="${TEARDOWN_RETRIES:-3}"
TEARDOWN_ESPERA="${TEARDOWN_ESPERA:-20}"      # segundos entre tentativas de delete
RESGATE_TIMEOUT="${RESGATE_TIMEOUT:-120}"     # teto do scp de resgate do CSV
LEAK_EXIT="${LEAK_EXIT:-70}"                  # status quando sobra recurso de pe

# Apaga com retry. Devolve 0 so se um delete passou.
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

    etapa "TEARDOWN (saida com status $rc) -- apagando VM e Cloud SQL"
    echo "--- a partir daqui INT/TERM/HUP sao IGNORADOS: a limpeza nao se interrompe ---"

    # Ultima chance de salvar o CSV: se a rodada morreu depois de avaliar mas antes de
    # coletar, o arquivo ainda esta na VM que estamos prestes a apagar. Com timeout: a
    # garantia de nao-vazamento vem antes do CSV.
    if [ "${RESULTADOS_COLETADOS:-0}" = "0" ] && [ -n "${ZONE:-}" ]; then
        echo "--- tentativa final de trazer os resultados (teto: ${RESGATE_TIMEOUT}s) ---"
        coletar_resultados || echo "--- resgate do CSV falhou ou estourou o tempo; seguindo para os deletes ---" >&2
    fi

    local zona
    for zona in ${ZONAS_TENTADAS:-}; do
        echo "--- apagando a VM $VM em $zona (se existir) ---"
        apagar_com_retry "delete da VM em $zona" \
            gcloud compute instances delete "$VM" --zone="$zona" \
                --project="$PROJECT_ID" --quiet
    done
    if [ -z "${ZONAS_TENTADAS:-}" ]; then
        echo "--- nenhuma VM foi criada nesta rodada ---"
    fi

    if [ "${SQL_TOCADA:-0}" = "1" ]; then
        echo "--- apagando a instancia Cloud SQL $INSTANCE (o indice morre com ela) ---"
        apagar_com_retry "delete do Cloud SQL $INSTANCE" \
            gcloud sql instances delete "$INSTANCE" --project="$PROJECT_ID" --quiet
    else
        echo "--- nenhuma instancia Cloud SQL foi tocada nesta rodada ---"
    fi

    # -- conferencia que ASSERTA (criterio de aceite do plano: nao pode sobrar nada) -----
    # Pequena espera para o estado da API assentar e nao gerar LEAK falso logo apos um
    # delete que ja retornou sucesso.
    run sleep 10
    echo "--- conferencia final: nada pode ter sobrado ---"
    local vazou=0
    for zona in ${ZONAS_TENTADAS:-}; do
        if existe gcloud compute instances describe "$VM" --zone="$zona" \
                  --project="$PROJECT_ID"; then
            vazou=1
            echo "!!! LEAK: a VM $VM AINDA EXISTE em $zona -- ela CONTINUA COBRANDO !!!" >&2
            echo "    apague a mao:" >&2
            echo "    gcloud compute instances delete $VM --zone=$zona --project=$PROJECT_ID --quiet" >&2
        fi
    done
    if [ "${SQL_TOCADA:-0}" = "1" ] \
       && existe gcloud sql instances describe "$INSTANCE" --project="$PROJECT_ID"; then
        vazou=1
        echo "!!! LEAK: a instancia Cloud SQL $INSTANCE AINDA EXISTE -- ela CONTINUA COBRANDO !!!" >&2
        echo "    apague a mao:" >&2
        echo "    gcloud sql instances delete $INSTANCE --project=$PROJECT_ID --quiet" >&2
    fi

    if [ -n "${TMP_DIR:-}" ] && [ -d "${TMP_DIR:-}" ]; then
        rm -rf "$TMP_DIR"
    fi

    if [ "$vazou" = "1" ]; then
        echo "!!! TEARDOWN INCOMPLETO: ha recurso pago de pe (veja os LEAK acima) !!!" >&2
        if [ "$rc" = "0" ]; then rc="$LEAK_EXIT"; fi
    else
        echo "--- conferencia ok: nem VM nem instancia Cloud SQL sobraram ---"
    fi

    echo "=== teardown concluido (status final: $rc) ==="
    exit "$rc"
}

# -------------------------------------------------------------------------------------
# Script remoto (roda DENTRO da VM)
#
# Gerado aqui e enviado por scp: consolida o bootstrap, o proxy do Cloud SQL, a indexacao
# por espaco e a avaliacao. A configuracao viaja num `slice_env.sh` separado, para o corpo
# do script nao virar um emaranhado de aspas.
# -------------------------------------------------------------------------------------
TMP_DIR=""

gerar_scripts_remotos() {
    TMP_DIR="$(mktemp -d)"

    cat > "$TMP_DIR/slice_env.sh" <<ENV
# Gerado por gcp/slice/run_slice.sh -- configuracao da rodada.
ARCH='$ARCH'
SPACES='$SPACES'
BUCKET='$BUCKET'
DB='$DB'
DB_USER='$DB_USER'
SECRET='$SECRET'
INSTANCE_CONN='$PROJECT_ID:$REGION:$INSTANCE'
SOURCE='$SOURCE'
DT='$DT'
BATCH_SIZE='$BATCH_SIZE'
MIXED_RECORDS='$MIXED_RECORDS'
AUDIO_RECORDS='$AUDIO_RECORDS'
EVAL_DIRS='$EVAL_DIRS'
OUT_CSV='$OUT_CSV'
ENV

    cat > "$TMP_DIR/slice_remote.sh" <<'REMOTE'
#!/usr/bin/env bash
# Gerado por gcp/slice/run_slice.sh -- roda DENTRO da VM L4. Fase em $1.
set -euo pipefail
FASE="${1:?uso: slice_remote.sh <bootstrap|index|eval> [arg]}"
# shellcheck disable=SC1091
source "$HOME/slice_env.sh"
echo "=== [vm $(hostname)] fase=$FASE ${2:-} $(date) ==="

REPO_DIR="$HOME/residencia"
AUDIO_VM_DIR="$HOME/govexe-audio"
PROXY_BIN="$HOME/cloud-sql-proxy"
PROXY_VER="v2.14.1"

# -- ambiente Python da VM ------------------------------------------------------------
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

# -- Cloud SQL Auth Proxy em 127.0.0.1:5432 -------------------------------------------
proxy() {
    if (echo >/dev/tcp/127.0.0.1/5432) 2>/dev/null; then
        echo "--- proxy ja de pe em 127.0.0.1:5432 ---"
        return 0
    fi
    if [ ! -x "$PROXY_BIN" ]; then
        curl -sSL -o "$PROXY_BIN" \
            "https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/$PROXY_VER/cloud-sql-proxy.linux.amd64"
        chmod +x "$PROXY_BIN"
    fi
    pkill -f cloud-sql-proxy || true
    nohup "$PROXY_BIN" "$INSTANCE_CONN" >"$HOME/proxy.log" 2>&1 &
    echo "--- aguardando o proxy abrir 127.0.0.1:5432 ---"
    for _ in $(seq 1 30); do
        if (echo >/dev/tcp/127.0.0.1/5432) 2>/dev/null; then echo "proxy up"; return 0; fi
        sleep 1
    done
    echo "ERRO: o Cloud SQL Auth Proxy nao abriu 127.0.0.1:5432" >&2
    cat "$HOME/proxy.log" || true
    return 1
}

# A senha NUNCA vem por flag: sai do Secret Manager direto para a variavel de ambiente.
credenciais() {
    PGPASSWORD="$(gcloud secrets versions access latest --secret="$SECRET")"
    export PGPASSWORD
    PG=(--pg-host 127.0.0.1 --pg-port 5432 --pg-db "$DB" --pg-user "$DB_USER")
}

# pgvector: o `embed_job` cria a tabela e o indice HNSW, mas nao a EXTENSAO. Feito aqui,
# pelo proxy, com a senha que ja esta na mao -- e nao por um psql interativo na maquina
# do operador. Idempotente.
extensao_vector() {
    python3 - "$DB" "$DB_USER" <<'PY'
import os, sys
import psycopg
dbname, user = sys.argv[1], sys.argv[2]
with psycopg.connect(host="127.0.0.1", port=5432, dbname=dbname, user=user,
                     password=os.environ["PGPASSWORD"], autocommit=True) as conexao:
    conexao.execute("CREATE EXTENSION IF NOT EXISTS vector")
print("extensao pgvector: ok")
PY
}

# Records de audio prontos para ESTA VM: reescreve `audio_path` para o diretorio enviado
# por scp e, quando pedido, troca a modalidade. O `item_id` e PRESERVADO de proposito --
# e ele que liga o mesmo clipe nos dois espacos, e e disso que o RRF vive.
preparar_records_de_audio() {
    local origem="$1" destino="$2" modalidade="${3:-}"
    python3 - "$origem" "$destino" "$modalidade" "$AUDIO_VM_DIR" <<'PY'
import json, os, sys
origem, destino, modalidade, audio_dir = sys.argv[1:5]
with open(origem, encoding="utf-8") as f_in, open(destino, "w", encoding="utf-8") as f_out:
    n = 0
    for linha in f_in:
        linha = linha.strip()
        if not linha:
            continue
        rec = json.loads(linha)
        caminho = rec.get("audio_path")
        if caminho:
            local = os.path.join(audio_dir, os.path.basename(caminho))
            if os.path.isfile(local):
                rec["audio_path"] = local
        if modalidade:
            rec["modality"] = modalidade      # item_id intacto: e a chave do RRF
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
print(f"records preparados: {n} -> {destino} (modalidade={modalidade or 'inalterada'})")
PY
}

case "$FASE" in

# =====================================================================================
bootstrap)
# =====================================================================================
    if [ ! -d "$REPO_DIR" ]; then
        mkdir -p "$REPO_DIR"
        tar xzf "$HOME/repo.tgz" -C "$REPO_DIR"
    fi
    cd "$REPO_DIR"
    export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
    export HF_HOME="$HOME/.cache/huggingface"
    export HF_XET_HIGH_PERFORMANCE=1
    mkdir -p "$HF_HOME"

    # ffmpeg: o decode de audio (yt-dlp/librosa) precisa dele e a imagem DLVM nao traz.
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "--- instalando ffmpeg ---"
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
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

    # Marcador por sha do requirements + sufixo desta fatia: mexer no requirements OU na
    # lista de adaptadores abaixo invalida o marcador e reinstala.
    REQ_HASH="$(sha256sum requirements-slurm.txt | cut -d' ' -f1)"
    DEPS_MARKER="$DEPS_MARKER_DIR/.residencia_deps_${REQ_HASH}_slice2"
    if [ -f "$DEPS_MARKER" ]; then
        echo "--- deps ja instaladas ($DEPS_MARKER) ---"
    else
        mkdir -p "$DEPS_MARKER_DIR"
        echo "--- instalando deps (requirements-slurm.txt) ---"
        python3 -m pip install --quiet -r requirements-slurm.txt
        echo "--- adaptadores desta fatia: gcs, psycopg[binary], yt-dlp ---"
        python3 -m pip install --quiet google-cloud-storage "psycopg[binary]" yt-dlp
        touch "$DEPS_MARKER"
    fi

    # Guard "preferir o torch da imagem": qualquer dep que declare torch instala um por
    # cima, SEM as libs CUDA do no (libnvJitLink.so.12...), e ele sombra o torch bom.
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

    # torchaudio ABI-incompativel derruba QUALQUER import de transformers 5.x.
    if python3 -c "import torchaudio" 2>/dev/null; then
        echo "--- torchaudio ok ---"
    else
        echo "--- reparando torchaudio casado com o torch da imagem ---"
        TA_VER="$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || true)"
        TA_VER="${TA_VER%%+*}"
        CU_RAW="$(python3 -c 'import torch;print(torch.version.cuda or str())' 2>/dev/null || true)"
        CU="cu${CU_RAW//./}"
        if [ -n "$TA_VER" ]; then
            python3 -m pip install --no-deps --force-reinstall "torchaudio==${TA_VER}" \
                --index-url "https://download.pytorch.org/whl/${CU}" \
                && python3 -c "import torchaudio" 2>/dev/null \
                && echo "--- torchaudio reparado ---" \
                || echo "aviso: torchaudio ainda nao importa (so afeta quem o exigir)"
        fi
    fi

    echo "=== validacao de imports + GPU ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    python3 - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__)
import psycopg; print("psycopg ok")
from google.cloud import storage; print("gcs ok")
from gcp.slice.embed_job import main as embed_main; print("embed_job ok")
from gcp.slice.eval import main as eval_main; print("eval ok")
PY
    # So valida o caminho do CLAP quando a arquitetura o usa (baixar/importar a toa custa).
    case " $SPACES " in
        *" clap "*)
            python3 - <<'PY'
import librosa, soundfile          # decode de audio do CLAP (import lazy no wrapper)
print("librosa", librosa.__version__, "| soundfile", soundfile.__version__)
from benchmark.models.clap_embed import embed_texts    # noqa: F401
print("clap_embed ok")
PY
            ;;
    esac
    echo "=== bootstrap concluido $(date) ==="
    ;;

# =====================================================================================
index)
# =====================================================================================
    ESPACO="${2:?uso: slice_remote.sh index <espaco>}"
    ambiente
    proxy
    credenciais
    extensao_vector

    # A lista de espacos do script bash tem de bater com o registro Python -- divergir
    # significaria indexar a arquitetura errada numa rodada paga.
    python3 - "$ARCH" "$SPACES" <<'PY'
import sys
from gcp.slice.spaces import ARCHITECTURES
arch, do_script = sys.argv[1], sys.argv[2].split()
do_registro = list(ARCHITECTURES[arch])
if do_script != do_registro:
    raise SystemExit(f"ERRO: run_slice.sh diz {do_script} para {arch!r}, mas o registro "
                     f"(gcp/slice/spaces.py) diz {do_registro}")
print(f"espacos de {arch}: {do_registro} (script e registro batem)")
PY

    case "$ESPACO" in
        qwen3vl)
            echo "=== [index qwen3vl] paginas do Bronze (doc_image) ==="
            python3 -m gcp.slice.embed_job \
                --bucket "$BUCKET" \
                --blob-prefix "bronze/fonte=$SOURCE/dt=$DT/" \
                --work-dir /tmp/govexe-bronze \
                --source "$SOURCE" \
                --space qwen3vl \
                --device cuda --batch-size "$BATCH_SIZE" "${PG[@]}"

            # Lote misto (foto + texto nativo), se a rodada o trouxe.
            if [ -n "$MIXED_RECORDS" ] && [ -f "$REPO_DIR/$MIXED_RECORDS" ]; then
                echo "=== [index qwen3vl] lote misto (image + text) ==="
                python3 -m gcp.slice.embed_job \
                    --records "$REPO_DIR/$MIXED_RECORDS" \
                    --records-store "$BUCKET" \
                    --work-dir /tmp/govexe-bronze \
                    --space qwen3vl \
                    --device cuda --batch-size "$BATCH_SIZE" "${PG[@]}"
            else
                echo "--- sem lote misto ($MIXED_RECORDS): pulando ---"
            fi

            # Audio pelo caminho POINTER (Whisper -> texto -> mesmo espaco).
            if [ -n "$AUDIO_RECORDS" ] && [ -f "$REPO_DIR/$AUDIO_RECORDS" ]; then
                echo "=== [index qwen3vl] audio via ASR (audio_asr) ==="
                preparar_records_de_audio "$REPO_DIR/$AUDIO_RECORDS" /tmp/audio_asr.jsonl
                python3 -m gcp.slice.embed_job \
                    --records /tmp/audio_asr.jsonl \
                    --work-dir /tmp/govexe-bronze \
                    --space qwen3vl \
                    --asr-backend faster-whisper --asr-model large-v3 --asr-language pt \
                    --device cuda --batch-size 1 "${PG[@]}"
            else
                echo "--- sem records de audio ($AUDIO_RECORDS): sem trilha audio_asr ---"
            fi
            ;;

        clap)
            # O CLAP indexa a FORMA DE ONDA: os mesmos clipes do caminho ASR, com a
            # modalidade trocada para `audio_native` e o mesmo item_id (chave do RRF).
            if [ -z "$AUDIO_RECORDS" ] || [ ! -f "$REPO_DIR/$AUDIO_RECORDS" ]; then
                echo "ERRO: o espaco clap nao tem o que indexar sem $AUDIO_RECORDS" >&2
                exit 1
            fi
            echo "=== [index clap] audio nativo (audio_native) ==="
            preparar_records_de_audio "$REPO_DIR/$AUDIO_RECORDS" /tmp/audio_native.jsonl \
                audio_native
            python3 -m gcp.slice.embed_job \
                --records /tmp/audio_native.jsonl \
                --work-dir /tmp/govexe-bronze \
                --space clap \
                --device cuda --batch-size 1 "${PG[@]}"
            ;;

        *) echo "ERRO: espaco desconhecido na VM: '$ESPACO'" >&2; exit 2 ;;
    esac
    echo "=== indexacao do espaco $ESPACO concluida $(date) ==="
    ;;

# =====================================================================================
eval)
# =====================================================================================
    ambiente
    proxy
    credenciais

    for DIR_AVAL in $EVAL_DIRS; do
        if [ ! -d "$REPO_DIR/$DIR_AVAL" ]; then
            echo "--- conjunto ausente: $DIR_AVAL (pulando) ---"
            continue
        fi
        echo "=== [eval] $ARCH sobre $DIR_AVAL ==="
        python3 -m gcp.slice.eval \
            --arch "$ARCH" \
            --data-dir "$REPO_DIR/$DIR_AVAL" \
            --out "$REPO_DIR/$OUT_CSV" \
            --device cuda "${PG[@]}"
    done

    echo "=== [eval] CSV comparativo ==="
    cat "$REPO_DIR/$OUT_CSV" || true

    echo "=== [eval] prova cross-modal + contagem por modalidade ==="
    python3 -m gcp.slice.query --text "metas para a saude publica" --k 5 "${PG[@]}" || true
    python3 - "$ARCH" "$DB" "$DB_USER" <<'PY'
import os, sys
import psycopg
from gcp.slice.spaces import ARCHITECTURES, SPACES
arch, dbname, user = sys.argv[1:4]
conexao = psycopg.connect(host="127.0.0.1", port=5432, dbname=dbname, user=user,
                          password=os.environ["PGPASSWORD"])
cur = conexao.cursor()
for nome in ARCHITECTURES[arch]:
    tabela = SPACES[nome].table
    cur.execute(f"SELECT modality, count(*) FROM {tabela} GROUP BY modality ORDER BY modality")
    print(f"{nome} ({tabela}):")
    for modalidade, n in cur.fetchall():
        print(f"  {modalidade:14s} {n}")
PY
    echo "=== eval concluida $(date) ==="
    ;;

*) echo "ERRO: fase desconhecida: '$FASE'" >&2; exit 2 ;;
esac
REMOTE
    echo "--- scripts remotos gerados em $TMP_DIR ---"
}

# -------------------------------------------------------------------------------------
# Provisionamento
# -------------------------------------------------------------------------------------
provisionar_sql() {
    etapa "[1/7] Cloud SQL ($INSTANCE, $REGION) + banco + senha"

    # A senha vem ANTES de criar a instancia: secret ausente ou sem permissao derruba a
    # rodada aqui, com zero recurso pago de pe. Ela so existe no Secret Manager -- nunca
    # em flag, arquivo ou log.
    local senha
    senha="$(senha_do_secret)"

    SQL_TOCADA=1          # a partir daqui o teardown e responsavel por apaga-la

    if existe gcloud sql instances describe "$INSTANCE" --project="$PROJECT_ID"; then
        echo "--- instancia $INSTANCE ja existe: reusando (sera APAGADA no teardown) ---"
    else
        run gcloud sql instances create "$INSTANCE" \
            --project="$PROJECT_ID" \
            --database-version=POSTGRES_16 \
            --edition=ENTERPRISE \
            --tier=db-custom-1-3840 \
            --region="$REGION" \
            --storage-size=10GB \
            --storage-type=SSD \
            --no-backup
    fi

    if existe gcloud sql databases describe "$DB" --instance="$INSTANCE" \
              --project="$PROJECT_ID"; then
        echo "--- banco $DB ja existe ---"
    else
        run gcloud sql databases create "$DB" --instance="$INSTANCE" --project="$PROJECT_ID"
    fi

    # Aqui a senha e aplicada ao usuario do banco. A linha NAO vai para o log
    # (`run_secreto`), mas ela ainda passa pela tabela de processos desta maquina enquanto
    # o gcloud roda -- limitacao do `--password` do gcloud, ja registrada no
    # README-slice.md secao 4.
    run_secreto "gcloud sql users set-password $DB_USER --instance=$INSTANCE --password=<secret>" \
        gcloud sql users set-password "$DB_USER" --instance="$INSTANCE" \
            --project="$PROJECT_ID" --password="$senha" --quiet \
    || run_secreto "gcloud sql users create $DB_USER --instance=$INSTANCE --password=<secret>" \
        gcloud sql users create "$DB_USER" --instance="$INSTANCE" \
            --project="$PROJECT_ID" --password="$senha" --quiet
    unset senha
    # A extensao `vector` NAO e criada daqui: `gcloud sql connect` abre um psql
    # interativo e liberaria o IP desta maquina na instancia. Quem a cria e a VM, pelo
    # proxy, no inicio da indexacao (fase `index` do script remoto).
}

provisionar_vm() {
    etapa "[2/7] VM L4 ($MACHINE_TYPE + $ACCELERATOR) com fallback de zona"
    echo "--- zonas na ordem: $ZONES ---"
    local zona
    for zona in $ZONES; do
        echo "--- tentando $zona ---"
        ZONAS_TENTADAS="$ZONAS_TENTADAS $zona"
        if run gcloud compute instances create "$VM" \
                --project="$PROJECT_ID" \
                --zone="$zona" \
                --machine-type="$MACHINE_TYPE" \
                --accelerator="type=$ACCELERATOR,count=1" \
                --maintenance-policy=TERMINATE \
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
    echo "ERRO: nenhuma zona da lista tinha capacidade de L4: $ZONES" >&2
    return 1
}

esperar_ssh() {
    etapa "[3/7] esperando o SSH da VM responder"
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] gcloud compute ssh %s --zone=%s --command "true"  (ate 20x)\n' \
            "$VM" "$ZONE"
        return 0
    fi
    local i
    for i in $(seq 1 20); do
        if gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT_ID" --quiet \
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
    etapa "[4/7] enviando repo, scripts e audio para a VM"
    # Arvore de trabalho (e nao `git archive`): o mini-set de audio costuma ainda estar
    # sem commit quando a rodada acontece.
    run tar czf "$TMP_DIR/repo.tgz" -C "$REPO_DIR" \
        --exclude=.git --exclude='.venv*' --exclude=__pycache__ \
        --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.benchmarks \
        --exclude=results --exclude=logs --exclude=.cache \
        --exclude='data/govexe/ptbr_planos/pages' \
        --exclude='data/govexe/ptbr_planos/pdf_cache' \
        --exclude=data/vidore_cache .

    # No Windows o `gcloud compute scp` delega ao `pscp.exe`, que NAO aceita `~` como
    # destino remoto nem varios arquivos de origem de uma vez. Dai o padrao usado em todo
    # upload daqui: um arquivo por chamada, com destino RELATIVO (o pscp resolve a partir
    # do home de login, entao equivale ao antigo `~/`).
    run gcloud compute scp "$TMP_DIR/repo.tgz" "$VM:repo.tgz" \
        --zone="$ZONE" --project="$PROJECT_ID" --quiet
    run gcloud compute scp "$TMP_DIR/slice_remote.sh" "$VM:slice_remote.sh" \
        --zone="$ZONE" --project="$PROJECT_ID" --quiet
    run gcloud compute scp "$TMP_DIR/slice_env.sh" "$VM:slice_env.sh" \
        --zone="$ZONE" --project="$PROJECT_ID" --quiet

    # Os binarios de audio nao cabem no repo: vao a parte, para o diretorio que o script
    # remoto usa ao reescrever `audio_path`. Vao num tar unico (em vez de `--recurse`)
    # pela mesma limitacao do pscp, e sao desempacotados na VM.
    if [ -d "$REPO_DIR/$AUDIO_DIR" ]; then
        run tar czf "$TMP_DIR/audio.tgz" -C "$REPO_DIR/$AUDIO_DIR" .
        run gcloud compute scp "$TMP_DIR/audio.tgz" "$VM:govexe-audio.tgz" \
            --zone="$ZONE" --project="$PROJECT_ID" --quiet
        remoto "mkdir -p ~/govexe-audio && tar xzf ~/govexe-audio.tgz -C ~/govexe-audio"
    else
        echo "--- sem diretorio de audio ($AUDIO_DIR): nada a enviar ---"
    fi

    # CSV anterior, para o eval.py ACUMULAR as linhas das rodadas antigas em vez de
    # reescrever o arquivo do zero.
    if [ -f "$REPO_DIR/$OUT_CSV" ]; then
        remoto "mkdir -p ~/residencia/$(dirname "$OUT_CSV")"
        run gcloud compute scp "$REPO_DIR/$OUT_CSV" "$VM:residencia/$OUT_CSV" \
            --zone="$ZONE" --project="$PROJECT_ID" --quiet
    fi
}

# Chamada no fluxo normal E dentro do teardown; por isso todo scp tem teto de tempo (um
# scp travado numa VM meio-bootada nao pode adiar a limpeza).
coletar_resultados() {
    echo "--- trazendo $OUT_CSV e os logs da VM (teto: ${RESGATE_TIMEOUT}s por arquivo) ---"
    run mkdir -p "$REPO_DIR/$(dirname "$OUT_CSV")" "$REPO_DIR/logs" || true
    com_timeout "$RESGATE_TIMEOUT" \
        gcloud compute scp "$VM:residencia/$OUT_CSV" "$REPO_DIR/$OUT_CSV" \
            --zone="$ZONE" --project="$PROJECT_ID" --quiet || return 1
    com_timeout "$RESGATE_TIMEOUT" \
        gcloud compute scp "$VM:proxy.log" "$REPO_DIR/logs/proxy-$VM.log" \
            --zone="$ZONE" --project="$PROJECT_ID" --quiet || true
    RESULTADOS_COLETADOS=1
    return 0
}

# -------------------------------------------------------------------------------------
# Preflight (local, antes de criar qualquer recurso pago)
# -------------------------------------------------------------------------------------
etapa "rodada: arquitetura $ARCH | espacos: $SPACES | dry-run: $DRY_RUN"
echo "    projeto=$PROJECT_ID  regiao=$REGION  instancia=$INSTANCE  vm=$VM"
echo "    bucket=$BUCKET  fonte=$SOURCE  dt=$DT  csv=$OUT_CSV"

if ! command -v gcloud >/dev/null 2>&1; then
    precondicao_ausente "gcloud nao esta no PATH"
fi

# Os caminhos de dados sao RELATIVOS a raiz do repo: eles viajam dentro do tarball e sao
# relidos na VM como "$REPO_DIR/<caminho>". Um caminho absoluto viraria concatenacao sem
# sentido -- falhar aqui e mais barato que descobrir isso na VM.
exige_relativo() {
    case "$2" in
        /*|[A-Za-z]:[\\/]*|*:*)
            echo "ERRO: $1 tem de ser relativo a raiz do repo ($REPO_DIR): '$2'" >&2
            exit 2 ;;
    esac
}
exige_relativo --audio-records "$AUDIO_RECORDS"
exige_relativo --audio-dir "$AUDIO_DIR"
exige_relativo --out "$OUT_CSV"

# O conjunto de avaliacao de audio entra em QUALQUER arquitetura, nao so nas que tem CLAP:
# o espaco qwen3vl ja indexa os mesmos clipes como `audio_asr` (Whisper -> texto), e e essa
# a baseline "A (ASR->texto)" que a tarefa de audio da spec (secao 5) compara com o CLAP
# nativo. Sem isto, `--arch A` indexava o audio e nunca o media.
#
# Gate humano: enquanto o mini-set nao for capturado e julgado, o diretorio nao existe e a
# rodada mede so a tarefa doc-visual -- aviso, nao erro.
if [ -d "$REPO_DIR/$EVAL_AUDIO_DIR" ]; then
    EVAL_DIRS="$EVAL_DIRS $EVAL_AUDIO_DIR"
else
    echo "--- aviso: sem $EVAL_AUDIO_DIR: a rodada mede so a tarefa doc-visual ---"
fi

for _dir in $EVAL_DIRS; do exige_relativo --eval-dirs "$_dir"; done

# O espaco clap so existe se houver audio para indexar -- descobrir isso DEPOIS de subir
# a L4 seria pagar GPU para falhar.
case " $SPACES " in
    *" clap "*)
        if [ ! -f "$REPO_DIR/$AUDIO_RECORDS" ]; then
            precondicao_ausente \
                "a arquitetura $ARCH indexa o espaco clap, mas nao existe" \
                "$AUDIO_RECORDS (records do mini-set de audio)." \
                "Capture com gcp/slice/audio_collector.py ou passe --audio-records."
        fi
        ;;
esac
echo "    conjuntos de avaliacao: $EVAL_DIRS"

gerar_scripts_remotos

# -------------------------------------------------------------------------------------
# A partir daqui existe recurso pago em jogo: o teardown passa a ser obrigatorio.
#
# INT/TERM/HUP viram uma saida normal para que o EXIT dispare de forma confiavel (bash nem
# sempre roda o trap de EXIT quando o shell morre por sinal nao tratado). HUP cobre o caso
# de uma rodada paga longa: terminal fechado ou ssh caido nao pode pular a limpeza.
# -------------------------------------------------------------------------------------
trap 'teardown' EXIT
trap 'echo ""; echo "--- interrompido (SIGINT) ---" >&2; exit 130' INT
trap 'echo ""; echo "--- terminado (SIGTERM) ---" >&2; exit 143' TERM
trap 'echo ""; echo "--- terminal caiu (SIGHUP) ---" >&2; exit 129' HUP

gancho_de_falha provision
provisionar_sql
provisionar_vm
esperar_ssh
enviar_repo

etapa "[5/7] bootstrap da VM (venv, deps, torchaudio, librosa/soundfile)"
gancho_de_falha bootstrap
remoto "bash ~/slice_remote.sh bootstrap"

for espaco in $SPACES; do
    etapa "[6/7] indexando o espaco $espaco (arquitetura $ARCH)"
    gancho_de_falha index
    remoto "bash ~/slice_remote.sh index $espaco"
done

etapa "[7/7] avaliacao (eval.py) -> $OUT_CSV"
gancho_de_falha eval
remoto "bash ~/slice_remote.sh eval"

etapa "resultados"
coletar_resultados || echo "aviso: nao foi possivel trazer o CSV (o teardown tentara de novo)"

etapa "rodada da arquitetura $ARCH concluida -- o teardown vem a seguir"
exit 0
