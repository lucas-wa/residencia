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
#   bash gcp/slice/run_slice.sh --arch B                # so a VM L4: SEM Cloud SQL
#   bash gcp/slice/run_slice.sh --arch B --specialist-batch-size 2   # menos VRAM na L4
#   bash gcp/slice/run_slice.sh --arch A --dry-run      # imprime tudo, nao toca a nuvem
#
#   # ensemble v2 nativo (5 especialistas -> RRF), em DUAS rodadas pagas separadas pelo
#   # gate humano dos qrels -- a semantica e a do ENSEMBLE_V2_PHASE do slurm_run.sh:
#   bash gcp/slice/run_slice.sh --arch ensemble --phase capture
#       -> captura o corpus Orcamento na VM e TRAZ records.jsonl + binarios para o repo
#          local ANTES do teardown; revise ensemble/eval/qrels_orcamento.tsv em cima dos
#          item_ids capturados (recapturar muda os ids e invalida o gabarito)
#   bash gcp/slice/run_slice.sh --arch ensemble --phase eval
#       -> SOBE o corpus congelado + binarios + queries + qrels, avalia e traz o CSV
#   bash gcp/slice/run_slice.sh --arch ensemble --phase all   # as duas na MESMA VM
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
#          bootstrap, index, capture, eval); nao tem uso em producao.
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
#   B         -> (nenhum espaco)      ensemble de especialistas: o eixo documento e o
#                                     ColQwen2.5 late-interaction (MaxSim em memoria)
#   ensemble  -> (nenhum espaco)      ensemble v2 NATIVO: 5 especialistas (texto BGE-M3,
#                                     imagem SigLIP2, audio CLAP, video WeMM, documento
#                                     ColQwen2.5) fundidos por RRF e comparados com os
#                                     baselines A (Qwen3-VL) e B (so documento). Os
#                                     modelos carregam UM EIXO POR VEZ -- cabe na L4 --
#                                     e os indices sao em memoria (`ensemble/`, nao
#                                     `gcp/slice/eval.py`).
#
# FASES DO ensemble (`--phase`, espelha ENSEMBLE_V2_PHASE do slurm_run.sh)
#
# Recapturar o corpus muda os `item_id` (hash da URL) e invalida o gabarito, entao captura
# e avaliacao sao DUAS rodadas pagas com um gate humano no meio. A persistencia entre elas
# e o proprio repo local: a fase `capture` TRAZ `records.jsonl` + os binarios por tar antes
# do teardown, e a fase `eval` os SOBE de volta (nao ha GCS no meio -- e ida-e-volta por
# scp, com os mesmos helpers do mini-set de audio). `all` roda as duas na mesma VM.
#
# ARQUITETURA SEM pgvector (`B` e `ensemble`)
#
# B nao tem espaco pgvector: multi-vetor nao cabe numa coluna `vector(n)` ranqueada por
# cosseno (ver gcp/slice/spaces.py). Entao a rodada de B PULA a etapa [1/7] (Cloud SQL) e
# a etapa [6/7] (indexacao) -- sobem so a VM L4 e as 93 paginas baixadas do Bronze, e o
# eval ranqueia tudo em memoria. O `ensemble` segue a MESMA rota (indices em memoria, RRF
# sobre as ordens), so que o corpus e o do ensemble v2 e a etapa [6/7] vira a CAPTURA.
# O teardown fica MAIS simples, nao mais fraco: `SQL_TOCADA`
# continua 0, entao nem o delete nem a conferencia final procuram uma instancia Cloud SQL
# que nunca existiu (procurar renderia um falso LEAK). A VM segue com a garantia de sempre.
#
# O lote do ColQwen2.5 e `--specialist-batch-size` (env `SPECIALIST_BATCH_SIZE`, default 4):
# o knob de VRAM da rodada, distinto do BATCH_SIZE da indexacao, que B nem faz. Se a L4
# estourar memoria, baixe para 2 ou 1 -- custa tempo e nao muda ranking.
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
# Lote do ranqueador late-interaction (arquitetura sem pgvector). Separado do BATCH_SIZE
# acima de proposito: aquele e da INDEXACAO (embed_job), este e do `encode_corpus` do
# ColQwen2.5, que guarda um vetor por token de imagem e e o que ja estourou os 24 GB da L4
# na prova #2. Baixar para 2 ou 1 troca tempo por VRAM e nao muda ranking.
SPECIALIST_BATCH_SIZE="${SPECIALIST_BATCH_SIZE:-4}"

# Conjuntos de avaliacao (um `eval.py` por conjunto; o CSV acumula as linhas).
EVAL_DIRS="${EVAL_DIRS:-data/govexe/ptbr_planos}"
EVAL_AUDIO_DIR="${EVAL_AUDIO_DIR:-data/govexe/audio_eval}"
# CSV de saida. O default DEPENDE da arquitetura (o ensemble v2 escreve o seu proprio
# comparativo, `results/comparacao_v2.csv`), entao ele e resolvido no `case "$ARCH"`
# abaixo -- aqui so se respeita o que vier do ambiente ou de `--out`.
OUT_CSV="${OUT_CSV:-}"

# Insumos da arquitetura `ensemble` (todos relativos a raiz do repo, como os demais: eles
# viajam no scp e sao relidos na VM como "$REPO_DIR/<caminho>"). Espelham as ENSEMBLE_V2_*
# do slurm_run.sh. O `bin/` guarda os binarios retidos pela captura (um subdiretorio por
# eixo) -- e o que `build_corpus` referencia em `meta["path"]`.
ENSEMBLE_CORPUS="${ENSEMBLE_CORPUS:-data/govexe/ensemble_orcamento/records.jsonl}"
ENSEMBLE_BIN_DIR="${ENSEMBLE_BIN_DIR:-data/govexe/ensemble_orcamento/bin}"
ENSEMBLE_QUERIES="${ENSEMBLE_QUERIES:-ensemble/eval/queries_orcamento_v2.txt}"
ENSEMBLE_QRELS="${ENSEMBLE_QRELS:-ensemble/eval/qrels_orcamento.tsv}"
ENSEMBLE_OUT_CSV="${ENSEMBLE_OUT_CSV:-results/comparacao_v2.csv}"
ENSEMBLE_LOG="${ENSEMBLE_LOG:-logs/ensemble_v2.txt}"
# Teto PROPRIO do scp dos binarios do corpus: o tar do `bin/` leva os videos das janelas e
# nao cabe nos 120s do resgate do CSV. So o tar usa este teto; o CSV segue no RESGATE_TIMEOUT.
ENSEMBLE_RESGATE_TIMEOUT="${ENSEMBLE_RESGATE_TIMEOUT:-900}"

ARCH="${ARCH:-A}"
# Fase da arquitetura `ensemble` (sem efeito nas demais): capture | eval | all.
PHASE="${PHASE:-all}"
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
    sed -n '2,97p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --arch)           ARCH="$2"; shift 2 ;;
        --phase)          PHASE="$2"; shift 2 ;;
        --zones)          ZONES="$2"; shift 2 ;;
        --project)        PROJECT_ID="$2"; shift 2 ;;
        --bucket)         BUCKET="$2"; shift 2 ;;
        --dt)             DT="$2"; shift 2 ;;
        --audio-records)  AUDIO_RECORDS="$2"; shift 2 ;;
        --audio-dir)      AUDIO_DIR="$2"; shift 2 ;;
        --eval-dirs)      EVAL_DIRS="$2"; shift 2 ;;
        --specialist-batch-size) SPECIALIST_BATCH_SIZE="$2"; shift 2 ;;
        --out)            OUT_CSV="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        uso 0 ;;
        *) echo "flag desconhecida: $1" >&2; uso 2 ;;
    esac
done

# A lista de espacos de cada arquitetura esta aqui em bash E em `gcp/slice/spaces.py`. A
# fonte de verdade e o registro Python; para a copia nao divergir em silencio, a VM
# confere as duas listas antes de indexar (fase `index`, no script remoto).
USA_ENSEMBLE=0          # 1 so na arquitetura `ensemble` (ensemble v2 nativo, 5 eixos)
case "$ARCH" in
    A)        SPACES="qwen3vl";      SPECIALISTS="" ;;
    A+audio)  SPACES="qwen3vl clap"; SPECIALISTS="" ;;
    # B: nenhum espaco pgvector (a string vazia e a declaracao disso, nao um esquecimento)
    # e um especialista late-interaction no eixo documento.
    B)        SPACES="";             SPECIALISTS="colqwen25_doc" ;;
    # ensemble: tambem sem espaco pgvector, e SEM especialista do registro
    # `gcp/slice/spaces.py` -- os 5 eixos sao os builders de `ensemble/native/embedders.py`
    # e quem os monta e `ensemble/eval/run_eval.py`, nao `gcp/slice/eval.py`. Por isso o
    # marcador e uma variavel PROPRIA e nao um nome enfiado em SPECIALISTS: aquele nome
    # seria procurado no registro e a rodada morreria com a GPU ja paga.
    ensemble) SPACES="";             SPECIALISTS="";  USA_ENSEMBLE=1
              : "${OUT_CSV:=$ENSEMBLE_OUT_CSV}" ;;
    *) echo "ERRO: arquitetura desconhecida: '$ARCH' (use A, A+audio, B ou ensemble)" >&2
       exit 2 ;;
esac
: "${OUT_CSV:=results/comparacao.csv}"

# `--phase` so existe para o ensemble (as demais arquiteturas nao tem o que separar), mas
# o VALOR e validado sempre: um `--phase captura` silenciosamente ignorado custaria uma
# rodada paga inteira.
case "$PHASE" in
    capture|eval|all) ;;
    *) echo "ERRO: fase desconhecida: '$PHASE' (use capture, eval ou all)" >&2; exit 2 ;;
esac
if [ "$USA_ENSEMBLE" != "1" ] && [ "$PHASE" != "all" ]; then
    echo "--- aviso: --phase $PHASE nao tem efeito em --arch $ARCH (so em --arch ensemble) ---" >&2
fi

# Deriva uma vez o que roteia a rodada inteira: com espaco pgvector ha Cloud SQL, proxy e
# indexacao; sem ele, nada disso existe.
if [ -n "$SPACES" ]; then USA_PGVECTOR=1; else USA_PGVECTOR=0; fi

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
PHASE='$PHASE'
SPACES='$SPACES'
SPECIALISTS='$SPECIALISTS'
USA_PGVECTOR='$USA_PGVECTOR'
USA_ENSEMBLE='$USA_ENSEMBLE'
ENSEMBLE_CORPUS='$ENSEMBLE_CORPUS'
ENSEMBLE_BIN_DIR='$ENSEMBLE_BIN_DIR'
ENSEMBLE_QUERIES='$ENSEMBLE_QUERIES'
ENSEMBLE_QRELS='$ENSEMBLE_QRELS'
ENSEMBLE_LOG='$ENSEMBLE_LOG'
BUCKET='$BUCKET'
DB='$DB'
DB_USER='$DB_USER'
SECRET='$SECRET'
INSTANCE_CONN='$PROJECT_ID:$REGION:$INSTANCE'
SOURCE='$SOURCE'
DT='$DT'
BATCH_SIZE='$BATCH_SIZE'
SPECIALIST_BATCH_SIZE='$SPECIALIST_BATCH_SIZE'
MIXED_RECORDS='$MIXED_RECORDS'
AUDIO_RECORDS='$AUDIO_RECORDS'
EVAL_DIRS='$EVAL_DIRS'
OUT_CSV='$OUT_CSV'
ENV

    cat > "$TMP_DIR/slice_remote.sh" <<'REMOTE'
#!/usr/bin/env bash
# Gerado por gcp/slice/run_slice.sh -- roda DENTRO da VM L4. Fase em $1.
set -euo pipefail
FASE="${1:?uso: slice_remote.sh <bootstrap|index|capture|eval> [arg]}"
# shellcheck disable=SC1091
source "$HOME/slice_env.sh"
echo "=== [vm $(hostname)] fase=$FASE ${2:-} $(date) ==="

REPO_DIR="$HOME/residencia"
AUDIO_VM_DIR="$HOME/govexe-audio"
# Paginas do Bronze para a arquitetura SEM pgvector: o tarball do repo exclui
# `data/govexe/ptbr_planos/pages` (tamanho), entao o eixo documento as busca no Bronze --
# o MESMO objeto que a arquitetura A indexa, e com o nome de arquivo preservado, que e o
# que faz o basename casar com o `item_id` do corpus/qrels.
PAGES_VM_DIR="$HOME/govexe-pages"
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

# Paginas do Bronze para o ranqueador em memoria (arquitetura sem pgvector). Reusa o
# `download_pages` do embed_job -- o mesmo download que a arquitetura A faz antes de
# indexar; a unica diferenca e o destino, que aqui e um diretorio e nao uma tabela.
baixar_paginas_do_bronze() {
    mkdir -p "$PAGES_VM_DIR"
    python3 - "$BUCKET" "bronze/fonte=$SOURCE/dt=$DT/" "$PAGES_VM_DIR" <<'PY'
import sys
from gcp.slice.collector import store_from_location
from gcp.slice.embed_job import download_pages
bucket, prefixo, destino = sys.argv[1:4]
paths, _ = download_pages(store_from_location(bucket), prefixo, destino)
if not paths:
    raise SystemExit(f"ERRO: nenhuma pagina em {bucket}/{prefixo} -- sem corpus para o "
                     f"eixo documento (confira --dt)")
print(f"paginas do Bronze: {len(paths)} -> {destino}")
PY
}

case "$FASE" in

# =====================================================================================
bootstrap)
# =====================================================================================
    # Guard por SENTINELA DE CODIGO, e nao por `[ -d "$REPO_DIR" ]`: a etapa [4/7] pode
    # PRE-CRIAR `~/residencia` ao subir o `results/comparacao.csv` anterior -- e ai o teste
    # de diretorio daria falso e o tarball nunca seria extraido (codigo e
    # requirements-slurm.txt ausentes, bootstrap morrendo no sha256sum). Extrair por cima
    # e seguro: o tarball exclui `results/`, entao o CSV recem-enviado fica intacto.
    if [ ! -f "$REPO_DIR/requirements-slurm.txt" ]; then
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

    # python3.10-venv: a imagem DLVM atual tambem nao traz, e sem ele o `python3 -m venv`
    # abaixo morre em "ensurepip is not available" e cai no fallback do python do sistema.
    # Passo PROPRIO (e nao junto do ffmpeg) para nao ser pulado quando o ffmpeg ja existe.
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
    # Idem para o especialista late-interaction: resolver a classe do colpali-engine aqui
    # custa segundos e evita descobrir um `ColQwen2_5` ausente depois da GPU ja estar paga.
    if [ -n "$SPECIALISTS" ]; then
        python3 - "$SPECIALISTS" <<'PY'
import sys
from gcp.slice.doc_specialist import build_doc_retriever    # noqa: F401
from gcp.slice.spaces import SPECIALISTS
from benchmark.models.colvision import resolve_classes
for nome in sys.argv[1].split():
    esp = SPECIALISTS[nome]
    modelo, proc = resolve_classes(esp.kind)
    print(f"{nome}: {esp.model_id} -> {modelo.__name__}/{proc.__name__}")
PY
    fi
    # Idem para o ensemble v2: as deps dele ja vem do requirements-slurm.txt
    # (sentence-transformers, colpali-engine, qwen-vl-utils, pypdfium2, librosa,
    # soundfile, faster-whisper, beautifulsoup4, pillow, transformers) mais o yt-dlp dos
    # adaptadores desta fatia e o ffmpeg do apt -- nada de pip extra. O que falta e
    # CONFERIR a fiacao antes da GPU estar paga: um `import yt_dlp` ausente ou uma classe
    # ColQwen2_5 irresoluvel descobertos no meio da captura custariam a rodada inteira.
    if [ "${USA_ENSEMBLE:-0}" = "1" ]; then
        python3 - <<'PY'
import bs4, librosa, pypdfium2, sentence_transformers, soundfile, yt_dlp   # noqa: F401
from PIL import Image                                                      # noqa: F401
from faster_whisper import WhisperModel                                    # noqa: F401
print("sentence-transformers", sentence_transformers.__version__,
      "| yt-dlp", yt_dlp.version.__version__)
from ensemble.build_corpus import build_corpus           # noqa: F401  (captura, rede)
from ensemble.eval.run_eval import DOC_KIND, DOC_MODEL, main   # noqa: F401
from ensemble.native.embedders import EMBEDDER_BUILDERS
print("eixos do ensemble:", sorted(EMBEDDER_BUILDERS))
from benchmark.models.colvision import resolve_classes
modelo, proc = resolve_classes(DOC_KIND)
print(f"documento: {DOC_MODEL} -> {modelo.__name__}/{proc.__name__}")
PY
        command -v ffmpeg >/dev/null 2>&1 \
            || { echo "ERRO: ffmpeg ausente -- a captura de video/audio nao roda" >&2; exit 1; }
    fi
    echo "=== bootstrap concluido $(date) ==="
    ;;

# =====================================================================================
capture)
# =====================================================================================
    # Captura do corpus Orcamento (5 eixos, PRECISA DE INTERNET -- a VM tem saida). O
    # destino e RELATIVO ao repo na VM de proposito: `build_corpus` grava
    # `meta["path"]` com o caminho que recebeu, e um caminho relativo continua valido
    # depois do round-trip para o repo local e de volta (um absoluto `/home/<user>/...`
    # nao continuaria). `ambiente` faz o cd para $REPO_DIR antes.
    #
    # ATENCAO: recapturar muda os item_ids (hash da URL) e INVALIDA os qrels -- e por
    # isso que esta fase e separada da avaliacao.
    ambiente
    echo "=== [capture] corpus do ensemble v2 -> $ENSEMBLE_CORPUS (bin: $ENSEMBLE_BIN_DIR) ==="
    python3 -m ensemble.build_corpus "$ENSEMBLE_CORPUS"
    wc -l "$ENSEMBLE_CORPUS"
    du -sh "$ENSEMBLE_BIN_DIR" || true
    echo "=== captura concluida $(date) ==="
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

    # O ensemble v2 tem driver PROPRIO (`ensemble/eval/run_eval.py`): ele monta os 5
    # especialistas nativos + A + B por import tardio, carregando UM EIXO POR VEZ, funde
    # por RRF e grava o comparativo por (sistema, eixo). Nao passa por `gcp.slice.eval`
    # nem pelas paginas do Bronze -- o corpus inteiro (records + binarios) ja esta no
    # repo da VM, posto la pela fase `capture` ou pelo upload da rodada de eval.
    if [ "${USA_ENSEMBLE:-0}" = "1" ]; then
        echo "=== [eval] ensemble v2 (5 especialistas -> RRF) vs. baselines A e B ==="
        mkdir -p "$(dirname "$OUT_CSV")" "$(dirname "$ENSEMBLE_LOG")"
        python3 -m ensemble.eval.run_eval \
            --corpus "$ENSEMBLE_CORPUS" \
            --queries "$ENSEMBLE_QUERIES" \
            --qrels "$ENSEMBLE_QRELS" \
            --out "$OUT_CSV" \
            --log "$ENSEMBLE_LOG"
        echo "=== [eval] CSV comparativo ==="
        cat "$OUT_CSV" || true
        echo "=== eval concluida $(date) ==="
        exit 0
    fi

    # Sem espaco pgvector nao ha Cloud SQL nesta rodada: nada de proxy nem de senha. O que
    # o ranqueador em memoria precisa sao as paginas, e elas vem do Bronze.
    EXTRA=()
    if [ "$USA_PGVECTOR" = "1" ]; then
        proxy
        credenciais
    else
        PG=()
        echo "--- $ARCH nao usa pgvector: sem proxy e sem credenciais de banco ---"
        baixar_paginas_do_bronze
        EXTRA=(--pages-dir "$PAGES_VM_DIR" --batch-size "$SPECIALIST_BATCH_SIZE")
    fi

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
            --device cuda "${EXTRA[@]}" "${PG[@]}"
    done

    echo "=== [eval] CSV comparativo ==="
    cat "$REPO_DIR/$OUT_CSV" || true

    if [ "$USA_PGVECTOR" != "1" ]; then
        echo "=== eval concluida $(date) ==="
        exit 0
    fi

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
    #
    # O corpus do ensemble (`data/govexe/ensemble_orcamento`) fica FORA do tarball pelo
    # mesmo motivo das paginas do ptbr_planos -- tamanho (ele carrega os videos das
    # janelas) -- e por um motivo a mais: o bootstrap EXTRAI o tarball por cima de
    # `~/residencia`, entao um corpus dentro dele sobrescreveria o que a rodada acabou de
    # subir. Quem o leva e `enviar_insumos_do_ensemble`, DEPOIS do bootstrap.
    run tar czf "$TMP_DIR/repo.tgz" -C "$REPO_DIR" \
        --exclude=.git --exclude='.venv*' --exclude=__pycache__ \
        --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.benchmarks \
        --exclude=results --exclude=logs --exclude=.cache \
        --exclude='data/govexe/ptbr_planos/pages' \
        --exclude='data/govexe/ptbr_planos/pdf_cache' \
        --exclude="$(dirname "$ENSEMBLE_CORPUS")" \
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
    if [ "$USA_PGVECTOR" != "1" ]; then
        echo "--- $ARCH nao avalia audio: nada de binarios de audio a enviar ---"
    elif [ -d "$REPO_DIR/$AUDIO_DIR" ]; then
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

# Insumos da arquitetura `ensemble`. Roda DEPOIS do bootstrap de proposito: o bootstrap
# extrai `repo.tgz` por cima de `~/residencia`, entao qualquer upload feito antes dele
# seria sobrescrito pelo tarball (a mesma armadilha que o guard por sentinela do bootstrap
# documenta). Os mecanismos sao os do mini-set de audio: arquivo-a-arquivo com destino
# RELATIVO (limitacao do pscp no Windows) e diretorio via tar + untar na VM.
enviar_insumos_do_ensemble() {
    if [ "$PHASE" = "eval" ]; then
        etapa "[5b/7] enviando os insumos do ensemble (queries, qrels, corpus + binarios)"
    else
        etapa "[5b/7] enviando os insumos do ensemble (queries e qrels)"
    fi

    # Queries e qrels sao pequenos e viajam SEMPRE: eles sao o gate humano da rodada, e a
    # versao que vale e a do repo local (editada a mao depois da fase `capture`).
    remoto "mkdir -p ~/residencia/$(dirname "$ENSEMBLE_QUERIES") ~/residencia/$(dirname "$ENSEMBLE_QRELS")"
    run gcloud compute scp "$REPO_DIR/$ENSEMBLE_QUERIES" "$VM:residencia/$ENSEMBLE_QUERIES" \
        --zone="$ZONE" --project="$PROJECT_ID" --quiet
    run gcloud compute scp "$REPO_DIR/$ENSEMBLE_QRELS" "$VM:residencia/$ENSEMBLE_QRELS" \
        --zone="$ZONE" --project="$PROJECT_ID" --quiet

    # O corpus so sobe quando a VM NAO vai captura-lo (fase `eval`): em `capture` e `all`
    # ele nasce na propria VM e subir um corpus velho so apagaria o novo.
    if [ "$PHASE" != "eval" ]; then
        echo "--- fase $PHASE: o corpus nasce na VM; nada de records/binarios a enviar ---"
        return 0
    fi
    remoto "mkdir -p ~/residencia/$(dirname "$ENSEMBLE_CORPUS")"
    run gcloud compute scp "$REPO_DIR/$ENSEMBLE_CORPUS" "$VM:residencia/$ENSEMBLE_CORPUS" \
        --zone="$ZONE" --project="$PROJECT_ID" --quiet
    run tar czf "$TMP_DIR/ensemble_bin.tgz" -C "$REPO_DIR/$ENSEMBLE_BIN_DIR" .
    run gcloud compute scp "$TMP_DIR/ensemble_bin.tgz" "$VM:ensemble_bin.tgz" \
        --zone="$ZONE" --project="$PROJECT_ID" --quiet
    remoto "mkdir -p ~/residencia/$ENSEMBLE_BIN_DIR && tar xzf ~/ensemble_bin.tgz -C ~/residencia/$ENSEMBLE_BIN_DIR"
}

# Volta do corpus capturado para o repo LOCAL -- a persistencia entre as duas rodadas
# pagas. Tem de rodar ANTES do teardown (a VM leva o corpus junto), e por isso tambem e
# chamada do resgate do teardown. Os binarios voltam num tar unico pela mesma limitacao
# do pscp da ida, com teto PROPRIO (eles sao grandes; o do CSV nao serve).
coletar_corpus_do_ensemble() {
    echo "--- trazendo o corpus capturado ($ENSEMBLE_CORPUS + $ENSEMBLE_BIN_DIR) ---"
    run mkdir -p "$REPO_DIR/$(dirname "$ENSEMBLE_CORPUS")" || true
    com_timeout "$RESGATE_TIMEOUT" \
        gcloud compute scp "$VM:residencia/$ENSEMBLE_CORPUS" "$REPO_DIR/$ENSEMBLE_CORPUS" \
            --zone="$ZONE" --project="$PROJECT_ID" --quiet || return 1
    remoto "tar czf ~/ensemble_bin.tgz -C ~/residencia/$ENSEMBLE_BIN_DIR ." || return 1
    com_timeout "$ENSEMBLE_RESGATE_TIMEOUT" \
        gcloud compute scp "$VM:ensemble_bin.tgz" "$TMP_DIR/ensemble_bin_volta.tgz" \
            --zone="$ZONE" --project="$PROJECT_ID" --quiet || return 1
    run mkdir -p "$REPO_DIR/$ENSEMBLE_BIN_DIR" || true
    run tar xzf "$TMP_DIR/ensemble_bin_volta.tgz" -C "$REPO_DIR/$ENSEMBLE_BIN_DIR" || return 1
    RESULTADOS_COLETADOS=1
    return 0
}

# Chamada no fluxo normal E dentro do teardown; por isso todo scp tem teto de tempo (um
# scp travado numa VM meio-bootada nao pode adiar a limpeza).
coletar_resultados() {
    # Na arquitetura `ensemble` o que tem de sobreviver depende da FASE: em `capture` o
    # resultado e o proprio corpus (nao existe CSV), em `all` sao os dois, em `eval` so o
    # CSV. Despachar aqui dentro mantem o resgate do teardown correto sem duplica-lo.
    if [ "${USA_ENSEMBLE:-0}" = "1" ]; then
        case "$PHASE" in
            capture) coletar_corpus_do_ensemble; return $? ;;
            all)     coletar_corpus_do_ensemble \
                        || echo "--- aviso: o corpus capturado nao voltou ---" >&2 ;;
        esac
    fi
    echo "--- trazendo $OUT_CSV e os logs da VM (teto: ${RESGATE_TIMEOUT}s por arquivo) ---"
    run mkdir -p "$REPO_DIR/$(dirname "$OUT_CSV")" "$REPO_DIR/logs" || true
    com_timeout "$RESGATE_TIMEOUT" \
        gcloud compute scp "$VM:residencia/$OUT_CSV" "$REPO_DIR/$OUT_CSV" \
            --zone="$ZONE" --project="$PROJECT_ID" --quiet || return 1
    com_timeout "$RESGATE_TIMEOUT" \
        gcloud compute scp "$VM:proxy.log" "$REPO_DIR/logs/proxy-$VM.log" \
            --zone="$ZONE" --project="$PROJECT_ID" --quiet || true
    if [ "${USA_ENSEMBLE:-0}" = "1" ]; then
        com_timeout "$RESGATE_TIMEOUT" \
            gcloud compute scp "$VM:residencia/$ENSEMBLE_LOG" "$REPO_DIR/$ENSEMBLE_LOG" \
                --zone="$ZONE" --project="$PROJECT_ID" --quiet || true
    fi
    RESULTADOS_COLETADOS=1
    return 0
}

# -------------------------------------------------------------------------------------
# Preflight (local, antes de criar qualquer recurso pago)
# -------------------------------------------------------------------------------------
etapa "rodada: arquitetura $ARCH | espacos: ${SPACES:-<nenhum>} | especialistas: ${SPECIALISTS:-<nenhum>} | dry-run: $DRY_RUN"
echo "    projeto=$PROJECT_ID  regiao=$REGION  instancia=$INSTANCE  vm=$VM"
echo "    bucket=$BUCKET  fonte=$SOURCE  dt=$DT  csv=$OUT_CSV"
if [ "$USA_ENSEMBLE" = "1" ]; then
    echo "    fase=$PHASE  corpus=$ENSEMBLE_CORPUS  bin=$ENSEMBLE_BIN_DIR"
    echo "    queries=$ENSEMBLE_QUERIES  qrels=$ENSEMBLE_QRELS  log=$ENSEMBLE_LOG"
fi

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
exige_relativo ENSEMBLE_CORPUS "$ENSEMBLE_CORPUS"
exige_relativo ENSEMBLE_BIN_DIR "$ENSEMBLE_BIN_DIR"
exige_relativo ENSEMBLE_QUERIES "$ENSEMBLE_QUERIES"
exige_relativo ENSEMBLE_QRELS "$ENSEMBLE_QRELS"
exige_relativo ENSEMBLE_LOG "$ENSEMBLE_LOG"

# O conjunto de avaliacao de audio entra em QUALQUER arquitetura, nao so nas que tem CLAP:
# o espaco qwen3vl ja indexa os mesmos clipes como `audio_asr` (Whisper -> texto), e e essa
# a baseline "A (ASR->texto)" que a tarefa de audio da spec (secao 5) compara com o CLAP
# nativo. Sem isto, `--arch A` indexava o audio e nunca o media.
#
# Gate humano: enquanto o mini-set nao for capturado e julgado, o diretorio nao existe e a
# rodada mede so a tarefa doc-visual -- aviso, nao erro.
#
# Arquitetura sem pgvector fica de fora disto: o unico braco de B hoje e o eixo DOCUMENTO
# (`doc_image`), e o registro nao lhe da especialista algum para audio -- anexar o conjunto
# de audio so derrubaria a rodada no `specialist_for_corpus`.
if [ "$USA_ENSEMBLE" = "1" ]; then
    echo "--- $ARCH: o corpus e o do ensemble v2 (5 eixos); EVAL_DIRS nao se aplica ---"
elif [ "$USA_PGVECTOR" != "1" ]; then
    echo "--- $ARCH: eixo documento apenas; o conjunto de audio fica fora da rodada ---"
elif [ -d "$REPO_DIR/$EVAL_AUDIO_DIR" ]; then
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

# Precondicoes do ensemble -- TODAS antes de provisionar: descobrir um qrels ausente com
# a L4 de pe seria pagar GPU para falhar (e os valores aqui sao de arquivo local, entao a
# checagem e gratuita). Em `--dry-run` viram aviso, como as demais precondicoes.
if [ "$USA_ENSEMBLE" = "1" ]; then
    if [ "$PHASE" != "capture" ]; then
        [ -f "$REPO_DIR/$ENSEMBLE_QUERIES" ] || precondicao_ausente \
            "a fase $PHASE avalia, mas nao existe $ENSEMBLE_QUERIES (queries do conjunto)."
        [ -f "$REPO_DIR/$ENSEMBLE_QRELS" ] || precondicao_ausente \
            "a fase $PHASE avalia, mas nao existe $ENSEMBLE_QRELS (gabarito revisado)."
    fi
    if [ "$PHASE" = "eval" ]; then
        # A fase `eval` NAO captura: o corpus tem de vir congelado da rodada `capture`.
        [ -f "$REPO_DIR/$ENSEMBLE_CORPUS" ] || precondicao_ausente \
            "--phase eval consome o corpus ja congelado, mas nao existe" \
            "$ENSEMBLE_CORPUS. Rode antes: bash gcp/slice/run_slice.sh --arch ensemble" \
            "--phase capture (ela traz records.jsonl + binarios para ca)."
        [ -d "$REPO_DIR/$ENSEMBLE_BIN_DIR" ] || precondicao_ausente \
            "--phase eval precisa dos binarios do corpus, mas nao existe" \
            "$ENSEMBLE_BIN_DIR. Rode antes: bash gcp/slice/run_slice.sh --arch ensemble" \
            "--phase capture."
    fi
else
    echo "    conjuntos de avaliacao: $EVAL_DIRS"
fi

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
if [ "$USA_PGVECTOR" = "1" ]; then
    provisionar_sql
else
    etapa "[1/7] Cloud SQL -- PULADO (a arquitetura $ARCH nao tem espaco pgvector)"
    echo "--- nenhuma instancia sera criada: SQL_TOCADA fica 0 e o teardown nem a procura ---"
fi
provisionar_vm
esperar_ssh
enviar_repo

etapa "[5/7] bootstrap da VM (venv, deps, torchaudio, librosa/soundfile)"
gancho_de_falha bootstrap
remoto "bash ~/slice_remote.sh bootstrap"

if [ "$USA_ENSEMBLE" = "1" ]; then
    # Ensemble v2: a etapa [6/7] nao e indexacao pgvector, e a CAPTURA do corpus (rede), e
    # a [7/7] e o driver proprio do ensemble. `--phase` liga/desliga cada uma; o teardown
    # por trap segue valendo nas tres fases.
    enviar_insumos_do_ensemble

    if [ "$PHASE" = "eval" ]; then
        etapa "[6/7] captura -- PULADA (--phase eval consome o corpus ja congelado)"
        echo "--- recapturar mudaria os item_ids e invalidaria o gabarito ---"
    else
        etapa "[6/7] captura do corpus do ensemble na VM (rede: RSS, PDF, yt-dlp, podcasts)"
        gancho_de_falha capture
        remoto "bash ~/slice_remote.sh capture"
    fi

    if [ "$PHASE" = "capture" ]; then
        etapa "[7/7] avaliacao -- PULADA (--phase capture apenas congela o corpus)"
        echo "--- proximo passo: revisar $ENSEMBLE_QRELS com os item_ids capturados e"
        echo "    rodar de novo com --phase eval ---"
    else
        etapa "[7/7] avaliacao do ensemble v2 (5 especialistas -> RRF) -> $OUT_CSV"
        gancho_de_falha eval
        remoto "bash ~/slice_remote.sh eval"
    fi
else
    if [ -z "$SPACES" ]; then
        etapa "[6/7] indexacao pgvector -- PULADA (a arquitetura $ARCH nao tem espaco)"
        echo "--- o especialista ($SPECIALISTS) embute o corpus na propria avaliacao ---"
    else
        for espaco in $SPACES; do
            etapa "[6/7] indexando o espaco $espaco (arquitetura $ARCH)"
            gancho_de_falha index
            remoto "bash ~/slice_remote.sh index $espaco"
        done
    fi

    etapa "[7/7] avaliacao (eval.py) -> $OUT_CSV"
    gancho_de_falha eval
    remoto "bash ~/slice_remote.sh eval"
fi

etapa "resultados"
coletar_resultados || echo "aviso: nao foi possivel trazer os resultados (o teardown tentara de novo)"

etapa "rodada da arquitetura $ARCH concluida -- o teardown vem a seguir"
exit 0
