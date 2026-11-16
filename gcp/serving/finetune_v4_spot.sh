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
#   [6/9] pairs: `finetune.make_pairs` (gerador template) ou, com --pairs-mode llm,
#         `finetune.gen_queries_llm` com SPLIT POR DOCUMENTO + o eval sintetico
#   [7/9] smoke: 2 passos reais com a VRAM instrumentada -- o gate e OOM-ou-nao
#   [8/9] finetune: `finetune/run_gcp_finetune.sh` (nDCG@10 do A no v4 ANTES e DEPOIS)
#         e, em --pairs-mode llm, a 2a medicao (conjunto sintetico) sobre o MESMO treino
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
#   # iteracao "pares via LLM + eval maior" (split por documento; ver o bloco PAIRS_MODE):
#   bash gcp/serving/finetune_v4_spot.sh --gcs-out gs://.../qwen3vl-qlora-llm --pairs-mode llm
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

# -------------------------------------------------------------------------------------
# PAIRS_MODE -- de onde vem o sinal de treino (e, de quebra, o 2o conjunto de avaliacao)
#
#   template (DEFAULT, o fluxo provado): `finetune.make_pairs` com o gerador
#     deterministico sobre o corpus inteiro, e UMA avaliacao (as 7 queries humanas).
#
#   llm: `finetune.gen_queries_llm` (Qwen2.5-7B-Instruct) com SPLIT POR DOCUMENTO --
#     a resposta as duas fraquezas da 1a rodada (loss do InfoNCE colapsando em poucos
#     passos por memorizacao de 88 pares; e 7 queries de eval = intervalo largo):
#
#       EVAL_DOCS = os doc_id do qrels de eval (derivados NA VM do arquivo staged).
#       treino    = consultas do LLM sobre os docs FORA de EVAL_DOCS (--exclude-docs):
#                   nenhum doc de avaliacao e alvo positivo de treino.
#       eval      = DOIS conjuntos, ambos ancorados so em EVAL_DOCS:
#                   (1) ancora-ouro: as 7 queries humanas + seus 24 qrels (headline);
#                   (2) SINTETICO: consultas do LLM sobre os 22 docs de eval
#                       (--include-docs), ~k por doc -- N maior, CI menor.
#
#     A geracao roda em SUBPROCESSO que SAI antes do treino: os ~15GB do 7B saem da VRAM
#     sozinhos, sem unload manual, e nao concorrem com o QLoRA.
# -------------------------------------------------------------------------------------
PAIRS_MODE="${PAIRS_MODE:-template}"            # template | llm
GEN_K="${GEN_K:-3}"                             # consultas por doc na geracao por LLM
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

DRY_RUN=0

# -------------------------------------------------------------------------------------
# Estado do teardown (o que ja foi tocado na nuvem)
# -------------------------------------------------------------------------------------
ZONE=""                 # zona onde a VM subiu de fato
ZONAS_TENTADAS=""       # toda zona em que um `create` foi tentado (o teardown varre todas)
UPLOAD_FEITO=0          # 1 depois que o adaptador chegou em $GCS_OUT
TMP_DIR=""
SAIDA_DO_DELETE=""      # saida do ultimo `instances delete` que falhou (vira o diagnostico
                        # de auth do teardown)

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
        --pairs-mode)     PAIRS_MODE="$2"; shift 2 ;;
        --gen-k)          GEN_K="$2"; shift 2 ;;
        --gen-model)      GEN_MODEL="$2"; shift 2 ;;
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

# -------------------------------------------------------------------------------------
# AUTH -- o token do gcloud e um recurso que EXPIRA no meio da rodada
#
# A conta que dispara o script pode estar sob politica de reauth periodica: o access token
# vence e o gcloud nao renova sozinho (precisa de `gcloud auth login` interativo). Numa
# rodada longa isso nao aparece no comeco -- aparece LA NA FRENTE, no upload do adaptador
# e, pior, no `instances delete` do teardown, que falha e deixa a L4 DE PE cobrando.
#
# `token_de_auth_ok` exercita o token e e a unica forma de tocar auth neste script. Em
# `--dry-run` ela so ANUNCIA a checagem: `print-access-token` e chamada de REDE, e o ensaio
# nao executa um gcloud sequer (mesma regra do `run`/`existe`).
token_de_auth_ok() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] gcloud auth print-access-token --project=%s   # (checagem de rede: o ensaio nao a executa)\n' \
            "$PROJECT"
        return 0
    fi
    gcloud auth print-access-token --project="$PROJECT" >/dev/null 2>&1
}

# Assinatura de token vencido na saida de um gcloud que falhou. E texto do proprio gcloud:
# "Reauthentication failed. cannot prompt during non-interactive execution."
falha_e_de_auth() {
    case "$1" in
        *"Reauthentication failed"*|*"cannot prompt during non-interactive execution"*)
            return 0 ;;
    esac
    return 1
}

# Assinatura de recurso INEXISTENTE na saida de um `instances delete` que falhou -- tambem
# texto do proprio gcloud: "The resource ... was not found" / "Could not fetch resource:".
# Nao se confunde com auth: token vencido fala de reauth, nao de recurso. O teardown varre
# TODAS as zonas tentadas, e as que nunca tiveram VM respondem exatamente isto.
delete_e_inexistente() {
    case "$1" in
        *"was not found"*|*"Could not fetch resource"*)
            return 0 ;;
    esac
    return 1
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
#      falha transitoria, e engoli-la com `|| true` seria vazar recurso pago. "not found"
#      e a excecao: nao ha o que apagar nessa zona, entao e sucesso e nao retry.
#   5. Conferencia final que ASSERTA: se algo sobreviveu, o script grita LEAK, imprime o
#      delete manual e sai com status != 0.
#   6. A saida de cada delete que falha e CAPTURADA (e reemitida): e nela que o teardown
#      le se a causa foi o token vencido -- e so nesse caso imprime o banner de
#      recuperacao manual, com os comandos exatos.
# -------------------------------------------------------------------------------------
# Banner do delete que falhou por AUTH: a VM esta de pe, o `gcloud` desta maquina nao
# consegue mais falar com a API, e o operador precisa dos comandos prontos -- inclusive o
# resgate do adaptador, que morre junto com a VM se ele apagar sem subir antes.
banner_de_recuperacao() {
    local zona="$1"
    {
        echo ""
        echo "=========== RECUPERACAO MANUAL NECESSARIA (auth do gcloud expirou) ==========="
        echo "  1. gcloud auth login"
        echo "  2. (resgatar adaptador, se o upload nao rodou:)"
        echo "     gcloud compute ssh $VM --zone=$zona --project=$PROJECT --command \\"
        echo "       \"cd ~/residencia && gcloud storage rsync -r finetune/runs/$BASE-$METHOD \\"
        echo "        $GCS_OUT/adapter --project=$PROJECT\""
        echo "  3. gcloud compute instances delete $VM --zone=$zona --project=$PROJECT --quiet"
        echo "  4. gcloud compute instances list --project=$PROJECT   # tem de dar 'Listed 0 items.'"
        echo "=============================================================================="
    } >&2
}

apagar_com_retry() {
    local rotulo="$1"; shift
    local tentativa=1
    local saida rc_del
    SAIDA_DO_DELETE=""
    while [ "$tentativa" -le "$TEARDOWN_RETRIES" ]; do
        # No ensaio nao ha saida a capturar: `run` imprime o comando e devolve 0, como em
        # todo o resto do script.
        if [ "$DRY_RUN" = "1" ]; then
            run "$@"
            return 0
        fi
        # Capturar (e reemitir) em vez de deixar passar direto: o teardown precisa LER o
        # motivo da falha para distinguir "operacao em andamento" de "token vencido".
        saida="$("$@" 2>&1)"
        rc_del=$?
        if [ -n "$saida" ]; then
            printf '%s\n' "$saida" >&2
        fi
        if [ "$rc_del" = "0" ]; then
            return 0
        fi
        # "not found" nao e falha a retentar: e a resposta de uma zona onde a VM nunca
        # existiu -- e o teardown passa por TODAS as zonas tentadas, nao so pela que pegou.
        # Retentar aqui queimaria TEARDOWN_RETRIES x TEARDOWN_ESPERA por zona vazia
        # enquanto a VM de verdade, na zona seguinte, continua cobrando.
        if delete_e_inexistente "$saida"; then
            echo "--- $rotulo: nada a apagar nessa zona (VM nao existe aqui) ---"
            return 0
        fi
        SAIDA_DO_DELETE="$SAIDA_DO_DELETE
$saida"
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

    # Exercitar o token ANTES dos deletes: se a credencial so estava stale, isto a renova e
    # a limpeza segue normal; se ja venceu de vez, pelo menos o aviso sai antes do retry e
    # o banner cobre o resto.
    if ! token_de_auth_ok; then
        echo "--- aviso: o gcloud desta maquina nao tem token valido AGORA;" \
             "o resgate e os deletes podem falhar (veja o banner no fim) ---" >&2
    fi

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
        if ! apagar_com_retry "delete da VM em $zona" \
                gcloud compute instances delete "$VM" --zone="$zona" \
                    --project="$PROJECT" --quiet; then
            # Falha de auth nao se resolve com retry nem com a conferencia: so com um
            # `gcloud auth login` humano. Entao o banner, aqui, com a VM e a zona reais.
            if falha_e_de_auth "$SAIDA_DO_DELETE"; then
                banner_de_recuperacao "$zona"
            fi
        fi
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
PAIRS_MODE='$PAIRS_MODE'
GEN_K='$GEN_K'
GEN_MODEL='$GEN_MODEL'
BRANCH='$BRANCH'
ENV

    cat > "$TMP_DIR/finetune_remote.sh" <<'REMOTE'
#!/usr/bin/env bash
# Gerado por gcp/serving/finetune_v4_spot.sh -- roda DENTRO da VM L4. Fase em $1.
set -euo pipefail
FASE="${1:?uso: finetune_remote.sh <bootstrap|stage|pairs|smoke|finetune|eval_syn|upload>}"
# shellcheck disable=SC1091
source "$HOME/finetune_env.sh"
echo "=== [vm $(hostname)] fase=$FASE $(date) ==="

REPO_DIR="$HOME/residencia"
RECORDS_REL="$CORPUS_DIR/records.jsonl"
BIN_REL="$CORPUS_DIR/bin"
QUERIES_REL="ensemble/eval/queries_v4.txt"
QRELS_REL="ensemble/eval/qrels_v4.tsv"
PAIRS_REL="finetune/data/pairs.jsonl"
# Split por DOCUMENTO + eval sintetico (so em PAIRS_MODE=llm). `EVAL_DOCS_REL` e DERIVADO
# do qrels staged -- nenhum arquivo de dados novo entra no repo nem no bucket.
EVAL_DOCS_REL="finetune/data/eval_docs.txt"
SYN_PAIRS_REL="finetune/data/pairs_eval_sintetico.jsonl"
SYN_QUERIES_REL="finetune/data/queries_syn.txt"
SYN_QRELS_REL="finetune/data/qrels_syn.tsv"
# Venv que `finetune/run_gcp_finetune.sh` cria (mesmo default dele): a 2a medicao precisa
# do torch/peft que estao LA, e nao no python do sistema.
VENV_REL=".venv-gcp"
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

    if [ "${PAIRS_MODE:-template}" != "llm" ]; then
        echo "=== [pairs] $RECORDS_REL -> $PAIRS_REL (gerador template) ==="
        PYTHONPATH="$REPO_DIR" python3 -m finetune.make_pairs \
            --records "$RECORDS_REL" --out "$PAIRS_REL"
        [ -s "$PAIRS_REL" ] || { echo "ERRO: $PAIRS_REL vazio" >&2; exit 1; }
        echo "--- pares gerados: $(wc -l < "$PAIRS_REL") ---"
        echo "=== pairs concluido $(date) ==="
        exit 0          # guarda: o fluxo atual termina AQUI, intacto (uma fase por invocacao)
    fi

    # -- PAIRS_MODE=llm: split por documento + eval sintetico ---------------------------
    # Deps do gerador. O DLVM pytorch normalmente ja traz transformers; quando nao traz,
    # instala SEM `--no-deps`: o torch e EXTRA do transformers, nao dep core, entao o pip
    # resolve so as deps pure-python (regex, filelock, requests, pyyaml, packaging, tqdm,
    # tokenizers, safetensors, huggingface_hub) e NAO arrasta um torch proprio para
    # ~/.local. Com `--no-deps` o import morria em `import regex as re`
    # (transformers/utils/auto_docstring.py). Ja o accelerate HARD-depende de torch, logo
    # nele o `--no-deps` fica -- mais `psutil`, a unica dep leve dele nao coberta pelo
    # transformers --, protegendo o torch da imagem (a mesma armadilha que
    # `prefer_host_torch` desfaz dentro do run_gcp_finetune.sh).
    python3 -c "import transformers" 2>/dev/null \
        || python3 -m pip install --user -q transformers
    python3 -c "import accelerate" 2>/dev/null \
        || python3 -m pip install --user -q --no-deps accelerate psutil
    # jinja2 do host (DLVM) e 3.0.3; o `apply_chat_template` do tokenizer exige >=3.1.0
    # (ImportError na geracao, DEPOIS de carregar o modelo). Guard por VERSAO, nao por import:
    # o 3.0.3 importa, so e velho demais. user-site vence o dist-packages da imagem.
    python3 -c "import jinja2; v=tuple(int(x) for x in jinja2.__version__.split('.')[:2]); exit(0 if v>=(3,1) else 1)" 2>/dev/null \
        || python3 -m pip install --user -q "jinja2>=3.1.0"

    # Reparo do torchaudio ABI-incompativel da imagem DLVM (pytorch-2-9-cu129), ESPELHO do
    # `ensure_torchaudio` de finetune/run_gcp_finetune.sh. Duplicacao consciente
    # (ref: run_gcp_finetune.sh:238 ensure_torchaudio): o caminho de treino ja esta provado
    # e nao pode regredir por uma refatoracao para helper compartilhado.
    # Sintoma: `OSError: .../torchaudio/lib/_torchaudio.abi3.so: undefined symbol:
    # torch_library_impl`, disparado pelo transformers ao RESOLVER a classe do modelo em
    # `AutoModelForCausalLM.from_pretrained`. A geracao roda no python do SISTEMA, ANTES do
    # venv do treino -- isto e, antes do `ensure_torchaudio` que faria esse reparo --, por
    # isso ele precisa acontecer aqui tambem. Sem venv, o pip e sempre `--user`: o user-site
    # tem prioridade sobre o dist-packages quebrado da imagem.
    if python3 -c "import torchaudio" 2>/dev/null; then
        echo "--- torchaudio ok (sem reparo) ---"
    else
        echo "--- torchaudio nao importa (ABI incompativel?): reinstalando casado com o torch ---"
        TA_VER="$(python3 -c "import torch;print(torch.__version__)" 2>/dev/null || true)"
        TA_VER="${TA_VER%%+*}"
        CU_RAW="$(python3 -c "import torch;print(torch.version.cuda or str())" 2>/dev/null || true)"
        CU="cu${CU_RAW//./}"
        if [ -z "$TA_VER" ]; then
            echo "aviso: nao foi possivel reparar torchaudio (siga; o erro aparecera no load se for necessario)"
        else
            echo "--- pip install torchaudio==$TA_VER (index: $CU) ---"
            python3 -m pip install --user -q --no-deps --force-reinstall "torchaudio==$TA_VER" \
                    --index-url "https://download.pytorch.org/whl/$CU" \
                && python3 -c "import torchaudio" 2>/dev/null \
                && echo "--- torchaudio reparado (import ok) ---" \
                || echo "aviso: nao foi possivel reparar torchaudio (siga; o erro aparecera no load se for necessario)"
        fi
    fi

    # 1. EVAL_DOCS: os doc_id julgados do conjunto de avaliacao, DERIVADOS do qrels que o
    #    `stage` baixou (3a coluna do TREC). Nenhum arquivo de dados novo foi criado.
    #    O mkdir e necessario: quem cria `finetune/data/` no fluxo atual e o proprio
    #    `make_pairs` (ele faz `parents=True` no `--out`), e aqui o awk escreve ANTES.
    mkdir -p "$(dirname "$EVAL_DOCS_REL")"
    awk '$0 !~ /^[[:space:]]*#/ && NF >= 3 {print $3}' "$QRELS_REL" | sort -u \
        > "$EVAL_DOCS_REL"
    [ -s "$EVAL_DOCS_REL" ] \
        || { echo "ERRO: nenhum doc_id em $QRELS_REL -- o split por documento ficaria vazio" >&2; exit 1; }
    echo "=== [pairs/llm] docs de AVALIACAO (banidos como alvo de treino): $(wc -l < "$EVAL_DOCS_REL") ==="

    # 2. TREINO: consultas do LLM sobre os docs FORA de EVAL_DOCS. Subprocesso que SAI ao
    #    final -- a VRAM do 7B volta inteira antes do smoke/treino, sem unload manual.
    echo "=== [pairs/llm] treino: $RECORDS_REL (sem os docs de eval) -> $PAIRS_REL | k=$GEN_K modelo=$GEN_MODEL ==="
    PYTHONPATH="$REPO_DIR" python3 -m finetune.gen_queries_llm \
        --records "$RECORDS_REL" --out "$PAIRS_REL" --k "$GEN_K" --model "$GEN_MODEL" \
        --exclude-docs "$EVAL_DOCS_REL"
    [ -s "$PAIRS_REL" ] || { echo "ERRO: $PAIRS_REL vazio" >&2; exit 1; }
    echo "--- pares de treino (LLM): $(wc -l < "$PAIRS_REL") ---"
    # Guard do vazamento, agora sobre o ARQUIVO e nao sobre a intencao: nenhum pos_id dos
    # pares de treino pode ser um doc de avaliacao.
    PYTHONPATH="$REPO_DIR" python3 - "$PAIRS_REL" "$EVAL_DOCS_REL" <<'PY'
import json, sys
pares, docs = sys.argv[1], sys.argv[2]
proibidos = {ln.strip() for ln in open(docs, encoding="utf-8") if ln.strip()}
vazou = {json.loads(ln)["pos_id"] for ln in open(pares, encoding="utf-8") if ln.strip()}
vazou &= proibidos
if vazou:
    raise SystemExit(f"ERRO: {len(vazou)} doc(s) de eval viraram positivo de treino: "
                     f"{sorted(vazou)[:5]}")
print("--- split por documento conferido: 0 doc de eval como positivo de treino ---")
PY

    # 3. EVAL SINTETICO: consultas do LLM sobre SO os docs de avaliacao (N maior, CI
    #    menor). Cada consulta -> o doc que a gerou, com label 1.
    echo "=== [pairs/llm] eval sintetico: so os docs de eval -> $SYN_PAIRS_REL | k=$GEN_K ==="
    PYTHONPATH="$REPO_DIR" python3 -m finetune.gen_queries_llm \
        --records "$RECORDS_REL" --out "$SYN_PAIRS_REL" --k "$GEN_K" --model "$GEN_MODEL" \
        --include-docs "$EVAL_DOCS_REL"
    [ -s "$SYN_PAIRS_REL" ] || { echo "ERRO: $SYN_PAIRS_REL vazio" >&2; exit 1; }

    # 4. pairs.jsonl -> o par (queries.txt, qrels.tsv) que o hook de avaliacao le.
    python3 - "$SYN_PAIRS_REL" "$SYN_QUERIES_REL" "$SYN_QRELS_REL" <<'PY'
import json, sys
pares, saida_queries, saida_qrels = sys.argv[1], sys.argv[2], sys.argv[3]
linhas = {}
for linha in open(pares, encoding="utf-8"):
    if not linha.strip():
        continue
    r = json.loads(linha)
    linhas[f"syn-{r['id']}"] = (r["query"], r["pos_id"])     # `id` ja e unico por par
with open(saida_queries, "w", encoding="utf-8") as q, \
     open(saida_qrels, "w", encoding="utf-8") as g:
    for qid, (texto, pos_id) in linhas.items():
        q.write(f"{qid}\t{texto}\n")
        g.write(f"{qid}\t0\t{pos_id}\t1\n")
print(f"--- eval sintetico: {len(linhas)} consultas sobre "
      f"{len({p for _, p in linhas.values()})} doc(s) de avaliacao ---")
PY
    [ -s "$SYN_QUERIES_REL" ] && [ -s "$SYN_QRELS_REL" ] \
        || { echo "ERRO: o eval sintetico saiu vazio" >&2; exit 1; }
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
eval_syn)
# =====================================================================================
    # A 2a MEDICAO do MESMO adaptador (PAIRS_MODE=llm): nDCG@10 antes/depois no conjunto
    # SINTETICO. O hook que o train_lora dispara so aceita um conjunto por treino, e
    # treinar de novo so para medir o segundo seria pagar a GPU duas vezes pela mesma
    # pergunta -- entao aqui e 1 TREINO, 2 MEDICOES: `finetune.eval_ensemble` roda a mesma
    # funcao `run_ensemble_eval_before_after` sobre o adaptador ja salvo em disco.
    #
    # A saida e APENDADA ao $TRAIN_LOG, que a fase `upload` leva para o bucket: os dois
    # deltas (ancora-ouro humana e sintetico) ficam no mesmo log da rodada.
    cd "$REPO_DIR"
    [ -s "$SYN_QUERIES_REL" ] && [ -s "$SYN_QRELS_REL" ] \
        || { echo "ERRO: eval sintetico ausente -- rode a fase pairs com PAIRS_MODE=llm" >&2; exit 1; }
    [ -d "$OUT_REL" ] \
        || { echo "ERRO: $OUT_REL ausente -- sem adaptador nao ha 'depois' a medir" >&2; exit 1; }

    # O torch/peft/Qwen3-VL estao no venv do treino, nao no python do sistema.
    if [ -f "$VENV_REL/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_REL/bin/activate"
    else
        echo "--- aviso: $VENV_REL ausente; tentando no python do sistema ---" >&2
    fi

    echo "=== [eval_syn] 2a medicao (SINTETICO): $(wc -l < "$SYN_QUERIES_REL") queries sobre os docs de eval ===" \
        | tee -a "$TRAIN_LOG"
    PYTHONPATH="$REPO_DIR" HF_HOME="$HOME/.cache/huggingface" \
        python3 -m finetune.eval_ensemble \
            --corpus "$RECORDS_REL" --queries "$SYN_QUERIES_REL" --qrels "$SYN_QRELS_REL" \
            --adapter "$OUT_REL" --base "$BASE" --method "$METHOD" \
            --tag "sintetico (consultas do LLM -- NAO e o gabarito humano)" \
        2>&1 | tee -a "$TRAIN_LOG"

    echo "--- os DOIS antes/depois desta rodada (ancora-ouro humana + sintetico) ---"
    grep -i "ndcg" "$TRAIN_LOG" || echo "    (nenhuma linha de nDCG no log)"
    echo "=== eval_syn concluido $(date) ==="
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
echo "    pares:  pairs_mode=$PAIRS_MODE gen_k=$GEN_K gen_model=$GEN_MODEL"
if [ "$PAIRS_MODE" = "llm" ]; then
    echo "            split por documento: os docs do qrels de eval NAO viram positivo de treino"
    echo "            avaliacao DUPLA: ancora-ouro (humana) + sintetica (LLM sobre os docs de eval)"
fi
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
# PAIRS_MODE decide a fase de pares INTEIRA na VM. Um valor errado so apareceria la,
# depois da VM paga -- e o `case` remoto cairia no fluxo template em silencio, treinando
# a rodada errada.
case "$PAIRS_MODE" in
    template|llm) ;;
    *) echo "ERRO: PAIRS_MODE invalido: '$PAIRS_MODE' (use template ou llm)" >&2
       echo "    template = gerador deterministico sobre o corpus inteiro (fluxo atual)" >&2
       echo "    llm      = consultas do LLM com split por documento + eval sintetico" >&2
       exit 2 ;;
esac

# GEN_K multiplica o custo da geracao (k consultas por doc, ~150 docs) -- um "0" produziria
# zero pares e o treino so descobriria isso na VM.
case "$GEN_K" in
    ''|*[!0-9]*) echo "ERRO: GEN_K tem de ser inteiro >= 1 (recebido: '$GEN_K')" >&2; exit 2 ;;
esac
[ "$GEN_K" -ge 1 ] || { echo "ERRO: GEN_K tem de ser >= 1 (recebido: '$GEN_K')" >&2; exit 2; }
[ -n "$GEN_MODEL" ] || { echo "ERRO: GEN_MODEL vazio (ex.: Qwen/Qwen2.5-7B-Instruct)" >&2; exit 2; }

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

# TOKEN VALIDO, aqui, antes de existir recurso pago. Um token ja vencido na largada nao
# impede a VM de subir (o `create` falharia so se a credencial sumisse de vez), mas condena
# o fim da rodada: o upload do adaptador e o delete do teardown passam pelo gcloud DESTA
# maquina. Falhar agora custa zero; falhar depois custou uma L4 orfa cobrando.
# Em `--dry-run` a checagem e apenas anunciada (ver `token_de_auth_ok`).
if ! token_de_auth_ok; then
    echo "ERRO: gcloud sem token valido. Rode 'gcloud auth login' (a conta ativa tem" \
         "politica de reauth que pode expirar em runs longos)." >&2
    exit 2
fi

# Aviso, nao barreira: contas academicas/institucionais costumam carregar a politica de
# reauth que derrubou a rodada anterior NO MEIO dela -- o token passa aqui e vence uma hora
# depois. Quem quer imunidade usa service account; quem so quer rodar segue em frente.
if [ "$DRY_RUN" != "1" ]; then
    CONTA_ATIVA="$(gcloud config get-value account 2>/dev/null || true)"
    case "$CONTA_ATIVA" in
        *.edu|*.edu.br|*discente*|*aluno*)
            echo "AVISO: conta $CONTA_ATIVA pode exigir reauth no meio de runs longos;" \
                 "para runs >1h considere uma service account." >&2 ;;
    esac
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

if [ "$PAIRS_MODE" = "llm" ]; then
    etapa "[6/9] pares de treino por LLM ($GEN_MODEL, k=$GEN_K) + eval sintetico"
    echo "--- split por documento: EVAL_DOCS vem do qrels staged (cut da 3a coluna) ---"
    echo "--- treino = docs FORA de EVAL_DOCS | eval sintetico = SO os docs de EVAL_DOCS ---"
else
    etapa "[6/9] pares de treino (finetune.make_pairs, gerador template)"
fi
remoto "bash ~/finetune_remote.sh pairs"

etapa "[7/9] smoke de VRAM: 2 passos reais na L4 (gate de OOM antes do treino inteiro)"
remoto "bash ~/finetune_remote.sh smoke"

etapa "[8/9] finetune QLoRA na VM (o log traz o nDCG@10 do A no v4 ANTES e DEPOIS)"
gancho_de_falha finetune
remoto "bash ~/finetune_remote.sh finetune"

# 2a MEDICAO do MESMO adaptador (1 treino, 2 medicoes) -- so no modo llm, que e quem
# produz o conjunto sintetico. NAO-FATAL de proposito: perder a medicao extra nao pode
# custar o upload do adaptador, que e o entregavel da rodada paga.
if [ "$PAIRS_MODE" = "llm" ]; then
    echo "--- [8/9b] 2a medicao: nDCG@10 antes/depois no conjunto SINTETICO ---"
    remoto "bash ~/finetune_remote.sh eval_syn" \
        || echo "--- aviso: a 2a medicao falhou; a ancora-ouro humana ja esta no log ---" >&2
fi

etapa "[9/9] persistindo o adaptador em $GCS_OUT (ANTES do teardown)"
gancho_de_falha upload
subir_resultado_para_gcs

etapa "finetune v4 concluido -- o teardown vem a seguir"
echo "    adaptador: $GCS_OUT/adapter/"
echo "    log do treino (antes/depois): $GCS_OUT/finetune_train.log"
exit 0
