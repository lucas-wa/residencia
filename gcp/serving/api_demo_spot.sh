#!/usr/bin/env bash
# =====================================================================================
# DEMO EFEMERA do subsistema [4] (API retrieval-only) numa L4 SPOT: a VM CONSTROI o
# indice com o adaptador finetunado, SOBE a API (uvicorn) e a maquina local bate
# `/consulta` DE FORA, por port-forward. Teardown GARANTIDO no fim.
#
#   [1/8] preflight local (ADAPTER, INDEX_URI, chave efemera, auth) -- nada pago ainda
#   [2/8] cria a VM L4 em SPOT, com fallback de zona            (a partir daqui ha custo)
#   [3/8] espera o SSH e envia repo + scripts
#   [4/8] bootstrap: untar do repo + venv-gcp (deps do Qwen3-VL)
#   [5/8] stage: baixa o corpus v4 ($GCS_CORPUS) E o adaptador ($ADAPTER_URI)
#   [6/8] build_index: `python -m api.build_index` na GPU -> sobe o indice para $INDEX_URI
#   [7/8] serve: `uvicorn api.app:app` em background NA VM + espera o /health responder
#   [8/8] prova DE FORA: port-forward local -> `/health` e 4 `/consulta` pt-BR via curl
#   [trap] teardown: fecha o tunel LOCAL, tenta o upload de novo, apaga a VM e ASSERTA
#          que nada sobrou
#
# O QUE SOBREVIVE A RODADA E SO O INDICE EM $INDEX_URI -- o artefato de runtime que
# faltava ao [4] (records.jsonl + matrix.npy + index_meta.json). Ele NAO e recurso
# cobrando: e um diretorio no bucket, e vira o `INDEX_PATH` de qualquer serving futuro
# (Cloud Run com `min-instances=0`, outra VM, o que for). A VM e descartavel; a API fica
# de pe so enquanto as consultas de prova rodam.
#
# -------------------------------------------------------------------------------------
# O QUE ESTE SCRIPT NAO FAZ (de proposito)
#
# Ele NAO reimplementa a API. `api/app.py`, `api/retriever.py`, `api/build_index.py` e
# `api/index_io.py` ja estao prontos e testados (tests/test_api.py); aqui so se monta o
# runtime que falta -- o indice -- e se DEMONSTRA o servico respondendo de fora.
#
# Ele tambem NAO abre porta na nuvem: a VM escuta em 127.0.0.1 e a unica forma de chegar
# nela e o tunel SSH que este script abre e fecha. Nenhuma regra de firewall, nenhum IP
# publico exposto, nenhuma chave vazando para a internet.
#
# PORTADO DE `gcp/serving/finetune_v4_spot.sh` (que por sua vez o portou de
# capture_v4_spot.sh / gcp/slice/run_slice.sh): as vars de estado, `token_de_auth_ok`,
# `falha_e_de_auth`, `delete_e_inexistente`, `banner_de_recuperacao`, `apagar_com_retry`,
# a funcao `teardown()` inteira (com a conferencia anti-leak), o preflight de auth, o
# guard de prefixo no GCS, `--instance-termination-action=DELETE` e o trap em EXIT. A
# COPIA e deliberada: o finetune esta rodando e refatorar os dois para um
# `gcp/serving/spot_common.sh` agora seria mexer na fiacao de teardown no meio de uma
# rodada paga. O commom pode vir depois, com os dois scripts parados.
#
# O QUE MUDA em relacao ao finetune: as fases de trabalho (`pairs|smoke|finetune|eval_syn`
# viram `build_index|serve`), o artefato que se resgata no teardown (o INDICE, nao o
# adaptador) e o port-forward LOCAL -- processo desta maquina, que o teardown tambem tem
# de matar para nao deixar tunel orfao.
#
# -------------------------------------------------------------------------------------
# RUN-BOOK
#
#   # ensaio completo, sem tocar a nuvem (nenhum gcloud e executado):
#   bash gcp/serving/api_demo_spot.sh --dry-run
#
#   # demo de verdade (L4 Spot; guarde a saida):
#   bash gcp/serving/api_demo_spot.sh \
#        --adapter gs://residencia-511517-corpus-v4/finetune/qwen3vl-qlora-llm/adapter \
#        2>&1 | tee logs/api-demo-$(date +%F).log
#
#   # conferir que NADA sobrou de pe (a resposta certa e "Listed 0 items."):
#   gcloud compute instances list --project=residencia-511517
#
#   # o que a rodada deixou no bucket (o indice do [4]):
#   gcloud storage ls -r gs://residencia-511517-corpus-v4/api/indice_v4 --project=residencia-511517
#
#   # servir esse indice em outro lugar (e o mesmo INDEX_PATH do Cloud Run):
#   INDEX_PATH=gs://residencia-511517-corpus-v4/api/indice_v4 ADAPTER_PATH=... \
#       API_KEY=... uvicorn api.app:app
#
#   # provar o teardown sem esperar o indice (falha injetada):
#   API_DEMO_FAIL_AT=serve bash gcp/serving/api_demo_spot.sh --dry-run
#
# PROJETO: toda chamada gcloud leva `--project=$PROJECT` explicito. Este script NUNCA roda
# `gcloud config set project`: o default do operador e dele.
# =====================================================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

PROJECT="${PROJECT:-residencia-511517}"

# ENTRADA 1: prefixo do corpus v4 no GCS, tal como `capture_v4_spot.sh` o deixou --
# `records.jsonl` e `bin/<eixo>/`. O indice embedda ESTES itens.
GCS_CORPUS="${GCS_CORPUS:-gs://residencia-511517-corpus-v4/ensemble/v4}"

# ENTRADA 2: o adaptador PEFT da Semana 10, tal como `finetune_v4_spot.sh` o deixou. E o
# MESMO adaptador que embedda os itens (build_index) e as queries (serving): se os dois
# divergissem, o `index_meta.json` faria o startup da API falhar -- ver `api/app.py`,
# `check_adapter_do_indice`.
ADAPTER_URI="${ADAPTER_URI:-gs://residencia-511517-corpus-v4/finetune/qwen3vl-qlora-llm/adapter}"

# SAIDA no GCS: o indice. UNICO artefato que sobrevive a rodada.
INDEX_URI="${INDEX_URI:-gs://residencia-511517-corpus-v4/api/indice_v4}"

BRANCH="${BRANCH:-feat/semana10-finetune-denso}"   # branch esperada no checkout local

VM="${VM:-govexe-api-demo}"             # nome PROPRIO: nao colide com govexe-finetune-v4,
                                        # govexe-capture-v4 nem govexe-embed
# Mesma ordem do finetune/capture: northamerica-northeast2-a foi a unica com capacidade de
# L4 na rodada do objetivo #2. O pool SPOT e outro, mas a ordem continua valendo.
ZONES="${ZONES:-northamerica-northeast2-a us-central1-a us-central1-b us-east1-c us-east4-a us-west1-a}"

MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"
ACCELERATOR="${ACCELERATOR:-nvidia-l4}"
# 150GB: corpus v4 (~1GB com os binarios) + pesos do Qwen3-VL-2B (~5GB) + venv + indice.
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-150GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
# SPOT e o default: a demo dura minutos e o indice ja esta no bucket quando a API sobe.
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"

# Arvore do corpus v4 no repo da VM (relativa a raiz, como todo caminho de dados daqui):
# `meta["path"]` dos records espera encontrar exatamente isto.
CORPUS_DIR="${CORPUS_DIR:-data/govexe/ensemble_orcamento_v4}"
# Onde o indice e escrito NA VM (mesmo default de `api/build_index.py`, DEFAULT_OUT).
INDICE_DIR="${INDICE_DIR:-data/govexe/indice_api_v4}"

DEVICE="${DEVICE:-cuda}"                # device do embedder na VM (a L4 existe para isso)
PORTA="${PORTA:-8080}"                  # porta do uvicorn DENTRO da VM (so em 127.0.0.1)
PORTA_LOCAL="${PORTA_LOCAL:-8080}"      # ponta LOCAL do port-forward
K="${K:-5}"                             # itens por consulta (a API aceita 1..100)
# Chave efemera do `X-API-Key`. Gerada aqui (ver o preflight) e impressa no log: ela vive
# o tempo da VM e morre com ela.
API_KEY="${API_KEY:-}"
# Arquivo local com os JSON das consultas de prova (results + latency_ms).
SAIDA="${SAIDA:-}"

# Consultas pt-BR de verdade sobre o dominio (orcamento/PLOA/legislativo). Sem acento de
# proposito: elas viajam por ssh/curl de um Git Bash no Windows, e o que se prova aqui e o
# ranking, nao o encoding. `--consulta` substitui a lista inteira (repetivel).
CONSULTAS=()
CONSULTAS_PADRAO=(
    "quanto foi destinado a saude no orcamento"
    "emendas parlamentares de relator"
    "despesas com pessoal da Uniao"
    "qual o valor previsto no PLOA para educacao"
)

DRY_RUN=0

# -------------------------------------------------------------------------------------
# Estado do teardown (o que ja foi tocado na nuvem -- e nesta maquina)
# -------------------------------------------------------------------------------------
ZONE=""                 # zona onde a VM subiu de fato
ZONAS_TENTADAS=""       # toda zona em que um `create` foi tentado (o teardown varre todas)
INDICE_SUBIU=0          # 1 depois que o indice chegou em $INDEX_URI
PF_PID=""               # pid do port-forward LOCAL (processo DESTA maquina)
TMP_DIR=""
SAIDA_DO_DELETE=""      # saida do ultimo `instances delete` que falhou (vira o diagnostico
                        # de auth do teardown)

TEARDOWN_RETRIES="${TEARDOWN_RETRIES:-3}"
TEARDOWN_ESPERA="${TEARDOWN_ESPERA:-20}"      # segundos entre tentativas de delete
UPLOAD_TIMEOUT="${UPLOAD_TIMEOUT:-1800}"      # teto do upload do indice
SERVE_TIMEOUT="${SERVE_TIMEOUT:-600}"         # teto do /health NA VM (o startup carrega o
                                              # Qwen3-VL: ~1-2 min no caso bom)
PF_TIMEOUT="${PF_TIMEOUT:-60}"                # teto do /health LOCAL (so o tunel subir)
LEAK_EXIT="${LEAK_EXIT:-70}"                  # status quando sobra recurso de pe

uso() {
    sed -n '3,/^# =====/p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --adapter)        ADAPTER_URI="$2"; shift 2 ;;
        --index-uri)      INDEX_URI="$2"; shift 2 ;;
        --gcs-corpus)     GCS_CORPUS="$2"; shift 2 ;;
        --project)        PROJECT="$2"; shift 2 ;;
        --zones)          ZONES="$2"; shift 2 ;;
        --vm)             VM="$2"; shift 2 ;;
        --branch)         BRANCH="$2"; shift 2 ;;
        --machine-type)   MACHINE_TYPE="$2"; shift 2 ;;
        --boot-disk-size) BOOT_DISK_SIZE="$2"; shift 2 ;;
        --corpus-dir)     CORPUS_DIR="$2"; shift 2 ;;
        --indice-dir)     INDICE_DIR="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --porta)          PORTA="$2"; shift 2 ;;
        --porta-local)    PORTA_LOCAL="$2"; shift 2 ;;
        --k)              K="$2"; shift 2 ;;
        --api-key)        API_KEY="$2"; shift 2 ;;
        --saida)          SAIDA="$2"; shift 2 ;;
        --consulta)       CONSULTAS+=("$2"); shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        uso 0 ;;
        *) echo "flag desconhecida: $1" >&2; uso 2 ;;
    esac
done

if [ "${#CONSULTAS[@]}" -eq 0 ]; then
    CONSULTAS=("${CONSULTAS_PADRAO[@]}")
fi

# -------------------------------------------------------------------------------------
# Primitivas de execucao (de gcp/serving/finetune_v4_spot.sh)
#
# Tudo que toca a nuvem passa por `run`/`existe`/`com_timeout`: e isso que faz o
# `--dry-run` imprimir a rodada inteira sem executar um unico gcloud -- nem o check de
# auth, nem o port-forward, nem o curl.
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
# Identico ao de `finetune_v4_spot.sh`: a conta pode estar sob politica de reauth, o token
# vence e o gcloud nao renova sozinho. Isso nao aparece no comeco -- aparece no upload do
# indice e, pior, no `instances delete` do teardown, que falha e deixa a L4 DE PE cobrando.
#
# Em `--dry-run` a funcao so ANUNCIA a checagem: `print-access-token` e chamada de REDE.
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

# Gancho de verificacao do trap: `API_DEMO_FAIL_AT=<etapa>` faz a etapa falhar de proposito
# (etapas: provision, bootstrap, stage, build_index, upload_index, serve, probe). Serve
# para provar que o teardown roda em qualquer saida -- e so isso.
gancho_de_falha() {
    if [ "${API_DEMO_FAIL_AT:-}" = "$1" ]; then
        echo "!!! falha INJETADA na etapa '$1' (API_DEMO_FAIL_AT) !!!" >&2
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
# De la o indice vai direto para o bucket pela rede da GCP, e nao depende de a maquina que
# disparou a rodada continuar viva. A VM sobe com `--scopes=cloud-platform`, entao o
# gcloud dela ja tem credencial para escrever no bucket do projeto.
#
# Chamada DUAS vezes de proposito (como o finetune faz com o adaptador): no fluxo normal,
# logo apos o build, e de novo de dentro do teardown se a primeira nao tiver acontecido --
# embeddar 150 itens na GPU custa minutos de L4 que nao se quer pagar duas vezes.
# -------------------------------------------------------------------------------------
subir_indice_para_gcs() {
    echo "--- subindo o indice da VM para $INDEX_URI (teto: ${UPLOAD_TIMEOUT}s) ---"
    echo "    [na VM] gcloud storage rsync -r <$INDICE_DIR> $INDEX_URI --project=$PROJECT"
    if remoto_com_timeout "$UPLOAD_TIMEOUT" "bash ~/api_demo_remote.sh upload_index"; then
        INDICE_SUBIU=1
        return 0
    fi
    return 1
}

# -------------------------------------------------------------------------------------
# PORT-FORWARD LOCAL -- a unica porta de entrada da API, e processo DESTA maquina
#
# A VM escuta so em 127.0.0.1: sem o tunel nao ha como chegar nela (nem com IP publico,
# nem com regra de firewall). O preco e que o tunel e um processo LOCAL, e processo local
# tambem vaza: um `ssh -N` orfao segura a porta 8080 desta maquina e mantem uma sessao
# aberta contra a VM. Por isso `fechar_port_forward` e chamada no fluxo normal E dentro do
# teardown, antes de qualquer delete.
# -------------------------------------------------------------------------------------
abrir_port_forward() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] gcloud compute ssh %s --zone=%s --project=%s --quiet -- -N -L %s:127.0.0.1:%s &   # (tunel: o ensaio nao o abre)\n' \
            "$VM" "$ZONE" "$PROJECT" "$PORTA_LOCAL" "$PORTA"
        return 0
    fi
    echo "--- abrindo o tunel 127.0.0.1:$PORTA_LOCAL -> $VM:$PORTA ---"
    gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --quiet \
        -- -N -L "$PORTA_LOCAL:127.0.0.1:$PORTA" &
    PF_PID=$!
    echo "--- port-forward local no pid $PF_PID ---"
}

fechar_port_forward() {
    if [ -z "${PF_PID:-}" ]; then
        return 0
    fi
    echo "--- fechando o port-forward local (pid $PF_PID) ---"
    # Matar os FILHOS antes do pai: o `gcloud compute ssh` delega a um `ssh` filho, e
    # matar so o pai deixaria o tunel orfao segurando a porta.
    if command -v pkill >/dev/null 2>&1; then
        pkill -P "$PF_PID" 2>/dev/null || true
    fi
    kill "$PF_PID" 2>/dev/null || true
    wait "$PF_PID" 2>/dev/null || true
    PF_PID=""
}

# -------------------------------------------------------------------------------------
# TEARDOWN -- a razao de ser deste script (o de capture/finetune, com duas adicoes)
#
# Registrado em `trap ... EXIT` logo antes do primeiro recurso ser criado, entao roda no
# sucesso, no erro (`set -e`), na interrupcao e na queda do terminal.
#
#   1. `trap '' INT TERM HUP`: o teardown IGNORA sinais. Um segundo Ctrl-C no meio dele
#      nao pode matar o shell deixando a L4 de pe, cobrando.
#   2. `set +eu +o pipefail`: nem erro, nem variavel nao-setada, nem pipe quebrado podem
#      abortar a limpeza antes dos deletes.
#   3. NOVO aqui: o port-forward LOCAL morre PRIMEIRO -- e processo desta maquina, nao
#      custa rede nenhuma matar, e um tunel orfao sobreviveria a VM.
#   4. O resgate do indice vem antes dos deletes (apagar a VM destroi o embed), mas com
#      TIMEOUT: um upload travado numa VM preemptada nao pode adiar a limpeza para sempre.
#      O uvicorn NAO e parado: ele morre junto com a VM, e um ssh a mais so atrasaria o
#      delete.
#   5. Todo delete tem RETRY: "operation in progress" logo apos um create interrompido e
#      falha transitoria, e engoli-la com `|| true` seria vazar recurso pago. "not found"
#      e a excecao: nao ha o que apagar nessa zona, entao e sucesso e nao retry.
#   6. Conferencia final que ASSERTA: se algo sobreviveu, o script grita LEAK, imprime o
#      delete manual e sai com status != 0.
#   7. A saida de cada delete que falha e CAPTURADA (e reemitida): e nela que o teardown
#      le se a causa foi o token vencido -- e so nesse caso imprime o banner.
# -------------------------------------------------------------------------------------
banner_de_recuperacao() {
    local zona="$1"
    {
        echo ""
        echo "=========== RECUPERACAO MANUAL NECESSARIA (auth do gcloud expirou) ==========="
        echo "  1. gcloud auth login"
        echo "  2. (resgatar o indice, se o upload nao rodou:)"
        echo "     gcloud compute ssh $VM --zone=$zona --project=$PROJECT --command \\"
        echo "       \"cd ~/residencia && gcloud storage rsync -r $INDICE_DIR \\"
        echo "        $INDEX_URI --project=$PROJECT\""
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

    etapa "TEARDOWN (saida com status $rc) -- fechando o tunel e apagando a VM $VM"
    echo "--- a partir daqui INT/TERM/HUP sao IGNORADOS: a limpeza nao se interrompe ---"

    # O tunel e local: matar nao custa rede e nao pode esperar o resto da limpeza.
    fechar_port_forward

    # Exercitar o token ANTES dos deletes: se a credencial so estava stale, isto a renova e
    # a limpeza segue normal; se ja venceu de vez, pelo menos o aviso sai antes do retry e
    # o banner cobre o resto.
    if ! token_de_auth_ok; then
        echo "--- aviso: o gcloud desta maquina nao tem token valido AGORA;" \
             "o resgate e os deletes podem falhar (veja o banner no fim) ---" >&2
    fi

    # Ultima chance de salvar o indice: se a rodada morreu depois do build mas antes de
    # subir, ele ainda esta na VM que estamos prestes a apagar. Se o build nem chegou la,
    # a fase falha limpo e seguimos para os deletes.
    if [ "${INDICE_SUBIU:-0}" = "0" ] && [ -n "${ZONE:-}" ]; then
        echo "--- tentativa final de subir o indice para $INDEX_URI ---"
        subir_indice_para_gcs || echo "--- o resgate do indice falhou ou estourou o tempo; seguindo para os deletes ---" >&2
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

    if [ "${INDICE_SUBIU:-0}" = "1" ]; then
        echo "--- o indice esta em $INDEX_URI (e so isso sobreviveu -- diretorio no bucket," \
             "nao recurso cobrando) ---"
    else
        echo "--- ATENCAO: o indice NAO chegou a $INDEX_URI; a rodada nao deixou nada ---" >&2
    fi

    echo "=== teardown concluido (status final: $rc) ==="
    exit "$rc"
}

# -------------------------------------------------------------------------------------
# Script remoto (roda DENTRO da VM)
#
# Gerado aqui e enviado por scp, como no finetune: a configuracao viaja num
# `api_demo_env.sh` separado para o corpo do script nao virar um emaranhado de aspas.
# -------------------------------------------------------------------------------------
gerar_scripts_remotos() {
    TMP_DIR="$(mktemp -d)"

    cat > "$TMP_DIR/api_demo_env.sh" <<ENV
# Gerado por gcp/serving/api_demo_spot.sh -- configuracao da demo da API [4].
PROJECT='$PROJECT'
GCS_CORPUS='$GCS_CORPUS'
ADAPTER_URI='$ADAPTER_URI'
INDEX_URI='$INDEX_URI'
CORPUS_DIR='$CORPUS_DIR'
INDICE_DIR='$INDICE_DIR'
DEVICE='$DEVICE'
PORTA='$PORTA'
API_KEY='$API_KEY'
SERVE_TIMEOUT='$SERVE_TIMEOUT'
BRANCH='$BRANCH'
ENV

    cat > "$TMP_DIR/api_demo_remote.sh" <<'REMOTE'
#!/usr/bin/env bash
# Gerado por gcp/serving/api_demo_spot.sh -- roda DENTRO da VM L4. Fase em $1.
set -euo pipefail
FASE="${1:?uso: api_demo_remote.sh <bootstrap|stage|build_index|upload_index|serve>}"
# shellcheck disable=SC1091
source "$HOME/api_demo_env.sh"
echo "=== [vm $(hostname)] fase=$FASE $(date) ==="

REPO_DIR="$HOME/residencia"
RECORDS_REL="$CORPUS_DIR/records.jsonl"
BIN_REL="$CORPUS_DIR/bin"
# Venv que `finetune/run_gcp_finetune.sh` cria (mesmo default dele): e dela que saem o
# torch/transformers/peft que o build_index e o serving precisam.
VENV_DIR="$REPO_DIR/.venv-gcp"
# ADAPTADOR: o MESMO caminho vai para o `--adapter` do build_index (que o carimba em
# `index_meta.json`) e para o `ADAPTER_PATH` do uvicorn. `api/app.py` COMPARA as duas
# strings no startup e se recusa a servir se divergirem -- itens e queries em espacos
# diferentes degradariam o ranking em silencio.
ADAPTER_LOCAL="$HOME/adapter"
INDICE_LOCAL="$REPO_DIR/$INDICE_DIR"
UVICORN_LOG="$HOME/uvicorn.log"
UVICORN_PID="$HOME/uvicorn.pid"

# /health da API local a VM, sem depender de nada instalado: curl se houver, senao o
# proprio python (que existe, porque e ele que serve).
health_ok() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsS -m 5 "http://127.0.0.1:$PORTA/health" >/dev/null 2>&1
    else
        python3 - "$PORTA" <<'PY' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5).read()
PY
    fi
}

case "$FASE" in

# =====================================================================================
bootstrap)
# =====================================================================================
    # Guard por SENTINELA DE CODIGO (e nao por `[ -d "$REPO_DIR" ]`), como no finetune:
    # outra etapa pode PRE-CRIAR `~/residencia` e o teste de diretorio daria falso. A
    # sentinela aqui e o proprio modulo que a fase `build_index` vai invocar.
    if [ ! -f "$REPO_DIR/api/build_index.py" ]; then
        mkdir -p "$REPO_DIR"
        tar xzf "$HOME/repo.tgz" -C "$REPO_DIR"
    fi
    cd "$REPO_DIR"
    [ -f "$REPO_DIR/requirements-slurm.txt" ] \
        || { echo "ERRO: requirements-slurm.txt ausente -- o tarball chegou incompleto" >&2; exit 1; }

    mkdir -p "$HOME/.cache/huggingface"

    # -- ambiente Python: ESPELHO do bloco de deps de finetune/run_gcp_finetune.sh -------
    # Duplicacao consciente (ref: run_gcp_finetune.sh, blocos "Ambiente Python" e
    # `prefer_host_torch`): o caminho de treino esta provado e rodando, e delegar a ele so
    # para montar o venv exigiria um pairs.jsonl valido (o DRY_RUN=1 dele valida os pares
    # antes de qualquer coisa) -- um arquivo de mentira so para enganar o validador seria
    # pior que estas 20 linhas.
    #
    # `--system-site-packages` e o que importa: o torch+CUDA vem da imagem DLVM, casado com
    # o driver. `requirements-slurm.txt` traz transformers/peft/accelerate/qwen-vl-utils/
    # pillow/numpy -- tudo que `api.build_index` e `api.app` importam, menos o servidor
    # (que entra na fase `serve`). `api/requirements.txt` NAO e usado aqui de proposito:
    # ele pina `transformers<4.51` (alvo do Cloud Run), e o Qwen3-VL-Embedding exige
    # >=4.57 -- instala-lo aqui rebaixaria o transformers e quebraria o embedder.
    if [ ! -d "$VENV_DIR" ]; then
        echo "--- criando venv (--system-site-packages) em $VENV_DIR ---"
        python3 -m venv --system-site-packages "$VENV_DIR" \
            || echo "--- venv falhou; seguindo no python do sistema (deps via pip --user) ---"
    fi
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        DEPS_MARKER_DIR="$VENV_DIR"
    else
        DEPS_MARKER_DIR="$HOME/.local"
    fi
    echo "--- python: $(command -v python3) ($(python3 --version 2>&1)) ---"

    REQ_HASH="$(sha256sum requirements-slurm.txt | cut -d" " -f1)"
    DEPS_MARKER="$DEPS_MARKER_DIR/.residencia_deps_$REQ_HASH"
    if [ -f "$DEPS_MARKER" ]; then
        echo "--- deps ja instaladas (marcador: $DEPS_MARKER) ---"
    else
        mkdir -p "$DEPS_MARKER_DIR"
        if [ -n "${VIRTUAL_ENV:-}" ]; then
            python3 -m pip install --quiet -r requirements-slurm.txt
        else
            python3 -m pip install --quiet --user -r requirements-slurm.txt
        fi
        touch "$DEPS_MARKER"
    fi

    # Guard "preferir o torch da imagem" (ESPELHO de `prefer_host_torch`): o pip resolve a
    # arvore inteira e qualquer dep que declare torch instala um torch proprio no venv --
    # sem as libs CUDA do no e, por vir antes no sys.path, sombreando o torch bom.
    if [ -n "${VIRTUAL_ENV:-}" ]; then TORCH_ROOT="$VIRTUAL_ENV"; else TORCH_ROOT="$HOME/.local"; fi
    for _sp in "$TORCH_ROOT"/lib/python*/site-packages; do
        [ -d "$_sp" ] || continue
        if [ -d "$_sp/torch" ]; then
            echo "--- torch encontrado em $_sp: removendo para o torch da imagem prevalecer ---"
            rm -rf "$_sp"/torch "$_sp"/torch-* "$_sp"/torchvision* \
                   "$_sp"/torchaudio* "$_sp"/nvidia* "$_sp"/functorch* || true
        fi
    done
    python3 -c "import torch; print('torch', torch.__version__, torch.__file__)" \
        || echo "aviso: import torch falha mesmo apos limpar $TORCH_ROOT -- verifique a VM antes de gastar GPU" >&2

    # GPU: informativo aqui (o driver pode ainda estar instalando no primeiro boot). O
    # gate de verdade e o build_index, que so termina se o embedder couber na L4.
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        || echo "--- sem GPU visivel ainda (driver instalando?) ---"
    echo "=== bootstrap concluido $(date) ==="
    ;;

# =====================================================================================
stage)
# =====================================================================================
    # Duas entradas do bucket: o corpus v4 (os ITENS que o indice embedda) e o adaptador
    # da Semana 10 (o espaco em que eles sao embeddados). A arvore do corpus tem de ser
    # reconstruida TAL E QUAL: `meta["path"]` dos records aponta para
    # `$CORPUS_DIR/bin/<eixo>/<arquivo>`, e `api.build_index` confere isso
    # (`check_corpus_paths`) ANTES de carregar o modelo.
    cd "$REPO_DIR"
    [ -n "$GCS_CORPUS" ] || { echo "ERRO: GCS_CORPUS vazio na VM" >&2; exit 2; }
    [ -n "$ADAPTER_URI" ] || { echo "ERRO: ADAPTER_URI vazio na VM" >&2; exit 2; }
    mkdir -p "$CORPUS_DIR" "$ADAPTER_LOCAL"

    echo "=== [stage] $GCS_CORPUS -> $CORPUS_DIR ==="
    gcloud storage rsync -r "$GCS_CORPUS/bin" "$BIN_REL" --project="$PROJECT"
    gcloud storage cp "$GCS_CORPUS/records.jsonl" "$RECORDS_REL" --project="$PROJECT"

    echo "=== [stage] $ADAPTER_URI -> $ADAPTER_LOCAL ==="
    gcloud storage rsync -r "$ADAPTER_URI" "$ADAPTER_LOCAL" --project="$PROJECT"

    # Falhar ALTO aqui: sem records/binarios o build apontaria para o vazio, e sem
    # `adapter_config.json` o PEFT nem carrega -- a demo embeddaria com a base crua e
    # provaria a coisa errada, depois de pagar a GPU.
    [ -s "$RECORDS_REL" ] \
        || { echo "ERRO: $RECORDS_REL vazio ou ausente apos o stage" >&2; exit 1; }
    [ -d "$BIN_REL" ] \
        || { echo "ERRO: $BIN_REL ausente apos o stage -- os records apontam para o vazio" >&2; exit 1; }
    [ -f "$ADAPTER_LOCAL/adapter_config.json" ] \
        || { echo "ERRO: $ADAPTER_LOCAL sem adapter_config.json -- $ADAPTER_URI nao e um adaptador PEFT" >&2; exit 1; }

    echo "--- records: $(wc -l < "$RECORDS_REL") linhas (esperado: 150) ---"
    du -sh "$BIN_REL" || true
    echo "--- adaptador: ---"
    ls -la "$ADAPTER_LOCAL"
    echo "=== stage concluido $(date) ==="
    ;;

# =====================================================================================
build_index)
# =====================================================================================
    # O artefato de runtime que faltava ao [4]: `records.jsonl` + `matrix.npy` +
    # `index_meta.json`, embeddados pelo lado de ITENS do baseline A com o adaptador.
    cd "$REPO_DIR"
    [ -s "$RECORDS_REL" ] || { echo "ERRO: $RECORDS_REL ausente -- rode a fase stage antes" >&2; exit 1; }
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    else
        echo "--- aviso: $VENV_DIR ausente; tentando no python do sistema ---" >&2
    fi

    echo "=== [build_index] $RECORDS_REL + $ADAPTER_LOCAL -> $INDICE_LOCAL ==="
    PYTHONPATH="$REPO_DIR" HF_HOME="$HOME/.cache/huggingface" \
        python3 -m api.build_index \
            --records "$RECORDS_REL" --out "$INDICE_LOCAL" --adapter "$ADAPTER_LOCAL"

    for arquivo in records.jsonl matrix.npy index_meta.json; do
        [ -s "$INDICE_LOCAL/$arquivo" ] \
            || { echo "ERRO: $INDICE_LOCAL/$arquivo ausente apos o build" >&2; exit 1; }
    done
    echo "--- indice em $INDICE_LOCAL ---"
    ls -la "$INDICE_LOCAL"
    echo "--- proveniencia carimbada (o serving compara com o ADAPTER_PATH dele) ---"
    cat "$INDICE_LOCAL/index_meta.json"
    echo "=== build_index concluido $(date) ==="
    ;;

# =====================================================================================
upload_index)
# =====================================================================================
    # O que sobrevive a rodada. `rsync -r` (retomavel, o que importa quando a primeira
    # tentativa morreu por preempcao) sobre o diretorio inteiro do indice.
    cd "$REPO_DIR"
    [ -n "$INDEX_URI" ] || { echo "ERRO: INDEX_URI vazio na VM" >&2; exit 2; }
    if [ ! -d "$INDICE_LOCAL" ]; then
        echo "ERRO: $INDICE_LOCAL nao existe -- o build nao terminou; nada a subir" >&2
        exit 1
    fi
    echo "=== [upload_index] $INDICE_LOCAL -> $INDEX_URI ==="
    gcloud storage rsync -r "$INDICE_LOCAL" "$INDEX_URI" --project="$PROJECT"
    echo "--- conferencia do que ficou no bucket ---"
    gcloud storage ls -r "$INDEX_URI" --project="$PROJECT"
    echo "=== upload_index concluido $(date) ==="
    ;;

# =====================================================================================
serve)
# =====================================================================================
    # Sobe o uvicorn em BACKGROUND e devolve o controle com a API ja respondendo: o
    # `nohup` e aqui, DENTRO da VM, porque a sessao ssh que dispara esta fase termina
    # assim que ela retorna -- e o servidor tem de sobreviver a ela para que o
    # port-forward da maquina local encontre alguem escutando.
    cd "$REPO_DIR"
    [ -d "$INDICE_LOCAL" ] || { echo "ERRO: $INDICE_LOCAL ausente -- rode build_index antes" >&2; exit 1; }
    [ -n "$API_KEY" ] || { echo "ERRO: API_KEY vazia -- api/app.py responderia 503 a toda consulta" >&2; exit 2; }
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    fi

    # Deps do SERVIDOR (o resto ja veio no bootstrap). Nenhuma delas arrasta torch, entao
    # instalar aqui nao ameaca o torch da imagem.
    python3 -c "import fastapi, uvicorn, pydantic" 2>/dev/null \
        || python3 -m pip install --quiet fastapi "uvicorn[standard]" pydantic

    # Idempotente: uma segunda chamada nao deixa dois uvicorn disputando a porta.
    if [ -f "$UVICORN_PID" ]; then
        kill "$(cat "$UVICORN_PID")" 2>/dev/null || true
        rm -f "$UVICORN_PID"
    fi

    echo "=== [serve] uvicorn api.app:app em 127.0.0.1:$PORTA (device=$DEVICE) ==="
    echo "--- INDEX_PATH=$INDICE_LOCAL  ADAPTER_PATH=$ADAPTER_LOCAL ---"
    # 127.0.0.1 de proposito: a API NAO fica exposta na rede da VM. Quem chega nela e o
    # port-forward ssh da maquina local -- nenhuma regra de firewall, nenhum IP publico.
    # `nohup` (e nao `setsid`) de proposito: o `nohup` EXECUTA o python no mesmo processo,
    # entao `$!` e o pid do uvicorn de verdade -- e e esse pid que a espera abaixo usa para
    # saber se o startup morreu. Com `setsid` o `$!` seria o do intermediario, que sai na
    # hora, e a espera acusaria morte falsa. Com stdin em /dev/null e a saida no log,
    # nenhum descritor fica preso na sessao ssh, e o SIGHUP do fim dela e ignorado.
    INDEX_PATH="$INDICE_LOCAL" ADAPTER_PATH="$ADAPTER_LOCAL" \
        API_KEY="$API_KEY" DEVICE="$DEVICE" \
        PYTHONPATH="$REPO_DIR" HF_HOME="$HOME/.cache/huggingface" \
        nohup python3 -m uvicorn api.app:app \
            --host 127.0.0.1 --port "$PORTA" \
        < /dev/null > "$UVICORN_LOG" 2>&1 &
    echo $! > "$UVICORN_PID"
    disown 2>/dev/null || true
    echo "--- uvicorn no pid $(cat "$UVICORN_PID"); log em $UVICORN_LOG ---"

    # O startup carrega o indice E o Qwen3-VL (+ adaptador) na GPU: ~1-2 min no caso bom.
    # Ate la o processo esta vivo mas a porta ainda nao aceita conexao.
    echo "--- esperando o /health responder (teto: ${SERVE_TIMEOUT}s) ---"
    ESPEROU=0
    while [ "$ESPEROU" -lt "$SERVE_TIMEOUT" ]; do
        if ! kill -0 "$(cat "$UVICORN_PID")" 2>/dev/null; then
            echo "ERRO: o uvicorn morreu durante o startup. Ultimas linhas do log:" >&2
            tail -n 40 "$UVICORN_LOG" >&2 || true
            exit 1
        fi
        if health_ok; then
            echo "--- /health respondeu apos ${ESPEROU}s ---"
            echo "SERVINDO em 127.0.0.1:$PORTA (pid $(cat "$UVICORN_PID"))"
            echo "=== serve concluido $(date) ==="
            exit 0
        fi
        sleep 5
        ESPEROU=$((ESPEROU + 5))
    done

    echo "ERRO: o /health nao respondeu em ${SERVE_TIMEOUT}s. Ultimas linhas do log:" >&2
    tail -n 40 "$UVICORN_LOG" >&2 || true
    kill "$(cat "$UVICORN_PID")" 2>/dev/null || true
    exit 1
    ;;

*)
    echo "ERRO: fase desconhecida: $FASE" >&2
    exit 2
    ;;
esac
REMOTE
    chmod +x "$TMP_DIR/api_demo_remote.sh"
}

# -------------------------------------------------------------------------------------
# Provisionamento (fallback de zona e flags de SPOT: de gcp/serving/finetune_v4_spot.sh)
# -------------------------------------------------------------------------------------
provisionar_vm() {
    etapa "[2/8] VM L4 ($MACHINE_TYPE + $ACCELERATOR, disco $BOOT_DISK_SIZE) com fallback de zona"
    echo "--- zonas na ordem: $ZONES ---"
    # `--instance-termination-action=DELETE` e o que importa para o teardown: numa
    # preempcao a VM se APAGA em vez de ficar STOPPED cobrando disco -- e o trap de saida
    # ainda tenta o delete depois, que vira no-op.
    local spot_flags=()
    if [ "$PROVISIONING_MODEL" = "SPOT" ]; then
        echo "--- provisionamento SPOT (a demo dura minutos; a preempcao quase nao pesa) ---"
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
    etapa "[3a/8] esperando o SSH da VM responder"
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
    etapa "[3b/8] enviando repo e scripts para a VM"
    # Arvore de trabalho (e nao `git archive`): o que vale e o checkout local, ja conferido
    # contra $BRANCH no preflight -- `api/` e o eixo documento do v4 so existem nesta
    # branch. O corpus v4 fica FORA do tarball: a fase `stage` o baixa do bucket.
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
    run gcloud compute scp "$TMP_DIR/api_demo_remote.sh" "$VM:api_demo_remote.sh" \
        --zone="$ZONE" --project="$PROJECT" --quiet
    run gcloud compute scp "$TMP_DIR/api_demo_env.sh" "$VM:api_demo_env.sh" \
        --zone="$ZONE" --project="$PROJECT" --quiet
}

# -------------------------------------------------------------------------------------
# A PROVA "de fora": tudo abaixo roda NESTA maquina, atravessando o tunel
# -------------------------------------------------------------------------------------
esperar_api_local() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] curl -fsS http://127.0.0.1:%s/health   (ate o tunel subir, teto %ss)\n' \
            "$PORTA_LOCAL" "$PF_TIMEOUT"
        return 0
    fi
    local esperou=0 corpo
    while [ "$esperou" -lt "$PF_TIMEOUT" ]; do
        if corpo="$(curl -fsS -m 5 "http://127.0.0.1:$PORTA_LOCAL/health" 2>/dev/null)"; then
            echo "--- GET /health (da maquina local, pelo tunel): $corpo ---"
            return 0
        fi
        sleep 3
        esperou=$((esperou + 3))
    done
    echo "ERRO: o /health nao respondeu pelo tunel em ${PF_TIMEOUT}s" >&2
    return 1
}

consultar_de_fora() {
    local pergunta="$1"
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] curl -sS -X POST http://127.0.0.1:%s/consulta -H "X-API-Key: ***" -d {"query": "%s", "k": %s}\n' \
            "$PORTA_LOCAL" "$pergunta" "$K"
        return 0
    fi
    local corpo
    # As consultas sao literais deste script (sem aspas nem barras), entao montar o corpo
    # JSON por interpolacao aqui e seguro -- e evita exigir jq na maquina de quem roda.
    if ! corpo="$(curl -sS -m 120 -X POST "http://127.0.0.1:$PORTA_LOCAL/consulta" \
            -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
            -d "{\"query\": \"$pergunta\", \"k\": $K}" 2>&1)"; then
        echo "--- aviso: a consulta '$pergunta' falhou: $corpo ---" >&2
        return 1
    fi
    case "$corpo" in
        *'"results"'*) ;;
        *) echo "--- aviso: resposta sem 'results' para '$pergunta': $corpo ---" >&2
           return 1 ;;
    esac
    printf '{"query": "%s", "k": %s, "resposta": %s}\n' "$pergunta" "$K" "$corpo" >> "$SAIDA"
    echo "--- POST /consulta \"$pergunta\" ---"
    echo "$corpo"
    return 0
}

# -------------------------------------------------------------------------------------
# [1/8] Preflight (local, antes de criar qualquer recurso pago)
# -------------------------------------------------------------------------------------
etapa "[1/8] demo da API [4] no v4 | projeto=$PROJECT | vm=$VM ($PROVISIONING_MODEL) | dry-run=$DRY_RUN"
echo "    corpus de entrada (GCS):   $GCS_CORPUS"
echo "    adaptador de entrada:      ${ADAPTER_URI:-<VAZIO>}"
echo "    indice de saida (GCS):     ${INDEX_URI:-<VAZIO>}"
echo "    disco=$BOOT_DISK_SIZE  maquina=$MACHINE_TYPE+$ACCELERATOR  device=$DEVICE"
echo "    corpus na VM:   $CORPUS_DIR (records.jsonl + bin/)"
echo "    indice na VM:   $INDICE_DIR"
echo "    porta: 127.0.0.1:$PORTA_LOCAL (local) -> $VM:$PORTA (so em 127.0.0.1 la)"
echo "    consultas de prova: ${#CONSULTAS[@]} (k=$K)"
echo "    branch esperada no checkout: ${BRANCH:-<checagem desligada>}"

# ADAPTER_URI e INDEX_URI sao o que define a rodada: sem adaptador a demo provaria a base
# crua, sem indice de saida ela pagaria a GPU para jogar o embed fora com a VM. Erro fatal
# mesmo em --dry-run: nao depende da nuvem, e um ensaio sem eles ensaiaria a rodada errada.
if [ -z "$ADAPTER_URI" ]; then
    echo "ERRO: ADAPTER_URI e obrigatorio -- sem ele o indice sairia da base crua." >&2
    echo "    use: --adapter gs://<bucket>/<prefixo>/adapter" >&2
    exit 2
fi
if [ -z "$INDEX_URI" ]; then
    echo "ERRO: INDEX_URI e obrigatorio -- sem ele o indice morre junto com a VM." >&2
    echo "    use: --index-uri gs://<bucket>/<prefixo>" >&2
    exit 2
fi
for _uri in ADAPTER_URI INDEX_URI GCS_CORPUS; do
    eval "_val=\$$_uri"
    case "$_val" in
        gs://*) ;;
        *) echo "ERRO: $_uri tem de ser um URI gs:// (recebido: '$_val')" >&2; exit 2 ;;
    esac
done
INDEX_URI="${INDEX_URI%/}"        # normaliza: os filhos sao montados com "$INDEX_URI/..."
ADAPTER_URI="${ADAPTER_URI%/}"
GCS_CORPUS="${GCS_CORPUS%/}"

# O indice nao pode cair DENTRO do corpus de entrada nem do adaptador: o `rsync -r` do
# indice escreveria por cima de `records.jsonl`/`bin/` (corrompendo o corpus que custou a
# rodada de captura) ou dos pesos PEFT (o entregavel da Semana 10).
if [ "$INDEX_URI" = "$GCS_CORPUS" ] || [ "${INDEX_URI#"$GCS_CORPUS"/}" != "$INDEX_URI" ]; then
    echo "ERRO: INDEX_URI ('$INDEX_URI') esta dentro do corpus de entrada ('$GCS_CORPUS')." >&2
    echo "    o upload do indice sobrescreveria records.jsonl/bin/: use outro prefixo." >&2
    exit 2
fi
if [ "$INDEX_URI" = "$ADAPTER_URI" ] || [ "${INDEX_URI#"$ADAPTER_URI"/}" != "$INDEX_URI" ]; then
    echo "ERRO: INDEX_URI ('$INDEX_URI') esta dentro do adaptador ('$ADAPTER_URI')." >&2
    echo "    o upload do indice sobrescreveria os pesos PEFT: use outro prefixo." >&2
    exit 2
fi

case "$PROVISIONING_MODEL" in
    STANDARD|SPOT) ;;
    *) echo "ERRO: PROVISIONING_MODEL invalido: '$PROVISIONING_MODEL' (use STANDARD ou SPOT)" >&2
       exit 2 ;;
esac

# k fora de [1,100] e 422 na API (`ConsultaRequest`): barrar aqui custa zero, barrar la
# custa a VM inteira para descobrir que as consultas de prova nao valiam.
case "$K" in
    ''|*[!0-9]*) echo "ERRO: K tem de ser inteiro entre 1 e 100 (recebido: '$K')" >&2; exit 2 ;;
esac
{ [ "$K" -ge 1 ] && [ "$K" -le 100 ]; } \
    || { echo "ERRO: K tem de estar entre 1 e 100 (recebido: '$K') -- a API responde 422 fora disso" >&2; exit 2; }

for _porta in PORTA PORTA_LOCAL; do
    eval "_val=\$$_porta"
    case "$_val" in
        ''|*[!0-9]*) echo "ERRO: $_porta tem de ser um numero de porta (recebido: '$_val')" >&2; exit 2 ;;
    esac
done

# Caminhos de dados sao RELATIVOS a raiz do repo: eles viajam no tarball e sao relidos na
# VM como "$REPO_DIR/<caminho>". Um absoluto viraria concatenacao sem sentido.
for _dir in CORPUS_DIR INDICE_DIR; do
    eval "_val=\$$_dir"
    case "$_val" in
        /*|[A-Za-z]:[\\/]*|*:*)
            echo "ERRO: $_dir tem de ser relativo a raiz do repo ($REPO_DIR): '$_val'" >&2
            exit 2 ;;
    esac
done

# CHAVE EFEMERA: vive o tempo da VM. Gerada aqui (nao na VM) porque quem a usa e o curl
# DESTA maquina; vai para a VM no `api_demo_env.sh` e some com ela no teardown. E impressa
# no log de proposito: nao ha segredo a proteger depois que a VM morre.
if [ -z "$API_KEY" ]; then
    if command -v openssl >/dev/null 2>&1; then
        API_KEY="$(openssl rand -hex 16)"
    else
        API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(16))' 2>/dev/null \
            || python -c 'import secrets; print(secrets.token_hex(16))' 2>/dev/null || true)"
    fi
fi
if [ -z "$API_KEY" ]; then
    echo "ERRO: nao foi possivel gerar a API_KEY (sem openssl e sem python no PATH)." >&2
    echo "    passe uma: --api-key <chave>" >&2
    exit 2
fi
echo "    API_KEY efemera desta rodada: $API_KEY"

if [ -z "$SAIDA" ]; then
    SAIDA="logs/api-demo-$(date +%Y%m%d-%H%M%S).jsonl"
fi
echo "    consultas serao gravadas em: $SAIDA"

if ! command -v gcloud >/dev/null 2>&1; then
    precondicao_ausente "gcloud nao esta no PATH"
fi
# O curl e a prova "de fora": sem ele a demo nao tem como bater na API desta maquina.
if ! command -v curl >/dev/null 2>&1; then
    precondicao_ausente "curl nao esta no PATH -- e ele quem prova a API de fora"
fi

# TOKEN VALIDO, aqui, antes de existir recurso pago. Um token ja vencido na largada nao
# impede a VM de subir, mas condena o fim da rodada: o upload do indice e o delete do
# teardown passam pelo gcloud DESTA maquina. Falhar agora custa zero; falhar depois custou
# uma L4 orfa cobrando. Em `--dry-run` a checagem e apenas anunciada.
if ! token_de_auth_ok; then
    echo "ERRO: gcloud sem token valido. Rode 'gcloud auth login' (a conta ativa tem" \
         "politica de reauth que pode expirar em runs longos)." >&2
    exit 2
fi

# Aviso, nao barreira: contas academicas/institucionais costumam carregar a politica de
# reauth que derrubou uma rodada anterior NO MEIO dela.
if [ "$DRY_RUN" != "1" ]; then
    CONTA_ATIVA="$(gcloud config get-value account 2>/dev/null || true)"
    case "$CONTA_ATIVA" in
        *.edu|*.edu.br|*discente*|*aluno*)
            echo "AVISO: conta $CONTA_ATIVA pode exigir reauth no meio de runs longos;" \
                 "para runs >1h considere uma service account." >&2 ;;
    esac
fi

# A VM recebe o CHECKOUT LOCAL (tarball da arvore de trabalho, sem .git): a branch que
# chega la e a que esta aqui. `--branch ''` desliga.
if [ -n "$BRANCH" ]; then
    BRANCH_LOCAL="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<sem git>')"
    if [ "$BRANCH_LOCAL" != "$BRANCH" ]; then
        precondicao_ausente \
            "o checkout local esta em '$BRANCH_LOCAL', mas a demo da API espera '$BRANCH'." \
            "Faca checkout da branch (ou passe --branch '$BRANCH_LOCAL' se for proposital)."
    else
        echo "    checkout local: $BRANCH_LOCAL (confere)"
    fi
fi

# O codigo do [4] viaja no tarball: se nao estiver no checkout, a VM so descobriria isso
# depois de paga.
for _modulo in api/build_index.py api/app.py; do
    if [ ! -f "$REPO_DIR/$_modulo" ]; then
        echo "ERRO: $_modulo ausente em $REPO_DIR -- a demo serve exatamente esse codigo." >&2
        exit 2
    fi
done

gerar_scripts_remotos

# -------------------------------------------------------------------------------------
# A partir daqui existe recurso pago em jogo: o teardown passa a ser obrigatorio.
#
# INT/TERM/HUP viram uma saida normal para que o EXIT dispare de forma confiavel (bash nem
# sempre roda o trap de EXIT quando o shell morre por sinal nao tratado). HUP cobre o caso
# do terminal fechado no meio da demo.
# -------------------------------------------------------------------------------------
trap 'teardown' EXIT
trap 'echo ""; echo "--- interrompido (SIGINT) ---" >&2; exit 130' INT
trap 'echo ""; echo "--- terminado (SIGTERM) ---" >&2; exit 143' TERM
trap 'echo ""; echo "--- terminal caiu (SIGHUP) ---" >&2; exit 129' HUP

gancho_de_falha provision
provisionar_vm
esperar_ssh
enviar_repo

etapa "[4/8] bootstrap da VM (untar do repo + venv-gcp com as deps do Qwen3-VL)"
gancho_de_falha bootstrap
remoto "bash ~/api_demo_remote.sh bootstrap"

etapa "[5/8] stage: corpus v4 de $GCS_CORPUS e adaptador de $ADAPTER_URI"
echo "--- records.jsonl + bin/<eixo>/ + o adaptador PEFT da Semana 10 ---"
gancho_de_falha stage
remoto "bash ~/api_demo_remote.sh stage"

etapa "[6/8] build_index na GPU (api.build_index com o adaptador) e upload para $INDEX_URI"
echo "--- o indice (records.jsonl + matrix.npy + index_meta.json) e o UNICO artefato que fica ---"
gancho_de_falha build_index
remoto "bash ~/api_demo_remote.sh build_index"

# O upload vem IMEDIATAMENTE depois do build, e nao no fim: se a Spot for preemptada
# durante a demo, o que custou GPU ja esta no bucket.
gancho_de_falha upload_index
subir_indice_para_gcs

etapa "[7/8] serve: uvicorn na VM (127.0.0.1:$PORTA) ate o /health responder"
echo "--- o startup carrega o indice + o Qwen3-VL com o adaptador: ~1-2 min ---"
gancho_de_falha serve
remoto "bash ~/api_demo_remote.sh serve"

etapa "[8/8] prova DE FORA: port-forward local 127.0.0.1:$PORTA_LOCAL -> $VM:$PORTA"
gancho_de_falha probe
abrir_port_forward
esperar_api_local
if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$(dirname "$SAIDA")"
    : > "$SAIDA"
fi
OK=0
for consulta in "${CONSULTAS[@]}"; do
    if consultar_de_fora "$consulta"; then
        OK=$((OK + 1))
    fi
done
fechar_port_forward
if [ "$DRY_RUN" != "1" ] && [ "$OK" = "0" ]; then
    echo "ERRO: nenhuma das ${#CONSULTAS[@]} consultas respondeu -- a demo nao provou nada." >&2
    exit 1
fi
if [ "$DRY_RUN" = "1" ]; then
    echo "--- [dry-run] as ${#CONSULTAS[@]} consultas seriam gravadas em $SAIDA ---"
else
    echo "--- $OK/${#CONSULTAS[@]} consultas responderam; JSON completo em $SAIDA ---"
fi

etapa "demo da API [4] concluida -- o teardown vem a seguir"
echo "    indice (o que sobrevive): $INDEX_URI"
echo "    consultas desta rodada:   $SAIDA"
exit 0
