#!/usr/bin/env bash
# =====================================================================================
# UM ciclo do fluxo event-driven [3]: cron -> Spot L4 -> captura -> indice no GCS -> nada.
#
#   [1/3] sobe a VM L4 em SPOT, captura o corpus e DERRUBA a VM   (gcp/slice/run_slice.sh)
#   [2/3] upsert dos records capturados no indice que vive no GCS (gcp/serving/upsert_index.py)
#   [3/3] sobe os binarios retidos do ciclo                       (gcloud storage cp)
#
# A etapa [1] NAO e reimplementada aqui: ela e uma chamada a `gcp/slice/run_slice.sh
# --arch ensemble --phase capture` com `PROVISIONING_MODEL=SPOT`. Todo o ciclo de vida
# dos recursos pagos -- fallback de zona, bootstrap, `trap EXIT` que apaga a VM no
# sucesso, no erro, no Ctrl-C e na queda do terminal, retry de cada delete e a asserção
# final que grita LEAK se algo sobrou -- e o daquele script, que ja foi exercitado numa
# rodada paga. Duplicar isso aqui seria duplicar a unica parte que nao pode falhar.
#
# CUSTO OCIOSO ZERO: entre dois ciclos nao existe VM, nao existe Cloud SQL e nao existe
# instancia de GPU parada. O que persiste e objeto no GCS (centavos) e, do lado do
# serving [4], um Cloud Run com `min-instances=0`.
#
# -------------------------------------------------------------------------------------
# RUN-BOOK
#
#   # ensaio completo, sem tocar a nuvem (nenhum gcloud e executado):
#   bash gcp/serving/trigger_cycle.sh --dry-run
#
#   # um ciclo de verdade (GPU paga por ~minutos; guarde a saida):
#   bash gcp/serving/trigger_cycle.sh 2>&1 | tee logs/ciclo-$(date +%F).log
#
#   # conferir que NADA sobrou de pe (a resposta certa e "Listed 0 items."):
#   gcloud compute instances list --project=residencia-511517
#
#   # o que o ciclo deixou no bucket:
#   gcloud storage ls -l gs://residencia-511517-bronze/serving/corpus/
#
# Habilitar/desabilitar o cron diario: `gcp/serving/create_scheduler.sh` (e o README).
#
# -------------------------------------------------------------------------------------
# O QUE ESTE SCRIPT NAO FAZ: a matriz do indice servido
#
# O indice que a API [4] le e um diretorio com `records.jsonl` + `matrix.npy` na MESMA
# ordem (`api/index_io.py`). Aqui so os records se movem, e num prefixo PROPRIO
# (`CORPUS_URI`), separado do indice servido (`INDEX_URI`): quem escreve o indice servido
# e `python -m api.build_index`, que precisa de GPU e grava os dois arquivos juntos,
# consistentes. Se o upsert mexesse direto no prefixo servido, a API passaria a ver `n`
# records para `m` linhas de matriz e recusaria a carga ate o proximo rebuild -- falha
# fechada, mas ainda assim indisponibilidade.
#
# Por isso o ciclo tambem SOBE OS BINARIOS retidos ($BIN_URI): sem eles, os records dos
# ciclos antigos apontariam para arquivos que morreram com a VM efemera e o rebuild da
# matriz nao teria o que embeddar. O passo de rebuild esta no README (secao "reconstruir
# o indice servido"), staged, porque e ele que exige GPU.
# =====================================================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
# `python -m gcp.serving.upsert_index` roda da raiz do repo venha de onde vier a chamada
# (o scheduler dispara o ciclo com o cwd do container, nao do checkout).
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

PROJECT_ID="${PROJECT_ID:-residencia-511517}"

# Prefixo de serving no Bronze. Os tres destinos abaixo saem dele por default e podem ser
# apontados um a um por ambiente.
SERVING_URI="${SERVING_URI:-gs://residencia-511517-bronze/serving}"
CORPUS_URI="${CORPUS_URI:-$SERVING_URI/corpus}"       # records.jsonl acumulado (upsert)
BIN_URI="${BIN_URI:-$SERVING_URI/bin}"                # binarios retidos, um tar por ciclo
INDEX_URI="${INDEX_URI:-$SERVING_URI/indice_api_v4}"  # indice servido (so o build_index escreve)

# Corpus DESTE ciclo, relativo a raiz do repo (e o que `run_slice.sh` traz da VM antes do
# teardown). Diretorio proprio para nao colidir com os corpora congelados do ensemble v2.
CYCLE_CORPUS="${CYCLE_CORPUS:-data/govexe/ciclo/records.jsonl}"
CYCLE_BIN_DIR="${CYCLE_BIN_DIR:-data/govexe/ciclo/bin}"

# Nome proprio de VM: um ciclo automatico nunca deve esbarrar numa rodada manual da fatia
# (que usa `govexe-embed`) -- e o teardown do run_slice apaga a VM pelo nome.
VM="${VM:-govexe-ciclo}"
ZONES="${ZONES:-northamerica-northeast2-a us-central1-a us-central1-b us-east1-c us-east4-a us-west1-a}"

SUBIR_BINARIOS="${SUBIR_BINARIOS:-1}"
CICLO_TAG="${CICLO_TAG:-$(date +%Y%m%dT%H%M%S)}"
DRY_RUN=0

uso() {
    sed -n '2,52p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --project)   PROJECT_ID="$2"; shift 2 ;;
        --corpus-uri) CORPUS_URI="$2"; shift 2 ;;
        --bin-uri)   BIN_URI="$2"; shift 2 ;;
        --zones)     ZONES="$2"; shift 2 ;;
        --vm)        VM="$2"; shift 2 ;;
        --no-binarios) SUBIR_BINARIOS=0; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        -h|--help)   uso 0 ;;
        *) echo "flag desconhecida: $1" >&2; uso 2 ;;
    esac
done

etapa() { echo ""; echo "=== $* ==="; }

run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '    [dry-run] %s\n' "$*"
        return 0
    fi
    "$@"
}

etapa "ciclo $CICLO_TAG | projeto=$PROJECT_ID | vm=$VM (SPOT) | dry-run=$DRY_RUN"
echo "    corpus do ciclo: $REPO_DIR/$CYCLE_CORPUS"
echo "    indice de records (upsert): $CORPUS_URI"
echo "    binarios retidos:           $BIN_URI (subir: $SUBIR_BINARIOS)"
echo "    indice servido por [4]:     $INDEX_URI (escrito so por api.build_index)"

# -------------------------------------------------------------------------------------
# [1/3] Rodada Spot: provisao, captura e teardown GARANTIDO sao do run_slice.sh
#
# A configuracao viaja por ambiente (o run_slice le todas elas de `${VAR:-default}`), e
# nao por flag, porque `--arch` e `--phase` sao as unicas flags que mudam a SEMANTICA da
# rodada -- e sao justamente as duas que este ciclo fixa.
# -------------------------------------------------------------------------------------
etapa "[1/3] ciclo Spot: captura na L4 e teardown (gcp/slice/run_slice.sh)"
slice_flags=(--arch ensemble --phase capture)
if [ "$DRY_RUN" = "1" ]; then slice_flags+=(--dry-run); fi

PROVISIONING_MODEL=SPOT \
PROJECT_ID="$PROJECT_ID" \
VM="$VM" \
ZONES="$ZONES" \
ENSEMBLE_CORPUS="$CYCLE_CORPUS" \
ENSEMBLE_BIN_DIR="$CYCLE_BIN_DIR" \
    bash "$REPO_DIR/gcp/slice/run_slice.sh" "${slice_flags[@]}"

# A partir daqui NAO ha recurso pago de pe: o `trap EXIT` do run_slice ja apagou a VM e
# ja asseriu que nada sobrou (ele sai com status 70 se sobrou, e o `set -e` daqui mataria
# o ciclo antes desta linha).

# -------------------------------------------------------------------------------------
# [2/3] Upsert: o que sobrevive ao ciclo e o records.jsonl no GCS
# -------------------------------------------------------------------------------------
etapa "[2/3] upsert dos records do ciclo em $CORPUS_URI"
if [ ! -f "$REPO_DIR/$CYCLE_CORPUS" ]; then
    if [ "$DRY_RUN" = "1" ]; then
        echo "--- aviso (dry-run): $CYCLE_CORPUS nao existe (a captura nao rodou de fato) ---" >&2
    else
        echo "ERRO: a captura terminou sem $CYCLE_CORPUS -- nao ha o que mesclar no indice" >&2
        exit 1
    fi
else
    wc -l "$REPO_DIR/$CYCLE_CORPUS"
fi

run python -m gcp.serving.upsert_index \
    --index "$CORPUS_URI" \
    --novos "$REPO_DIR/$CYCLE_CORPUS"

# -------------------------------------------------------------------------------------
# [3/3] Binarios retidos: um tar por ciclo, para o rebuild da matriz ter o que embeddar
#
# Um tar POR CICLO (e nao um objeto por arquivo) pelo mesmo motivo do resgate do corpus no
# run_slice: sao muitos arquivos pequenos e alguns videos. Os caminhos dentro do tar sao
# relativos a $CYCLE_BIN_DIR, entao extrair todos os tars por cima do mesmo diretorio
# reconstroi a arvore que os `meta["path"]` dos records esperam.
# -------------------------------------------------------------------------------------
if [ "$SUBIR_BINARIOS" != "1" ]; then
    etapa "[3/3] binarios -- PULADO (--no-binarios)"
    echo "--- os records novos apontam para arquivos que morreram com a VM ---" >&2
elif [ -d "$REPO_DIR/$CYCLE_BIN_DIR" ] || [ "$DRY_RUN" = "1" ]; then
    etapa "[3/3] subindo os binarios do ciclo -> $BIN_URI/bin-$CICLO_TAG.tgz"
    TAR_LOCAL="${TMPDIR:-/tmp}/bin-$CICLO_TAG.tgz"
    run tar czf "$TAR_LOCAL" -C "$REPO_DIR/$CYCLE_BIN_DIR" .
    run gcloud storage cp "$TAR_LOCAL" "$BIN_URI/bin-$CICLO_TAG.tgz" --project="$PROJECT_ID"
    run rm -f "$TAR_LOCAL"
else
    etapa "[3/3] binarios -- nada a subir ($CYCLE_BIN_DIR nao existe)"
fi

# -------------------------------------------------------------------------------------
# Conferencia de custo: o criterio de aceite do subsistema e "nenhum recurso vivo".
# O run_slice ja assertou a VM dele; esta listagem e a prova para o log do ciclo.
# -------------------------------------------------------------------------------------
etapa "conferencia de custo: instancias vivas no projeto (o esperado e NENHUMA)"
run gcloud compute instances list --project="$PROJECT_ID" || true

etapa "ciclo $CICLO_TAG concluido"
echo "    indice de records: $CORPUS_URI"
echo "    para republicar o indice servido (precisa de GPU), veja gcp/serving/README.md"
exit 0
