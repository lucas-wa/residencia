#!/usr/bin/env bash
# =====================================================================================
# FINETUNING QLoRA (4-bit) numa VM UNICA com GPU na GCP (L4, 24 GB).
#
# Equivalente do finetune/slurm_finetune.sh sem Slurm e sem Singularity: roda direto na
# VM, no mesmo padrao do run_gcp.sh (imagem "Deep Learning VM" com driver+CUDA+PyTorch,
# venv com --system-site-packages, deps em cache por hash do requirements).
#
# CAMINHO SEGURO: DRY_RUN=1 e o DEFAULT -- valida config + pares + batches, estima
# parametros treinaveis/VRAM, imprime o resumo e SAI SEM TREINAR. Treinar exige
# DRY_RUN=0 explicito.
#
# -------------------------------------------------------------------------------------
# USO
#
#   # dry run (nao precisa nem de GPU):
#   bash finetune/run_gcp_finetune.sh
#
#   # dry run carregando a base de verdade (embrulha o LoRA; baixa o checkpoint):
#   LOAD_MODEL=1 bash finetune/run_gcp_finetune.sh
#
#   # TREINO de verdade (so sob ok explicito):
#   DRY_RUN=0 EPOCHS=1 bash finetune/run_gcp_finetune.sh
#
#   # base 8B (opcao) na L4 -- ver a nota de VRAM abaixo antes:
#   BASE=colembed8b BATCH_SIZE=1 GRAD_ACCUM=16 MAX_NEG=1 \
#       DRY_RUN=0 bash finetune/run_gcp_finetune.sh
#
#   # cache do HF num disco de dados (os checkpoints sao grandes):
#   HF_CACHE_DIR=/mnt/disks/data/hf bash finetune/run_gcp_finetune.sh
#
# Maquina: g2-standard-4/8 (1x NVIDIA L4, 24 GB), disco >= 150 GB. Ligar/desligar a VM
# no uso (mesma disciplina da fatia GCP: nada de VM ociosa cobrando).
#
# Saida: finetune/runs/<base>-<metodo>/ (adaptador PEFT + train_config.json)
#
# -------------------------------------------------------------------------------------
# NOTA go/no-go de VRAM -- "8B-QLoRA cabe na L4 de 24 GB?"
#
#   Estimativa (a mesma conta de `estimate_vram_gb` em finetune/train_lora.py; o dry-run
#   imprime estes numeros na linha `VRAM estimada`). Com BASE=colembed8b METHOD=qlora,
#   BATCH_SIZE=1, MAX_NEG=1 e gradient checkpointing LIGADO:
#
#     pesos NF4 (8e9 params x 0.5 byte + ~6% de constantes de double quant) ~  3.9 GB
#     LoRA r=32 (~87 M params x 12 bytes: bf16 + grad + 2 estados AdamW fp32) ~ 1.0 GB
#     ativacoes: 2 paginas x 1280 tokens x hidden 4096 x 2 bytes x 36 camadas ~ 0.7 GB
#     contexto CUDA + fragmentacao do allocator                              ~ 1.5 GB
#     ------------------------------------------------------------------------------
#     total                                                                  ~ 7.1 GB
#
#   >>> GO para o caso nominal (7 GB de 24) -- e NAO sao os pesos que decidem. <<<
#
#   O que decide sao as ATIVACOES de pagina inteira: elas escalam com o numero de tokens
#   visuais que o processor gera por pagina (resolucao dinamica do Qwen-VL: uma pagina A4
#   densa pode passar bem dos 1280 tokens assumidos aqui), com as imagens por passo
#   (BATCH_SIZE x (1 + MAX_NEG)) e, sobretudo, com o gradient checkpointing. A mesma
#   conta com BATCH_SIZE=2, MAX_NEG=2 e NO_GRAD_CKPT=1 da ~27 GB: NO-GO na L4.
#
#   PONTO A CONFIRMAR NA 1a EXECUCAO (nao adivinhar):
#     1. rodar com DRY_RUN=0 MAX_STEPS=2 BATCH_SIZE=1 MAX_NEG=1 e ler
#        `torch.cuda.max_memory_allocated()` / `nvidia-smi --query-gpu=memory.used`;
#     2. se couber, subir BATCH_SIZE/MAX_NEG ate o pico ficar ~<=20 GB (margem de 4 GB);
#     3. se estourar, limitar os pixels por pagina no processor (`max_pixels` do
#        Qwen2.5-VL/Qwen3-VL) antes de mexer no batch.
#
#   Para ColQwen2.5 (~3B) em QLoRA a folga e bem maior (~1.5 GB so de pesos, ~4 GB no
#   total): GO confortavel -- e por isso que ela e a base PRIMARIA aqui.
#   Fora do caminho recomendado: 8B em LoRA bf16 na L4 -- so os pesos pedem ~15 GB e
#   sobram ~7 GB para ativacoes+otimizador, sem margem para pagina densa ou batch>1.
#   Esse caso e o da H100, via finetune/slurm_finetune.sh.
# =====================================================================================

set -euo pipefail

# -------------------------------------------------------------------------------------
# Parametros (todos sobrescreviveis por variavel de ambiente na chamada)
# -------------------------------------------------------------------------------------

# Na VM unica nao ha bind nenhum: o repo e o diretorio ACIMA deste script (finetune/).
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# 1 (DEFAULT) => so valida e sai. 0 => TREINA de verdade.
DRY_RUN="${DRY_RUN:-1}"
# 1 => no dry run, carrega a base e embrulha o LoRA (reporta os params reais).
LOAD_MODEL="${LOAD_MODEL:-0}"

PAIRS="${PAIRS:-finetune/data/pairs.jsonl}"
BASE="${BASE:-colqwen2.5}"          # colqwen2.5 (primaria, ~3B) | colembed8b (opcao)
METHOD="${METHOD:-qlora}"           # L4: QLoRA 4-bit (LoRA bf16 e o caminho da H100)
OUTPUT_DIR="${OUTPUT_DIR:-}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"       # L4: comece em 1 e suba conferindo a VRAM
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-1e-4}"
MAX_NEG="${MAX_NEG:-1}"             # 0 => so in-batch negatives
LORA_R="${LORA_R:-32}"
MAX_STEPS="${MAX_STEPS:-}"          # vazio => epocas completas
LIMIT="${LIMIT:-}"                  # usa so os N primeiros pares (smoke)
# 1 => desliga o gradient checkpointing. NA L4 ISSO E O QUE ESTOURA A VRAM: so ligue
# depois de medir o pico com ele ativo.
NO_GRAD_CKPT="${NO_GRAD_CKPT:-0}"
EVAL_CONFIG="${EVAL_CONFIG:-}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
NO_EVAL="${NO_EVAL:-0}"

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv-gcp}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN invalido: '$DRY_RUN' (use 0|1)" >&2; exit 2 ;; esac
case "$BASE" in colqwen2.5|colembed8b) ;;
    *) echo "BASE invalida: '$BASE' (use colqwen2.5|colembed8b)" >&2; exit 2 ;; esac
case "$METHOD" in lora|qlora) ;;
    *) echo "METHOD invalido: '$METHOD' (use lora|qlora)" >&2; exit 2 ;; esac

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p logs results finetune/runs "$HF_HOME"

echo "=== VM $(hostname) | $(date) ==="
echo "    DRY_RUN=$DRY_RUN  LOAD_MODEL=$LOAD_MODEL  BASE=$BASE  METHOD=$METHOD"
echo "    PAIRS=$PAIRS  EPOCHS=$EPOCHS  BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM"
echo "    LR=$LR  MAX_NEG=$MAX_NEG  LORA_R=$LORA_R  LIMIT=${LIMIT:-<nenhum>}"
echo "    REPO_DIR=$REPO_DIR  HF_CACHE_DIR=$HF_CACHE_DIR"
if [ "$DRY_RUN" = "1" ]; then
    echo "    >>> DRY RUN: nenhum passo de treino sera executado <<<"
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "--- sem GPU visivel (ok no DRY_RUN=1 sem LOAD_MODEL) ---"

# -------------------------------------------------------------------------------------
# Ambiente Python -- identico ao run_gcp.sh: venv com --system-site-packages para herdar
# o torch+CUDA da imagem base; se o venv nao puder ser criado, cai para pip --user.
# -------------------------------------------------------------------------------------
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

# Deps: mesmo cache por sha256 do requirements-slurm.txt do run_gcp.sh (o finetuning
# acrescentou `peft` la). bitsandbytes NAO esta no requirements de proposito -- ele so
# serve ao 4-bit do QLoRA e, listado la, entraria na resolucao de TODO job arrastando
# torch junto; aqui ele e instalado sob demanda, com --no-deps, por ensure_bitsandbytes.
# Para forcar a reinstalacao: rm "$DEPS_MARKER_DIR"/.residencia_deps_*
REQ_HASH="$(sha256sum requirements-slurm.txt | cut -d" " -f1)"
DEPS_MARKER="$DEPS_MARKER_DIR/.residencia_deps_$REQ_HASH"
if [ -f "$DEPS_MARKER" ]; then
    echo "--- deps ja instaladas (marcador: $DEPS_MARKER) ---"
else
    mkdir -p "$DEPS_MARKER_DIR"
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        if command -v uv >/dev/null 2>&1; then
            echo "--- instalando deps no venv com uv (fallback: pip) ---"
            uv pip install -r requirements-slurm.txt \
                || python3 -m pip install --quiet -r requirements-slurm.txt
        else
            echo "--- instalando deps no venv com pip ---"
            python3 -m pip install --quiet -r requirements-slurm.txt
        fi
    else
        echo "--- sem venv: instalando deps com pip --user (alvo: $HOME/.local) ---"
        python3 -m pip install --quiet --user -r requirements-slurm.txt
    fi
    touch "$DEPS_MARKER"
fi

# -------------------------------------------------------------------------------------
# Guard "preferir o torch da imagem" (ver comentario longo no run_gcp.sh): o pip resolve
# a arvore inteira do requirements e qualquer dep que declare torch instala um torch
# proprio no venv/--user. Esse torch vem sem as libs CUDA do no (`ImportError:
# libnvJitLink.so.12`) e, por vir antes no sys.path, sombra o torch bom da imagem.
# Removemos o torch do alvo (venv ou $HOME/.local) para o da imagem prevalecer. Roda em
# TODA rodada: mudar o requirements invalida o marcador de deps e o torch indesejado
# volta.
# -------------------------------------------------------------------------------------
prefer_host_torch() {
    # Mesma deteccao de alvo do bloco de deps / ensure_torchaudio.
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        TORCH_ROOT="$VIRTUAL_ENV"
    else
        TORCH_ROOT="$HOME/.local"
    fi

    LOCAL_TORCH=""
    for _f in "$TORCH_ROOT"/lib/python*/site-packages/torch/__init__.py; do
        if [ -f "$_f" ]; then LOCAL_TORCH="$_f"; fi
    done
    if [ -z "$LOCAL_TORCH" ]; then
        echo "--- sem torch em $TORCH_ROOT: o torch da imagem prevalece ---"
        return 0
    fi

    echo "--- torch encontrado em $LOCAL_TORCH: removendo para o torch da imagem prevalecer ---"
    for _sp in "$TORCH_ROOT"/lib/python*/site-packages; do
        [ -d "$_sp" ] || continue
        rm -rf "$_sp"/torch "$_sp"/torch-* "$_sp"/torchvision* \
               "$_sp"/torchaudio* "$_sp"/nvidia* "$_sp"/functorch* || true
    done

    python3 -c "import torch; print(torch.__version__, torch.__file__)" \
        || echo "aviso: import torch AINDA falha apos limpar $TORCH_ROOT -- o torch da imagem tambem esta quebrado; verifique a VM antes de gastar GPU"
}

prefer_host_torch

# -------------------------------------------------------------------------------------
# Reparo do torchaudio ABI-incompativel (ver comentario longo no run_gcp.sh): foi visto
# exatamente nesta Deep Learning VM, e derruba qualquer `import colpali_engine`.
# -------------------------------------------------------------------------------------
ensure_torchaudio() {
    if python3 -c "import torchaudio" 2>/dev/null; then
        echo "--- torchaudio ok (sem reparo) ---"
        return 0
    fi
    echo "--- torchaudio nao importa (ABI incompativel?): reinstalando casado com o torch ---"
    TA_VER="$(python3 -c "import torch;print(torch.__version__)" 2>/dev/null || true)"
    TA_VER="${TA_VER%%+*}"
    CU_RAW="$(python3 -c "import torch;print(torch.version.cuda or str())" 2>/dev/null || true)"
    CU="cu${CU_RAW//./}"
    if [ -z "$TA_VER" ]; then
        echo "aviso: nao foi possivel reparar torchaudio (siga; o erro aparecera no load se for necessario)"
        return 0
    fi
    PIP_USER=""
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        PIP_USER="--user"
    fi
    echo "--- pip install torchaudio==$TA_VER (index: $CU) ---"
    # shellcheck disable=SC2086
    python3 -m pip install ${PIP_USER} --no-deps --force-reinstall "torchaudio==${TA_VER}" \
            --index-url "https://download.pytorch.org/whl/${CU}" \
        && python3 -c "import torchaudio" 2>/dev/null \
        && echo "--- torchaudio reparado (import ok) ---" \
        || echo "aviso: nao foi possivel reparar torchaudio (siga; o erro aparecera no load se for necessario)"
}

ensure_torchaudio

# -------------------------------------------------------------------------------------
# bitsandbytes: SO o QLoRA precisa (kernels NF4 do `BitsAndBytesConfig(load_in_4bit)`).
# Ele saiu do requirements-slurm.txt de proposito -- la, ele entrava na resolucao de todo
# job e arrastava um torch proprio para o venv/--user, justamente o que o
# `prefer_host_torch` acima tem de desfazer. Aqui ele e instalado sob demanda e com
# `--no-deps`, que impede o pip de puxar torch junto: o torch da imagem ja esta no
# ambiente e o wheel do bitsandbytes so precisa encontra-lo em runtime.
#
# Nao-fatal: se o indice estiver inacessivel ou o wheel nao casar com a CUDA da imagem,
# `set -e` nao pode derrubar a rodada -- o erro reaparece no load, e o `--dry-run` do
# treinador roda sem bitsandbytes (so avisa que faltaria).
# -------------------------------------------------------------------------------------
ensure_bitsandbytes() {
    if [ "$METHOD" != "qlora" ]; then
        echo "--- METHOD=$METHOD: bitsandbytes nao e necessario (so o QLoRA usa) ---"
        return 0
    fi
    if python3 -c "import bitsandbytes" 2>/dev/null; then
        echo "--- bitsandbytes ok (ja instalado) ---"
        return 0
    fi

    # Mesma deteccao de alvo do bloco de deps / ensure_torchaudio.
    PIP_USER=""
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        PIP_USER="--user"
    fi

    echo "--- METHOD=qlora: instalando bitsandbytes (--no-deps, para nao arrastar torch) ---"
    # shellcheck disable=SC2086
    python3 -m pip install --quiet ${PIP_USER} --no-deps "bitsandbytes>=0.43" \
        && python3 -c "import bitsandbytes" 2>/dev/null \
        && echo "--- bitsandbytes instalado (import ok) ---" \
        || echo "aviso: bitsandbytes indisponivel -- o QLoRA vai falhar no load (o --dry-run segue)"
}

ensure_bitsandbytes

# -------------------------------------------------------------------------------------
# Argumentos: cada opcional so entra quando a variavel esta setada.
# -------------------------------------------------------------------------------------
ARGS=(--pairs "$PAIRS" --base "$BASE" --method "$METHOD"
      --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM"
      --lr "$LR" --max-neg "$MAX_NEG" --lora-r "$LORA_R")
if [ -n "${OUTPUT_DIR:-}" ];  then ARGS+=(--output-dir "$OUTPUT_DIR"); fi
if [ -n "${MAX_STEPS:-}" ];   then ARGS+=(--max-steps "$MAX_STEPS"); fi
if [ -n "${LIMIT:-}" ];       then ARGS+=(--limit "$LIMIT"); fi
if [ -n "${EVAL_CONFIG:-}" ]; then ARGS+=(--eval-config "$EVAL_CONFIG"); fi
if [ -n "${EVAL_LIMIT:-}" ];  then ARGS+=(--eval-limit "$EVAL_LIMIT"); fi
if [ "${NO_GRAD_CKPT:-0}" = "1" ]; then ARGS+=(--no-gradient-checkpointing); fi
if [ "${NO_EVAL:-0}" = "1" ];      then ARGS+=(--no-eval); fi
if [ "$DRY_RUN" = "1" ]; then
    ARGS+=(--dry-run)
    if [ "${LOAD_MODEL:-0}" = "1" ]; then ARGS+=(--load-model); fi
fi

echo "=== finetune/train_lora.py ${ARGS[*]} ==="
python3 finetune/train_lora.py "${ARGS[@]}"

if [ "$DRY_RUN" = "1" ]; then
    echo "=== DRY RUN concluido: nada foi treinado (use DRY_RUN=0 para treinar) ==="
else
    echo "=== treino concluido: $(date) ==="
fi
