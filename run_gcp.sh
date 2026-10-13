#!/usr/bin/env bash
# =====================================================================================
# Execucao do pipeline de benchmark da residencia numa VM UNICA com GPU (GCP, L4).
#
# Equivalente ao slurm_run.sh, mas SEM Slurm e SEM Singularity: aqui nao ha fila nem
# container -- o script roda direto na VM, assumindo a imagem "Deep Learning VM" da GCP,
# que ja vem com driver NVIDIA + CUDA + PyTorch instalados no Python do sistema.
#
# Mesmos estagios e mesmas variaveis do slurm_run.sh, parametrizado por STAGE:
#
#   preflight -> so a checagem de versoes das libs (CPU, segundos)
#   build   -> constroi o conjunto pt-BR dos planos GovExe (CPU-only: baixa PDFs por
#              HTTP e rasteriza as paginas com pypdfium2; NAO usa GPU)
#   vidore  -> preflight + avalia ViDoRe v2 (ColQwen2 + ColPali)         [precisa GPU]
#   govexe  -> preflight + avalia o conjunto pt-BR ja construido         [precisa GPU]
#   all     -> preflight + build + vidore + govexe (default)
#
# -------------------------------------------------------------------------------------
# USO
#
#   # bash run_gcp.sh                         # tudo (preflight+build+vidore+govexe)
#   # STAGE=preflight bash run_gcp.sh
#   # STAGE=build bash run_gcp.sh             # CPU (fetch+render dos PDFs)
#   # LIMIT=50 STAGE=vidore bash run_gcp.sh   # smoke
#   # FORCE=1 STAGE=build bash run_gcp.sh     # refaz o build do zero
#   # HF_CACHE_DIR=/mnt/disks/data/hf bash run_gcp.sh   # cache num disco de dados
#   # STRICT_PREFLIGHT=1 STAGE=vidore bash run_gcp.sh   # warn do preflight vira erro
#
# Maquina sugerida: g2-standard-8 (1x NVIDIA L4, 24 GB), imagem Deep Learning VM com
# CUDA/PyTorch, disco >= 100 GB (os PNGs dos planos + os checkpoints do HF pesam).
#
# Resultados: results/vidore.csv e results/govexe_ptbr.csv (uma linha por execucao)
# =====================================================================================

set -euo pipefail

# -------------------------------------------------------------------------------------
# Parametros (todos sobrescreviveis por variavel de ambiente na chamada)
# -------------------------------------------------------------------------------------

# Na VM unica nao ha bind nenhum: o repo e simplesmente o diretorio deste script.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)}"

STAGE="${STAGE:-all}"
LIMIT="${LIMIT:-}"
# 1 => passa --force ao build (ignora caches e refaz o conjunto pt-BR). So afeta o build.
FORCE="${FORCE:-}"
# 1 => o preflight trata warn (ex.: cair no ColQwen2 nativo) como erro e aborta a rodada.
STRICT_PREFLIGHT="${STRICT_PREFLIGHT:-0}"
# Cache do HuggingFace. Numa VM com disco de dados separado, aponte para la
# (ex.: HF_CACHE_DIR=/mnt/disks/data/hf) -- os checkpoints sao grandes.
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
# Venv criado ao lado do repo, herdando o torch/CUDA da imagem base (ver abaixo).
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv-gcp}"

case "$STAGE" in
    preflight|build|vidore|govexe|all) ;;
    *) echo "STAGE invalido: '$STAGE' (use preflight|build|vidore|govexe|all)" >&2; exit 2 ;;
esac

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Caches do HuggingFace (modelo/dataset) fora do $HOME quando HF_CACHE_DIR apontar
# para um disco de dados.
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"

# Download do Hub: o backend Xet (nativo do huggingface_hub recente) paraleliza por chunk
# e e bem mais rapido que o HTTP simples nos checkpoints grandes. NAO usamos hf_transfer:
# ele e mais fragil (sem retomada, erros opacos) e o Xet ja cobre o ganho. Se a rede da VM
# estiver instavel, desligue o Xet:
#     export HF_HUB_DISABLE_XET=1
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p logs results data/govexe/ptbr_planos "$HF_HOME"

echo "=== VM $(hostname) | $(date) ==="
echo "    STAGE=$STAGE  LIMIT=${LIMIT:-<nenhum>}  FORCE=${FORCE:-<nao>}  REPO_DIR=$REPO_DIR"
echo "    STRICT_PREFLIGHT=$STRICT_PREFLIGHT  HF_CACHE_DIR=$HF_CACHE_DIR"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "--- sem GPU visivel (esperado no STAGE=build; nos demais a rodada sera em CPU) ---"

# -------------------------------------------------------------------------------------
# Ambiente Python
#
# A Deep Learning VM ja traz torch + CUDA casados com o driver. Reinstalar torch por pip
# quebraria isso, entao o venv e criado com --system-site-packages: ele ENXERGA o torch da
# base e isola so o resto (transformers, colpali-engine, ...). Se o venv nao puder ser
# criado (python3-venv ausente, disco cheio), caimos para instalar no ambiente ativo.
# -------------------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "--- criando venv (--system-site-packages) em $VENV_DIR ---"
    python3 -m venv --system-site-packages "$VENV_DIR" \
        || echo "--- venv falhou; seguindo no ambiente Python ativo ---"
fi
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    DEPS_MARKER_DIR="$VENV_DIR"
else
    DEPS_MARKER_DIR="$HOME/.local"
fi

echo "--- python: $(command -v python3) ($(python3 --version 2>&1)) ---"

# Deps SEM torch (a imagem ja traz torch+CUDA) e SEM faiss (benchmark/common/index.py cai
# para numpy; os corpora aqui sao pequenos).
#
# Cache de instalacao: o conjunto instalado e identificado pelo sha256 do
# requirements-slurm.txt. Se o marcador correspondente existir, o arquivo nao mudou desde
# a ultima instalacao e o pip e pulado inteiro. Editar o requirements muda o hash -> novo
# marcador -> reinstala. Para forcar: rm "$VENV_DIR"/.residencia_deps_*
REQ_HASH="$(sha256sum requirements-slurm.txt | cut -d" " -f1)"
DEPS_MARKER="$DEPS_MARKER_DIR/.residencia_deps_$REQ_HASH"
if [ -f "$DEPS_MARKER" ]; then
    echo "--- deps ja instaladas (marcador: $DEPS_MARKER) ---"
else
    mkdir -p "$DEPS_MARKER_DIR"
    if command -v uv >/dev/null 2>&1; then
        # `uv` resolve/instala muito mais rapido. Sem o fallback, uma falha do uv
        # derrubaria a rodada inteira por causa do `set -e`.
        echo "--- instalando deps com uv (fallback: pip) ---"
        uv pip install -r requirements-slurm.txt \
            || python3 -m pip install --quiet -r requirements-slurm.txt
    else
        echo "--- instalando deps com pip ---"
        python3 -m pip install --quiet -r requirements-slurm.txt
    fi
    touch "$DEPS_MARKER"
fi

# --limit so e repassado quando LIMIT nao esta vazio.
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

run_preflight() {
    echo "=== [preflight] versoes de transformers/colpali-engine/torch ==="
    # Com `set -e`, status=fail (nenhum caminho de carregamento do ColQwen2) aborta a
    # rodada AQUI, antes de reservar modelo/GPU por 20 minutos a toa.
    STRICT_ARGS=()
    if [ "${STRICT_PREFLIGHT:-0}" = "1" ]; then
        STRICT_ARGS=(--strict)
    fi
    python3 -m benchmark.check_env ${STRICT_ARGS[@]+"${STRICT_ARGS[@]}"}
}

run_build() {
    echo "=== [build] conjunto pt-BR dos planos GovExe (CPU) ==="
    # FORCE=1 ignora os caches (saidas, PNGs ja rasterizados, PDFs baixados) e refaz tudo.
    # So vale aqui: run_vidore/run_govexe nao tem --force.
    FORCE_ARGS=()
    if [ "${FORCE:-}" = "1" ]; then
        FORCE_ARGS=(--force)
    fi
    # --parquet "" => baixa GovExe/gov-br-linguagem-simples (config planos) do Hub.
    python3 -m data.govexe.build_planos_benchmark \
        --parquet "" \
        --out-dir data/govexe/ptbr_planos \
        ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
        ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}
}

run_vidore() {
    echo "=== [vidore] ViDoRe v2 (GPU) ==="
    for cfg in benchmark/configs/vidore_colqwen2.yaml \
               benchmark/configs/vidore_colpali.yaml; do
        echo "--- $cfg ---"
        python3 -m benchmark.run_vidore \
            --config "$cfg" \
            ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
    done
}

run_govexe() {
    echo "=== [govexe] conjunto pt-BR (GPU) ==="
    python3 -m benchmark.run_govexe_ptbr \
        --config benchmark/configs/govexe_colqwen2.yaml \
        --data-dir data/govexe/ptbr_planos \
        --out results/govexe_ptbr.csv \
        ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
}

case "$STAGE" in
    preflight) run_preflight ;;
    build)     run_build ;;
    vidore)    run_preflight; run_vidore ;;
    govexe)    run_preflight; run_govexe ;;
    all)       run_preflight; run_build; run_vidore; run_govexe ;;
esac

echo "=== STAGE=$STAGE concluido: $(date) ==="
