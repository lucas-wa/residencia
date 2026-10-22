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
# criado (python3-venv/ensurepip ausente, disco cheio), caimos para `pip install --user`
# no python do sistema -- que ja tem o torch -- sem precisar de sudo.
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

# Deps SEM torch (a imagem ja traz torch+CUDA) e SEM faiss (benchmark/common/index.py cai
# para numpy; os corpora aqui sao pequenos).
#
# Cache de instalacao: o conjunto instalado e identificado pelo sha256 do
# requirements-slurm.txt. Se o marcador correspondente existir, o arquivo nao mudou desde
# a ultima instalacao e o pip e pulado inteiro. Editar o requirements muda o hash -> novo
# marcador -> reinstala. Para forcar: rm "$DEPS_MARKER_DIR"/.residencia_deps_*
REQ_HASH="$(sha256sum requirements-slurm.txt | cut -d" " -f1)"
DEPS_MARKER="$DEPS_MARKER_DIR/.residencia_deps_$REQ_HASH"
if [ -f "$DEPS_MARKER" ]; then
    echo "--- deps ja instaladas (marcador: $DEPS_MARKER) ---"
else
    mkdir -p "$DEPS_MARKER_DIR"
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        # Venv ativo: ele proprio e o alvo da instalacao.
        if command -v uv >/dev/null 2>&1; then
            # `uv` resolve/instala muito mais rapido. Sem o fallback, uma falha do uv
            # derrubaria a rodada inteira por causa do `set -e`.
            echo "--- instalando deps no venv com uv (fallback: pip) ---"
            uv pip install -r requirements-slurm.txt \
                || python3 -m pip install --quiet -r requirements-slurm.txt
        else
            echo "--- instalando deps no venv com pip ---"
            python3 -m pip install --quiet -r requirements-slurm.txt
        fi
    else
        # Sem venv (algumas Deep Learning VMs nao trazem `ensurepip`/python3-venv, entao
        # `python3 -m venv` falha): instalamos em $HOME/.local com `--user`. NAO usamos
        # `uv` aqui -- sem venv ativo ele nao tem alvo -- nem instalamos no site do
        # sistema, que exigiria sudo. O torch ja vem no python do sistema e e enxergado
        # por cima dessa camada --user, entao as deps que dependem dele nao o reinstalam.
        echo "--- sem venv: instalando deps com pip --user (alvo: $HOME/.local) ---"
        python3 -m pip install --quiet --user -r requirements-slurm.txt
    fi
    touch "$DEPS_MARKER"
fi

# -------------------------------------------------------------------------------------
# Guard "preferir o torch da imagem".
#
# A imagem base (Deep Learning VM) ja traz torch + CUDA casados com o driver, e nem o
# venv (--system-site-packages) nem a camada --user devem conter um torch proprio. Mas o
# pip resolve a arvore INTEIRA do requirements e qualquer dep que declare torch
# (colpali-engine, peft, ...) instala um por cima. Esse torch vem sem as libs CUDA do no
# -- o sintoma real foi
#     ImportError: libnvJitLink.so.12: cannot open shared object file
# -- e, por vir ANTES no sys.path, SOMBRA o torch funcional da imagem: o `import torch`
# de benchmark/models/colvision.py quebra e derruba a rodada.
#
# Correcao: depois de instalar as deps, remover do alvo (venv ou $HOME/.local) o torch e
# os pacotes colados nele. Roda em TODA rodada de proposito: editar o requirements
# invalida o marcador de deps, dispara a reinstalacao e o torch indesejado volta. Quando
# nao ha nada a limpar, o custo e um teste de existencia de diretorio.
# -------------------------------------------------------------------------------------
prefer_host_torch() {
    # Mesma deteccao de alvo do bloco de deps / ensure_torchaudio: com venv ativo o
    # torch indesejado estaria dentro do venv; sem venv, em $HOME/.local (--user).
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
# Reparo do torchaudio (algumas imagens GPU vem com ele ABI-incompativel com o torch).
#
# Visto na Deep Learning VM da GCP: torch 2.9.1+cu129 ok, mas o torchaudio da imagem
# estoura "_torchaudio.abi3.so: undefined symbol: torch_library_impl". Como transformers
# 5.x importa torchaudio de forma ansiosa (processing_utils -> audio_utils), QUALQUER
# `import colpali_engine` quebra junto. Reinstalar so o torchaudio (--no-deps, para nao
# mexer no torch da imagem) na mesma versao/tag CUDA do torch instalado resolve.
# -------------------------------------------------------------------------------------
ensure_torchaudio() {
    if python3 -c "import torchaudio" 2>/dev/null; then
        echo "--- torchaudio ok (sem reparo) ---"
        return 0
    fi
    echo "--- torchaudio nao importa (ABI incompativel?): reinstalando casado com o torch ---"

    # Versao e tag CUDA derivadas do torch INSTALADO (nada hardcodado). As fatias de
    # string ficam no bash para o mesmo trecho poder ser colado no slurm_run.sh, cujo
    # corpo roda dentro de aspas simples (nenhuma aspa simples pode aparecer aqui).
    TA_VER="$(python3 -c "import torch;print(torch.__version__)" 2>/dev/null || true)"
    TA_VER="${TA_VER%%+*}"
    CU_RAW="$(python3 -c "import torch;print(torch.version.cuda or str())" 2>/dev/null || true)"
    CU="cu${CU_RAW//./}"
    if [ -z "$TA_VER" ]; then
        echo "aviso: nao foi possivel reparar torchaudio (siga; o erro aparecera no load se for necessario)"
        return 0
    fi

    # Mesma logica do bloco de deps: sem venv ativo o alvo e $HOME/.local (--user), que
    # tem precedencia sobre o site-packages do sistema; com venv, o alvo e o venv.
    PIP_USER=""
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        PIP_USER="--user"
    fi

    echo "--- pip install torchaudio==$TA_VER (index: $CU) ---"
    # Nao-fatal de ponta a ponta: se o indice do PyTorch estiver inacessivel, o `set -e`
    # nao pode derrubar a rodada -- o erro reaparece no load, se for mesmo necessario.
    # shellcheck disable=SC2086
    python3 -m pip install ${PIP_USER} --no-deps --force-reinstall "torchaudio==${TA_VER}" \
            --index-url "https://download.pytorch.org/whl/${CU}" \
        && python3 -c "import torchaudio" 2>/dev/null \
        && echo "--- torchaudio reparado (import ok) ---" \
        || echo "aviso: nao foi possivel reparar torchaudio (siga; o erro aparecera no load se for necessario)"
}

ensure_torchaudio

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
