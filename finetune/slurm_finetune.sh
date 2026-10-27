#!/bin/bash
# =====================================================================================
# Submissao Slurm do FINETUNING LoRA (bf16) no cluster UFG (H100 80 GB).
#
# Mesmo padrao do slurm_run.sh (container Singularity, deps em cache por hash do
# requirements, propagacao via SINGULARITYENV_*/APPTAINERENV_*), mas o comando final e
# `finetune/train_lora.py` em vez do benchmark.
#
# CAMINHO SEGURO: DRY_RUN=1 e o DEFAULT. Nesse modo o script valida config + pares +
# batches, estima parametros treinaveis/VRAM, imprime o resumo e SAI SEM TREINAR.
# Para treinar de verdade e preciso passar DRY_RUN=0 explicitamente.
#
# -------------------------------------------------------------------------------------
# USO
#
#   cd /raid/user_lucasalves/ufg/residencia && mkdir -p logs
#
#   # 0) gerar os pares (CPU, zero GPU) -- ver finetune/make_pairs.py:
#   #    python3 -m finetune.make_pairs --corpus data/govexe/ptbr_planos \
#   #        --out finetune/data/pairs.jsonl
#
#   # 1) dry run SEM GPU (valida tudo e sai; nao baixa pesos):
#   sbatch --gres=gpu:h100:0 finetune/slurm_finetune.sh
#
#   # 2) dry run COM a base carregada de verdade (embrulha o LoRA e reporta o numero
#   #    REAL de parametros treinaveis; baixa/le o checkpoint, entao pede GPU):
#   LOAD_MODEL=1 sbatch finetune/slurm_finetune.sh
#
#   # 3) TREINO de verdade (so sob ok explicito):
#   DRY_RUN=0 EPOCHS=1 sbatch finetune/slurm_finetune.sh
#
#   # base 8B (opcao; licenca NAO comercial) e hiperparametros:
#   BASE=colembed8b BATCH_SIZE=1 GRAD_ACCUM=16 sbatch finetune/slurm_finetune.sh
#
#   # avaliacao antes/depois no ViDoRe V3 (hook do train_lora.py; LIMIT para smoke):
#   DRY_RUN=0 EVAL_CONFIG=benchmark/configs/v3_colqwen25.yaml EVAL_LIMIT=50 \
#       sbatch finetune/slurm_finetune.sh
#
# Logs:   logs/slurm_<jobid>_finetune.{out,err}
# Saida:  finetune/runs/<base>-<metodo>/ (adaptador PEFT + train_config.json)
#
# -------------------------------------------------------------------------------------
# NOTA go/no-go de VRAM (H100 80 GB, LoRA bf16)
#
#   ColQwen2.5 (~3B): pesos bf16 ~6 GB + LoRA/AdamW ~0.7 GB + ativacoes de pagina
#   inteira. GO folgado -- da para subir BATCH_SIZE (4-8) e ate desligar o gradient
#   checkpointing (NO_GRAD_CKPT=1) se quiser velocidade.
#
#   ColEmbed 8B: pesos bf16 ~16 GB + LoRA/AdamW ~1 GB + ativacoes. GO com
#   BATCH_SIZE<=2 e checkpointing ligado.
#
#   O numero que vale e o da 1a execucao: o resumo do dry-run imprime a estimativa
#   (`VRAM estimada`) e o treino real deve ser conferido com
#   `nvidia-smi` / `torch.cuda.max_memory_allocated()`. A estimativa do L4/QLoRA, que e
#   a apertada, esta em finetune/run_gcp_finetune.sh.
# =====================================================================================

#SBATCH --job-name=residencia-finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodelist=dgx-H100-02
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/slurm_%j_finetune.out
#SBATCH --error=logs/slurm_%j_finetune.err

set -euo pipefail

# -------------------------------------------------------------------------------------
# Parametros (todos sobrescreviveis por variavel de ambiente na submissao)
# -------------------------------------------------------------------------------------

# Mesmo bind do slurm_run.sh: /raid/user_lucasalves/ -> /work.
RESIDENCIA_DIR="${RESIDENCIA_DIR:-/work/ufg/residencia}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/raid/user_lucasalves/images/vllm-openai_latest.sif}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/work/.cache/huggingface}"

# 1 (DEFAULT) => so valida e sai. 0 => TREINA de verdade.
DRY_RUN="${DRY_RUN:-1}"
# 1 => no dry run, carrega a base e embrulha o LoRA (reporta os params reais).
LOAD_MODEL="${LOAD_MODEL:-0}"

PAIRS="${PAIRS:-finetune/data/pairs.jsonl}"
BASE="${BASE:-colqwen2.5}"          # colqwen2.5 (primaria, ~3B) | colembed8b (opcao)
METHOD="${METHOD:-lora}"            # H100: LoRA bf16 (QLoRA e o caminho da L4)
OUTPUT_DIR="${OUTPUT_DIR:-}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
MAX_NEG="${MAX_NEG:-2}"             # 0 => so in-batch negatives
LORA_R="${LORA_R:-32}"
MAX_STEPS="${MAX_STEPS:-}"          # vazio => epocas completas
LIMIT="${LIMIT:-}"                  # usa so os N primeiros pares (smoke)
# 1 => desliga o gradient checkpointing (mais rapido, MUITO mais VRAM).
NO_GRAD_CKPT="${NO_GRAD_CKPT:-0}"
# Hook de avaliacao antes/depois (vazio => o YAML default da base).
EVAL_CONFIG="${EVAL_CONFIG:-}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
NO_EVAL="${NO_EVAL:-0}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN invalido: '$DRY_RUN' (use 0|1)" >&2; exit 2 ;; esac
case "$BASE" in colqwen2.5|colembed8b) ;;
    *) echo "BASE invalida: '$BASE' (use colqwen2.5|colembed8b)" >&2; exit 2 ;; esac
case "$METHOD" in lora|qlora) ;;
    *) echo "METHOD invalido: '$METHOD' (use lora|qlora)" >&2; exit 2 ;; esac

mkdir -p logs

echo "=== Job ${SLURM_JOB_ID:-local} em $(hostname) | $(date) ==="
echo "    DRY_RUN=$DRY_RUN  LOAD_MODEL=$LOAD_MODEL  BASE=$BASE  METHOD=$METHOD"
echo "    PAIRS=$PAIRS  EPOCHS=$EPOCHS  BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM"
echo "    LR=$LR  MAX_NEG=$MAX_NEG  LORA_R=$LORA_R  LIMIT=${LIMIT:-<nenhum>}"
echo "    RESIDENCIA_DIR=$RESIDENCIA_DIR  CONTAINER_IMAGE=$CONTAINER_IMAGE"
if [ "$DRY_RUN" = "1" ]; then
    echo "    >>> DRY RUN: nenhum passo de treino sera executado <<<"
fi

# -------------------------------------------------------------------------------------
# Propagacao para DENTRO do container (mesma tecnica do slurm_run.sh: o corpo do
# `bash -c '...'` usa aspas SIMPLES e nada e interpolado pelo shell externo; as variaveis
# entram pelo mecanismo nativo SINGULARITYENV_*/APPTAINERENV_*).
# -------------------------------------------------------------------------------------
for _v in RESIDENCIA_DIR HF_CACHE_DIR DRY_RUN LOAD_MODEL PAIRS BASE METHOD OUTPUT_DIR \
          EPOCHS BATCH_SIZE GRAD_ACCUM LR MAX_NEG LORA_R MAX_STEPS LIMIT NO_GRAD_CKPT \
          EVAL_CONFIG EVAL_LIMIT NO_EVAL; do
    eval "export SINGULARITYENV_${_v}=\"\${${_v}}\""
    eval "export APPTAINERENV_${_v}=\"\${${_v}}\""
done

srun singularity exec \
    --nv \
    --no-home \
    --bind "/raid/user_lucasalves/:/work" \
    "$CONTAINER_IMAGE" \
    bash -c '
        set -euo pipefail

        # O container e read-only: HOME e cache do pip vao para o bind /work, e o pip
        # instala em modo user-local ($HOME/.local).
        export HOME=/work/.home
        export PIP_CACHE_DIR=/work/.cache/pip
        mkdir -p "$HOME" "$PIP_CACHE_DIR"
        export PATH="/work/.home/.local/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

        # Caches do HuggingFace no bind (o $HOME do container some entre jobs).
        export HF_HOME="$HF_CACHE_DIR"
        export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
        export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
        export HF_XET_HIGH_PERFORMANCE=1

        cd "$RESIDENCIA_DIR"
        export PYTHONPATH="$RESIDENCIA_DIR:${PYTHONPATH:-}"
        mkdir -p logs results finetune/runs "$HF_CACHE_DIR"

        echo "--- python: $(command -v python3) ($(python3 --version 2>&1)) ---"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
            || echo "--- sem GPU visivel (ok no DRY_RUN=1 sem LOAD_MODEL) ---"

        # Deps: mesmo cache por sha256 do requirements-slurm.txt do slurm_run.sh (o
        # finetuning acrescentou `peft` la; bitsandbytes NAO entra -- ele so serve ao
        # QLoRA da L4 e e instalado sob demanda em finetune/run_gcp_finetune.sh. O LoRA
        # bf16 daqui nao o usa). Para forcar a reinstalacao:
        #     rm "$HOME"/.local/.residencia_deps_*
        REQ_HASH="$(sha256sum requirements-slurm.txt | cut -d" " -f1)"
        DEPS_MARKER="$HOME/.local/.residencia_deps_$REQ_HASH"
        if [ -f "$DEPS_MARKER" ]; then
            echo "--- deps ja instaladas (marcador: $DEPS_MARKER) ---"
        else
            mkdir -p "$HOME/.local"
            if command -v uv >/dev/null 2>&1; then
                echo "--- instalando deps com uv (fallback: pip) ---"
                uv pip install --system -r requirements-slurm.txt \
                    || python3 -m pip install --quiet -r requirements-slurm.txt
            else
                echo "--- instalando deps com pip ---"
                python3 -m pip install --quiet -r requirements-slurm.txt
            fi
            touch "$DEPS_MARKER"
        fi

        # Guard "preferir o torch do container" (ver comentario longo no slurm_run.sh):
        # quando o `uv pip install --system` falha e caimos no `pip install --user`, o
        # pip arrasta um torch PROPRIO para $HOME/.local (puxado por colpali-engine/
        # peft). Esse torch vem sem as libs CUDA do no (`ImportError:
        # libnvJitLink.so.12`) e, por vir antes no sys.path, sombra o torch bom da
        # imagem. Removemos o torch do --user para o da imagem prevalecer. Roda em TODO
        # job: mudar o requirements invalida o marcador de deps e o torch --user volta.
        prefer_host_torch() {
            USER_TORCH=""
            for _f in "$HOME"/.local/lib/python*/site-packages/torch/__init__.py; do
                if [ -f "$_f" ]; then USER_TORCH="$_f"; fi
            done
            if [ -z "$USER_TORCH" ]; then
                echo "--- sem torch em --user: o torch do container prevalece ---"
                return 0
            fi

            echo "--- torch encontrado em --user ($USER_TORCH): removendo para o torch do container prevalecer ---"
            for _sp in "$HOME"/.local/lib/python*/site-packages; do
                [ -d "$_sp" ] || continue
                rm -rf "$_sp"/torch "$_sp"/torch-* "$_sp"/torchvision* \
                       "$_sp"/torchaudio* "$_sp"/nvidia* "$_sp"/functorch* || true
            done

            python3 -c "import torch; print(torch.__version__, torch.__file__)" \
                || echo "aviso: import torch AINDA falha apos limpar o --user -- o torch da imagem tambem esta quebrado; verifique o container antes de gastar GPU"
        }

        prefer_host_torch

        # Reparo do torchaudio ABI-incompativel (ver comentario longo no slurm_run.sh):
        # transformers 5.x importa torchaudio de forma ansiosa e derruba qualquer
        # `import colpali_engine`. Nao-fatal de ponta a ponta.
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
            python3 -m pip install ${PIP_USER} --no-deps --force-reinstall "torchaudio==${TA_VER}" \
                    --index-url "https://download.pytorch.org/whl/${CU}" \
                && python3 -c "import torchaudio" 2>/dev/null \
                && echo "--- torchaudio reparado (import ok) ---" \
                || echo "aviso: nao foi possivel reparar torchaudio (siga; o erro aparecera no load se for necessario)"
        }

        ensure_torchaudio

        # ---------------------------------------------------------------------------
        # Montagem dos argumentos. Cada opcional so entra quando a variavel esta setada
        # -- nada de passar string vazia para o argparse.
        # ---------------------------------------------------------------------------
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
            echo "=== treino concluido ==="
        fi
    '

echo "=== Concluido: $(date) ==="
