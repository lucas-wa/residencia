#!/bin/bash
# =====================================================================================
# Submissao Slurm do pipeline de benchmark da residencia (cluster UFG, no "H100 2").
#
# Um unico arquivo, parametrizado pela variavel de ambiente STAGE:
#
#   preflight -> so a checagem de versoes das libs (CPU, segundos)
#   build   -> constroi o conjunto pt-BR dos planos GovExe (CPU-only: baixa PDFs por
#              HTTP e rasteriza as paginas com pypdfium2; NAO usa GPU)
#   vidore  -> preflight + avalia ViDoRe V3 com o trio SOTA 2026         [precisa GPU]
#              (ColEmbed V2 8B + Qwen3-VL-Embedding 8B + ColQwen2.5)
#   govexe  -> preflight + avalia o conjunto pt-BR ja construido         [precisa GPU]
#              (mesmo trio)
#   ensemble-> preflight + demo cross-modal do ensemble (indice textual   [precisa GPU]
#              unico BGE-M3): ranking anotado por modalidade em logs/
#   ensemble_v2-> preflight + captura (internet) + eval nativo: os 5       [precisa GPU]
#              especialistas nativos fundidos por RRF vs. baselines A/B ->
#              results/comparacao_v2.csv + logs/ensemble_v2.txt
#   all     -> preflight + build + vidore + govexe (default)
#
# -------------------------------------------------------------------------------------
# USO
#
#   # 0) uma vez, no cluster: clonar/copiar o repo e criar logs/
#   #    (o diretorio no HOST e /raid/user_lucasalves/ufg/residencia, que aparece
#   #     como /work/ufg/residencia dentro do container)
#   cd /raid/user_lucasalves/ufg/residencia && mkdir -p logs
#
#   # 1) construir o conjunto pt-BR -- CPU-only, pede ZERO GPU:
#   STAGE=build sbatch --gres=gpu:h100:0 slurm_run.sh
#
#   # o dataset dos planos e PRIVADO no Hub (da 401 sem auth). Copie o parquet para o
#   # cluster e aponte PARQUET para ele (caminho DENTRO do bind /work):
#   #   (no host) scp train.parquet \
#   #       <user>@<cluster>:/raid/user_lucasalves/ufg/residencia/data/govexe/planos_train.parquet
#   PARQUET=/work/ufg/residencia/data/govexe/planos_train.parquet \
#       STAGE=build sbatch --gres=gpu:h100:0 slurm_run.sh
#
#   # 2) ViDoRe V3 com o trio SOTA 2026 (GPU):
#   STAGE=vidore sbatch slurm_run.sh
#
#   # so um modelo do trio (ou os configs v2 antigos, mantidos como referencia):
#   VIDORE_CONFIGS=benchmark/configs/v3_colqwen25.yaml STAGE=vidore sbatch slurm_run.sh
#   VIDORE_CONFIGS="benchmark/configs/vidore_colqwen2.yaml benchmark/configs/vidore_colpali.yaml" \
#       STAGE=vidore sbatch slurm_run.sh
#   GOVEXE_CONFIGS=benchmark/configs/govexe_colqwen2.yaml STAGE=govexe sbatch slurm_run.sh
#
#   # 3) pt-BR GovExe (GPU), depois que o build terminou:
#   STAGE=govexe sbatch slurm_run.sh
#
#   # 3b) demo cross-modal do ensemble (GPU, para o BGE-M3); grava o ranking anotado em
#   #     logs/ensemble_demo.txt. Os records vem de ENSEMBLE_RECORDS (default abaixo):
#   STAGE=ensemble sbatch --gres=gpu:h100:1 slurm_run.sh
#   ENSEMBLE_RECORDS=/work/ufg/residencia/data/govexe/ensemble_orcamento/records.jsonl \
#       STAGE=ensemble sbatch --gres=gpu:h100:1 slurm_run.sh
#
#   # 3c) ensemble v2 NATIVO (GPU): captura o corpus Orcamento (internet; o no tem saida)
#   #     retendo os binarios no bind /work, depois avalia os 5 especialistas nativos
#   #     fundidos por RRF vs. os baselines A/B -> results/comparacao_v2.csv +
#   #     logs/ensemble_v2.txt. Corpus/queries/qrels sao parametrizaveis por env:
#   STAGE=ensemble_v2 sbatch --gres=gpu:h100:1 slurm_run.sh
#   ENSEMBLE_V2_CORPUS=/work/ufg/residencia/data/govexe/ensemble_orcamento/records.jsonl \
#       STAGE=ensemble_v2 sbatch --gres=gpu:h100:1 slurm_run.sh
#
#   # 3c.1) humano-no-loop: ENSEMBLE_V2_PHASE separa captura e eval. Recapturar muda os
#   #       item_ids (hash da URL) e invalida o gabarito, entao o corpus e CONGELADO
#   #       primeiro, os qrels sao propostos/revisados em cima dele, e so entao avalia:
#   ENSEMBLE_V2_PHASE=capture STAGE=ensemble_v2 sbatch --gres=gpu:h100:0 slurm_run.sh
#   #       -> preencher/revisar ensemble/eval/qrels_orcamento.tsv com os ids capturados
#   ENSEMBLE_V2_PHASE=eval STAGE=ensemble_v2 sbatch --gres=gpu:h100:1 slurm_run.sh
#   #       (ENSEMBLE_V2_PHASE=all, o default, roda captura+eval num passo so)
#
#   # tudo de uma vez (build + vidore + govexe), GPU:
#   sbatch slurm_run.sh
#
#   # smoke test barato (so as N primeiras queries):
#   LIMIT=50 STAGE=vidore sbatch slurm_run.sh
#
#   # ignorar os caches do build (PDFs/PNGs/saidas) e reconstruir tudo:
#   FORCE=1 STAGE=build sbatch --gres=gpu:h100:0 slurm_run.sh  # refaz o conjunto pt-BR do zero
#
#   # so a checagem de versoes (barata, nao precisa de GPU):
#   STAGE=preflight sbatch --gres=gpu:h100:0 slurm_run.sh
#
#   # tratar warn do preflight como erro (aborta antes de gastar GPU):
#   STRICT_PREFLIGHT=1 STAGE=vidore sbatch slurm_run.sh
#
#   # o no nao isola GPU por job: por padrao o runner escolhe sozinho a GPU mais livre;
#   # GPU_AUTOSELECT=0 desliga e respeita o CUDA_VISIBLE_DEVICES herdado:
#   GPU_AUTOSELECT=0 STAGE=vidore sbatch slurm_run.sh
#
#   # apontar para outro checkout:
#   RESIDENCIA_DIR=/work/ufg/residencia-dev STAGE=vidore sbatch slurm_run.sh
#
# Logs:       logs/slurm_<jobid>_residencia.{out,err}
# Resultados: results/vidore.csv e results/govexe_ptbr.csv (uma linha por execucao)
# =====================================================================================

#SBATCH --job-name=residencia-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --nodelist=dgx-H100-02
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/slurm_%j_residencia.out
#SBATCH --error=logs/slurm_%j_residencia.err

set -euo pipefail

# -------------------------------------------------------------------------------------
# Parametros (todos sobrescreviveis por variavel de ambiente na submissao)
# -------------------------------------------------------------------------------------

# Caminho do repo DENTRO do bind /work. O bind e /raid/user_lucasalves/ -> /work, logo
# /work/ufg/residencia == /raid/user_lucasalves/ufg/residencia no host.
# Ajuste aqui (ou via env) se clonar o repo em outro lugar.
RESIDENCIA_DIR="${RESIDENCIA_DIR:-/work/ufg/residencia}"

STAGE="${STAGE:-all}"
LIMIT="${LIMIT:-}"
# 1 => passa --force ao build (ignora caches e refaz o conjunto pt-BR). So afeta o build.
FORCE="${FORCE:-}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/raid/user_lucasalves/images/vllm-openai_latest.sif}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/work/.cache/huggingface}"
# 1 => o preflight trata warn (ex.: cair no ColQwen2 nativo) como erro e aborta o job.
STRICT_PREFLIGHT="${STRICT_PREFLIGHT:-0}"
# 1 (default) => antes dos estagios de GPU, escolhe sozinho a GPU com MAIS memoria livre
# e exporta CUDA_VISIBLE_DEVICES para ela. 0 => desliga e respeita o
# CUDA_VISIBLE_DEVICES herdado do Slurm/usuario. Ver select_gpu() no corpo do container.
GPU_AUTOSELECT="${GPU_AUTOSELECT:-1}"
# Parquet LOCAL dos planos (caminho DENTRO do bind /work). Vazio => baixa do Hub o
# espelho PUBLICO lucas-wa/gov-br-linguagem-simples, que nao exige token. Aponte para o
# parquet copiado ao cluster para rodar offline / fixar a versao, ex.:
# /work/ufg/residencia/data/govexe/planos_train.parquet
PARQUET="${PARQUET:-}"

# Configs avaliados em cada estagio (lista separada por espaco). O default e o trio SOTA
# 2026: ColEmbed V2 8B (multi-vetor), Qwen3-VL-Embedding 8B (denso) e ColQwen2.5
# (baseline atualizado). Os configs v2 seguem no repo como referencia -- rode-os
# explicitamente se precisar refazer o historico.
VIDORE_CONFIGS="${VIDORE_CONFIGS:-benchmark/configs/v3_colpali.yaml benchmark/configs/v3_colqwen2.yaml benchmark/configs/v3_colembed8b.yaml benchmark/configs/v3_jinav4.yaml benchmark/configs/v3_colqwen25.yaml}"
GOVEXE_CONFIGS="${GOVEXE_CONFIGS:-benchmark/configs/govexe_colembed8b.yaml benchmark/configs/govexe_qwen3vl8b.yaml benchmark/configs/govexe_colqwen25.yaml}"

# records.jsonl consumido pela STAGE=ensemble (caminho DENTRO do bind /work). Parametriza
# o insumo como o GOVEXE_CONFIGS faz com os configs; aponte para outro conjunto por env.
ENSEMBLE_RECORDS="${ENSEMBLE_RECORDS:-data/govexe/ensemble_orcamento/records.jsonl}"

# Insumos da STAGE=ensemble_v2 (caminhos DENTRO do bind /work). O corpus e (re)construido
# pela propria captura da stage; queries/qrels sao os julgamentos do conjunto Orcamento.
# Parametrizados por env como o ENSEMBLE_RECORDS acima.
ENSEMBLE_V2_CORPUS="${ENSEMBLE_V2_CORPUS:-data/govexe/ensemble_orcamento/records.jsonl}"
ENSEMBLE_V2_QUERIES="${ENSEMBLE_V2_QUERIES:-ensemble/eval/queries_orcamento_v2.txt}"
ENSEMBLE_V2_QRELS="${ENSEMBLE_V2_QRELS:-ensemble/eval/qrels_orcamento.tsv}"
# Fase da STAGE=ensemble_v2: capture (so a captura, congela o corpus) | eval (so o eval,
# consome o corpus ja congelado + os qrels revisados) | all (os dois, default).
ENSEMBLE_V2_PHASE="${ENSEMBLE_V2_PHASE:-all}"

case "$ENSEMBLE_V2_PHASE" in
    capture|eval|all) ;;
    *) echo "ENSEMBLE_V2_PHASE invalido: '$ENSEMBLE_V2_PHASE' (use capture|eval|all)" >&2; exit 2 ;;
esac

case "$STAGE" in
    preflight|build|vidore|govexe|ensemble|ensemble_v2|all) ;;
    *) echo "STAGE invalido: '$STAGE' (use preflight|build|vidore|govexe|ensemble|ensemble_v2|all)" >&2; exit 2 ;;
esac

mkdir -p logs

echo "=== Job ${SLURM_JOB_ID:-local} em $(hostname) | $(date) ==="
echo "    STAGE=$STAGE  LIMIT=${LIMIT:-<nenhum>}  FORCE=${FORCE:-<nao>}  RESIDENCIA_DIR=$RESIDENCIA_DIR"
echo "    STRICT_PREFLIGHT=$STRICT_PREFLIGHT  GPU_AUTOSELECT=$GPU_AUTOSELECT"
echo "    PARQUET=${PARQUET:-<Hub (privado)>}"
echo "    CONTAINER_IMAGE=$CONTAINER_IMAGE  HF_CACHE_DIR=$HF_CACHE_DIR"
echo "    VIDORE_CONFIGS=$VIDORE_CONFIGS"
echo "    GOVEXE_CONFIGS=$GOVEXE_CONFIGS"
echo "    ENSEMBLE_RECORDS=$ENSEMBLE_RECORDS"
echo "    ENSEMBLE_V2_CORPUS=$ENSEMBLE_V2_CORPUS  ENSEMBLE_V2_QUERIES=$ENSEMBLE_V2_QUERIES  ENSEMBLE_V2_QRELS=$ENSEMBLE_V2_QRELS"
echo "    ENSEMBLE_V2_PHASE=$ENSEMBLE_V2_PHASE"

# -------------------------------------------------------------------------------------
# Propagacao das variaveis para DENTRO do container.
#
# O corpo do `bash -c '...'` usa aspas SIMPLES de proposito (nada e expandido pelo shell
# externo -> sem risco de quebrar o script com aspas/espacos vindos do ambiente). Para as
# variaveis chegarem la dentro usamos o mecanismo nativo do Singularity: toda variavel
# SINGULARITYENV_FOO do host vira FOO no ambiente do container (APPTAINERENV_FOO e o
# equivalente no Apptainer -- exportamos os dois para funcionar nas duas versoes).
# E mais robusto que interpolar os valores no texto do comando.
# -------------------------------------------------------------------------------------
export SINGULARITYENV_RESIDENCIA_DIR="$RESIDENCIA_DIR"
export SINGULARITYENV_STAGE="$STAGE"
export SINGULARITYENV_LIMIT="$LIMIT"
export SINGULARITYENV_FORCE="$FORCE"
export SINGULARITYENV_HF_CACHE_DIR="$HF_CACHE_DIR"
export SINGULARITYENV_STRICT_PREFLIGHT="$STRICT_PREFLIGHT"
export SINGULARITYENV_GPU_AUTOSELECT="$GPU_AUTOSELECT"
export SINGULARITYENV_PARQUET="$PARQUET"
export SINGULARITYENV_VIDORE_CONFIGS="$VIDORE_CONFIGS"
export SINGULARITYENV_GOVEXE_CONFIGS="$GOVEXE_CONFIGS"
export SINGULARITYENV_ENSEMBLE_RECORDS="$ENSEMBLE_RECORDS"
export SINGULARITYENV_ENSEMBLE_V2_CORPUS="$ENSEMBLE_V2_CORPUS"
export SINGULARITYENV_ENSEMBLE_V2_QUERIES="$ENSEMBLE_V2_QUERIES"
export SINGULARITYENV_ENSEMBLE_V2_QRELS="$ENSEMBLE_V2_QRELS"
export SINGULARITYENV_ENSEMBLE_V2_PHASE="$ENSEMBLE_V2_PHASE"

export APPTAINERENV_RESIDENCIA_DIR="$RESIDENCIA_DIR"
export APPTAINERENV_STAGE="$STAGE"
export APPTAINERENV_LIMIT="$LIMIT"
export APPTAINERENV_FORCE="$FORCE"
export APPTAINERENV_HF_CACHE_DIR="$HF_CACHE_DIR"
export APPTAINERENV_STRICT_PREFLIGHT="$STRICT_PREFLIGHT"
export APPTAINERENV_GPU_AUTOSELECT="$GPU_AUTOSELECT"
export APPTAINERENV_PARQUET="$PARQUET"
export APPTAINERENV_VIDORE_CONFIGS="$VIDORE_CONFIGS"
export APPTAINERENV_GOVEXE_CONFIGS="$GOVEXE_CONFIGS"
export APPTAINERENV_ENSEMBLE_RECORDS="$ENSEMBLE_RECORDS"
export APPTAINERENV_ENSEMBLE_V2_CORPUS="$ENSEMBLE_V2_CORPUS"
export APPTAINERENV_ENSEMBLE_V2_QUERIES="$ENSEMBLE_V2_QUERIES"
export APPTAINERENV_ENSEMBLE_V2_QRELS="$ENSEMBLE_V2_QRELS"
export APPTAINERENV_ENSEMBLE_V2_PHASE="$ENSEMBLE_V2_PHASE"

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

        # Caches do HuggingFace tambem no bind (o $HOME do container some entre jobs).
        export HF_HOME="$HF_CACHE_DIR"
        export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
        export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"

        # Download do Hub: o backend Xet (nativo do huggingface_hub recente) paraleliza
        # por chunk e e bem mais rapido que o HTTP simples nos checkpoints grandes.
        # NAO usamos hf_transfer: ele e mais fragil (sem retomada, erros opacos) e o Xet
        # ja cobre o ganho. Se a rede do no estiver instavel, desligue o Xet:
        #     export HF_HUB_DISABLE_XET=1
        export HF_XET_HIGH_PERFORMANCE=1

        cd "$RESIDENCIA_DIR"
        export PYTHONPATH="$RESIDENCIA_DIR:${PYTHONPATH:-}"

        mkdir -p logs results data/govexe/ptbr_planos "$HF_CACHE_DIR"

        echo "--- python: $(command -v python3) ($(python3 --version 2>&1)) ---"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
            || echo "--- sem GPU visivel (esperado no STAGE=build) ---"

        # Deps SEM torch (a imagem vLLM ja traz torch+CUDA) e SEM faiss (benchmark.
        # common.index cai para numpy; os corpora aqui sao pequenos).
        #
        # Cache de instalacao: o conjunto instalado em $HOME/.local e identificado pelo
        # sha256 do requirements-slurm.txt. Se o marcador correspondente existir, o
        # arquivo nao mudou desde a ultima instalacao e o pip e pulado inteiro (economiza
        # alguns minutos em CADA submissao). Editar o requirements muda o hash -> novo
        # marcador -> reinstala. Para forcar: rm "$HOME"/.local/.residencia_deps_*
        REQ_HASH="$(sha256sum requirements-slurm.txt | cut -d" " -f1)"
        DEPS_MARKER="$HOME/.local/.residencia_deps_$REQ_HASH"
        if [ -f "$DEPS_MARKER" ]; then
            echo "--- deps ja instaladas (marcador: $DEPS_MARKER) ---"
        else
            mkdir -p "$HOME/.local"
            # `uv` resolve/instala muito mais rapido; usa quando a imagem o traz.
            if command -v uv >/dev/null 2>&1; then
                # `uv` resolve/instala muito mais rapido. `--system` falha se o
                # site-packages do container for read-only -> cai para o pip user-local
                # (modo --user), que e o caminho comprovado no cluster. Sem o fallback, a
                # falha do uv derrubaria o job inteiro por causa do `set -e`.
                echo "--- instalando deps com uv (fallback: pip) ---"
                uv pip install --system -r requirements-slurm.txt \
                    || python3 -m pip install --quiet -r requirements-slurm.txt
            else
                echo "--- instalando deps com pip ---"
                python3 -m pip install --quiet -r requirements-slurm.txt
            fi
            touch "$DEPS_MARKER"
        fi

        # ---------------------------------------------------------------------------
        # Guard "preferir o torch do container".
        #
        # Quando o `uv pip install --system` falha (site-packages do container
        # read-only) e caimos no `pip install --user`, o pip resolve a arvore INTEIRA e
        # qualquer dep que declare torch (colpali-engine, peft, ...) instala um torch
        # PROPRIO em $HOME/.local. Esse torch vem sem as libs CUDA casadas com o no --
        # o sintoma real no cluster foi
        #     ImportError: libnvJitLink.so.12: cannot open shared object file
        # -- e, como o site-packages --user vem ANTES do site-packages do container no
        # sys.path, ele SOMBRA o torch funcional da imagem vLLM: o `import torch` de
        # benchmark/models/colvision.py quebra e derruba o run_vidore inteiro.
        #
        # Correcao: depois de instalar as deps, remover do --user o torch e os pacotes
        # que vem colados nele (torchvision/torchaudio/nvidia-*/functorch), para o torch
        # da imagem voltar a prevalecer. Roda em TODO job de proposito: editar o
        # requirements invalida o marcador de deps, dispara a reinstalacao e o torch
        # --user volta. O custo quando nao ha nada a limpar e um teste de existencia de
        # diretorio.
        # ---------------------------------------------------------------------------
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

        # ---------------------------------------------------------------------------
        # Reparo do torchaudio (algumas imagens GPU vem com ele ABI-incompativel com o
        # torch). Visto na Deep Learning VM da GCP: torch 2.9.1+cu129 ok, mas o
        # torchaudio da imagem estoura "_torchaudio.abi3.so: undefined symbol:
        # torch_library_impl". Como transformers 5.x importa torchaudio de forma
        # ansiosa (processing_utils -> audio_utils), QUALQUER `import colpali_engine`
        # quebra junto. Reinstalar so o torchaudio (--no-deps, para nao mexer no torch
        # da imagem) na mesma versao/tag CUDA do torch instalado resolve.
        # Mantido aqui por paridade com run_gcp.sh (a imagem vLLM do cluster costuma
        # estar sa; neste caso a funcao so imprime e sai).
        # ---------------------------------------------------------------------------
        ensure_torchaudio() {
            if python3 -c "import torchaudio" 2>/dev/null; then
                echo "--- torchaudio ok (sem reparo) ---"
                return 0
            fi
            echo "--- torchaudio nao importa (ABI incompativel?): reinstalando casado com o torch ---"

            # Versao e tag CUDA derivadas do torch INSTALADO (nada hardcodado). As
            # fatias de string ficam no bash porque este corpo roda dentro de aspas
            # simples -- nenhuma aspa simples pode aparecer aqui.
            TA_VER="$(python3 -c "import torch;print(torch.__version__)" 2>/dev/null || true)"
            TA_VER="${TA_VER%%+*}"
            CU_RAW="$(python3 -c "import torch;print(torch.version.cuda or str())" 2>/dev/null || true)"
            CU="cu${CU_RAW//./}"
            if [ -z "$TA_VER" ]; then
                echo "aviso: nao foi possivel reparar torchaudio (siga; o erro aparecera no load se for necessario)"
                return 0
            fi

            # Mesma logica do bloco de deps: sem venv ativo o alvo e $HOME/.local
            # (--user), que tem precedencia sobre o site-packages do container
            # read-only; com venv, o alvo e o venv.
            PIP_USER=""
            if [ -z "${VIRTUAL_ENV:-}" ]; then
                PIP_USER="--user"
            fi

            echo "--- pip install torchaudio==$TA_VER (index: $CU) ---"
            # Nao-fatal de ponta a ponta: no cluster o indice do PyTorch pode estar
            # inacessivel e o `set -e` nao pode derrubar o job por isso -- o erro
            # reaparece no load, se for mesmo necessario.
            python3 -m pip install ${PIP_USER} --no-deps --force-reinstall "torchaudio==${TA_VER}" \
                    --index-url "https://download.pytorch.org/whl/${CU}" \
                && python3 -c "import torchaudio" 2>/dev/null \
                && echo "--- torchaudio reparado (import ok) ---" \
                || echo "aviso: nao foi possivel reparar torchaudio (siga; o erro aparecera no load se for necessario)"
        }

        ensure_torchaudio

        # ---------------------------------------------------------------------------
        # Escolha da GPU mais livre.
        #
        # O no dgx-H100-02 (particao h100n2) NAO isola GPU por job: o job enxerga as 8
        # GPUs e a memoria ocupada pelos processos de outros usuarios. Como os configs
        # usam device: cuda, o PyTorch cai no cuda:0 FISICO -- que pode estar cheio de
        # outro job -- e o modelo estoura torch.OutOfMemoryError ja no load, mesmo com
        # outras GPUs do no completamente livres.
        #
        # Correcao: logo antes do trabalho de GPU (depois das deps/torch, para encurtar
        # a janela em que outro job pode tomar a GPU escolhida), consultar o nvidia-smi,
        # pegar o indice com MAIS memoria livre e exportar CUDA_VISIBLE_DEVICES para
        # ele; assim o cuda:0 do PyTorch passa a apontar para a GPU livre.
        # GPU_AUTOSELECT=0 desliga tudo e respeita o CUDA_VISIBLE_DEVICES herdado do
        # Slurm/usuario. Sem GPU visivel (STAGE=build com --gres=gpu:h100:0, ou
        # nvidia-smi ausente) a funcao apenas avisa e segue -- nada aqui pode derrubar
        # o job por causa do `set -e`.
        # ---------------------------------------------------------------------------
        select_gpu() {
            if [ "${GPU_AUTOSELECT:-1}" != "1" ]; then
                echo "--- GPU_AUTOSELECT=${GPU_AUTOSELECT:-1}: auto-selecao desligada (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<herdado>}) ---"
                return 0
            fi

            GPU_LIST="$(nvidia-smi -L 2>/dev/null || true)"
            case "$GPU_LIST" in
                GPU*) ;;
                *) echo "--- sem GPU visivel: sem auto-selecao ---"; return 0 ;;
            esac

            # Linha do nvidia-smi com a maior memoria livre: "indice, livre_MiB".
            # O head fecha o pipe cedo (SIGPIPE no nvidia-smi) e o pipefail esta ativo,
            # dai o `|| true`: a falta de resposta vira string vazia, tratada abaixo.
            GPU_TOP="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
                | sort -t, -k2 -nr | head -n1 || true)"
            GPU_IDX="$(printf "%s" "$GPU_TOP" | cut -d, -f1 | tr -d " ")"
            GPU_FREE="$(printf "%s" "$GPU_TOP" | cut -d, -f2 | tr -d " ")"

            # Indice vazio ou nao-numerico => NAO exporta nada (melhor o default do
            # PyTorch que um CUDA_VISIBLE_DEVICES invalido, que esconde todas as GPUs).
            case "$GPU_IDX" in
                ""|*[!0-9]*)
                    echo "aviso: nao foi possivel determinar a GPU mais livre: mantendo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<herdado>}"
                    return 0
                    ;;
            esac

            export CUDA_VISIBLE_DEVICES="$GPU_IDX"
            echo "--- GPU auto-selecionada: indice $GPU_IDX (mais livre, ${GPU_FREE:-?} MiB) | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ---"
        }

        # --limit so e repassado quando LIMIT nao esta vazio.
        LIMIT_ARGS=()
        if [ -n "${LIMIT:-}" ]; then
            LIMIT_ARGS=(--limit "$LIMIT")
        fi

        run_preflight() {
            echo "=== [preflight] versoes de transformers/colpali-engine/torch ==="
            # Com `set -e`, status=fail (nenhum caminho de carregamento do ColQwen2)
            # aborta o job AQUI, antes de reservar modelo/GPU por 20 minutos a toa.
            STRICT_ARGS=()
            if [ "${STRICT_PREFLIGHT:-0}" = "1" ]; then
                STRICT_ARGS=(--strict)
            fi
            python3 -m benchmark.check_env ${STRICT_ARGS[@]+"${STRICT_ARGS[@]}"}
        }

        run_build() {
            echo "=== [build] conjunto pt-BR dos planos GovExe (CPU) ==="
            # FORCE=1 ignora os caches (saidas, PNGs ja rasterizados, PDFs baixados) e
            # refaz tudo. So vale aqui: run_vidore/run_govexe nao tem --force.
            FORCE_ARGS=()
            if [ "${FORCE:-}" = "1" ]; then
                FORCE_ARGS=(--force)
            fi
            # PARQUET vazio => baixa do Hub o espelho PUBLICO
            # lucas-wa/gov-br-linguagem-simples (config planos), que nao exige token. Com
            # PARQUET apontando para o parquet local no bind /work, le do arquivo e nao
            # toca no Hub (modo offline / versao fixa).
            python3 -m data.govexe.build_planos_benchmark \
                --parquet "${PARQUET:-}" \
                --out-dir data/govexe/ptbr_planos \
                ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
                ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}
        }

        # Trio SOTA 2026 avaliado no ViDoRe V3. Os configs v2 (vidore_colqwen2/
        # vidore_colpali) continuam no repo como REFERENCIA historica e NAO rodam por
        # padrao -- para refaze-los, use VIDORE_CONFIGS (ver bloco de USO no topo).
        VIDORE_CONFIGS="${VIDORE_CONFIGS:-benchmark/configs/v3_colpali.yaml benchmark/configs/v3_colqwen2.yaml benchmark/configs/v3_colembed8b.yaml benchmark/configs/v3_jinav4.yaml benchmark/configs/v3_colqwen25.yaml}"
        GOVEXE_CONFIGS="${GOVEXE_CONFIGS:-benchmark/configs/govexe_colembed8b.yaml benchmark/configs/govexe_qwen3vl8b.yaml benchmark/configs/govexe_colqwen25.yaml}"

        run_vidore() {
            echo "=== [vidore] ViDoRe V3 (GPU) ==="
            for cfg in $VIDORE_CONFIGS; do
                echo "--- $cfg ---"
                python3 -m benchmark.run_vidore \
                    --config "$cfg" \
                    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
            done
        }

        run_govexe() {
            echo "=== [govexe] conjunto pt-BR (GPU) ==="
            for cfg in $GOVEXE_CONFIGS; do
                echo "--- $cfg ---"
                python3 -m benchmark.run_govexe_ptbr \
                    --config "$cfg" \
                    --data-dir data/govexe/ptbr_planos \
                    --out results/govexe_ptbr.csv \
                    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
            done
        }

        run_ensemble() {
            echo "=== [ensemble] demo cross-modal (indice textual unico BGE-M3, GPU) ==="
            # records.jsonl parametrizado por ENSEMBLE_RECORDS (como o govexe parametriza
            # os configs); queries fixas do conjunto orcamento; saida anotada em logs/.
            # O embedder BGE-M3 so e construido aqui (import tardio) -> precisa de GPU.
            python3 -m ensemble.run_ensemble \
                --records "${ENSEMBLE_RECORDS:-data/govexe/ensemble_orcamento/records.jsonl}" \
                --queries ensemble/queries_orcamento.txt \
                --out logs/ensemble_demo.txt
        }

        run_ensemble_v2_capture() {
            # captura: (re)constroi o corpus Orcamento baixando os binarios de cada eixo
            # e RETENDO-OS no bind /work (ao lado do records.jsonl). Precisa de internet
            # -- o no do cluster tem saida. ENSEMBLE_V2_CORPUS parametriza o destino.
            # ATENCAO: recapturar muda os item_ids (hash da URL) e invalida os qrels.
            python3 -m ensemble.build_corpus \
                "${ENSEMBLE_V2_CORPUS:-data/govexe/ensemble_orcamento/records.jsonl}"
        }

        run_ensemble_v2_eval() {
            # eval: monta os 5 especialistas nativos + retriever A + B (import tardio ->
            # os modelos/checkpoints carregam AQUI, por isso precisa de GPU), funde por
            # RRF e grava o CSV comparativo por (sistema, eixo) + o log da execucao.
            python3 -m ensemble.eval.run_eval \
                --corpus "${ENSEMBLE_V2_CORPUS:-data/govexe/ensemble_orcamento/records.jsonl}" \
                --queries "${ENSEMBLE_V2_QUERIES:-ensemble/eval/queries_orcamento_v2.txt}" \
                --qrels "${ENSEMBLE_V2_QRELS:-ensemble/eval/qrels_orcamento.tsv}" \
                --out results/comparacao_v2.csv \
                --log logs/ensemble_v2.txt
        }

        run_ensemble_v2() {
            # ENSEMBLE_V2_PHASE separa as fases para o humano-no-loop: congelar o corpus
            # (capture), revisar os qrels em cima dele, e so entao avaliar (eval).
            echo "=== [ensemble_v2] fase=${ENSEMBLE_V2_PHASE:-all} (captura=internet, eval=GPU) ==="
            case "${ENSEMBLE_V2_PHASE:-all}" in
                capture) run_ensemble_v2_capture ;;
                eval)    run_ensemble_v2_eval ;;
                all)     run_ensemble_v2_capture; run_ensemble_v2_eval ;;
                *) echo "ENSEMBLE_V2_PHASE invalido: ${ENSEMBLE_V2_PHASE:-all}" >&2; return 2 ;;
            esac
        }

        select_gpu

        case "$STAGE" in
            preflight) run_preflight ;;
            build)     run_build ;;
            vidore)    run_preflight; run_vidore ;;
            govexe)    run_preflight; run_govexe ;;
            ensemble)  run_preflight; run_ensemble ;;
            ensemble_v2) run_preflight; run_ensemble_v2 ;;
            all)       run_preflight; run_build; run_vidore; run_govexe ;;
        esac

        echo "=== STAGE=$STAGE concluido ==="
    '

echo "=== Concluido: $(date) ==="
