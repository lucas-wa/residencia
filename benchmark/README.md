# Benchmark de RI multimodal — como rodar

Pipeline de avaliação de modelos de embeddings multimodais em recuperação de
**documento visual** (ViDoRe v2 + mini-benchmark pt-BR dos planos de governo do GovExe).

Ambiente-alvo das rodadas reais: **WSL2 (Linux)** na máquina com RTX 4090 (24 GB).
Os testes automatizados rodam em **CPU, offline**.

## 1. Ambiente

```bash
python3.11 -m venv .venv            # venv isolado; NÃO reusar o .venv-dataset do GovExe
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# GPU (WSL2): ver os comentários do arquivo, as rodas são instaladas à parte
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
pip install faiss-gpu-cu12
```

Para só rodar a suíte de testes bastam `numpy pyyaml pytest`.

### Fallbacks

| falta | o que acontece |
|---|---|
| `flash-attn` | wrappers recarregam o modelo com `attn_implementation="eager"` |
| `faiss-gpu` / `faiss-cpu` | `dense_search` cai para produto interno em numpy (mesmo top-k) |

## 2. Construir o conjunto pt-BR (planos GovExe)

Baixa os PDFs dos 27 planos e rasteriza cada página em PNG a 150 dpi.
Plano com PDF corrompido/ilegível é **pulado com aviso**, sem abortar o lote.

```bash
python -m data.govexe.build_planos_benchmark \
  --parquet /mnt/d/projects/govexe/dataset/planos/train.parquet \
  --out-dir data/govexe/ptbr_planos
# sem --parquet local: usa load_dataset("GovExe/gov-br-linguagem-simples", "planos")
```

Saída: `data/govexe/ptbr_planos/{pages/*.png, corpus.jsonl, queries.jsonl, qrels.tsv}`
(+ `pdf_cache/<seq>.pdf` com os PDFs crus).
Queries vêm de `destaque`, `resumo` e de cada item de `eixos`; a relevância é em nível
de documento (query de um plano → todas as páginas daquele plano).

**O build é resumível e idempotente** (job interrompido não recomeça do zero):

| cache | efeito |
|---|---|
| `corpus.jsonl`+`queries.jsonl`+`qrels.tsv` já existem e não estão vazios | sai cedo: *"benchmark já construído (use --force para refazer)"* |
| `pages/<seq>_p*.png` já existem | reusa os PNGs do plano — não baixa o PDF nem rasteriza |
| `pdf_cache/<seq>.pdf` já existe | reusa o PDF baixado (2ª linha de defesa, se a rasterização morreu no meio) |

`--force` ignora os três e refaz tudo. Reconstruir **não duplica linhas**: corpus,
queries e qrels são indexados por id (`<seq>_p<NN>` / `<seq>_<sufixo>`).

## 3. Rodar os experimentos

ViDoRe v2 (validação estabelecida):

```bash
python -m benchmark.run_vidore --config benchmark/configs/vidore_colqwen2.yaml --limit 50
python -m benchmark.run_vidore --config benchmark/configs/vidore_colpali.yaml  --limit 50
```

pt-BR GovExe:

```bash
python -m benchmark.run_govexe_ptbr \
  --config benchmark/configs/govexe_colqwen2.yaml \
  --data-dir data/govexe/ptbr_planos
```

`--limit N` amostra as N primeiras queries (smoke test barato antes da rodada cheia).

## 4. Execução no cluster (Slurm/H100)

Rodada real na H100 do cluster da UFG (nó `dgx-H100-02`, partição `h100n2`), dentro do
container Singularity `vllm-openai_latest.sif`. Tudo sai de um único script parametrizado:
[`slurm_run.sh`](../slurm_run.sh) na raiz do repo.

### 4.1 Preparar

O bind do cluster é `/raid/user_lucasalves/` → `/work`, então o repo deve ficar em:

```bash
# no host (nó de login)
mkdir -p /raid/user_lucasalves/ufg
git clone <repo> /raid/user_lucasalves/ufg/residencia
cd /raid/user_lucasalves/ufg/residencia
mkdir -p logs            # necessário ANTES do primeiro sbatch (--output aponta p/ logs/)
```

Dentro do container esse caminho aparece como `/work/ufg/residencia`, que é o default de
`RESIDENCIA_DIR` no script. Clonou em outro lugar? Sobrescreva na submissão:
`RESIDENCIA_DIR=/work/outro/caminho sbatch slurm_run.sh`.

As dependências são instaladas no job a partir de [`requirements-slurm.txt`](../requirements-slurm.txt)
— **sem torch** (a imagem vLLM já traz torch+CUDA casados com o driver) e **sem faiss**
(o fallback numpy de `common/index.py` dá o mesmo top-k nestes corpora pequenos). O
container é read-only, então o `pip` instala em `$HOME/.local` com `HOME=/work/.home`.

**Cache de dependências**: o job calcula o `sha256` de `requirements-slurm.txt` e grava
o marcador `$HOME/.local/.residencia_deps_<hash>`. Se o marcador já existe, o
`pip install` é **pulado inteiro** (economiza minutos em cada submissão); editar o
requirements muda o hash e dispara a reinstalação. Quando a imagem tem `uv`, a instalação
usa `uv pip install --system` (bem mais rápido); senão cai para `python3 -m pip install`.
Para forçar: `rm "$HOME"/.local/.residencia_deps_*`.

**Download do Hub**: o corpo exporta `HF_XET_HIGH_PERFORMANCE=1` — o backend Xet do
`huggingface_hub` baixa por chunk e acelera bastante os checkpoints (`hf_transfer` **não**
é usado: mais frágil, sem retomada). Rede instável? Descomente `HF_HUB_DISABLE_XET=1`.

### 4.2 Preflight de versões

Antes de qualquer etapa de GPU, o job roda o preflight:

```bash
python -m benchmark.check_env            # relatório; sai 1 só se status=fail
python -m benchmark.check_env --strict   # sai 1 também em status=warn
```

Ele lê as versões instaladas (`transformers`, `colpali-engine`, `torch`,
`huggingface_hub`, …) e decide o caminho de carregamento do ColQwen2:

| `colqwen_path` | status | quando |
|---|---|---|
| `colpali-engine` | ok | `colpali-engine >0.3.4` **e** `transformers >4.46.1` → checkpoint `vidore/colqwen2-v1.0` |
| `transformers-native` | warn | sem engine compatível, mas `transformers >=4.49` → exige o checkpoint convertido `vidore/colqwen2-v1.0-hf` |
| `none` | fail | nenhum caminho viável → com `set -e` o job **aborta aqui**, antes de gastar GPU |

`STRICT_PREFLIGHT=1` (default `0`) passa `--strict`, tratando o caminho nativo (warn)
como erro.

### 4.3 As submissões

O estágio é escolhido pela variável `STAGE`
(`preflight` | `build` | `vidore` | `govexe` | `all`):

```bash
# 0) só a checagem de versões (segundos, não precisa de GPU):
STAGE=preflight sbatch --gres=gpu:h100:0 slurm_run.sh

# 1) construir o conjunto pt-BR: baixa PDFs e rasteriza páginas — é CPU-only,
#    então pede ZERO GPU (não ocupa a H100 à toa):
STAGE=build sbatch --gres=gpu:h100:0 slurm_run.sh

# 2) ViDoRe v2 (ColQwen2 + ColPali) — GPU (roda o preflight antes):
STAGE=vidore sbatch slurm_run.sh

# 3) pt-BR GovExe, depois que o build terminou — GPU (idem):
STAGE=govexe sbatch slurm_run.sh

# 1b) refazer o conjunto pt-BR do zero (ignora caches: saídas, PNGs e PDFs):
FORCE=1 STAGE=build sbatch --gres=gpu:h100:0 slurm_run.sh

# warn do preflight tratado como erro:
STRICT_PREFLIGHT=1 STAGE=vidore sbatch slurm_run.sh
```

`sbatch slurm_run.sh` sem `STAGE` roda `all` (preflight + build + vidore + govexe) num
job só. `LIMIT=50 STAGE=vidore sbatch slurm_run.sh` faz o smoke test barato (`--limit` só
é repassado quando `LIMIT` não está vazio). `FORCE=1` repassa o `--force` do build (§2) e
só tem efeito no `STAGE=build` — `vidore`/`govexe` não têm essa opção.

> As variáveis (`STAGE`, `LIMIT`, `FORCE`, `RESIDENCIA_DIR`, `HF_CACHE_DIR`, `STRICT_PREFLIGHT`)
> chegam dentro do container via `SINGULARITYENV_*`/`APPTAINERENV_*` — o corpo do
> `bash -c '…'` usa aspas simples e não interpola nada vindo do ambiente.

### 4.4 Onde olham os resultados

- **Logs**: `logs/slurm_<jobid>_residencia.out` e `.err` (no repo, lado do host).
- **Resultados**: `results/vidore.csv` e `results/govexe_ptbr.csv` — append, uma linha por
  execução (ver seção 5).
- **Caches HF**: `/work/.cache/huggingface` (`HF_CACHE_DIR`), fora do container, para os
  downloads de modelo/dataset sobreviverem entre jobs.

### 4.5 TODOs de runtime (checar na primeira rodada)

- **Subset ViDoRe v2**: `benchmark/configs/vidore_*.yaml` usam
  `vidore/biomedical_lectures_v2`. Os quatro subconjuntos do v2 são
  [`biomedical_lectures_v2`](https://huggingface.co/datasets/vidore/biomedical_lectures_v2),
  [`economics_reports_v2`](https://huggingface.co/datasets/vidore/economics_reports_v2),
  [`esg_reports_v2`](https://huggingface.co/datasets/vidore/esg_reports_v2) e
  [`esg_reports_human_labeled_v2`](https://huggingface.co/datasets/vidore/esg_reports_human_labeled_v2)
  (coleção [`vidore/vidore-benchmark-v2`](https://huggingface.co/collections/vidore/vidore-benchmark-v2)).
  Nomes confirmados no Hub (atualizados em 2 jul. 2025); para avaliar outro subconjunto,
  troque `dataset:` nos YAMLs por um dos quatro acima.
- **`colpali-engine` × `transformers`**: agora é o `preflight` (§4.2) que responde isso —
  ele diz se o caminho é `colpali-engine` ou o nativo do `transformers` (que exige o
  checkpoint `vidore/colqwen2-v1.0-hf`). Se cair no nativo, ajustar `model_id` nos YAMLs e
  as classes importadas em `benchmark/models/colvision.py`.
- **`flash-attn`** não é instalado: os wrappers caem para `attn_implementation="eager"`
  (mais lento, mesmo resultado).

### 4.6 Execução em VM única (GCP, L4)

Alternativa ao cluster, quando a fila do Slurm não compensa: uma **VM única com GPU** na
GCP. Aqui **não** há `sbatch` nem Singularity — o `slurm_run.sh` não é usado. O
equivalente é [`run_gcp.sh`](../run_gcp.sh) na raiz do repo, com os **mesmos estágios e as
mesmas variáveis** (`STAGE`, `LIMIT`, `FORCE`, `STRICT_PREFLIGHT`, `HF_CACHE_DIR`).

**Máquina sugerida**: `g2-standard-8` (8 vCPU, 32 GB RAM, **1× NVIDIA L4** de 24 GB),
imagem **Deep Learning VM** (já traz driver NVIDIA + CUDA + PyTorch), disco ≥ **100 GB**
(os PNGs dos 27 planos e os checkpoints do HF ocupam bastante).

```bash
# na VM, após o SSH:
git clone <repo> ~/residencia
cd ~/residencia

bash run_gcp.sh                         # tudo: preflight + build + vidore + govexe
STAGE=preflight bash run_gcp.sh         # só a checagem de versões (segundos)
STAGE=build bash run_gcp.sh             # CPU: baixa os PDFs e rasteriza as páginas
STAGE=vidore bash run_gcp.sh            # ViDoRe v2 (preflight + GPU)
STAGE=govexe bash run_gcp.sh            # pt-BR GovExe (preflight + GPU)

LIMIT=50 STAGE=vidore bash run_gcp.sh   # smoke test barato
FORCE=1 STAGE=build bash run_gcp.sh     # refaz o conjunto pt-BR do zero
STRICT_PREFLIGHT=1 STAGE=vidore bash run_gcp.sh   # warn do preflight vira erro

# cache do HF num disco de dados (checkpoints são grandes):
HF_CACHE_DIR=/mnt/disks/data/hf bash run_gcp.sh
```

**Ambiente Python**: o script cria (uma vez) `.venv-gcp` com
`python3 -m venv --system-site-packages` — o venv **enxerga o torch+CUDA da imagem base**
e isola só o resto (`transformers`, `colpali-engine`, …), instalado do mesmo
[`requirements-slurm.txt`](../requirements-slurm.txt) usado no cluster. Reinstalar torch
por pip quebraria o casamento com o driver. Se a criação do venv falhar, o script segue
instalando no ambiente Python ativo. O **cache por hash** é o mesmo do §4.1 (marcador
`.venv-gcp/.residencia_deps_<hash>`; para forçar, `rm .venv-gcp/.residencia_deps_*`).

Logs vão para o stdout da sessão (não há `logs/slurm_<jobid>_*`); para uma rodada longa,
use `tmux`/`nohup`, p.ex. `nohup bash run_gcp.sh > logs/gcp_$(date +%F_%H%M).log 2>&1 &`.
Os resultados continuam em `results/vidore.csv` e `results/govexe_ptbr.csv`.

## 5. Resultados

Cada execução **acrescenta uma linha** ao CSV, com a config inteira (modelo, dtype,
batch, device, seed) + as métricas `recall@k`, `ndcg@k` e `mrr`:

- `results/vidore.csv`
- `results/govexe_ptbr.csv`

## 6. Configs

Um YAML por experimento em `benchmark/configs/`. Campos: `name`, `model_id`,
`model_kind` (`clip|siglip|colpali|colqwen2|text`), `dataset`, `dtype`, `batch_size`,
`device`, `k_values`, `seed`. `model_kind` também decide o regime de busca:
`colpali`/`colqwen2` → multi-vetor (MaxSim); o resto → dense (faiss/numpy).

## 7. Testes

```bash
python -m pytest -q
```

Rodam em CPU e sem rede: os wrappers de modelo são testados por contrato
(substituição de `_embed_*` via `__new__`) e os loaders usam fixtures locais.
