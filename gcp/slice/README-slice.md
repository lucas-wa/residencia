# Fatia vertical GovExe na GCP — multimodal → GCS → embedding (L4) → Cloud SQL/pgvector → consulta

Primeira fatia **fim a fim** da arquitetura de `fusao/coleta-gcp.md`, com o **modelo
unificado** `Qwen/Qwen3-VL-Embedding-2B` (o mesmo wrapper do benchmark:
`benchmark/models/qwen3vl_embed.py`).

É a validação prática da **Abordagem A** de `fusao/multimodal.md` (§2): **quatro
modalidades no mesmo índice, pelo mesmo modelo, no mesmo espaço vetorial** — uma consulta
textual recupera página de documento, foto, texto nativo e áudio transcrito da **mesma**
tabela `govexe_qwen3vl`.

Sobre ela a fatia ganhou um **segundo espaço**: `govexe_clap` (áudio **nativo**). Um
**espaço** é um modelo com a sua própria geometria — checkpoint, dimensão e **tabela
própria** —, e uma **arquitetura** é a lista de espaços que ela consulta:

| Arquitetura | Espaços | Tabelas |
|---|---|---|
| `A` | `qwen3vl` | `govexe_qwen3vl` |
| `A+áudio` | `qwen3vl` + `clap` | `govexe_qwen3vl` + `govexe_clap` |
| `B` | — (nenhum) | — (nenhuma) |

`B` é o **ensemble de especialistas** de `fusao/multimodal.md` §3, e o seu primeiro braço é
o **eixo documento**: ColQwen2.5 *late interaction*. Ele não cabe num espaço: emite um vetor
por **token** e pontua por **MaxSim**, enquanto um espaço é "1 vetor denso por item, cosseno,
1 tabela pgvector". Por isso `B` tem a lista de espaços **vazia** e vive num segundo registro
(`SPECIALISTS`, no mesmo `spaces.py`), e a sua rodada **dispensa Cloud SQL inteiro** — as 93
páginas são ranqueadas em memória (`gcp/slice/doc_specialist.py`). `spaces.uses_pgvector(arch)`
é o que roteia tanto o `eval.py` quanto o `run_slice.sh`.

O registro declarativo (fonte única de verdade de tabela, checkpoint, dimensão e
modalidades de cada espaço) é `gcp/slice/spaces.py`; `embed_job --space` e a consulta
multi-espaço só o leem. Espaços **não** se somam: os dois rankings se encontram na **fusão
por RRF** (`gcp/slice/fusion.py`) — posição, nunca score, porque a similaridade de espaços
diferentes não se compara.

```
páginas (*.png) · fotos (*.jpg) · textos · áudios (*.mp3)
        │  collector.py            (Bronze, particionado por fonte/data; JSONL de records)
        ▼
  gs://<bucket>/bronze/fonte=<fonte>/dt=YYYY-MM-DD/<arquivo>
        │  embed_job.py --space    (VM L4 g2-standard-4, EFÊMERA)
        │
        │  --space qwen3vl                           --space clap
        │     ├─ doc_image · image → embed_images() ─┐    └─ audio_native →
        │     ├─ text              → embed_texts()  ─┤       embed_audio() (CLAP)
        │     └─ audio_asr → Whisper (asr.py)       ─┘       forma de onda, SEM ASR
        ▼                                            ▼
  Cloud SQL (PostgreSQL + pgvector) — UMA TABELA POR ESPAÇO, UPSERT por item_id
    govexe_qwen3vl  VECTOR(1024)                 govexe_clap  VECTOR(512)
        │  query.py       (espaço único: texto → top-k por `<=>`, `--modality`)
        │  eval.py        (multi-espaço: top-k de cada espaço → RRF → métricas)
        ▼
  top-k itens — de qualquer modalidade (cross-modal), com crédito e pointer
```

| Script | Onde roda | O que faz |
|---|---|---|
| `gcp/slice/audio_collector.py` | máquina local ou Cloud Shell | baixa áudio de fonte pública (yt-dlp) e emite os records `audio_asr` |
| `gcp/slice/collector.py` | máquina local ou Cloud Shell | sobe páginas, fotos e áudios para o bucket **Bronze**; enriquece o JSONL de records |
| `gcp/slice/asr.py` | VM L4 (dentro do job) | **Whisper large-v3** atrás da interface `Transcriber` (áudio → texto) |
| `gcp/slice/spaces.py` | — (registro) | declara espaços (tabela, checkpoint, dimensão, modalidades) e arquiteturas |
| `gcp/slice/embed_job.py` | **VM L4 efêmera** | baixa do Bronze, roteia por modalidade, embute no espaço do `--space` e faz **upsert** no pgvector |
| `gcp/slice/fusion.py` | onde a consulta rodar | **RRF**: funde os rankings de espaços diferentes por posição |
| `gcp/slice/query.py` | VM ou Cloud Shell (com proxy) | CLI de **espaço único**: texto → top-k, com filtro `--modality`; e `search_architecture`, a consulta multi-espaço |
| `gcp/slice/eval.py` | VM ou Cloud Shell (com proxy) | roda um conjunto de avaliação por arquitetura → recall@k/nDCG@k/MRR + latência no CSV comparativo |
| `gcp/slice/run_slice.sh` | máquina local | orquestra a rodada paga inteira (provisiona → indexa → avalia → **teardown garantido**) |

## Modalidades indexadas

| `modality` | Espaço | O que é | Como vira vetor | Campos típicos |
|---|---|---|---|---|
| `doc_image` | `qwen3vl` | página de plano de governo rasterizada | `embed_images()` | `uri` = objeto no Bronze |
| `image` | `qwen3vl` | foto de agência pública (EBC, Agência Câmara…) | `embed_images()` | `credit` **obrigatório**, `pointer_uri` = foto original |
| `text` | `qwen3vl` | texto nativo (matéria, legenda, ementa) | `embed_texts()` | `content_text`, `uri` da matéria; `pointer_uri` NULL |
| `audio_asr` | `qwen3vl` | áudio (sessão, RadioAgência) | ASR (Whisper) → `embed_texts()` | `content_text` = **transcrição**, `pointer_uri` = áudio original |
| `audio_native` | `clap` | o **mesmo** áudio, como forma de onda | `embed_audio()` (CLAP) | sem `content_text` (não há transcrição); `pointer_uri` = áudio original |

> **No espaço `qwen3vl` o áudio não tem embedder próprio.** Ele é colapsado em texto pelo
> Whisper (*pointer vector*, `fusao/multimodal.md` §3 e §4) e embutido pelo **mesmo**
> `Qwen3-VL-Embedding` que embute as imagens — por isso as quatro modalidades caem no
> mesmo espaço e um único índice basta. O embedding **nativo** de áudio é o espaço `clap`
> (`--space clap`, tabela própria): o **mesmo clipe**, com o **mesmo `item_id`**, indexado
> em paralelo pelos dois caminhos — e é esse `item_id` compartilhado que permite ao RRF
> somar as contribuições. *Ensemble* multi-modelo sobre a mesma modalidade continua fora
> desta fatia.

**Custo e disciplina:** a VM L4 só fica ligada durante a ingestão (`start` → job →
`stop`/`delete`); o Cloud SQL é parado ou apagado no fim (§9). A senha do banco **nunca**
aparece em flag, em código ou no git — vem do **Secret Manager** para a variável
`PGPASSWORD` (§3).

---

## 0. Pré-requisitos e variáveis

`gcloud` autenticado (`gcloud auth login`), faturamento ativo no projeto, e o corpus já
construído localmente (`STAGE=build bash run_gcp.sh`, ou qualquer diretório de PNGs).

```bash
export PROJECT_ID="govexe-residencia"           # ajuste
export REGION="us-central1"
export ZONE="us-central1-a"                     # zona com L4 disponível
export BUCKET="gs://${PROJECT_ID}-bronze"
export INSTANCE="govexe-pg"
export DB="govexe"
export DB_USER="postgres"                       # o superusuário que a instância já traz
export SECRET="govexe-pg-password"
export VM="govexe-embed"
export DT="$(date +%F)"                         # partição dt=YYYY-MM-DD

gcloud config set project "$PROJECT_ID"
gcloud config set compute/region "$REGION"
```

## 1. Habilitar as APIs

```bash
gcloud services enable \
    compute.googleapis.com \
    sqladmin.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com
```

## 2. Bucket Bronze

```bash
gcloud storage buckets create "$BUCKET" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
```

## 3. Senha do banco no Secret Manager

A senha é gerada localmente e **só** existe no Secret Manager — nenhum arquivo, nenhum
flag, nenhum commit.

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))" \
  | tr -d '\n' \
  | gcloud secrets create "$SECRET" --data-file=- --replication-policy=automatic

# leitura (é assim que a VM popula a variável de ambiente):
gcloud secrets versions access latest --secret="$SECRET"
```

## 4. Cloud SQL (PostgreSQL) + extensão pgvector + tabela

```bash
gcloud sql instances create "$INSTANCE" \
    --project="$PROJECT_ID" \
    --database-version=POSTGRES_16 \
    --edition=ENTERPRISE \
    --tier=db-custom-1-3840 \
    --region="$REGION" \
    --storage-size=10GB \
    --storage-type=SSD \
    --no-backup

gcloud sql databases create "$DB" --instance="$INSTANCE"

# `postgres` já existe na instância: o que falta é dar a ele a senha do Secret Manager.
gcloud sql users set-password "$DB_USER" \
    --instance="$INSTANCE" \
    --password="$(gcloud secrets versions access latest --secret=$SECRET)"
```

> `--tier=db-g1-small` é mais barato e também dá conta desta fatia (algumas centenas de
> páginas). O `--password=` acima deixa a senha no histórico do shell da sua máquina (e na
> tabela de processos enquanto o `gcloud` roda) — se isso incomodar, use
> `gcloud sql users set-password "$DB_USER" --instance="$INSTANCE" --prompt-for-password`
> e cole o valor do secret. O `run_slice.sh` faz exatamente este `set-password` (com
> `users create` como fallback) e mantém a linha fora do log.

**Habilitar o pgvector** (uma vez, dentro do banco):

```bash
gcloud sql connect "$INSTANCE" --user="$DB_USER" --database="$DB"
# senha: a do Secret Manager (§3). Já no psql:
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;
\dx                       -- confirma: vector | 0.8.x | public
```

**Tabelas — uma por espaço.** O `embed_job.py` as cria sozinho no primeiro upsert (ele lê
tabela e dimensão do registro, `gcp/slice/spaces.py`, a partir do `--space`). Duas tabelas,
mesmo DDL, dimensões diferentes — espaços distintos não são comparáveis no mesmo índice:

| `--space` | Tabela | `VECTOR(n)` | Checkpoint | Modalidades |
|---|---|---|---|---|
| `qwen3vl` (default) | `govexe_qwen3vl` | 1024 | `Qwen/Qwen3-VL-Embedding-2B` | `doc_image`, `image`, `text`, `audio_asr` |
| `clap` | `govexe_clap` | 512 | `laion/larger_clap_general` | `audio_native` |

Para criar à mão, use a dimensão que o espaço declara (`{table}`/`{dim}` abaixo saem do
registro — é literalmente o DDL do `embed_job.py`):

```sql
CREATE TABLE IF NOT EXISTS govexe_qwen3vl (     -- govexe_clap: idem, com VECTOR(512)
    item_id        TEXT PRIMARY KEY,
    modality       TEXT NOT NULL,          -- doc_image | image | text | audio_asr
    source         TEXT NOT NULL,
    uri            TEXT NOT NULL,          -- fonte da representação indexada
    pointer_uri    TEXT,                   -- binário original (foto/áudio); NULL p/ texto
    credit         TEXT,                   -- crédito (autor/fonte) — o pointer auditável
    content_text   TEXT,                   -- texto nativo OU transcrição ASR
    content_sha256 TEXT NOT NULL,
    model_id       TEXT NOT NULL,
    embedding      VECTOR(1024) NOT NULL,  -- = a dimensão do espaço (teto do HNSW: 2000)
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS govexe_qwen3vl_embedding_cos
    ON govexe_qwen3vl USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS govexe_qwen3vl_modality
    ON govexe_qwen3vl (modality);          -- apoia o `--modality` do query.py
```

> **Teto de 2000 dimensões.** O índice do pgvector (`hnsw` e também `ivfflat`) não aceita
> coluna acima de **2000 dimensões** — passar disso faz o `CREATE INDEX` estourar
> `column cannot have more than 2000 dimensions for hnsw index`. O
> `Qwen/Qwen3-VL-Embedding-2B` entrega **2048** nativas, então a fatia indexa em **1024**
> por padrão (`--dim`, Matryoshka/MRL: trunca o vetor e renormaliza, preservando o
> cosseno). Indexar as 2048 nativas exigiria abrir mão do índice (busca exata, sem HNSW);
> com `--dim` acima de 2000 o `embed_job.py` falha de propósito, antes do DDL, com um
> recado apontando este teto. A consulta do §8 usa a mesma dimensão por padrão — se você
> indexar com outro `--dim`, passe o mesmo valor ao `query.py`.

`item_id` é a chave primária e é o que torna a ingestão idempotente:

- **páginas** (`doc_image`): o **stem do arquivo** (`<tse_seq>_p<NN>`), o mesmo id de
  página do `corpus.jsonl`;
- **foto/texto/áudio**: o `item_id` do record, quando ele traz um (ex.: `foto:ebc-1`);
  senão, `"<modality>:<16 hex do sha256 da uri>"` — determinístico e estável entre
  execuções (`collector.default_item_id`, reexportado pelo `embed_job`).

`content_sha256` registra o **conteúdo que gerou o vetor**: os bytes do arquivo nas
modalidades de imagem e o próprio texto embutido (a transcrição, no caso do áudio) nas de
texto.

> **Tabelas antigas (`govexe_pages`, `govexe_items`).** Nenhuma das duas é alterada ou
> apagada pelo job — e nenhuma é mais usada: cada espaço tem a **sua** tabela
> (`govexe_qwen3vl`, `govexe_clap`), que é o que torna possível indexar geometrias
> diferentes lado a lado. Como o Cloud SQL desta fatia é **efêmero** (o `run_slice.sh` o
> apaga no teardown de toda rodada), não há migração a fazer: o caminho é **reingerir**,
> e o `embed_job.py` cria tabela e índices no primeiro upsert. Para aproveitar uma
> `govexe_items` já populada com a mesma dimensão, sem reprocessar GPU:
>
> ```sql
> INSERT INTO govexe_qwen3vl
>     (item_id, modality, source, uri, pointer_uri, credit, content_text,
>      content_sha256, model_id, embedding, updated_at)
> SELECT item_id, modality, source, uri, pointer_uri, credit, content_text,
>        content_sha256, model_id, embedding, updated_at
> FROM govexe_items
> ON CONFLICT (item_id) DO NOTHING;
> ```
>
> (Rode o `embed_job.py` uma vez antes, para a tabela e os índices existirem.) `--table
> <nome>` ainda existe e sobrescreve a tabela do espaço — mas ele **não** muda checkpoint
> nem dimensão, então apontá-lo para uma tabela de outro espaço é erro na certa; use-o só
> para indexar o MESMO espaço num nome diferente, e passe o mesmo `--table` ao `query.py`.

## 5. Coletor → Bronze (da máquina local)

```bash
python -m gcp.slice.collector \
    --bucket "$BUCKET" \
    --images-dir data/govexe/ptbr_planos/pages \
    --source tse_planos \
    --date "$DT" \
    --limit 200                  # tire o --limit para subir o corpus inteiro

gcloud storage ls "$BUCKET/bronze/fonte=tse_planos/dt=$DT/" | head
```

Rodar de novo não reenvia nada (o coletor pula objetos existentes); `--force`
sobrescreve. Áudio entra pelo mesmo caminho (`--glob '**/*.mp3'`): a modalidade sai da
extensão (`.mp3/.wav/.m4a/.ogg/.opus/.flac` → `audio_asr`); para as demais, use
`--modality` (`doc_image` por padrão, `image` para fotos).

### 5.1 JSONL de *records* — o lote misto

Foto, texto nativo e áudio entram por um **JSONL** (um objeto por linha). Campos:

| Campo | Obrigatório | Observação |
|---|---|---|
| `modality` | **sim** | `doc_image` \| `image` \| `text` \| `audio_asr` |
| `image_path` / `audio_path` | binário local | caminho do arquivo a subir/embutir |
| `blob` | alternativa | objeto já no Bronze (o job resolve com `--records-store`) |
| `text` | em `text` | o texto nativo a embutir |
| `uri` | — | fonte da representação; default: o objeto no Bronze ou o caminho local |
| `pointer_uri` | — | binário/URL original preservado |
| `credit` | nas fontes que exigem | autor/fonte — **obrigatório** para foto de agência |
| `item_id` | — | id natural; sem ele, o default da §4 |
| `source` | — | sobrescreve o `--source` do CLI, por linha |

```jsonl
{"modality": "doc_image", "image_path": "pages/123456_p07.png", "source": "tse_planos"}
{"item_id": "foto:ebc-1", "modality": "image", "image_path": "fotos/hospital.jpg", "pointer_uri": "https://agenciabrasil.ebc.com.br/.../hospital.jpg", "credit": "Foto: Fulano/Agência Brasil", "source": "ebc"}
{"item_id": "texto:camara-1", "modality": "text", "text": "meta de ampliar a atenção básica em saúde", "uri": "https://www.camara.leg.br/noticias/1", "credit": "Agência Câmara", "source": "agencia_camara"}
{"item_id": "audio:camara-1", "modality": "audio_asr", "audio_path": "audios/sessao.mp3", "pointer_uri": "https://www.youtube.com/watch?v=exemplo", "credit": "TV Câmara", "source": "tv_camara"}
```

O coletor sobe os binários para o Bronze e **reescreve** o JSONL com `blob`/`uri` (e
`pointer_uri`, quando o record não trouxe um) — essa saída é a entrada do `embed_job.py`:

```bash
python -m gcp.slice.collector \
    --bucket "$BUCKET" \
    --records lote.jsonl \
    --out-records lote-bronze.jsonl \
    --date "$DT"
```

Records de texto nativo não têm binário e passam intactos. `credit` nunca é descartado —
é ele que cumpre a exigência de licença das fontes de `fusao/coleta-gcp.md` §6.

### 5.2 Captura de áudio (`audio_collector.py`)

Os records `audio_asr` da §5.1 podem ser escritos à mão, mas quem os produz em escala é o
`audio_collector.py`: dada a URL pública de uma sessão (YouTube da **SenadoTV**/**TV
Câmara**) ou de uma mídia direta (**RadioAgência**, **Senado Multimídia**), ele baixa o
áudio e emite o JSONL. Cada etapa fica no seu lugar — **ele não sobe nada e não
transcreve nada**:

```
audio_collector.py   URL pública  → áudio local + records      (local / Cloud Shell)
collector.py         records      → binário no Bronze + blob/uri
embed_job.py         records      → ASR (asr.py) → embed_texts → govexe_qwen3vl
                     (ou, com --space clap, o MESMO clipe → embed_audio → govexe_clap)
```

```bash
pip install -r gcp/slice/requirements-collect.txt    # yt-dlp (+ google-cloud-storage)
sudo apt-get install -y ffmpeg                       # requisito de SISTEMA do yt-dlp

python -m gcp.slice.audio_collector \
    --url "https://www.youtube.com/watch?v=<id-da-sessao>" \
    --url "https://www.camara.leg.br/radio/programas/<id>.mp3" \
    --source tv_camara --credit "TV Câmara" \
    --audio-dir data/govexe/audio \
    --out-records audios.jsonl \
    --max-seconds 600                 # sessão legislativa passa de uma hora
```

```jsonl
{"item_id": "audio_asr:552e77e1bc4d355a", "modality": "audio_asr", "audio_path": "data/govexe/audio/<id>.wav", "pointer_uri": "https://www.youtube.com/watch?v=<id-da-sessao>", "credit": "TV Câmara", "source": "tv_camara"}
```

Esse JSONL entra **sem alteração** no `collector.py --records` (§5.1) e depois no
`embed_job.py --records` (§7). Detalhes que importam:

- **`item_id` segue a fonte**, não o arquivo: `audio_asr:<hash da URL de origem>`
  (`collector.default_item_id`). Recapturar a mesma sessão cai no mesmo item — upsert, não
  duplicata — mesmo que o arquivo baixado mude de nome.
- **`pointer_uri` = a URL pública** e `credit` é **obrigatório** (flag `--credit`): são o
  *pointer* auditável exigido pelas licenças de `fusao/coleta-gcp.md` §6.
- **ffmpeg é requisito de sistema** (não é pacote pip): o yt-dlp o usa para extrair a
  trilha de áudio e para o corte de `--max-seconds`. Sem ele, o download falha no
  pós-processamento.
- **Só fonte pública, pelo endereço público.** O downloader não expõe cookie, login nem
  chave — nada de DRM, paywall ou área logada. Captura de **sinal** de TV/rádio (que exige
  ofício) está fora desta fatia.
- URL repetida no lote baixa uma vez só; URL que falha é registrada no log e **pulada** —
  um lote não morre inteiro por causa de uma sessão fora do ar (o CLI sai com `1` se
  nenhuma URL vingar).
- `--downloader fake` grava um WAV canônico (silêncio, duração = `--max-seconds`) **sem
  rede nem yt-dlp**: é como ensaiar o fluxo inteiro offline (§11).

## 6. VM L4 efêmera (g2-standard-4)

Permissões do service account da VM (Storage, Cloud SQL e Secret Manager):

```bash
PROJECT_NUM="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
SA="${PROJECT_NUM}-compute@developer.gserviceaccount.com"
for ROLE in roles/storage.objectViewer roles/cloudsql.client roles/secretmanager.secretAccessor; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${SA}" --role="$ROLE"
done
```

Criar a VM (driver NVIDIA instalado pela imagem Deep Learning):

```bash
gcloud compute instances create "$VM" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --maintenance-policy=TERMINATE \
    --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --metadata="install-nvidia-driver=True" \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

> **A família de imagem importa.** `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` (torch 2.9 +
> CUDA 12.9 + driver 580) é a que funcionou de ponta a ponta — é o default do
> `run_slice.sh`. O venv da VM é criado com `--system-site-packages` **de propósito**: ele
> enxerga o torch da imagem, já casado com o driver. Reinstalar torch por `pip` (direta ou
> indiretamente, por uma dependência que o declare) sombreia o torch bom com um sem as
> libs CUDA do nó — o `run_slice.sh` chega a remover um torch de venv se encontrar.
>
> **L4 dá STOCKOUT.** Na rodada real nenhuma zona `us-*` tinha capacidade de
> `g2-standard-4`; só `northamerica-northeast2-a`. Por isso o `run_slice.sh` tenta uma
> lista de zonas em ordem (`--zones`) em vez de uma só.

Liga/desliga (a VM parada **não** cobra vCPU/GPU, só o disco):

```bash
gcloud compute instances stop  "$VM" --zone="$ZONE"
gcloud compute instances start "$VM" --zone="$ZONE"
```

## 7. Na VM: ambiente, proxy e job de embedding

```bash
gcloud compute ssh "$VM" --zone="$ZONE"
```

```bash
nvidia-smi                                   # confirma a L4 (print nº 1)

sudo apt-get update && sudo apt-get install -y git
git clone <URL-DO-REPO> residencia && cd residencia
python3 -m venv --system-site-packages .venv-gcp && source .venv-gcp/bin/activate
pip install -r requirements-slurm.txt        # mesmas deps do run_gcp.sh (sem torch)
pip install google-cloud-storage "psycopg[binary]"   # adaptadores desta fatia
pip install librosa soundfile                # só para --space clap (decode de áudio)
```

> O venv é `--system-site-packages` **de propósito**: o torch tem de ser o da imagem (§6).

O `faster-whisper` (ASR da modalidade `audio_asr`) já vem no `requirements-slurm.txt`;
os pesos do `large-v3` só são baixados na **primeira transcrição** (import *lazy*), e um
lote sem áudio nunca os toca. `librosa`/`soundfile` são a mesma história para o CLAP: o
wrapper os importa *lazy*, então uma rodada `--arch A` não precisa deles — mas sem eles o
`--space clap` só quebraria lá na frente, com a GPU já paga (por isso o `run_slice.sh` os
instala no bootstrap e valida o import antes de indexar).

Cloud SQL Auth Proxy (expõe o banco em `127.0.0.1:5432`, sem abrir IP público):

```bash
CONN="$(gcloud sql instances describe "$INSTANCE" --format='value(connectionName)')"
curl -o cloud-sql-proxy \
    https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.14.1/cloud-sql-proxy.linux.amd64
chmod +x cloud-sql-proxy
./cloud-sql-proxy "$CONN" &
```

Senha via Secret Manager e ingestão:

```bash
export PGPASSWORD="$(gcloud secrets versions access latest --secret=govexe-pg-password)"

python -m gcp.slice.embed_job \
    --bucket "$BUCKET" \
    --blob-prefix "bronze/fonte=tse_planos/dt=$DT/" \
    --work-dir /tmp/govexe-bronze \
    --source tse_planos \
    --space qwen3vl \
    --device cuda --batch-size 8 \
    --pg-host 127.0.0.1 --pg-port 5432 --pg-db "$DB" --pg-user "$DB_USER"
```

Saída esperada: `espaco qwen3vl: tabela govexe_qwen3vl, checkpoint …, dim 1024` e, no fim,
`N itens gravados; total no índice: N`.

`--space` é **a** flag que decide a rodada: dela saem tabela, checkpoint, dimensão e as
modalidades aceitas (tudo de `gcp/slice/spaces.py`). O default é `qwen3vl` — a Abordagem
A —, então a linha acima é a mesma de sempre. Modalidade fora do espaço é erro antes de
qualquer GPU: um record `audio_native` em `--space qwen3vl` (ou `doc_image` em
`--space clap`) para na validação do lote.

**Lote misto** (foto + texto + áudio, do JSONL da §5.1) — o mesmo job, mesmo espaço, mesmo
checkpoint; o que muda é só a entrada:

```bash
python -m gcp.slice.embed_job \
    --records lote-bronze.jsonl \
    --records-store "$BUCKET" \
    --work-dir /tmp/govexe-bronze \
    --asr-backend faster-whisper --asr-model large-v3 --asr-language pt \
    --device cuda --batch-size 8 \
    --pg-host 127.0.0.1 --pg-port 5432 --pg-db "$DB" --pg-user "$DB_USER"
```

O job separa o lote internamente (`doc_image`/`image` → `embed_images`;
`text`/`audio_asr` → `embed_texts`), transcreve os áudios antes de embutir (log
`asr: <arquivo> -> N caracteres`) e grava tudo na tabela do espaço, na ordem do lote.

**Áudio nativo (arquitetura A+áudio)** — os **mesmos** clipes, agora como forma de onda,
no segundo espaço. O `item_id` é preservado (é ele que liga o clipe nos dois espaços e o
que o RRF soma); o que muda é a `modality`, que passa a `audio_native`:

```bash
# o run_slice.sh reescreve `modality` para audio_native antes desta chamada
python -m gcp.slice.embed_job \
    --records audios-native.jsonl \
    --space clap \
    --device cuda --batch-size 1 \
    --pg-host 127.0.0.1 --pg-port 5432 --pg-db "$DB" --pg-user "$DB_USER"
```

Aqui **não há ASR**: o CLAP embute o binário direto (`benchmark/models/clap_embed.py`,
que importa `librosa`/`soundfile` de forma *lazy* — instale-os na VM). A tabela é
`govexe_clap`, com `VECTOR(512)`.

> **A dimensão não é negociável por flag.** Ela é o *tipo da coluna* e vem do espaço
> (1024 no `qwen3vl`, 512 no `clap`). `--dim` diferente do registro só é aceito com
> `--local-store` (índice de desenvolvimento) e em backbone Matryoshka — no Cloud SQL o
> job falha **antes** do DDL, em vez de produzir um upsert que o pgvector recusaria
> (`expected 1024 dimensions, not …`). Se uma tabela antiga ficou com a dimensão errada, o
> `CREATE TABLE IF NOT EXISTS` não a altera: derrube-a (`DROP TABLE IF EXISTS
> govexe_qwen3vl;` leva o índice HNSW junto) e reingira.

## 8. Consulta cross-modal e prova de idempotência

```bash
python -m gcp.slice.query \
    --text "metas para a saude publica" --k 5 \
    --pg-host 127.0.0.1 --pg-db "$DB" --pg-user "$DB_USER"
```

```
consulta: 'metas para a saude publica'  (top-5)
 1. 0.5312  [doc_image]  123456_p07  gs://.../dt=2026-10-22/123456_p07.png
 2. 0.4980  [text]       texto:camara-1  https://www.camara.leg.br/noticias/1
      credito: Agência Câmara
      texto:   meta de ampliar a atenção básica em saúde
 3. 0.4712  [audio_asr]  audio:camara-1  gs://.../dt=2026-10-22/sessao.mp3
      credito: TV Câmara
      pointer: https://www.youtube.com/watch?v=exemplo
      texto:   ... (transcrição Whisper)
 4. 0.4503  [image]      foto:ebc-1  gs://.../dt=2026-10-22/hospital.jpg
      credito: Foto: Fulano/Agência Brasil
      pointer: https://agenciabrasil.ebc.com.br/.../hospital.jpg
 ...
```

Uma consulta, um índice, **quatro modalidades** no resultado — é a prova da Abordagem A.
Para restringir a uma delas (`WHERE modality = …`, apoiado pelo btree da §4):

```bash
python -m gcp.slice.query --text "sessao sobre saneamento" --modality audio_asr --k 5 \
    --pg-host 127.0.0.1 --pg-db "$DB" --pg-user "$DB_USER"
```

> A CLI do `query.py` é de **espaço único**: sem `--table` ela consulta a tabela do espaço
> default (`govexe_qwen3vl`). Quem funde espaços é `search_architecture` (a consulta
> multi-espaço do mesmo módulo), usada pelo `eval.py` da §8.1.

### 8.1 Avaliação por arquitetura (o CSV comparativo)

```bash
python -m gcp.slice.eval --arch A       --data-dir data/govexe/ptbr_planos \
    --out results/comparacao.csv --pg-host 127.0.0.1 --pg-db "$DB" --pg-user "$DB_USER"
python -m gcp.slice.eval --arch A+audio --data-dir data/govexe/ptbr_planos \
    --out results/comparacao.csv --pg-host 127.0.0.1 --pg-db "$DB" --pg-user "$DB_USER"

# B: sem pgvector, sem proxy, sem PGPASSWORD. `--pages-dir` aponta para onde as páginas
# do Bronze foram baixadas (o tarball do repo as exclui por tamanho).
python -m gcp.slice.eval --arch B --data-dir data/govexe/ptbr_planos \
    --out results/comparacao.csv --pages-dir ~/govexe-pages
```

Cada execução escreve **uma linha por (arquitetura, tarefa)** no mesmo CSV (recall@k,
nDCG@k, MRR e latência média por consulta); a linha da rodada anterior fica. `--table`
não se aplica aqui (cada espaço tem a sua, pelo registro) e `--local-store` pede um
**diretório** (`<dir>/<espaço>.json`, um índice por espaço).

Quem decide os espaços da rodada é a dupla **arquitetura × corpus**: só entram na fusão os
espaços que indexam a modalidade daquele conjunto de avaliação. Num corpus de páginas
(`doc_image`), o `clap` não indexa nada do corpus e fica de fora — por isso `A` e
`A+áudio` dão a **mesma** linha na tarefa doc-visual, como o protocolo de avaliação exige.
Num corpus de clipes, `A` mede a trilha ASR→texto e `A+áudio` funde ASR + CLAP nativo.

A mesma regra vale para `B`, pelo outro registro (`specialists_for_modalities`): o eixo
documento só declara `doc_image`, então `B` avalia a tarefa doc-visual e **recusa** um
corpus de áudio em vez de devolver 0.0 em tudo. Os ids do ranking de `B` são os `Doc.id`
do `corpus.jsonl` — os mesmos do `qrels.tsv` —, e é isso que torna a linha
`B,ptbr_planos,…` comparável **direto** com a linha `A,ptbr_planos,…`.

**Idempotência** — rode o `embed_job.py` do §7 uma segunda vez, sem mudar nada, e
compare as contagens antes/depois:

```sql
-- gcloud sql connect "$INSTANCE" --user="$DB_USER" --database="$DB"
SELECT count(*) AS linhas, count(DISTINCT item_id) AS itens FROM govexe_qwen3vl;
SELECT modality, count(*) FROM govexe_qwen3vl GROUP BY modality ORDER BY modality;
SELECT item_id, modality, left(uri, 50) AS uri, left(credit, 30) AS credito,
       left(content_sha256, 12) AS sha, updated_at
FROM govexe_qwen3vl ORDER BY item_id LIMIT 10;

-- A+áudio: o mesmo clipe nos dois espaços, sob o MESMO item_id (a chave do RRF)
SELECT modality, count(*) FROM govexe_clap GROUP BY modality;
SELECT q.item_id FROM govexe_qwen3vl q JOIN govexe_clap c USING (item_id);
```

`linhas == itens` e o total **não muda** entre as duas execuções: o upsert é
`ON CONFLICT (item_id) DO UPDATE` — só `updated_at` avança.

## 9. Teardown (obrigatório — custo residual)

```bash
gcloud compute instances delete "$VM" --zone="$ZONE" --quiet       # VM + disco

# Cloud SQL: parar (mantém os dados, continua cobrando o disco) ...
gcloud sql instances patch "$INSTANCE" --activation-policy=NEVER --quiet
# ... ou apagar de vez (não reversível):
gcloud sql instances delete "$INSTANCE" --quiet

gcloud storage rm --recursive "$BUCKET"                            # opcional
gcloud secrets delete "$SECRET" --quiet                            # opcional

# conferência final: as duas listas têm de sair vazias (ou com a instância STOPPED)
gcloud compute instances list
gcloud sql instances list
```

> **Numa rodada orquestrada, isto é automático.** O `gcp/slice/run_slice.sh` apaga VM e
> instância Cloud SQL num `trap … EXIT` — no sucesso, no erro, no Ctrl-C e na queda do
> terminal —, traz o CSV **antes** dos deletes e **asserta** no fim que nada sobrou (se
> sobrou, grita `LEAK` e sai com status 70). Ele é **dono do ciclo de vida** desses dois
> recursos: não o aponte para uma instância que você queira manter.
>
> ```bash
> bash gcp/slice/run_slice.sh --arch A --dry-run     # imprime a rodada, não toca a nuvem
> bash gcp/slice/run_slice.sh --arch A+audio 2>&1 | tee logs/run_slice.log
> ```

## 10. Onde tirar os prints (prova da entrega)

| # | Print | Onde, no console |
|---|---|---|
| 1 | **VM L4 ativa** | Compute Engine → *Instâncias de VM*: `govexe-embed`, `g2-standard-4`, status **RUNNING** (+ o `nvidia-smi` do §7 no terminal) |
| 2 | **Bronze populado** | Cloud Storage → o bucket → `bronze/fonte=tse_planos/dt=<data>/` com os PNGs |
| 3 | **Tabela populada** | SQL → a instância → **Cloud SQL Studio** → resultado do `SELECT count(*) …`, do `GROUP BY modality` e das 10 linhas do §8 (em `A+áudio`, também o `GROUP BY` de `govexe_clap`) |
| 4 | **Consulta cross-modal** | terminal da VM com a saída do `query.py` (top-5 com score, `[modalidade]`, crédito e `gs://…`) + a mesma consulta com `--modality` |
| 5 | **Idempotência** | os dois `SELECT count(*)` (antes/depois da 2ª ingestão) com o mesmo total |
| 6 | **Teardown** | Compute Engine sem instâncias e SQL apagado/STOPPED (saída do §9) |

## 11. Modo dev (sem GCP, para ensaiar os comandos)

Os mesmos scripts rodam com os *backends* falsos — nada de nuvem, nada de banco:
`--local-root`/`--local-bronze` (GCS), `--local-store` (Cloud SQL) e `--asr-backend fake`
(Whisper). `--local-store` sozinho guarda o índice só em memória; **com um caminho
`.json`** ele persiste, e é assim que o `query.py` enxerga o que o `embed_job.py`
indexou num processo anterior.

```bash
python -m gcp.slice.collector --local-root /tmp/bronze-fake \
    --images-dir data/govexe/ptbr_planos/pages --limit 4

# embute e indexa (precisa de GPU/modelo; use --device cpu para um teste lento)
python -m gcp.slice.embed_job --local-bronze /tmp/bronze-fake \
    --local-store /tmp/indice-fake.json --device cpu

# reaproveitando vetores já gravados por scripts/embed_qwen3vl.py (sem carregar modelo):
python scripts/embed_qwen3vl.py --images-dir data/govexe/ptbr_planos/pages \
    --out data/embeddings/planos_qwen3vl --limit 8
python -m gcp.slice.embed_job --vectors data/embeddings/planos_qwen3vl --local-store
python -m gcp.slice.query --text "educacao" --vectors data/embeddings/planos_qwen3vl
```

**Captura de áudio sem rede** (`--downloader fake`: grava um WAV canônico no lugar do
download, sem yt-dlp nem ffmpeg) — o JSONL resultante é o mesmo da captura real:

```bash
python -m gcp.slice.audio_collector \
    --url "https://www.youtube.com/watch?v=sessao-camara-1" \
    --url "https://www.camara.leg.br/radio/programas/999999.mp3" \
    --source tv_camara --credit "TV Camara" --downloader fake --max-seconds 120 \
    --audio-dir /tmp/audio --out-records /tmp/audios.jsonl
#    INFO captura: 2 de 2 urls baixadas para /tmp/audio
#    2 records -> /tmp/audios.jsonl
#      audio_asr:552e77e1bc4d355a  /tmp/audio/watch-552e77e1.wav
#      audio_asr:2aba99074d83898c  /tmp/audio/999999-2aba9907.wav

python -m gcp.slice.embed_job --records /tmp/audios.jsonl \
    --asr-backend fake --local-store /tmp/indice-fake.json --device cpu
#    INFO asr: /tmp/audio/watch-552e77e1.wav -> 40 caracteres
#    2 itens gravados; total no índice: 2
```

**Lote misto, fim a fim, sem nuvem.** `lote.jsonl` com 2 páginas + 1 foto (com legenda e
crédito) + 1 texto nativo + 1 áudio (transcrito pelo *fake* do Whisper):

```bash
cat > /tmp/lote.jsonl <<'JSONL'
{"modality": "doc_image", "image_path": "/tmp/dados/plano_saude_p01.png", "source": "tse_planos"}
{"modality": "doc_image", "image_path": "/tmp/dados/plano_educacao_p02.png", "source": "tse_planos"}
{"item_id": "foto:ebc-1", "modality": "image", "image_path": "/tmp/dados/foto_hospital.jpg", "pointer_uri": "https://agenciabrasil.ebc.com.br/foto_hospital.jpg", "credit": "Foto: Fulano de Tal/Agencia Brasil", "source": "ebc"}
{"item_id": "texto:camara-1", "modality": "text", "text": "meta de ampliar a atencao basica em saude e reduzir filas no hospital", "uri": "https://www.camara.leg.br/noticias/1", "credit": "Agencia Camara", "source": "agencia_camara"}
{"item_id": "audio:camara-1", "modality": "audio_asr", "audio_path": "/tmp/dados/sessao_saneamento.mp3", "pointer_uri": "https://www.youtube.com/watch?v=exemplo", "credit": "TV Camara", "source": "tv_camara"}
JSONL

# 1) binários para o Bronze fake; o JSONL sai enriquecido com blob/uri
python -m gcp.slice.collector --local-root /tmp/bronze-fake \
    --records /tmp/lote.jsonl --out-records /tmp/lote-bronze.jsonl --date 2026-11-02

# 2) embute as 4 modalidades no mesmo índice (ASR fake: nenhum peso é baixado)
python -m gcp.slice.embed_job \
    --records /tmp/lote-bronze.jsonl --records-store /tmp/bronze-fake \
    --asr-backend fake --local-store /tmp/indice-fake.json --device cpu
#    INFO asr: /tmp/dados/sessao_saneamento.mp3 -> 43 caracteres
#    5 itens gravados; total no índice: 5
#    (rode de novo: continua 5 — idempotente por item_id)

# 3) consulta cross-modal: um índice, modalidades diferentes no mesmo top-k
python -m gcp.slice.query --text "atencao basica em saude no hospital" --k 5 \
    --local-store /tmp/indice-fake.json --device cpu
#     1. 1.0000  [text]       texto:camara-1  https://www.camara.leg.br/noticias/1
#     2. 0.3536  [doc_image]  plano_saude_p01  /tmp/bronze-fake/bronze/fonte=tse_planos/...
#     3. 0.3536  [image]      foto:ebc-1      /tmp/bronze-fake/bronze/fonte=ebc/...
#           credito: Foto: Fulano de Tal/Agencia Brasil
#     5. 0.0000  [audio_asr]  audio:camara-1  .../sessao_saneamento.mp3
#           texto:   transcricao fake do audio sessao_saneamento

# 4) o mesmo índice, filtrado por modalidade
python -m gcp.slice.query --text "saude" --k 5 --modality image \
    --local-store /tmp/indice-fake.json --device cpu
```

> Os passos 2–4 ainda carregam o `Qwen3-VL-Embedding` de verdade (é só o GCS, o banco e o
> Whisper que estão *fakes*). Para ensaiar a fatia **sem nenhum peso**, os testes offline
> (`tests/test_gcp_slice.py`) fazem exatamente esse percurso com o encode monkeypatchado.

Os testes offline da lógica (`tests/test_gcp_slice.py`) usam esses mesmos *fakes*
(`LocalObjectStore`, `InMemoryVectorStore`, `FakeTranscriber`, `FakeDownloader`): `pytest
tests/test_gcp_slice.py`.
