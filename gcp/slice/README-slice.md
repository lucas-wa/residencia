# Fatia vertical GovExe na GCP — multimodal → GCS → embedding (L4) → Cloud SQL/pgvector → consulta

Primeira fatia **fim a fim** da arquitetura de `fusao/coleta-gcp.md`, com o **modelo
unificado** `Qwen/Qwen3-VL-Embedding-2B` (o mesmo wrapper do benchmark:
`benchmark/models/qwen3vl_embed.py`).

É a validação prática da **Abordagem A** de `fusao/multimodal.md` (§2): **quatro
modalidades no mesmo índice, pelo mesmo modelo, no mesmo espaço vetorial** — uma consulta
textual recupera página de documento, foto, texto nativo e áudio transcrito do **mesmo**
`govexe_items`.

```
páginas (*.png) · fotos (*.jpg) · textos · áudios (*.mp3)
        │  collector.py            (Bronze, particionado por fonte/data; JSONL de records)
        ▼
  gs://<bucket>/bronze/fonte=<fonte>/dt=YYYY-MM-DD/<arquivo>
        │  embed_job.py            (VM L4 g2-standard-4, EFÊMERA)
        │     ├─ doc_image · image  → embed_images()   ─┐  MESMO checkpoint,
        │     ├─ text               → embed_texts()    ─┤  MESMO dim, L2-norm
        │     └─ audio_asr → Whisper large-v3 (asr.py) → embed_texts()  ─┘
        ▼
  Cloud SQL (PostgreSQL + pgvector) · tabela govexe_items · UPSERT por item_id
        │  query.py                (texto → embedding → top-k por `<=>`, `--modality`)
        ▼
  top-k itens — de qualquer modalidade (cross-modal), com crédito e pointer
```

| Script | Onde roda | O que faz |
|---|---|---|
| `gcp/slice/collector.py` | máquina local ou Cloud Shell | sobe páginas, fotos e áudios para o bucket **Bronze**; enriquece o JSONL de records |
| `gcp/slice/asr.py` | VM L4 (dentro do job) | **Whisper large-v3** atrás da interface `Transcriber` (áudio → texto) |
| `gcp/slice/embed_job.py` | **VM L4 efêmera** | baixa do Bronze, roteia por modalidade, embute e faz **upsert** no pgvector |
| `gcp/slice/query.py` | VM ou Cloud Shell (com proxy) | consulta texto → top-k itens, com filtro `--modality` |

## Modalidades indexadas

| `modality` | O que é | Como vira vetor | Campos típicos |
|---|---|---|---|
| `doc_image` | página de plano de governo rasterizada | `embed_images()` | `uri` = objeto no Bronze |
| `image` | foto de agência pública (EBC, Agência Câmara…) | `embed_images()` | `credit` **obrigatório**, `pointer_uri` = foto original |
| `text` | texto nativo (matéria, legenda, ementa) | `embed_texts()` | `content_text`, `uri` da matéria; `pointer_uri` NULL |
| `audio_asr` | áudio (sessão, RadioAgência) | ASR (Whisper) → `embed_texts()` | `content_text` = **transcrição**, `pointer_uri` = áudio original |

> **Áudio não tem embedder próprio aqui.** Ele é colapsado em texto pelo Whisper
> (*pointer vector*, `fusao/multimodal.md` §3 e §4) e embutido pelo **mesmo**
> `Qwen3-VL-Embedding` que embute as imagens — por isso tudo cai no mesmo espaço e um
> único índice basta. Embedding **nativo** de áudio (CLAP) e *ensemble* multi-modelo são
> outros objetivos, fora desta fatia.

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
export DB_USER="govexe"
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

gcloud sql users create "$DB_USER" \
    --instance="$INSTANCE" \
    --password="$(gcloud secrets versions access latest --secret=$SECRET)"
```

> `--tier=db-g1-small` é mais barato e também dá conta desta fatia (algumas centenas de
> páginas). O `--password=` acima deixa a senha no histórico do shell da sua máquina —
> se isso incomodar, use `gcloud sql users set-password "$DB_USER" --instance="$INSTANCE"
> --prompt-for-password` e cole o valor do secret.

**Habilitar o pgvector** (uma vez, dentro do banco):

```bash
gcloud sql connect "$INSTANCE" --user="$DB_USER" --database="$DB"
# senha: a do Secret Manager (§3). Já no psql:
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;
\dx                       -- confirma: vector | 0.8.x | public
```

**Tabela**: o `embed_job.py` a cria sozinho no primeiro upsert (ele conhece a dimensão
com que está indexando). Para criar à mão, use a mesma dimensão do `--dim` do job (1024
por padrão):

```sql
CREATE TABLE IF NOT EXISTS govexe_items (
    item_id        TEXT PRIMARY KEY,
    modality       TEXT NOT NULL,          -- doc_image | image | text | audio_asr
    source         TEXT NOT NULL,
    uri            TEXT NOT NULL,          -- fonte da representação indexada
    pointer_uri    TEXT,                   -- binário original (foto/áudio); NULL p/ texto
    credit         TEXT,                   -- crédito (autor/fonte) — o pointer auditável
    content_text   TEXT,                   -- texto nativo OU transcrição ASR
    content_sha256 TEXT NOT NULL,
    model_id       TEXT NOT NULL,
    embedding      VECTOR(1024) NOT NULL,  -- = --dim do embed_job (teto do HNSW: 2000)
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS govexe_items_embedding_cos
    ON govexe_items USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS govexe_items_modality
    ON govexe_items (modality);            -- apoia o `--modality` do query.py
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
  execuções (`embed_job.default_item_id`).

`content_sha256` registra o **conteúdo que gerou o vetor**: os bytes do arquivo nas
modalidades de imagem e o próprio texto embutido (a transcrição, no caso do áudio) nas de
texto.

> **Migração `govexe_pages` → `govexe_items`.** A tabela antiga não é alterada nem
> apagada pelo job: `govexe_items` é uma tabela **nova** (superset — `page_id` virou
> `item_id` e entraram `modality`, `pointer_uri`, `credit`, `content_text`). Como o
> Cloud SQL desta fatia é efêmero, o caminho normal é **reingerir** o corpus no
> `govexe_items` (o `embed_job.py` cria tabela e índices no primeiro upsert). Para
> aproveitar uma `govexe_pages` já populada com a mesma `--dim`, sem reprocessar GPU:
>
> ```sql
> INSERT INTO govexe_items
>     (item_id, modality, source, uri, content_sha256, model_id, embedding, updated_at)
> SELECT page_id, 'doc_image', source, uri, content_sha256, model_id, embedding, updated_at
> FROM govexe_pages
> ON CONFLICT (item_id) DO NOTHING;
> ```
>
> (Rode o `embed_job.py` uma vez antes, para a tabela e os índices existirem.) Quem
> quiser continuar na tabela antiga passa `--table govexe_pages` — mas aí o DDL novo não
> bate com a tabela velha; prefira reingerir.

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
    --image-family=common-cu124-ubuntu-2204-py310 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --metadata="install-nvidia-driver=True" \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

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
```

O `faster-whisper` (ASR da modalidade `audio_asr`) já vem no `requirements-slurm.txt`;
os pesos do `large-v3` só são baixados na **primeira transcrição** (import *lazy*), e um
lote sem áudio nunca os toca.

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
    --device cuda --batch-size 8 --dim 1024 \
    --pg-host 127.0.0.1 --pg-port 5432 --pg-db "$DB" --pg-user "$DB_USER"
```

Saída esperada: `N itens gravados; total no índice: N`.

**Lote misto** (foto + texto + áudio, do JSONL da §5.1) — o mesmo job, mesmo índice,
mesmo checkpoint; o que muda é só a entrada:

```bash
python -m gcp.slice.embed_job \
    --records lote-bronze.jsonl \
    --records-store "$BUCKET" \
    --work-dir /tmp/govexe-bronze \
    --asr-backend faster-whisper --asr-model large-v3 --asr-language pt \
    --device cuda --batch-size 8 --dim 1024 \
    --pg-host 127.0.0.1 --pg-port 5432 --pg-db "$DB" --pg-user "$DB_USER"
```

O job separa o lote internamente (`doc_image`/`image` → `embed_images`;
`text`/`audio_asr` → `embed_texts`), transcreve os áudios antes de embutir (log
`asr: <arquivo> -> N caracteres`) e grava tudo na mesma tabela, na ordem do lote.

> **Migração: já ingeriu com `--dim 2048`?** O default do `--dim` passou de **2048** para
> **1024** (§4). A dimensão faz parte do *tipo da coluna*, então uma tabela `govexe_pages`
> criada antes disso continua `VECTOR(2048)` e **recusa** os vetores de 1024 no upsert
> (`expected 2048 dimensions, not 1024`) — o `CREATE TABLE IF NOT EXISTS` do job não
> altera tabela existente. Antes de reingerir, derrube a tabela (o `embed_job.py` a recria
> com a dimensão nova, junto do índice HNSW, no primeiro upsert):
>
> ```sql
> DROP TABLE IF EXISTS govexe_pages;    -- leva o índice HNSW junto
> ```
>
> Para preservar a ingestão antiga, indexe em outra tabela: passe o mesmo `--table <nome>`
> ao `embed_job.py` e ao `query.py` (§8).

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

**Idempotência** — rode o `embed_job.py` do §7 uma segunda vez, sem mudar nada, e
compare as contagens antes/depois:

```sql
-- gcloud sql connect "$INSTANCE" --user="$DB_USER" --database="$DB"
SELECT count(*) AS linhas, count(DISTINCT item_id) AS itens FROM govexe_items;
SELECT modality, count(*) FROM govexe_items GROUP BY modality ORDER BY modality;
SELECT item_id, modality, left(uri, 50) AS uri, left(credit, 30) AS credito,
       left(content_sha256, 12) AS sha, updated_at
FROM govexe_items ORDER BY item_id LIMIT 10;
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

## 10. Onde tirar os prints (prova da entrega)

| # | Print | Onde, no console |
|---|---|---|
| 1 | **VM L4 ativa** | Compute Engine → *Instâncias de VM*: `govexe-embed`, `g2-standard-4`, status **RUNNING** (+ o `nvidia-smi` do §7 no terminal) |
| 2 | **Bronze populado** | Cloud Storage → o bucket → `bronze/fonte=tse_planos/dt=<data>/` com os PNGs |
| 3 | **Tabela populada** | SQL → a instância → **Cloud SQL Studio** → resultado do `SELECT count(*) …`, do `GROUP BY modality` e das 10 linhas do §8 |
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
(`LocalObjectStore`, `InMemoryVectorStore`, `FakeTranscriber`): `pytest
tests/test_gcp_slice.py`.
