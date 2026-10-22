# Fatia vertical GovExe na GCP — páginas → GCS → embedding (L4) → Cloud SQL/pgvector → consulta

Primeira fatia **fim a fim** da arquitetura de `fusao/coleta-gcp.md`, com o **modelo
unificado** `Qwen/Qwen3-VL-Embedding-2B` (o mesmo wrapper do benchmark:
`benchmark/models/qwen3vl_embed.py`).

```
data/govexe/ptbr_planos/pages/*.png
        │  collector.py            (Bronze, particionado por fonte/data)
        ▼
  gs://<bucket>/bronze/fonte=tse_planos/dt=YYYY-MM-DD/*.png
        │  embed_job.py            (VM L4 g2-standard-4, EFÊMERA)
        ▼
  Cloud SQL (PostgreSQL + pgvector) · tabela govexe_pages · UPSERT por page_id
        │  query.py                (texto → embedding → top-k por `<=>`)
        ▼
  top-k páginas de plano de governo
```

| Script | Onde roda | O que faz |
|---|---|---|
| `gcp/slice/collector.py` | máquina local ou Cloud Shell | sobe as páginas-imagem para o bucket **Bronze** |
| `gcp/slice/embed_job.py` | **VM L4 efêmera** | baixa do Bronze, embute e faz **upsert** no pgvector |
| `gcp/slice/query.py` | VM ou Cloud Shell (com proxy) | consulta texto → top-k páginas |

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
real do checkpoint). Para criar à mão, troque `<dim>` pela dimensão do modelo:

```sql
CREATE TABLE IF NOT EXISTS govexe_pages (
    page_id        TEXT PRIMARY KEY,
    source         TEXT NOT NULL,
    uri            TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    model_id       TEXT NOT NULL,
    embedding      VECTOR(<dim>) NOT NULL,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS govexe_pages_embedding_cos
    ON govexe_pages USING hnsw (embedding vector_cosine_ops);
```

`page_id` é a chave primária e é o **stem do arquivo** (`<tse_seq>_p<NN>`), o mesmo id de
página do `corpus.jsonl` — é ele que torna a ingestão idempotente.

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
sobrescreve.

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
    --device cuda --batch-size 8 \
    --pg-host 127.0.0.1 --pg-port 5432 --pg-db "$DB" --pg-user "$DB_USER"
```

Saída esperada: `N páginas gravadas; total no índice: N`.

## 8. Consulta fim a fim e prova de idempotência

```bash
python -m gcp.slice.query \
    --text "metas para a saude publica" --k 5 \
    --pg-host 127.0.0.1 --pg-db "$DB" --pg-user "$DB_USER"
```

```
consulta: 'metas para a saude publica'  (top-5)
 1. 0.5312  123456_p07  gs://.../bronze/fonte=tse_planos/dt=2026-10-22/123456_p07.png
 ...
```

**Idempotência** — rode o `embed_job.py` do §7 uma segunda vez, sem mudar nada, e
compare as contagens antes/depois:

```sql
-- gcloud sql connect "$INSTANCE" --user="$DB_USER" --database="$DB"
SELECT count(*) AS linhas, count(DISTINCT page_id) AS paginas FROM govexe_pages;
SELECT page_id, left(uri, 60) AS uri, left(content_sha256, 12) AS sha, updated_at
FROM govexe_pages ORDER BY page_id LIMIT 10;
```

`linhas == paginas` e o total **não muda** entre as duas execuções: o upsert é
`ON CONFLICT (page_id) DO UPDATE` — só `updated_at` avança.

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
| 3 | **Tabela populada** | SQL → a instância → **Cloud SQL Studio** → resultado do `SELECT count(*) …` e das 10 linhas do §8 |
| 4 | **Consulta fim a fim** | terminal da VM com a saída do `query.py` (top-5 com score e `gs://…`) |
| 5 | **Idempotência** | os dois `SELECT count(*)` (antes/depois da 2ª ingestão) com o mesmo total |
| 6 | **Teardown** | Compute Engine sem instâncias e SQL apagado/STOPPED (saída do §9) |

## 11. Modo dev (sem GCP, para ensaiar os comandos)

Os mesmos scripts rodam com os *backends* falsos — nada de nuvem, nada de banco:

```bash
python -m gcp.slice.collector --local-root /tmp/bronze-fake \
    --images-dir data/govexe/ptbr_planos/pages --limit 4

# embute e indexa em memória (precisa de GPU/modelo; use --device cpu para um teste lento)
python -m gcp.slice.embed_job --local-bronze /tmp/bronze-fake --local-store --device cpu

# reaproveitando vetores já gravados por scripts/embed_qwen3vl.py (sem carregar modelo):
python scripts/embed_qwen3vl.py --images-dir data/govexe/ptbr_planos/pages \
    --out data/embeddings/planos_qwen3vl --limit 8
python -m gcp.slice.embed_job --vectors data/embeddings/planos_qwen3vl --local-store
python -m gcp.slice.query --text "educacao" --vectors data/embeddings/planos_qwen3vl
```

Os testes offline da lógica (`tests/test_gcp_slice.py`) usam esses mesmos *fakes*:
`pytest tests/test_gcp_slice.py`.
