"""Job de embedding da fatia GCP: GCS (Bronze) -> Qwen3-VL-Embedding -> Cloud SQL/pgvector.

Roda na **VM L4 efemera** (`g2-standard-4`, ver `gcp/slice/README-slice.md`): baixa os
itens do bucket Bronze, embute cada um com `Qwen/Qwen3-VL-Embedding-2B` e faz **upsert**
no Cloud SQL + pgvector.

**Abordagem A -- modelo unificado** (`fusao/multimodal.md` secao 2): QUATRO modalidades no
MESMO indice (`govexe_items`), pelo MESMO checkpoint e no MESMO espaco vetorial:

    doc_image   pagina de documento rasterizada  -> embed_images()
    image       foto de agencia publica          -> embed_images()
    text        texto nativo (materia, legenda)  -> embed_texts()
    audio_asr   audio -> transcricao Whisper     -> embed_texts()

O audio NAO tem embedder proprio aqui: ele e colapsado em texto pelo ASR (`gcp/slice/asr.py`,
Whisper large-v3 atras da interface `Transcriber`) e embutido pelo mesmo `embed_texts` --
o *pointer vector* de `fusao/multimodal.md` (secoes 3 e 4). Embedding nativo de audio
(CLAP) e
outro objetivo. O binario original fica em `pointer_uri` e o credito obrigatorio em
`credit`.

Modos de entrada:

    --bucket gs://... --blob-prefix bronze/fonte=.../dt=...   paginas vindas do GCS
    --local-bronze <dir>                                      bucket fake (modo dev)
    --images-dir <dir>                                        diretorio local de paginas
    --records <arquivo.jsonl>                                 lote MISTO (foto/texto/audio,
                                                              formato no README-slice.md);
                                                              `--records-store` resolve o
                                                              campo `blob` contra o Bronze
    --vectors <prefixo>                                       reusa `<prefixo>.npy` +
                                                              `<prefixo>.manifest.json`
                                                              de `scripts/embed_qwen3vl.py`
                                                              (nao carrega modelo nenhum)

**Idempotencia** (requisito do plano): a chave e o `item_id` -- o stem do arquivo para
paginas (o mesmo id do `corpus.jsonl`, `<tse_seq>_p<NN>`) e `<modalidade>:<hash da uri>`
(ou o id natural do record) para foto/texto/audio. O SQL e
`INSERT ... ON CONFLICT (item_id) DO UPDATE`, entao rodar o job duas vezes sobre o mesmo
lote deixa exatamente as mesmas linhas -- so `updated_at` avanca. `content_sha256`
registra o conteudo que gerou aquele vetor: os bytes do arquivo nas modalidades de imagem
e o proprio texto embutido (a transcricao, no caso do audio) nas de texto.

**Segredo**: a senha do Cloud SQL vem da variavel de ambiente `PGPASSWORD` (nome
configuravel por `--password-env`), populada a partir do **Secret Manager**:

    export PGPASSWORD="$(gcloud secrets versions access latest --secret=govexe-pg-password)"

Nada de senha em flag, em codigo ou no git.

**Dimensao**: a fatia indexa em 1024 dimensoes (`--dim`, Matryoshka/MRL: trunca e
renormaliza, preservando o cosseno). O checkpoint entrega 2048 nativas, e o indice HNSW
do pgvector nao aceita colunas acima de 2000 -- ver `HNSW_MAX_DIM`/`DEFAULT_DIM`.

Uso:
    python -m gcp.slice.embed_job --bucket gs://govexe-bronze --blob-prefix bronze/ \
        --pg-host 127.0.0.1 --pg-db govexe --pg-user govexe
    python -m gcp.slice.embed_job --records lote.jsonl --asr-model large-v3 \
        --pg-host 127.0.0.1 --pg-db govexe --pg-user govexe
    python -m gcp.slice.embed_job --records lote.jsonl --asr-backend fake \
        --local-store /tmp/indice-fake.json --device cpu --dim 4            # dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

# permite tanto `python gcp/slice/embed_job.py` quanto `python -m gcp.slice.embed_job`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.models.qwen3vl_embed import (  # noqa: E402
    DEFAULT_MODEL_ID,
    embed_images,
    embed_texts,
)
from gcp.slice.asr import Transcriber, add_asr_arguments, transcriber_from_args  # noqa: E402
from gcp.slice.collector import (  # noqa: E402
    DEFAULT_PREFIX,
    DEFAULT_SOURCE,
    IMAGE_SUFFIXES,
    ObjectStore,
    default_item_id,
    load_records,
    open_store,
    store_from_location,
)
from scripts.embed_qwen3vl import build_manifest, collect_paths  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_TABLE = "govexe_items"
DEFAULT_PASSWORD_ENV = "PGPASSWORD"
DEFAULT_WORK_DIR = "/tmp/govexe-bronze"
TOP_K = 5

# -- modalidades (Abordagem A: todas no mesmo indice, pelo mesmo checkpoint) --------------
MODALITY_DOC_IMAGE = "doc_image"   # pagina de documento rasterizada
MODALITY_IMAGE = "image"           # foto de agencia publica (com credito)
MODALITY_TEXT = "text"             # texto nativo (materia, legenda, ementa)
MODALITY_AUDIO = "audio_asr"       # audio colapsado em texto pelo Whisper
MODALITIES = (MODALITY_DOC_IMAGE, MODALITY_IMAGE, MODALITY_TEXT, MODALITY_AUDIO)
# Quem vai por `embed_images` e quem vai por `embed_texts` -- mesmo checkpoint nos dois.
IMAGE_MODALITIES = (MODALITY_DOC_IMAGE, MODALITY_IMAGE)
TEXT_MODALITIES = (MODALITY_TEXT, MODALITY_AUDIO)

# Teto do indice do pgvector: uma coluna indexada por `hnsw` (ou `ivfflat`) nao pode passar
# de 2000 dimensoes -- acima disso o CREATE INDEX estoura
# `column cannot have more than 2000 dimensions for hnsw index`.
HNSW_MAX_DIM = 2000
# Dimensao indexada por esta fatia. O Qwen3-VL-Embedding entrega 2048 nativas, que NAO
# cabem no HNSW; 1024 e o corte Matryoshka/MRL validado na execucao real (`--dim` trunca e
# renormaliza, entao o cosseno continua valendo).
DEFAULT_DIM = 1024

# DDL identica a do README-slice.md (secao "tabela"): o job cria a tabela se ela nao
# existir, mas a criacao tambem pode ser feita a mao no Cloud SQL Studio/psql.
# `govexe_items` e superset da antiga `govexe_pages` -- ver a nota de migracao no README.
DDL_TABLE = """CREATE TABLE IF NOT EXISTS {table} (
    item_id        TEXT PRIMARY KEY,
    modality       TEXT NOT NULL,
    source         TEXT NOT NULL,
    uri            TEXT NOT NULL,
    pointer_uri    TEXT,
    credit         TEXT,
    content_text   TEXT,
    content_sha256 TEXT NOT NULL,
    model_id       TEXT NOT NULL,
    embedding      VECTOR({dim}) NOT NULL,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
)"""

DDL_INDEX = ("CREATE INDEX IF NOT EXISTS {table}_embedding_cos "
             "ON {table} USING hnsw (embedding vector_cosine_ops)")

# btree de apoio ao filtro `--modality` do query.py (o HNSW nao filtra por coluna comum).
DDL_INDEX_MODALITY = ("CREATE INDEX IF NOT EXISTS {table}_modality "
                      "ON {table} (modality)")

# O upsert idempotente: chave = item_id. Reprocessar o mesmo lote atualiza a linha.
SQL_UPSERT = """INSERT INTO {table}
    (item_id, modality, source, uri, pointer_uri, credit, content_text,
     content_sha256, model_id, embedding, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, now())
ON CONFLICT (item_id) DO UPDATE SET
    modality       = EXCLUDED.modality,
    source         = EXCLUDED.source,
    uri            = EXCLUDED.uri,
    pointer_uri    = EXCLUDED.pointer_uri,
    credit         = EXCLUDED.credit,
    content_text   = EXCLUDED.content_text,
    content_sha256 = EXCLUDED.content_sha256,
    model_id       = EXCLUDED.model_id,
    embedding      = EXCLUDED.embedding,
    updated_at     = now()"""

SQL_SEARCH = """SELECT item_id, modality, uri, pointer_uri, credit, content_text,
       1 - (embedding <=> %s::vector) AS score
FROM {table}
{where}ORDER BY embedding <=> %s::vector
LIMIT %s"""

SQL_WHERE_MODALITY = "WHERE modality = %s\n"


# -- linhas -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemRow:
    """Um item pronto para o indice: metadados + vetor L2-normalizado.

    Vale para as quatro modalidades -- o que muda e quais campos opcionais vem
    preenchidos: `content_text` nas de texto (incluindo a transcricao do audio),
    `pointer_uri`/`credit` no que tem binario original e autoria a preservar.
    """

    item_id: str
    modality: str
    source: str
    uri: str
    content_sha256: str
    model_id: str
    embedding: np.ndarray = field(repr=False)
    pointer_uri: str | None = None
    credit: str | None = None
    content_text: str | None = None


@dataclass(frozen=True)
class Hit:
    """Resultado de busca (score = cosseno, porque os vetores saem L2-normalizados)."""

    item_id: str
    modality: str
    uri: str
    score: float
    pointer_uri: str | None = None
    credit: str | None = None
    content_text: str | None = None


def content_sha256(path: str | Path) -> str:
    """sha256 do arquivo. Item ausente (manifest de outra maquina) -> hash do caminho."""
    p = Path(path)
    if not p.is_file():
        log.warning("arquivo ausente, hash derivado do caminho: %s", p)
        return hashlib.sha256(str(p).encode()).hexdigest()
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    """sha256 do texto embutido (nas modalidades de texto e ele que gera o vetor)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_rows(items: list[dict], vecs: np.ndarray, *, model_id: str = DEFAULT_MODEL_ID,
               source: str = DEFAULT_SOURCE,
               uris: dict[str, str] | None = None,
               modality: str = MODALITY_DOC_IMAGE) -> list[ItemRow]:
    """Manifest + matriz -> linhas do indice (caminho das paginas-documento).

    `items` segue o manifest de `scripts/embed_qwen3vl.py` (`{"row", "id", "path"}`), entao
    o `item_id` ja e o stem do arquivo. `uris` mapeia caminho local -> URI de origem
    (`gs://...`) quando a pagina veio do Bronze; sem ele, a URI e o proprio caminho.
    """
    vecs = np.atleast_2d(np.asarray(vecs, dtype="float32"))
    if len(items) != len(vecs):
        raise ValueError(f"manifest ({len(items)}) e vetores ({len(vecs)}) nao batem")
    uris = uris or {}
    linhas = []
    for item in items:
        path = item["path"]
        linhas.append(ItemRow(
            item_id=item["id"],
            modality=modality,
            source=source,
            uri=uris.get(path, str(path)),
            content_sha256=content_sha256(path),
            model_id=model_id,
            embedding=np.asarray(vecs[item["row"]], dtype="float32"),
        ))
    return linhas


# -- records: lote misto (foto, texto nativo, audio->ASR) ---------------------------------


def prepare_record(rec: dict, *, source: str = DEFAULT_SOURCE,
                   store: ObjectStore | None = None,
                   work_dir: str | Path = DEFAULT_WORK_DIR,
                   transcriber: Transcriber | None = None) -> dict:
    """Record cru -> item normalizado, ja com `content_text` (ASR incluido) e hash.

    Resolve, nesta ordem: o binario (caminho local ou `blob` baixado do Bronze), a URI da
    representacao, o `item_id` e -- para `audio_asr` sem texto -- a transcricao pelo
    `Transcriber`. O vetor NAO sai daqui: quem embute e o `embed_records`, que agrupa por
    modalidade para chamar o encoder certo do mesmo checkpoint.
    """
    modality = rec.get("modality")
    if modality not in MODALITIES:
        raise ValueError(f"modalidade desconhecida: {modality!r} (use uma de {MODALITIES})")

    path = rec.get("image_path") or rec.get("audio_path") or rec.get("path")
    uri = rec.get("uri")
    blob = rec.get("blob")
    if blob and not path:
        if store is None:
            raise ValueError(f"record com `blob` ({blob}) exige --records-store para "
                             "resolver o objeto no Bronze")
        path = str(store.download(blob, work_dir).resolve())
    if blob and store is not None and not uri:
        uri = store.uri(blob)
    if not uri:
        uri = str(path) if path else rec.get("pointer_uri")
    if not uri:
        raise ValueError(f"record sem uri, caminho ou blob: {rec!r}")

    content_text = rec.get("text") or rec.get("content_text")
    if modality == MODALITY_AUDIO and not content_text:
        if not path:
            raise ValueError(f"record de audio sem arquivo nem transcricao: {rec!r}")
        if transcriber is None:
            raise ValueError("record `audio_asr` sem transcricao exige um transcritor "
                             "(--asr-backend / parametro `transcriber`)")
        content_text = transcriber.transcribe(path)
        log.info("asr: %s -> %d caracteres", path, len(content_text))
    if modality == MODALITY_TEXT and not content_text:
        raise ValueError(f"record de texto sem campo `text`: {rec!r}")
    if modality in IMAGE_MODALITIES and not path:
        raise ValueError(f"record de imagem sem `image_path` nem `blob`: {rec!r}")

    item_id = rec.get("item_id")
    if not item_id:
        item_id = (Path(path).stem if modality == MODALITY_DOC_IMAGE and path
                   else default_item_id(modality, uri))
    return {
        "item_id": item_id,
        "modality": modality,
        "source": rec.get("source") or source,
        "uri": uri,
        "pointer_uri": rec.get("pointer_uri"),
        "credit": rec.get("credit"),
        "content_text": content_text if modality in TEXT_MODALITIES else None,
        "path": str(path) if path else None,
        "content_sha256": (text_sha256(content_text) if modality in TEXT_MODALITIES
                           else content_sha256(path)),
    }


def embed_records(records: list[dict], *, model_id: str = DEFAULT_MODEL_ID,
                  device: str = "cuda", dtype: str = "bfloat16", batch_size: int = 8,
                  dim: int | None = DEFAULT_DIM, source: str = DEFAULT_SOURCE,
                  transcriber: Transcriber | None = None,
                  store: ObjectStore | None = None,
                  work_dir: str | Path = DEFAULT_WORK_DIR) -> list[ItemRow]:
    """Lote MISTO -> linhas do indice, cada modalidade pelo encoder certo.

    Imagens (`doc_image`/`image`) vao por `embed_images`, texto e transcricao
    (`text`/`audio_asr`) por `embed_texts` -- mesmo `model_id`, mesmo `dim`, mesma
    L2-normalizacao, logo **mesmo espaco vetorial** (Abordagem A). As duas chamadas sao
    separadas so por eficiencia de batch; a ordem do lote e reconstituida no fim.
    """
    preparados = [prepare_record(r, source=source, store=store, work_dir=work_dir,
                                 transcriber=transcriber) for r in records]
    kw = dict(model_id=model_id, device=device, dtype=dtype, batch_size=batch_size, dim=dim)
    vetores: list[np.ndarray | None] = [None] * len(preparados)

    idx_img = [i for i, p in enumerate(preparados) if p["modality"] in IMAGE_MODALITIES]
    if idx_img:
        m = np.atleast_2d(embed_images([preparados[i]["path"] for i in idx_img], **kw))
        for linha, i in enumerate(idx_img):
            vetores[i] = np.asarray(m[linha], dtype="float32")

    idx_txt = [i for i, p in enumerate(preparados) if p["modality"] in TEXT_MODALITIES]
    if idx_txt:
        m = np.atleast_2d(embed_texts([preparados[i]["content_text"] for i in idx_txt], **kw))
        for linha, i in enumerate(idx_txt):
            vetores[i] = np.asarray(m[linha], dtype="float32")

    return [ItemRow(item_id=p["item_id"], modality=p["modality"], source=p["source"],
                    uri=p["uri"], content_sha256=p["content_sha256"], model_id=model_id,
                    embedding=v, pointer_uri=p["pointer_uri"], credit=p["credit"],
                    content_text=p["content_text"])
            for p, v in zip(preparados, vetores, strict=True)]


def load_vectors(prefix: str | Path) -> tuple[dict, np.ndarray]:
    """Le `<prefixo>.npy` + `<prefixo>.manifest.json` gravados por `scripts/embed_qwen3vl.py`."""
    base = Path(prefix)
    if base.suffix == ".npy":
        base = base.with_suffix("")
    manifest = json.loads(base.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    return manifest, np.load(base.with_suffix(".npy"))


# -- interface de banco ------------------------------------------------------------------


class VectorStore(Protocol):
    """Minimo que a fatia precisa do indice vetorial (pgvector real ou fake em memoria)."""

    def ensure_schema(self, dim: int) -> None:
        ...

    def upsert(self, rows: list[ItemRow]) -> int:
        """Grava as linhas por `item_id` (idempotente) e devolve quantas foram gravadas."""

    def count(self) -> int:
        ...

    def search(self, vec: np.ndarray, k: int = TOP_K,
               modality: str | None = None) -> list[Hit]:
        ...


def rank_top_k(q_vec: np.ndarray, matrix: np.ndarray, ids: list[str],
               k: int = TOP_K) -> list[tuple[str, float]]:
    """Top-k por produto interno (= cosseno com vetores L2-normalizados), maior primeiro.

    E a logica pura por tras do indice em memoria, e produz a mesma ordenacao que o
    `ORDER BY embedding <=> ...` do pgvector no backend real -- sem banco nenhum.
    """
    if not ids or np.size(matrix) == 0:
        return []
    scores = np.asarray(q_vec, dtype="float32").ravel() @ np.atleast_2d(matrix).T
    ordem = np.argsort(-scores)[:k]
    return [(ids[i], float(scores[i])) for i in ordem]


class InMemoryVectorStore:
    """FAKE de `VectorStore`: dicionario por `item_id` + produto interno. Usado nos testes
    e no modo dev (`--local-store`, `--vectors` do `query.py`), nunca em producao.

    Com `path`, o dicionario e espelhado num JSON a cada upsert: e so o que falta para o
    modo dev atravessar dois processos (`embed_job.py` e depois `query.py`) sem Cloud SQL.
    """

    def __init__(self, rows: list[ItemRow] | None = None, path: str | Path | None = None):
        self.rows: dict[str, ItemRow] = {}
        self.path = Path(path) if path else None
        if self.path and self.path.is_file():
            self._load()
        if rows:
            self.upsert(rows)

    # -- persistencia opcional (so modo dev) -------------------------------------
    @staticmethod
    def _to_json(row: ItemRow) -> dict:
        d = {k: v for k, v in row.__dict__.items() if k != "embedding"}
        d["embedding"] = [float(x) for x in np.asarray(row.embedding).ravel()]
        return d

    def _load(self) -> None:
        for d in json.loads(self.path.read_text(encoding="utf-8")):
            d = dict(d)
            d["embedding"] = np.asarray(d["embedding"], dtype="float32")
            self.rows[d["item_id"]] = ItemRow(**d)

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([self._to_json(r) for r in self.rows.values()],
                                        ensure_ascii=False), encoding="utf-8")

    # -- VectorStore -------------------------------------------------------------
    def ensure_schema(self, dim: int) -> None:
        self.dim = dim

    def upsert(self, rows: list[ItemRow]) -> int:
        for row in rows:
            self.rows[row.item_id] = row       # mesma chave -> substitui, nao duplica
        self._save()
        return len(rows)

    def count(self) -> int:
        return len(self.rows)

    def search(self, vec: np.ndarray, k: int = TOP_K,
               modality: str | None = None) -> list[Hit]:
        ids = [i for i, r in self.rows.items() if modality in (None, r.modality)]
        if not ids:
            return []
        matriz = np.vstack([self.rows[i].embedding for i in ids])
        return [Hit(iid, self.rows[iid].modality, self.rows[iid].uri, score,
                    self.rows[iid].pointer_uri, self.rows[iid].credit,
                    self.rows[iid].content_text)
                for iid, score in rank_top_k(vec, matriz, ids, k)]


class PgVectorStore:
    """Backend real: Cloud SQL (PostgreSQL) + extensao pgvector, via psycopg.

    O vetor viaja como literal `'[...]'::vector` -- funciona com ou sem o adaptador
    `pgvector.psycopg` instalado no cliente; quem calcula `<=>` e a extensao no servidor.
    """

    def __init__(self, conn, table: str = DEFAULT_TABLE):
        self._conn = conn
        self.table = table

    @classmethod
    def connect(cls, *, host: str, port: int, dbname: str, user: str,
                password_env: str = DEFAULT_PASSWORD_ENV,
                table: str = DEFAULT_TABLE) -> PgVectorStore:
        """Conecta usando a senha da variavel de ambiente (vinda do Secret Manager)."""
        # a checagem do segredo vem ANTES do import: falhar por senha ausente e mais util
        # (e mais rapido) do que falhar por driver ausente.
        password = os.environ.get(password_env)
        if not password:
            raise RuntimeError(
                f"${password_env} vazia: exporte a senha do Secret Manager antes de rodar "
                f'(export {password_env}="$(gcloud secrets versions access latest '
                '--secret=govexe-pg-password)")')
        import psycopg

        conn = psycopg.connect(host=host, port=port, dbname=dbname, user=user,
                               password=password)
        return cls(conn, table=table)

    @staticmethod
    def vector_literal(vec: np.ndarray) -> str:
        return "[" + ",".join(f"{float(x):.8g}" for x in np.asarray(vec).ravel()) + "]"

    def ensure_schema(self, dim: int) -> None:
        dim = int(dim)
        if dim > HNSW_MAX_DIM:
            # checado aqui para falhar com um recado acionavel em vez da
            # `psycopg.errors.ProgramLimitExceeded` crua do CREATE INDEX.
            raise ValueError(
                f"dimensao {dim} acima do teto do indice HNSW do pgvector "
                f"({HNSW_MAX_DIM} dimensoes): reduza o vetor com --dim <= {HNSW_MAX_DIM} "
                f"(Matryoshka/MRL -- o default da fatia e {DEFAULT_DIM}, que trunca e "
                "renormaliza, preservando o cosseno)")
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(DDL_TABLE.format(table=self.table, dim=dim))
            cur.execute(DDL_INDEX.format(table=self.table))
            cur.execute(DDL_INDEX_MODALITY.format(table=self.table))
        self._conn.commit()

    def upsert(self, rows: list[ItemRow]) -> int:
        if not rows:
            return 0
        params = [(r.item_id, r.modality, r.source, r.uri, r.pointer_uri, r.credit,
                   r.content_text, r.content_sha256, r.model_id,
                   self.vector_literal(r.embedding)) for r in rows]
        with self._conn.cursor() as cur:
            cur.executemany(SQL_UPSERT.format(table=self.table), params)
        self._conn.commit()
        return len(rows)

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return int(cur.fetchone()[0])

    def search(self, vec: np.ndarray, k: int = TOP_K,
               modality: str | None = None) -> list[Hit]:
        literal = self.vector_literal(vec)
        where = SQL_WHERE_MODALITY if modality else ""
        params = [literal] + ([modality] if modality else []) + [literal, int(k)]
        with self._conn.cursor() as cur:
            cur.execute(SQL_SEARCH.format(table=self.table, where=where), tuple(params))
            return [Hit(item_id, mod, uri, float(score), pointer_uri, credit, content_text)
                    for item_id, mod, uri, pointer_uri, credit, content_text, score
                    in cur.fetchall()]


def open_vector_store(args: argparse.Namespace) -> VectorStore:
    """`--local-store` -> fake em memoria (dev/testes); senao, Cloud SQL + pgvector."""
    local = getattr(args, "local_store", None)
    if local:
        # `--local-store` sozinho = so memoria; com um caminho, o indice fake persiste
        # naquele JSON (e so assim embed_job e query se enxergam em processos diferentes).
        return InMemoryVectorStore(path=local if isinstance(local, str) else None)
    return PgVectorStore.connect(host=args.pg_host, port=args.pg_port, dbname=args.pg_db,
                                 user=args.pg_user, password_env=args.password_env,
                                 table=args.table)


def add_pg_arguments(ap: argparse.ArgumentParser) -> None:
    """Flags de conexao compartilhadas por `embed_job.py` e `query.py`."""
    ap.add_argument("--pg-host", default=os.environ.get("PGHOST", "127.0.0.1"),
                    help="host do Postgres (default: 127.0.0.1, o Cloud SQL Auth Proxy)")
    ap.add_argument("--pg-port", type=int, default=int(os.environ.get("PGPORT", 5432)))
    ap.add_argument("--pg-db", default=os.environ.get("PGDATABASE", "govexe"))
    ap.add_argument("--pg-user", default=os.environ.get("PGUSER", "govexe"))
    ap.add_argument("--password-env", default=DEFAULT_PASSWORD_ENV,
                    help=f"variavel de ambiente com a senha (default: {DEFAULT_PASSWORD_ENV}); "
                         "preencha a partir do Secret Manager -- nunca por flag")
    ap.add_argument("--table", default=DEFAULT_TABLE,
                    help=f"tabela do indice (default: {DEFAULT_TABLE})")
    ap.add_argument("--local-store", nargs="?", const=True, default=None,
                    metavar="ARQUIVO.json",
                    help="usa o indice fake em memoria (modo dev; nao toca o Cloud SQL). "
                         "Com um caminho .json, o indice fake e persistido nesse arquivo, "
                         "para o query.py reabrir o mesmo lote em outro processo")


# -- download do Bronze -------------------------------------------------------------------


def download_pages(store: ObjectStore, prefix: str, dest_dir: str | Path,
                   limit: int | None = None) -> tuple[list[str], dict[str, str]]:
    """Baixa as imagens sob `prefix` e devolve (caminhos locais, caminho -> URI de origem)."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    nomes = [n for n in store.list(prefix) if n.lower().endswith(IMAGE_SUFFIXES)]
    if limit:
        nomes = nomes[:limit]
    paths, uris = [], {}
    for nome in nomes:
        # resolve(): as chaves de `uris` precisam bater com o campo "path" do manifest,
        # que `scripts/embed_qwen3vl.build_manifest` grava ja resolvido.
        local = store.download(nome, dest).resolve()
        paths.append(str(local))
        uris[str(local)] = store.uri(nome)
    log.info("bronze -> %s: %d paginas baixadas de %s", dest, len(paths), prefix)
    return paths, uris


# -- CLI ------------------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, store: VectorStore | None = None,
         transcriber: Transcriber | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    entrada = ap.add_mutually_exclusive_group(required=True)
    entrada.add_argument("--bucket", help="bucket GCS com o Bronze (ex.: gs://govexe-bronze)")
    entrada.add_argument("--local-bronze", help="diretorio que simula o bucket (modo dev)")
    entrada.add_argument("--images-dir", help="le as paginas direto de um diretorio local")
    entrada.add_argument("--records",
                         help="JSONL de records do lote misto (foto/texto/audio; formato "
                              "no README-slice.md)")
    entrada.add_argument("--vectors",
                         help="prefixo .npy/.manifest.json de scripts/embed_qwen3vl.py "
                              "(nao carrega modelo)")
    ap.add_argument("--records-store", default=None,
                    help="Bronze onde resolver o campo `blob` dos records "
                         "(gs://bucket ou diretorio local)")
    ap.add_argument("--blob-prefix", default=f"{DEFAULT_PREFIX}/",
                    help=f"prefixo dos objetos no bucket (default: {DEFAULT_PREFIX}/)")
    ap.add_argument("--work-dir", default=DEFAULT_WORK_DIR,
                    help=f"onde o job baixa as paginas do GCS (default: {DEFAULT_WORK_DIR})")
    ap.add_argument("--glob", default=None, help="padrao das imagens em --images-dir")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="fonte gravada em cada linha")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help=f"checkpoint do embedder (default: {DEFAULT_MODEL_ID})")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM,
                    help="trunca o vetor para N dimensoes (Matryoshka/MRL) e renormaliza "
                         f"(default: {DEFAULT_DIM}; o indice HNSW do pgvector so aceita ate "
                         f"{HNSW_MAX_DIM} dimensoes)")
    ap.add_argument("--limit", type=int, default=None, help="processa so os N primeiros itens")
    add_asr_arguments(ap)
    add_pg_arguments(ap)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.records:
        registros = load_records(a.records)
        if a.limit:
            registros = registros[:a.limit]
        if not registros:
            log.error("nenhum record em %s", a.records)
            return 1
        objetos = store_from_location(a.records_store) if a.records_store else None
        log.info("embutindo %d records (%s) com %s", len(registros),
                 ", ".join(sorted({r.get("modality", "?") for r in registros})), a.model_id)
        rows = embed_records(registros, model_id=a.model_id, device=a.device, dtype=a.dtype,
                             batch_size=a.batch_size, dim=a.dim, source=a.source,
                             transcriber=transcriber or transcriber_from_args(a),
                             store=objetos, work_dir=a.work_dir)
        return _index(rows, a, store, dim=len(rows[0].embedding))

    uris: dict[str, str] = {}
    if a.vectors:
        manifest, vecs = load_vectors(a.vectors)
        items, model_id = manifest["items"], manifest.get("model_id", a.model_id)
        if a.limit:
            items, vecs = items[:a.limit], vecs[:a.limit]
    else:
        if a.images_dir:
            paths = collect_paths(a.images_dir, a.glob or "**/*.png", a.limit)
        else:
            objetos = open_store(a.bucket, a.local_bronze)
            paths, uris = download_pages(objetos, a.blob_prefix, a.work_dir, a.limit)
        if not paths:
            log.error("nenhuma pagina para embutir")
            return 1
        log.info("embutindo %d paginas com %s", len(paths), a.model_id)
        vecs = embed_images(paths, model_id=a.model_id, device=a.device, dtype=a.dtype,
                            batch_size=a.batch_size, dim=a.dim)
        items = build_manifest(paths, a.model_id, vecs.shape[1])["items"]
        model_id = a.model_id

    rows = build_rows(items, vecs, model_id=model_id, source=a.source, uris=uris)
    return _index(rows, a, store, dim=int(np.shape(vecs)[1]))


def _index(rows: list[ItemRow], a: argparse.Namespace, store: VectorStore | None,
           *, dim: int) -> int:
    """Upsert das linhas no indice (o mesmo fim para todos os modos de entrada)."""
    store = store or open_vector_store(a)
    store.ensure_schema(dim)
    gravadas = store.upsert(rows)
    total = store.count()
    log.info("upsert: %d linhas gravadas; indice com %d itens", gravadas, total)
    print(f"{gravadas} itens gravados; total no indice: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
