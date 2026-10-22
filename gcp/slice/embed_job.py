"""Job de embedding da fatia GCP: GCS (Bronze) -> Qwen3-VL-Embedding -> Cloud SQL/pgvector.

Roda na **VM L4 efemera** (`g2-standard-4`, ver `gcp/slice/README-slice.md`): baixa as
paginas do bucket Bronze, embute cada uma com `Qwen/Qwen3-VL-Embedding-2B`
(`benchmark.models.qwen3vl_embed.embed_images`) e faz **upsert** no Cloud SQL + pgvector.

Tres modos de entrada:

    --bucket gs://... --blob-prefix bronze/fonte=.../dt=...   paginas vindas do GCS
    --images-dir <dir>                                        diretorio local (modo dev)
    --vectors <prefixo>                                       reusa `<prefixo>.npy` +
                                                              `<prefixo>.manifest.json`
                                                              de `scripts/embed_qwen3vl.py`
                                                              (nao carrega modelo nenhum)

**Idempotencia** (requisito do plano): a chave e o `page_id` -- o stem do arquivo, o mesmo
id de pagina do `corpus.jsonl` (`<tse_seq>_p<NN>`). O SQL e
`INSERT ... ON CONFLICT (page_id) DO UPDATE`, entao rodar o job duas vezes sobre o mesmo
Bronze deixa exatamente as mesmas linhas; `content_sha256` registra o conteudo que gerou
aquele vetor (pagina reemitida com bytes diferentes atualiza a linha, nao cria outra).

**Segredo**: a senha do Cloud SQL vem da variavel de ambiente `PGPASSWORD` (nome
configuravel por `--password-env`), populada a partir do **Secret Manager**:

    export PGPASSWORD="$(gcloud secrets versions access latest --secret=govexe-pg-password)"

Nada de senha em flag, em codigo ou no git.

Uso:
    python -m gcp.slice.embed_job --bucket gs://govexe-bronze --blob-prefix bronze/ \
        --pg-host 127.0.0.1 --pg-db govexe --pg-user govexe
    python -m gcp.slice.embed_job --images-dir pages --local-store /tmp/pgvector-fake  # dev
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

from benchmark.models.qwen3vl_embed import DEFAULT_MODEL_ID, embed_images  # noqa: E402
from gcp.slice.collector import (  # noqa: E402
    DEFAULT_PREFIX,
    DEFAULT_SOURCE,
    IMAGE_SUFFIXES,
    ObjectStore,
    open_store,
)
from scripts.embed_qwen3vl import build_manifest, collect_paths  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_TABLE = "govexe_pages"
DEFAULT_PASSWORD_ENV = "PGPASSWORD"
DEFAULT_WORK_DIR = "/tmp/govexe-bronze"
TOP_K = 5

# DDL identica a do README-slice.md (secao "tabela"): o job cria a tabela se ela nao
# existir, mas a criacao tambem pode ser feita a mao no Cloud SQL Studio/psql.
DDL_TABLE = """CREATE TABLE IF NOT EXISTS {table} (
    page_id        TEXT PRIMARY KEY,
    source         TEXT NOT NULL,
    uri            TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    model_id       TEXT NOT NULL,
    embedding      VECTOR({dim}) NOT NULL,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
)"""

DDL_INDEX = ("CREATE INDEX IF NOT EXISTS {table}_embedding_cos "
             "ON {table} USING hnsw (embedding vector_cosine_ops)")

# O upsert idempotente: chave = page_id. Reprocessar o mesmo Bronze atualiza a linha.
SQL_UPSERT = """INSERT INTO {table}
    (page_id, source, uri, content_sha256, model_id, embedding, updated_at)
VALUES (%s, %s, %s, %s, %s, %s::vector, now())
ON CONFLICT (page_id) DO UPDATE SET
    source         = EXCLUDED.source,
    uri            = EXCLUDED.uri,
    content_sha256 = EXCLUDED.content_sha256,
    model_id       = EXCLUDED.model_id,
    embedding      = EXCLUDED.embedding,
    updated_at     = now()"""

SQL_SEARCH = """SELECT page_id, uri, 1 - (embedding <=> %s::vector) AS score
FROM {table}
ORDER BY embedding <=> %s::vector
LIMIT %s"""


# -- linhas -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PageRow:
    """Uma pagina pronta para o indice: metadados + vetor L2-normalizado."""

    page_id: str
    source: str
    uri: str
    content_sha256: str
    model_id: str
    embedding: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class Hit:
    """Resultado de busca (score = cosseno, porque os vetores saem L2-normalizados)."""

    page_id: str
    uri: str
    score: float


def content_sha256(path: str | Path) -> str:
    """sha256 do arquivo. Pagina ausente (manifest de outra maquina) -> hash da URI."""
    p = Path(path)
    if not p.is_file():
        log.warning("arquivo ausente, hash derivado do caminho: %s", p)
        return hashlib.sha256(str(p).encode()).hexdigest()
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def build_rows(items: list[dict], vecs: np.ndarray, *, model_id: str = DEFAULT_MODEL_ID,
               source: str = DEFAULT_SOURCE,
               uris: dict[str, str] | None = None) -> list[PageRow]:
    """Manifest + matriz -> linhas do indice.

    `items` segue o manifest de `scripts/embed_qwen3vl.py` (`{"row", "id", "path"}`), entao
    o `page_id` ja e o stem do arquivo. `uris` mapeia caminho local -> URI de origem
    (`gs://...`) quando a pagina veio do Bronze; sem ele, a URI e o proprio caminho.
    """
    vecs = np.atleast_2d(np.asarray(vecs, dtype="float32"))
    if len(items) != len(vecs):
        raise ValueError(f"manifest ({len(items)}) e vetores ({len(vecs)}) nao batem")
    uris = uris or {}
    linhas = []
    for item in items:
        path = item["path"]
        linhas.append(PageRow(
            page_id=item["id"],
            source=source,
            uri=uris.get(path, str(path)),
            content_sha256=content_sha256(path),
            model_id=model_id,
            embedding=np.asarray(vecs[item["row"]], dtype="float32"),
        ))
    return linhas


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

    def upsert(self, rows: list[PageRow]) -> int:
        """Grava as linhas por `page_id` (idempotente) e devolve quantas foram gravadas."""

    def count(self) -> int:
        ...

    def search(self, vec: np.ndarray, k: int = TOP_K) -> list[Hit]:
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
    """FAKE de `VectorStore`: dicionario por `page_id` + produto interno. Usado nos testes
    e no modo dev (`--vectors` do `query.py`), nunca em producao."""

    def __init__(self, rows: list[PageRow] | None = None):
        self.rows: dict[str, PageRow] = {}
        if rows:
            self.upsert(rows)

    def ensure_schema(self, dim: int) -> None:
        self.dim = dim

    def upsert(self, rows: list[PageRow]) -> int:
        for row in rows:
            self.rows[row.page_id] = row       # mesma chave -> substitui, nao duplica
        return len(rows)

    def count(self) -> int:
        return len(self.rows)

    def search(self, vec: np.ndarray, k: int = TOP_K) -> list[Hit]:
        if not self.rows:
            return []
        ids = list(self.rows)
        matriz = np.vstack([self.rows[i].embedding for i in ids])
        return [Hit(pid, self.rows[pid].uri, score)
                for pid, score in rank_top_k(vec, matriz, ids, k)]


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
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(DDL_TABLE.format(table=self.table, dim=int(dim)))
            cur.execute(DDL_INDEX.format(table=self.table))
        self._conn.commit()

    def upsert(self, rows: list[PageRow]) -> int:
        if not rows:
            return 0
        params = [(r.page_id, r.source, r.uri, r.content_sha256, r.model_id,
                   self.vector_literal(r.embedding)) for r in rows]
        with self._conn.cursor() as cur:
            cur.executemany(SQL_UPSERT.format(table=self.table), params)
        self._conn.commit()
        return len(rows)

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return int(cur.fetchone()[0])

    def search(self, vec: np.ndarray, k: int = TOP_K) -> list[Hit]:
        literal = self.vector_literal(vec)
        with self._conn.cursor() as cur:
            cur.execute(SQL_SEARCH.format(table=self.table), (literal, literal, int(k)))
            return [Hit(page_id, uri, float(score)) for page_id, uri, score in cur.fetchall()]


def open_vector_store(args: argparse.Namespace) -> VectorStore:
    """`--local-store` -> fake em memoria (dev/testes); senao, Cloud SQL + pgvector."""
    if getattr(args, "local_store", None):
        return InMemoryVectorStore()
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
    ap.add_argument("--local-store", action="store_true",
                    help="usa o indice fake em memoria (modo dev; nao toca o Cloud SQL)")


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


def main(argv: list[str] | None = None, *, store: VectorStore | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    entrada = ap.add_mutually_exclusive_group(required=True)
    entrada.add_argument("--bucket", help="bucket GCS com o Bronze (ex.: gs://govexe-bronze)")
    entrada.add_argument("--local-bronze", help="diretorio que simula o bucket (modo dev)")
    entrada.add_argument("--images-dir", help="le as paginas direto de um diretorio local")
    entrada.add_argument("--vectors",
                         help="prefixo .npy/.manifest.json de scripts/embed_qwen3vl.py "
                              "(nao carrega modelo)")
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
    ap.add_argument("--dim", type=int, default=None,
                    help="trunca o vetor para N dimensoes (Matryoshka/MRL) e renormaliza")
    ap.add_argument("--limit", type=int, default=None, help="processa so as N primeiras paginas")
    add_pg_arguments(ap)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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
    store = store or open_vector_store(a)
    store.ensure_schema(int(np.shape(vecs)[1]))
    gravadas = store.upsert(rows)
    total = store.count()
    log.info("upsert: %d linhas gravadas; indice com %d paginas", gravadas, total)
    print(f"{gravadas} paginas gravadas; total no indice: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
