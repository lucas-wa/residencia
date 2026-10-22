"""Fatia GCP (`gcp/slice/`): coletor Bronze, job de embedding e consulta top-k.

Tudo offline: o GCS e substituido pelo `LocalObjectStore` (fake de filesystem), o Cloud SQL
pelo `InMemoryVectorStore`, e o encode do Qwen3-VL e monkeypatchado -- nenhum teste toca a
nuvem, a rede ou pesos de modelo.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from gcp.slice import collector, embed_job, query


def _pages(tmp_path, n, prefix="BR001"):
    """Cria N paginas-imagem falsas (bytes distintos, para o sha256 diferir)."""
    pages = tmp_path / "pages"
    pages.mkdir(exist_ok=True)
    for i in range(n):
        (pages / f"{prefix}_p{i:02d}.png").write_bytes(b"png" + bytes([i]))
    return pages


def _fake_embed(paths, **kw):
    """Vetores canonicos e0, e1, ... (L2=1): o ranking fica previsivel."""
    return np.eye(len(paths), 4, dtype="float32")


# -- collector -------------------------------------------------------------------------


def test_blob_name_particiona_por_fonte_e_data():
    assert collector.blob_name("/x/y/BR001_p00.png", "tse_planos", "2026-10-22") == (
        "bronze/fonte=tse_planos/dt=2026-10-22/BR001_p00.png")


def test_collect_sobe_tudo_e_na_segunda_vez_nao_reenvia(tmp_path):
    pages = _pages(tmp_path, 3)
    store = collector.LocalObjectStore(tmp_path / "bucket")
    paths = collector.collect_paths(str(pages))

    primeira = collector.collect(paths, store, day="2026-10-22")
    assert [r["uploaded"] for r in primeira] == [True, True, True]
    assert store.list("bronze/") == [
        f"bronze/fonte=tse_planos/dt=2026-10-22/BR001_p{i:02d}.png" for i in range(3)]

    segunda = collector.collect(paths, store, day="2026-10-22")
    assert [r["uploaded"] for r in segunda] == [False, False, False]
    assert len(store.list("bronze/")) == 3             # nenhum objeto a mais
    assert [r["blob"] for r in segunda] == [r["blob"] for r in primeira]

    terceira = collector.collect(paths, store, day="2026-10-22", force=True)
    assert all(r["uploaded"] for r in terceira) and len(store.list("bronze/")) == 3


def test_collector_main_modo_dev_sobe_para_o_fake(tmp_path):
    pages = _pages(tmp_path, 2)
    root = tmp_path / "bucket"
    rc = collector.main(["--local-root", str(root), "--images-dir", str(pages),
                         "--date", "2026-10-22", "--source", "tse_planos"])
    assert rc == 0
    assert (root / "bronze/fonte=tse_planos/dt=2026-10-22/BR001_p00.png").is_file()


def test_collector_main_sem_paginas_sai_1(tmp_path):
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    assert collector.main(["--local-root", str(tmp_path / "b"), "--images-dir", str(vazio)]) == 1


def test_open_store_exige_exatamente_um_destino(tmp_path):
    with pytest.raises(ValueError):
        collector.open_store(None, None)
    with pytest.raises(ValueError):
        collector.open_store("gs://b", str(tmp_path))


# -- embed_job: manifest -> linhas --------------------------------------------------------


def test_build_rows_do_manifest_usa_o_stem_como_chave(tmp_path):
    pages = _pages(tmp_path, 2)
    paths = collector.collect_paths(str(pages))
    vecs = _fake_embed(paths)
    items = [{"row": i, "id": Path(p).stem, "path": p} for i, p in enumerate(paths)]

    rows = embed_job.build_rows(items, vecs, model_id="Qwen/X", source="tse_planos",
                                uris={paths[0]: "gs://b/bronze/BR001_p00.png"})

    assert [r.page_id for r in rows] == ["BR001_p00", "BR001_p01"]
    assert rows[0].uri == "gs://b/bronze/BR001_p00.png"     # veio do Bronze
    assert rows[1].uri == paths[1]                          # sem URI mapeada: caminho local
    assert rows[0].content_sha256 != rows[1].content_sha256
    assert rows[0].model_id == "Qwen/X"
    np.testing.assert_allclose(rows[0].embedding, vecs[0])


def test_build_rows_recusa_manifest_e_matriz_de_tamanhos_diferentes(tmp_path):
    items = [{"row": 0, "id": "a", "path": str(tmp_path / "a.png")}]
    with pytest.raises(ValueError):
        embed_job.build_rows(items, np.zeros((2, 4), dtype="float32"))


def test_content_sha256_muda_com_o_conteudo(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    assert embed_job.content_sha256(a) != embed_job.content_sha256(b)
    assert embed_job.content_sha256(a) == embed_job.content_sha256(a)


# -- embed_job: upsert idempotente ---------------------------------------------------------


def _row(page_id, vec, sha="h0", uri="gs://b/x.png"):
    return embed_job.PageRow(page_id=page_id, source="tse_planos", uri=uri,
                             content_sha256=sha, model_id="Qwen/X",
                             embedding=np.asarray(vec, dtype="float32"))


def test_upsert_em_memoria_e_idempotente_por_page_id():
    store = embed_job.InMemoryVectorStore()
    rows = [_row("BR001_p00", [1, 0, 0, 0]), _row("BR001_p01", [0, 1, 0, 0])]

    store.upsert(rows)
    store.upsert(rows)
    assert store.count() == 2

    store.upsert([_row("BR001_p00", [0, 0, 1, 0], sha="h1")])   # mesma pagina, novo conteudo
    assert store.count() == 2
    assert store.rows["BR001_p00"].content_sha256 == "h1"
    np.testing.assert_allclose(store.rows["BR001_p00"].embedding, [0, 0, 1, 0])


def test_embed_job_main_duas_ingestoes_nao_duplicam(tmp_path, monkeypatch, capsys):
    """Fatia inteira offline: paginas -> fake de GCS -> embed mock -> indice -> 2x sem duplicar."""
    pages = _pages(tmp_path, 3)
    bucket = tmp_path / "bucket"
    bronze = collector.LocalObjectStore(bucket)
    collector.collect(collector.collect_paths(str(pages)), bronze, day="2026-10-22")

    monkeypatch.setattr(embed_job, "embed_images", _fake_embed)
    store = embed_job.InMemoryVectorStore()
    argv = ["--local-bronze", str(bucket), "--blob-prefix", "bronze/",
            "--work-dir", str(tmp_path / "work"), "--device", "cpu"]

    assert embed_job.main(argv, store=store) == 0
    assert store.count() == 3
    primeiras = {pid: row.content_sha256 for pid, row in store.rows.items()}

    assert embed_job.main(argv, store=store) == 0          # reprocesso do mesmo Bronze
    assert store.count() == 3
    assert {pid: row.content_sha256 for pid, row in store.rows.items()} == primeiras
    assert sorted(store.rows) == ["BR001_p00", "BR001_p01", "BR001_p02"]
    assert store.rows["BR001_p00"].uri.startswith("gs://") is False  # fake: caminho local
    assert "bronze/fonte=tse_planos/dt=2026-10-22" in store.rows["BR001_p00"].uri
    assert "3 paginas gravadas" in capsys.readouterr().out


def test_embed_job_main_reusa_npy_e_manifest(tmp_path, monkeypatch):
    """Modo --vectors: le a saida de scripts/embed_qwen3vl.py sem carregar modelo."""
    pages = _pages(tmp_path, 2)
    paths = collector.collect_paths(str(pages))
    vecs = _fake_embed(paths)
    base = tmp_path / "emb" / "planos"
    base.parent.mkdir(parents=True)
    np.save(base.with_suffix(".npy"), vecs)
    base.with_suffix(".manifest.json").write_text(json.dumps(
        {"model_id": "Qwen/X", "dim": 4, "count": 2,
         "items": [{"row": i, "id": Path(p).stem, "path": p}
                   for i, p in enumerate(paths)]}), encoding="utf-8")

    def nao_deveria(*a, **kw):
        raise AssertionError("--vectors nao pode chamar o embedder")

    monkeypatch.setattr(embed_job, "embed_images", nao_deveria)
    store = embed_job.InMemoryVectorStore()
    assert embed_job.main(["--vectors", str(base)], store=store) == 0
    assert sorted(store.rows) == ["BR001_p00", "BR001_p01"]
    assert store.rows["BR001_p00"].model_id == "Qwen/X"


# -- embed_job: SQL do backend real (sem banco) --------------------------------------------


class _FakeCursor:
    def __init__(self, sql_log, resultado=None):
        self._log = sql_log
        self._resultado = resultado or []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._log.append((" ".join(sql.split()), params))

    def executemany(self, sql, params):
        self._log.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self._resultado

    def fetchone(self):
        return (len(self._resultado),)


class _FakeConn:
    def __init__(self, resultado=None):
        self.sql = []
        self.commits = 0
        self._resultado = resultado

    def cursor(self):
        return _FakeCursor(self.sql, self._resultado)

    def commit(self):
        self.commits += 1


def test_pgvector_upsert_usa_on_conflict_na_chave_de_pagina():
    conn = _FakeConn()
    store = embed_job.PgVectorStore(conn, table="govexe_pages")

    store.ensure_schema(4)
    store.upsert([_row("BR001_p00", [1, 0, 0, 0])])

    ddl = conn.sql[1][0]
    assert "CREATE EXTENSION IF NOT EXISTS vector" in conn.sql[0][0]
    assert "govexe_pages" in ddl and "VECTOR(4)" in ddl and "page_id TEXT PRIMARY KEY" in ddl
    assert "USING hnsw (embedding vector_cosine_ops)" in conn.sql[2][0]

    sql, params = conn.sql[3]
    assert "INSERT INTO govexe_pages" in sql
    assert "ON CONFLICT (page_id) DO UPDATE SET" in sql
    assert params[0][0] == "BR001_p00" and params[0][5] == "[1,0,0,0]"
    assert conn.commits == 2


def test_pgvector_search_ordena_pelo_operador_de_cosseno():
    conn = _FakeConn(resultado=[("BR001_p01", "gs://b/p01.png", 0.9)])
    store = embed_job.PgVectorStore(conn)

    hits = store.search(np.array([0, 1, 0, 0], dtype="float32"), k=3)

    sql, params = conn.sql[0]
    assert "ORDER BY embedding <=> %s::vector" in sql and params[2] == 3
    assert hits == [embed_job.Hit("BR001_p01", "gs://b/p01.png", 0.9)]


def test_pgvector_connect_sem_senha_no_ambiente_falha(monkeypatch):
    monkeypatch.delenv("PGPASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="PGPASSWORD"):
        embed_job.PgVectorStore.connect(host="127.0.0.1", port=5432, dbname="govexe",
                                        user="govexe")


# -- query ---------------------------------------------------------------------------------


def test_rank_top_k_ordena_do_maior_para_o_menor():
    matriz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.7, 0.7, 0, 0]], dtype="float32")
    ids = ["p0", "p1", "p2"]

    top = query.rank_top_k(np.array([0, 1, 0, 0], dtype="float32"), matriz, ids, k=2)

    assert [pid for pid, _ in top] == ["p1", "p2"]
    assert top[0][1] > top[1][1]
    assert query.rank_top_k(np.zeros(4), np.zeros((0, 4)), [], k=3) == []


def test_query_main_imprime_o_ranking(monkeypatch, capsys):
    store = embed_job.InMemoryVectorStore([
        _row("BR001_p00", [1, 0, 0, 0], uri="gs://b/p00.png"),
        _row("BR001_p01", [0, 1, 0, 0], uri="gs://b/p01.png"),
    ])
    monkeypatch.setattr(query, "embed_texts",
                        lambda textos, **kw: np.array([[0, 1, 0, 0]], dtype="float32"))

    rc = query.main(["--text", "metas de saude", "--k", "2", "--device", "cpu"], store=store)

    assert rc == 0
    linhas = capsys.readouterr().out.splitlines()
    assert "metas de saude" in linhas[0]
    assert "BR001_p01" in linhas[1] and "gs://b/p01.png" in linhas[1]
    assert "BR001_p00" in linhas[2]


def test_query_main_modo_dev_le_npy_e_manifest(tmp_path, monkeypatch, capsys):
    pages = _pages(tmp_path, 2)
    paths = collector.collect_paths(str(pages))
    base = tmp_path / "emb"
    np.save(base.with_suffix(".npy"), _fake_embed(paths))
    base.with_suffix(".manifest.json").write_text(json.dumps(
        {"model_id": "Qwen/X", "dim": 4, "count": 2,
         "items": [{"row": i, "id": f"BR001_p{i:02d}", "path": p}
                   for i, p in enumerate(paths)]}), encoding="utf-8")
    monkeypatch.setattr(query, "embed_texts",
                        lambda textos, **kw: np.array([[1, 0, 0, 0]], dtype="float32"))

    assert query.main(["--text", "saude", "--vectors", str(base), "--device", "cpu"]) == 0
    assert "BR001_p00" in capsys.readouterr().out


def test_query_main_indice_vazio_sai_1(monkeypatch):
    monkeypatch.setattr(query, "embed_texts",
                        lambda textos, **kw: np.zeros((1, 4), dtype="float32"))
    assert query.main(["--text", "x"], store=embed_job.InMemoryVectorStore()) == 1
