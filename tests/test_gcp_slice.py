"""Fatia GCP (`gcp/slice/`): coletor Bronze, job de embedding multimodal e consulta top-k.

Tudo offline: o GCS e substituido pelo `LocalObjectStore` (fake de filesystem), o Cloud SQL
pelo `InMemoryVectorStore`, o Whisper pelo `FakeTranscriber` e o encode do Qwen3-VL (e o do
CLAP) e monkeypatchado -- nenhum teste toca a nuvem, a rede, GPU ou pesos de modelo.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from gcp.slice import asr, audio_collector, collector, embed_job, query, spaces


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


def _fake_embed_texts(texts, **kw):
    """Texto -> vetor canonico: e3 para transcricao de audio, e2 para texto nativo.

    Fica no MESMO espaco de 4 dimensoes do `_fake_embed` das imagens -- e isso que
    permite exercitar a busca cross-modal sem modelo nenhum.
    """
    linhas = []
    for t in texts:
        v = np.zeros(4, dtype="float32")
        v[3 if asr.FAKE_PREFIX in t else 2] = 1.0
        linhas.append(v)
    return np.vstack(linhas)


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

    assert [r.item_id for r in rows] == ["BR001_p00", "BR001_p01"]
    assert {r.modality for r in rows} == {embed_job.MODALITY_DOC_IMAGE}
    assert rows[0].content_text is None          # pagina-imagem nao tem texto embutido
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


def _row(item_id, vec, sha="h0", uri="gs://b/x.png",
         modality=embed_job.MODALITY_DOC_IMAGE, **kw):
    return embed_job.ItemRow(item_id=item_id, modality=modality, source="tse_planos", uri=uri,
                             content_sha256=sha, model_id="Qwen/X",
                             embedding=np.asarray(vec, dtype="float32"), **kw)


def test_upsert_em_memoria_e_idempotente_por_item_id():
    store = embed_job.InMemoryVectorStore()
    rows = [_row("BR001_p00", [1, 0, 0, 0]), _row("BR001_p01", [0, 1, 0, 0])]

    store.upsert(rows)
    store.upsert(rows)
    assert store.count() == 2

    store.upsert([_row("BR001_p00", [0, 0, 1, 0], sha="h1")])   # mesma pagina, novo conteudo
    assert store.count() == 2
    assert store.rows["BR001_p00"].content_sha256 == "h1"
    np.testing.assert_allclose(store.rows["BR001_p00"].embedding, [0, 0, 1, 0])


def test_in_memory_store_com_arquivo_sobrevive_ao_processo(tmp_path):
    """`--local-store <arquivo>`: o indice fake persiste entre duas execucoes do CLI."""
    arquivo = tmp_path / "indice.json"
    primeira = embed_job.InMemoryVectorStore(path=arquivo)
    primeira.upsert([_row("foto:abc", [0, 0, 1, 0], modality=embed_job.MODALITY_IMAGE,
                          credit="Foto: Fulano/Agencia Brasil",
                          pointer_uri="https://agencia/foto.jpg")])

    segunda = embed_job.InMemoryVectorStore(path=arquivo)
    assert segunda.count() == 1
    linha = segunda.rows["foto:abc"]
    assert linha.modality == embed_job.MODALITY_IMAGE
    assert linha.credit == "Foto: Fulano/Agencia Brasil"
    assert linha.pointer_uri == "https://agencia/foto.jpg"
    np.testing.assert_allclose(linha.embedding, [0, 0, 1, 0])


def test_embed_job_main_duas_ingestoes_nao_duplicam(tmp_path, monkeypatch, capsys):
    """Fatia inteira offline: paginas -> fake de GCS -> embed mock -> indice -> 2x sem duplicar."""
    pages = _pages(tmp_path, 3)
    bucket = tmp_path / "bucket"
    bronze = collector.LocalObjectStore(bucket)
    collector.collect(collector.collect_paths(str(pages)), bronze, day="2026-10-22")

    monkeypatch.setattr(embed_job, "embed_images", _fake_embed)
    store = embed_job.InMemoryVectorStore()
    # `--dim 4` = a largura do embedder fake. So passa porque o indice e o fake em memoria
    # (no Cloud SQL a largura e sempre a que o espaco declara -- ver os testes de --space).
    argv = ["--local-bronze", str(bucket), "--blob-prefix", "bronze/",
            "--work-dir", str(tmp_path / "work"), "--device", "cpu", "--dim", "4"]

    assert embed_job.main(argv, store=store) == 0
    assert store.count() == 3
    primeiras = {pid: row.content_sha256 for pid, row in store.rows.items()}

    assert embed_job.main(argv, store=store) == 0          # reprocesso do mesmo Bronze
    assert store.count() == 3
    assert {pid: row.content_sha256 for pid, row in store.rows.items()} == primeiras
    assert sorted(store.rows) == ["BR001_p00", "BR001_p01", "BR001_p02"]
    assert store.rows["BR001_p00"].uri.startswith("gs://") is False  # fake: caminho local
    assert "bronze/fonte=tse_planos/dt=2026-10-22" in store.rows["BR001_p00"].uri
    assert "3 itens gravados" in capsys.readouterr().out


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
    # `--dim 4` = a largura do .npy ja gravado; aceito por ser indice local (em memoria)
    assert embed_job.main(["--vectors", str(base), "--dim", "4"], store=store) == 0
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


def test_pgvector_upsert_usa_on_conflict_na_chave_de_item():
    conn = _FakeConn()
    store = embed_job.PgVectorStore(conn, table="govexe_items")

    store.ensure_schema(4)
    store.upsert([_row("audio:abc", [1, 0, 0, 0], modality=embed_job.MODALITY_AUDIO,
                       credit="Radio Camara", pointer_uri="gs://b/audio.mp3",
                       content_text="transcricao")])

    ddl = conn.sql[1][0]
    assert "CREATE EXTENSION IF NOT EXISTS vector" in conn.sql[0][0]
    assert "govexe_items" in ddl and "VECTOR(4)" in ddl and "item_id TEXT PRIMARY KEY" in ddl
    assert "modality TEXT NOT NULL" in ddl and "credit TEXT" in ddl
    assert "USING hnsw (embedding vector_cosine_ops)" in conn.sql[2][0]
    assert "(modality)" in conn.sql[3][0]                      # btree do filtro por modalidade

    sql, params = conn.sql[4]
    assert "INSERT INTO govexe_items" in sql
    assert "ON CONFLICT (item_id) DO UPDATE SET" in sql
    assert params[0][:2] == ("audio:abc", embed_job.MODALITY_AUDIO)
    assert params[0][-1] == "[1,0,0,0]"
    assert conn.commits == 2


def test_pgvector_search_ordena_pelo_operador_de_cosseno():
    conn = _FakeConn(resultado=[("BR001_p01", "doc_image", "gs://b/p01.png",
                                 None, None, None, 0.9)])
    store = embed_job.PgVectorStore(conn)

    hits = store.search(np.array([0, 1, 0, 0], dtype="float32"), k=3)

    sql, params = conn.sql[0]
    assert "ORDER BY embedding <=> %s::vector" in sql and params[-1] == 3
    assert "WHERE" not in sql
    assert hits == [embed_job.Hit("BR001_p01", "doc_image", "gs://b/p01.png", 0.9)]


def test_pgvector_search_filtra_por_modalidade():
    conn = _FakeConn(resultado=[])
    store = embed_job.PgVectorStore(conn)

    store.search(np.array([0, 1, 0, 0], dtype="float32"), k=3,
                 modality=embed_job.MODALITY_AUDIO)

    sql, params = conn.sql[0]
    assert "WHERE modality = %s" in sql
    assert embed_job.MODALITY_AUDIO in params and params[-1] == 3


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
    assert embed_job.MODALITY_DOC_IMAGE in linhas[1]
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


# -- ASR (Whisper atras de interface) --------------------------------------------------------


def test_fake_transcriber_e_deterministico_e_nao_baixa_modelo(tmp_path):
    audio = tmp_path / "sessao_camara.mp3"
    audio.write_bytes(b"ID3fake")
    t = asr.FakeTranscriber()

    texto = t.transcribe(audio)

    assert texto == t.transcribe(audio)                  # deterministico
    assert asr.FAKE_PREFIX in texto and "sessao_camara" in texto
    assert t.calls == [str(audio), str(audio)]


def test_fake_transcriber_aceita_transcricao_canonica_por_arquivo(tmp_path):
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"RIFF")
    t = asr.FakeTranscriber({"radio": "metas de saneamento no orcamento"})
    assert t.transcribe(audio) == "metas de saneamento no orcamento"


def test_whisper_transcriber_nao_carrega_pesos_no_construtor():
    """Import do Whisper e lazy: construir o transcritor nao toca a rede nem a GPU."""
    import sys

    t = asr.WhisperTranscriber(model="large-v3", device="cpu")

    assert t.model == "large-v3" and t.backend == asr.DEFAULT_ASR_BACKEND
    assert t._model is None
    assert "faster_whisper" not in sys.modules and "whisper" not in sys.modules


def test_open_transcriber_fake_nao_instancia_whisper():
    assert isinstance(asr.open_transcriber("fake"), asr.FakeTranscriber)
    assert isinstance(asr.open_transcriber(asr.DEFAULT_ASR_BACKEND), asr.WhisperTranscriber)
    with pytest.raises(ValueError, match="backend"):
        asr.open_transcriber("nao-existe")


# -- records: id estavel, roteamento por modalidade e ASR ------------------------------------


def test_default_item_id_e_estavel_e_marcado_pela_modalidade():
    a = embed_job.default_item_id(embed_job.MODALITY_IMAGE, "https://agencia/foto.jpg")
    b = embed_job.default_item_id(embed_job.MODALITY_IMAGE, "https://agencia/foto.jpg")
    c = embed_job.default_item_id(embed_job.MODALITY_IMAGE, "https://agencia/outra.jpg")

    assert a == b and a != c
    assert a.startswith(f"{embed_job.MODALITY_IMAGE}:")


def test_prepare_record_texto_nativo_hasheia_o_proprio_texto():
    rec = {"modality": "text", "text": "meta: ampliar a atencao basica",
           "uri": "https://camara/materia/1", "credit": "Agencia Camara"}

    p = embed_job.prepare_record(rec, source="agencia_camara")

    assert p["content_text"] == "meta: ampliar a atencao basica"
    assert p["content_sha256"] == embed_job.text_sha256(rec["text"])
    assert p["pointer_uri"] is None and p["uri"] == "https://camara/materia/1"
    assert p["item_id"] == embed_job.default_item_id("text", "https://camara/materia/1")
    assert p["source"] == "agencia_camara"


def test_prepare_record_audio_chama_o_transcritor(tmp_path):
    audio = tmp_path / "sessao.mp3"
    audio.write_bytes(b"ID3fake")
    t = asr.FakeTranscriber()

    p = embed_job.prepare_record(
        {"modality": "audio_asr", "audio_path": str(audio),
         "pointer_uri": "https://youtube/watch?v=x", "credit": "TV Camara"},
        source="tv_camara", transcriber=t)

    assert t.calls == [str(audio)]
    assert p["content_text"] == t.transcribe(audio)
    assert p["pointer_uri"] == "https://youtube/watch?v=x" and p["credit"] == "TV Camara"


def test_prepare_record_audio_sem_transcritor_falha(tmp_path):
    audio = tmp_path / "sessao.mp3"
    audio.write_bytes(b"ID3fake")
    with pytest.raises(ValueError, match="transcritor"):
        embed_job.prepare_record({"modality": "audio_asr", "audio_path": str(audio)})


class _SpyStore(collector.LocalObjectStore):
    """`LocalObjectStore` que anota cada download -- para provar que ele NAO aconteceu."""

    def __init__(self, root):
        super().__init__(root)
        self.baixados: list[str] = []

    def download(self, name, dest_dir):
        self.baixados.append(name)
        return super().download(name, dest_dir)


def _store_com_foto(tmp_path, blob="bronze/fonte=camara/dt=2026-10-22/dep_204379.jpg"):
    """Bronze fake com a foto ja gravada sob `blob`."""
    store = _SpyStore(tmp_path / "bucket")
    origem = tmp_path / "origem.jpg"
    origem.write_bytes(b"jpgfake")
    store.upload(origem, blob)
    return store, blob


def test_prepare_record_baixa_do_bronze_quando_o_caminho_local_nao_existe(tmp_path):
    """Record gerado em outra maquina: `image_path` nao existe aqui, mas o `blob` sim."""
    store, blob = _store_com_foto(tmp_path)
    ausente = tmp_path / "outra_maquina" / "data" / "govexe" / "fotos" / "dep_204379.jpg"
    rec = {"modality": "image", "image_path": str(ausente),
           "blob": blob, "credit": "Camara dos Deputados"}

    p = embed_job.prepare_record(rec, store=store, work_dir=tmp_path / "work")

    assert store.baixados == [blob]
    assert Path(p["path"]).is_file() and p["path"] != rec["image_path"]
    assert p["uri"] == store.uri(blob)
    assert p["content_sha256"] == embed_job.content_sha256(p["path"])


def test_prepare_record_prefere_o_arquivo_local_quando_ele_existe(tmp_path):
    """Com o binario no disco, o Bronze nem e tocado (nada de download inutil)."""
    store, blob = _store_com_foto(tmp_path)
    local = tmp_path / "fotos" / "dep_204379.jpg"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"jpgfake-local")

    p = embed_job.prepare_record({"modality": "image", "image_path": str(local),
                                  "blob": blob}, store=store, work_dir=tmp_path / "work")

    assert store.baixados == []
    assert p["path"] == str(local)
    assert p["content_sha256"] == embed_job.content_sha256(local)


def test_prepare_record_com_blob_e_sem_store_falha(tmp_path):
    ausente = tmp_path / "outra_maquina" / "data" / "govexe" / "fotos" / "dep_204379.jpg"
    rec = {"modality": "image", "image_path": str(ausente),
           "blob": "bronze/fonte=camara/dt=2026-10-22/dep_204379.jpg"}
    with pytest.raises(ValueError, match="records-store"):
        embed_job.prepare_record(rec)


def test_prepare_record_modalidade_desconhecida_falha():
    with pytest.raises(ValueError, match="modalidade"):
        embed_job.prepare_record({"modality": "video", "uri": "x"})


def test_embed_records_roteia_cada_modalidade_para_o_encoder_certo(tmp_path, monkeypatch):
    """Lote misto: imagem vai por embed_images, texto/ASR por embed_texts -- mesmo espaco."""
    pages = _pages(tmp_path, 1)
    foto = tmp_path / "foto.jpg"
    foto.write_bytes(b"jpgfake")
    audio = tmp_path / "sessao.mp3"
    audio.write_bytes(b"ID3fake")
    vistos: dict[str, list] = {"img": [], "txt": []}

    def espia_imagens(paths, **kw):
        vistos["img"].append((list(paths), kw.get("model_id"), kw.get("dim")))
        return _fake_embed(paths)

    def espia_textos(textos, **kw):
        vistos["txt"].append((list(textos), kw.get("model_id"), kw.get("dim")))
        return _fake_embed_texts(textos)

    monkeypatch.setattr(embed_job, "embed_images", espia_imagens)
    monkeypatch.setattr(embed_job, "embed_texts", espia_textos)

    t = asr.FakeTranscriber()
    rows = embed_job.embed_records(
        [{"modality": "doc_image", "image_path": str(pages / "BR001_p00.png")},
         {"modality": "text", "text": "meta de saude", "uri": "https://x/1"},
         {"modality": "image", "image_path": str(foto), "credit": "Foto: EBC"},
         {"modality": "audio_asr", "audio_path": str(audio), "credit": "TV Camara"}],
        model_id="Qwen/X", dim=4, transcriber=t, source="govexe")

    # ordem do lote preservada, apesar de imagens e textos irem em chamadas separadas
    assert [r.modality for r in rows] == ["doc_image", "text", "image", "audio_asr"]
    assert vistos["img"][0][0] == [str(pages / "BR001_p00.png"), str(foto)]
    assert vistos["txt"][0][0] == ["meta de saude", t.transcribe(audio)]
    # mesmo checkpoint e mesma dimensao nos dois lados == mesmo espaco vetorial
    assert vistos["img"][0][1:] == vistos["txt"][0][1:] == ("Qwen/X", 4)
    assert {r.model_id for r in rows} == {"Qwen/X"}
    np.testing.assert_allclose([np.linalg.norm(r.embedding) for r in rows], 1.0, atol=1e-6)
    # a transcricao do Whisper (fake) e o texto que foi embutido, e fica persistida
    audio_row = rows[3]
    assert audio_row.content_text == t.transcribe(audio)
    np.testing.assert_allclose(audio_row.embedding, [0, 0, 0, 1])
    assert rows[0].content_text is None


# -- fatia multimodal fim a fim: um indice, quatro modalidades -------------------------------


def _records_jsonl(tmp_path):
    """JSONL de um lote misto: 2 paginas + 1 foto com credito + 1 texto + 1 audio."""
    pages = _pages(tmp_path, 2)
    foto = tmp_path / "foto_ebc.jpg"
    foto.write_bytes(b"jpgfake")
    audio = tmp_path / "sessao_camara.mp3"
    audio.write_bytes(b"ID3fake")
    registros = [
        {"modality": "doc_image", "image_path": str(pages / "BR001_p00.png")},
        {"modality": "doc_image", "image_path": str(pages / "BR001_p01.png")},
        {"item_id": "foto:ebc-1", "modality": "image", "image_path": str(foto),
         "pointer_uri": "https://agenciabrasil/foto_ebc.jpg",
         "credit": "Foto: Fulano/Agencia Brasil", "source": "ebc"},
        {"item_id": "texto:camara-1", "modality": "text",
         "text": "meta de ampliar a atencao basica em saude",
         "uri": "https://camara/materia/1", "credit": "Agencia Camara", "source": "camara"},
        {"item_id": "audio:camara-1", "modality": "audio_asr", "audio_path": str(audio),
         "pointer_uri": "https://youtube/watch?v=x", "credit": "TV Camara",
         "source": "tv_camara"},
    ]
    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in registros),
                     encoding="utf-8")
    return jsonl


def test_embed_job_main_lote_misto_indexa_tudo_e_nao_duplica(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(embed_job, "embed_images", _fake_embed)
    monkeypatch.setattr(embed_job, "embed_texts", _fake_embed_texts)
    jsonl = _records_jsonl(tmp_path)
    store = embed_job.InMemoryVectorStore()
    argv = ["--records", str(jsonl), "--asr-backend", "fake", "--device", "cpu", "--dim", "4"]

    assert embed_job.main(argv, store=store) == 0
    assert store.count() == 5
    assert {r.modality for r in store.rows.values()} == {
        "doc_image", "image", "text", "audio_asr"}

    foto = store.rows["foto:ebc-1"]
    assert foto.credit == "Foto: Fulano/Agencia Brasil"
    assert foto.pointer_uri == "https://agenciabrasil/foto_ebc.jpg"
    assert foto.source == "ebc" and foto.content_text is None

    audio = store.rows["audio:camara-1"]
    assert asr.FAKE_PREFIX in audio.content_text          # passou pelo Transcriber
    assert audio.credit == "TV Camara"

    # mesmo model_id e mesma dimensao em TODAS as modalidades (um unico espaco)
    assert len({r.model_id for r in store.rows.values()}) == 1
    assert {len(r.embedding) for r in store.rows.values()} == {4}
    assert "5 itens gravados" in capsys.readouterr().out

    assert embed_job.main(argv, store=store) == 0         # reingestao do mesmo lote
    assert store.count() == 5
    assert sorted(store.rows) == ["BR001_p00", "BR001_p01", "audio:camara-1",
                                  "foto:ebc-1", "texto:camara-1"]


def test_query_cross_modal_recupera_as_quatro_modalidades(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(embed_job, "embed_images", _fake_embed)
    monkeypatch.setattr(embed_job, "embed_texts", _fake_embed_texts)
    store = embed_job.InMemoryVectorStore()
    embed_job.main(["--records", str(_records_jsonl(tmp_path)), "--asr-backend", "fake",
                    "--device", "cpu", "--dim", "4"], store=store)

    # consulta equidistante das 4 direcoes canonicas: o top-5 traz o indice inteiro
    monkeypatch.setattr(query, "embed_texts",
                        lambda textos, **kw: np.full((1, 4), 0.5, dtype="float32"))
    hits = query.search("saude publica", store, k=5)
    assert {h.modality for h in hits} == {"doc_image", "image", "text", "audio_asr"}

    # consulta alinhada com a transcricao: o audio vence a pagina-imagem (cross-modal real)
    monkeypatch.setattr(query, "embed_texts",
                        lambda textos, **kw: np.array([[0, 0, 0, 1]], dtype="float32"))
    capsys.readouterr()                                   # descarta a saida da ingestao
    assert query.main(["--text", "sessao da camara", "--k", "3", "--device", "cpu"],
                      store=store) == 0
    saida = capsys.readouterr().out
    primeira = saida.splitlines()[1]
    assert "audio:camara-1" in primeira and "audio_asr" in primeira
    assert "TV Camara" in saida and "https://youtube/watch?v=x" in saida


def test_query_main_filtra_por_modalidade(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(embed_job, "embed_images", _fake_embed)
    monkeypatch.setattr(embed_job, "embed_texts", _fake_embed_texts)
    store = embed_job.InMemoryVectorStore()
    embed_job.main(["--records", str(_records_jsonl(tmp_path)), "--asr-backend", "fake",
                    "--device", "cpu", "--dim", "4"], store=store)
    monkeypatch.setattr(query, "embed_texts",
                        lambda textos, **kw: np.full((1, 4), 0.5, dtype="float32"))

    assert query.main(["--text", "saude", "--k", "5", "--modality", "image",
                       "--device", "cpu"], store=store) == 0

    saida = capsys.readouterr().out
    assert "foto:ebc-1" in saida
    assert "BR001_p00" not in saida and "audio:camara-1" not in saida
    assert "Foto: Fulano/Agencia Brasil" in saida


def test_embed_job_main_records_resolve_blob_do_bronze(tmp_path, monkeypatch):
    """Record com `blob` (sem caminho local): o job baixa do Bronze antes de embutir."""
    monkeypatch.setattr(embed_job, "embed_images", _fake_embed)
    monkeypatch.setattr(embed_job, "embed_texts", _fake_embed_texts)
    pages = _pages(tmp_path, 1)
    bucket = tmp_path / "bucket"
    bronze = collector.LocalObjectStore(bucket)
    [rec] = collector.collect(collector.collect_paths(str(pages)), bronze, day="2026-10-22")
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps({"modality": "doc_image", "blob": rec["blob"]}),
                     encoding="utf-8")

    store = embed_job.InMemoryVectorStore()
    assert embed_job.main(["--records", str(jsonl), "--records-store", str(bucket),
                           "--work-dir", str(tmp_path / "work"), "--device", "cpu",
                           "--dim", "4"], store=store) == 0
    assert "BR001_p00" in store.rows
    assert store.rows["BR001_p00"].uri.endswith("dt=2026-10-22/BR001_p00.png")


# -- collector: audio no Bronze e JSONL de records -------------------------------------------


def test_collect_marca_a_modalidade_e_aceita_audio(tmp_path):
    audio_dir = tmp_path / "audios"
    audio_dir.mkdir()
    (audio_dir / "sessao.mp3").write_bytes(b"ID3fake")
    store = collector.LocalObjectStore(tmp_path / "bucket")

    registros = collector.collect([str(audio_dir / "sessao.mp3")], store,
                                  source="tv_camara", day="2026-10-22")

    assert registros[0]["modality"] == embed_job.MODALITY_AUDIO
    assert registros[0]["item_id"] == "sessao"
    assert store.list("bronze/") == ["bronze/fonte=tv_camara/dt=2026-10-22/sessao.mp3"]


def test_collector_main_records_sobe_binarios_e_reescreve_o_jsonl(tmp_path):
    foto = tmp_path / "foto.jpg"
    foto.write_bytes(b"jpgfake")
    audio = tmp_path / "sessao.mp3"
    audio.write_bytes(b"ID3fake")
    entrada = tmp_path / "in.jsonl"
    entrada.write_text("\n".join(json.dumps(r) for r in [
        {"modality": "image", "image_path": str(foto), "credit": "Foto: EBC"},
        {"modality": "audio_asr", "audio_path": str(audio), "credit": "TV Camara"},
        {"modality": "text", "text": "meta de saude", "uri": "https://x/1"},
    ]), encoding="utf-8")
    saida = tmp_path / "out.jsonl"
    root = tmp_path / "bucket"

    rc = collector.main(["--local-root", str(root), "--records", str(entrada),
                         "--out-records", str(saida), "--source", "ebc",
                         "--date", "2026-10-22"])

    assert rc == 0
    linhas = [json.loads(ln) for ln in saida.read_text(encoding="utf-8").splitlines()]
    assert linhas[0]["blob"] == "bronze/fonte=ebc/dt=2026-10-22/foto.jpg"
    assert linhas[0]["credit"] == "Foto: EBC"                 # credito preservado
    assert linhas[0]["pointer_uri"].endswith("foto.jpg")      # aponta o binario no Bronze
    assert linhas[1]["blob"].endswith("sessao.mp3")
    assert "blob" not in linhas[2] and linhas[2]["text"] == "meta de saude"
    assert (root / "bronze/fonte=ebc/dt=2026-10-22/sessao.mp3").is_file()


# -- audio_collector: captura de audio -> records ----------------------------------------


URL_CAMARA = "https://www.youtube.com/watch?v=sessao-camara-1"
URL_RADIO = "https://www.camara.leg.br/radio/programas/999999.mp3"


def test_fake_downloader_grava_audio_deterministico_e_respeita_max_seconds(tmp_path):
    d = audio_collector.FakeDownloader()

    a = d.download(URL_CAMARA, tmp_path / "audio")
    b = d.download(URL_CAMARA, tmp_path / "audio")
    curto = d.download(URL_RADIO, tmp_path / "curto", max_seconds=2)
    longo = d.download(URL_RADIO, tmp_path / "longo", max_seconds=8)

    assert a == b and a.is_file() and a.read_bytes()[:4] == b"RIFF"
    assert a.suffix == audio_collector.DEFAULT_AUDIO_SUFFIX
    assert a.parent == (tmp_path / "audio") and a != curto
    assert curto.stat().st_size < longo.stat().st_size      # max_seconds vale no contrato
    assert d.calls[0] == (URL_CAMARA, str(tmp_path / "audio"), None)
    assert d.calls[2][2] == 2


def test_collect_audio_emite_record_no_formato_do_fluxo_records(tmp_path):
    d = audio_collector.FakeDownloader()

    [rec] = audio_collector.collect_audio([URL_CAMARA], source="tv_camara",
                                          credit="TV Camara", dest_dir=tmp_path / "audio",
                                          downloader=d)

    assert rec["modality"] == embed_job.MODALITY_AUDIO
    assert rec["pointer_uri"] == URL_CAMARA            # a URL de origem e o pointer
    assert rec["credit"] == "TV Camara" and rec["source"] == "tv_camara"
    assert Path(rec["audio_path"]).is_file()
    # id estavel, derivado da URL de origem -- o mesmo esquema do embed_job
    assert rec["item_id"] == collector.default_item_id(embed_job.MODALITY_AUDIO, URL_CAMARA)
    # so campos que o fluxo --records entende
    assert set(rec) <= {"item_id", "modality", "audio_path", "pointer_uri", "credit",
                        "source", "uri"}


def test_collect_audio_repassa_max_seconds_e_deduplica_urls(tmp_path):
    d = audio_collector.FakeDownloader()

    recs = audio_collector.collect_audio([URL_CAMARA, URL_RADIO, URL_CAMARA],
                                         source="tv_camara", credit="TV Camara",
                                         dest_dir=tmp_path / "audio", downloader=d,
                                         max_seconds=30)

    assert [c[2] for c in d.calls] == [30, 30]          # a URL repetida nao baixa de novo
    assert [r["pointer_uri"] for r in recs] == [URL_CAMARA, URL_RADIO]


def test_collect_audio_pula_url_que_falha(tmp_path, caplog):
    class _Quebrado:
        def download(self, url, dest_dir, max_seconds=None):
            if "radio" in url:
                raise RuntimeError("404")
            return audio_collector.FakeDownloader().download(url, dest_dir, max_seconds)

    recs = audio_collector.collect_audio([URL_RADIO, URL_CAMARA], source="tv_camara",
                                         credit="TV Camara", dest_dir=tmp_path / "a",
                                         downloader=_Quebrado())

    assert [r["pointer_uri"] for r in recs] == [URL_CAMARA]


def test_ytdlp_downloader_e_lazy_e_pede_bestaudio(tmp_path):
    import sys

    d = audio_collector.YtDlpDownloader()
    opts = d._options(tmp_path / "audio")

    assert "yt_dlp" not in sys.modules                  # import so dentro do download()
    assert opts["format"] == "bestaudio/best"
    assert str(tmp_path / "audio") in opts["outtmpl"]
    assert opts["postprocessors"][0]["preferredcodec"] == audio_collector.DEFAULT_AUDIO_FORMAT
    assert opts["noplaylist"] is True


def test_open_downloader_escolhe_o_backend():
    assert isinstance(audio_collector.open_downloader("fake"), audio_collector.FakeDownloader)
    assert isinstance(audio_collector.open_downloader("yt-dlp"),
                      audio_collector.YtDlpDownloader)
    with pytest.raises(ValueError, match="downloader"):
        audio_collector.open_downloader("curl")


def test_audio_collector_main_escreve_o_jsonl_offline(tmp_path, capsys):
    saida = tmp_path / "audios.jsonl"

    rc = audio_collector.main(["--url", URL_CAMARA, "--url", URL_RADIO,
                               "--source", "tv_camara", "--credit", "TV Camara",
                               "--audio-dir", str(tmp_path / "audio"),
                               "--out-records", str(saida), "--max-seconds", "60",
                               "--downloader", "fake"])

    assert rc == 0
    linhas = [json.loads(ln) for ln in saida.read_text(encoding="utf-8").splitlines()]
    assert [ln["modality"] for ln in linhas] == ["audio_asr", "audio_asr"]
    assert all(Path(ln["audio_path"]).is_file() for ln in linhas)
    assert "2 records" in capsys.readouterr().out


def test_audio_collector_main_sem_nenhum_download_sai_1(tmp_path):
    class _Quebrado:
        def download(self, url, dest_dir, max_seconds=None):
            raise RuntimeError("404")

    rc = audio_collector.main(["--url", URL_RADIO, "--source", "x", "--credit", "y",
                               "--audio-dir", str(tmp_path / "a"),
                               "--out-records", str(tmp_path / "o.jsonl")],
                              downloader=_Quebrado())
    assert rc == 1


def test_record_capturado_e_ingerido_pelo_embed_job_sem_alteracao(tmp_path, monkeypatch):
    """Fim a fim da plumbing: captura -> JSONL -> `embed_job --records` -> indice."""
    monkeypatch.setattr(embed_job, "embed_images", _fake_embed)
    monkeypatch.setattr(embed_job, "embed_texts", _fake_embed_texts)
    jsonl = tmp_path / "audios.jsonl"
    assert audio_collector.main(["--url", URL_CAMARA, "--source", "tv_camara",
                                 "--credit", "TV Camara", "--downloader", "fake",
                                 "--audio-dir", str(tmp_path / "audio"),
                                 "--out-records", str(jsonl)]) == 0

    store = embed_job.InMemoryVectorStore()
    assert embed_job.main(["--records", str(jsonl), "--asr-backend", "fake",
                           "--device", "cpu", "--dim", "4"], store=store) == 0

    [linha] = store.rows.values()
    assert linha.modality == embed_job.MODALITY_AUDIO
    assert linha.item_id == collector.default_item_id(embed_job.MODALITY_AUDIO, URL_CAMARA)
    assert linha.credit == "TV Camara" and linha.pointer_uri == URL_CAMARA
    assert asr.FAKE_PREFIX in linha.content_text         # transcrito pelo FakeTranscriber
    np.testing.assert_allclose(linha.embedding, [0, 0, 0, 1])   # embutido por embed_texts


# -- embed_job --space: roteamento por espaco vetorial (A e A+audio) -------------------------


def _fake_embed_audio(paths, **kw):
    """CLAP fake: um vetor canonico por arquivo, na dimensao declarada pelo espaco."""
    return np.eye(len(paths), spaces.SPACES["clap"].dim, dtype="float32")


def _nao_deveria(*a, **kw):
    raise AssertionError("encoder do espaco errado foi chamado")


def _audio_native_jsonl(tmp_path, item_id="audio:camara-1"):
    """JSONL com um record de audio NATIVO (forma de onda, sem ASR) + o arquivo."""
    audio = tmp_path / "sessao_camara.m4a"
    audio.write_bytes(b"m4afake")
    jsonl = tmp_path / "audio_native.jsonl"
    jsonl.write_text(json.dumps(
        {"item_id": item_id, "modality": "audio_native", "audio_path": str(audio),
         "pointer_uri": URL_CAMARA, "credit": "TV Camara", "source": "tv_camara"}),
        encoding="utf-8")
    return jsonl, audio


def _fake_pg(monkeypatch, resultado=None):
    """Substitui o `connect` do pgvector por um `PgVectorStore` sobre `_FakeConn`.

    Devolve `(conn, capturado)`: `capturado` guarda os kwargs da conexao -- e assim que o
    teste ve QUAL tabela o job escolheu, sem Cloud SQL nenhum.
    """
    conn = _FakeConn(resultado)
    capturado: dict = {}

    def fake_connect(**kw):
        capturado.update(kw)
        return embed_job.PgVectorStore(conn, table=kw["table"])

    monkeypatch.setattr(embed_job.PgVectorStore, "connect", staticmethod(fake_connect))
    return conn, capturado


def test_prepare_record_audio_native_hasheia_o_binario_e_dispensa_asr(tmp_path):
    """`audio_native` NAO passa pelo Whisper: o vetor vem da onda, o hash do arquivo."""
    audio = tmp_path / "sessao.m4a"
    audio.write_bytes(b"m4afake")

    p = embed_job.prepare_record(
        {"modality": "audio_native", "audio_path": str(audio),
         "pointer_uri": URL_CAMARA, "credit": "TV Camara"}, source="tv_camara")

    assert p["modality"] == embed_job.MODALITY_AUDIO_NATIVE
    assert p["content_text"] is None                     # nao e texto: nao ha transcricao
    assert p["content_sha256"] == embed_job.content_sha256(audio)
    assert p["path"] == str(audio) and p["credit"] == "TV Camara"


def test_space_clap_roteia_audio_native_para_embed_audio(tmp_path, monkeypatch):
    """Espaco `clap`: o audio nativo vai por `embed_audio` -- nunca por texto/imagem."""
    audio = tmp_path / "sessao.m4a"
    audio.write_bytes(b"m4afake")
    vistos: dict = {}

    def espia_audio(paths, **kw):
        vistos["paths"], vistos["kw"] = list(paths), kw
        return _fake_embed_audio(paths)

    monkeypatch.setattr(embed_job, "embed_audio", espia_audio)
    monkeypatch.setattr(embed_job, "embed_images", _nao_deveria)
    monkeypatch.setattr(embed_job, "embed_texts", _nao_deveria)

    rows = embed_job.embed_records(
        [{"item_id": "audio:camara-1", "modality": "audio_native",
          "audio_path": str(audio), "pointer_uri": URL_CAMARA, "credit": "TV Camara"}],
        space="clap", device="cpu", source="tv_camara")

    assert vistos["paths"] == [str(audio)]               # o CAMINHO do binario, nao texto
    assert vistos["kw"]["model_id"] == spaces.SPACES["clap"].model_id
    assert [r.modality for r in rows] == [embed_job.MODALITY_AUDIO_NATIVE]
    assert len(rows[0].embedding) == spaces.SPACES["clap"].dim
    assert rows[0].model_id == spaces.SPACES["clap"].model_id
    assert rows[0].content_text is None
    assert rows[0].pointer_uri == URL_CAMARA and rows[0].credit == "TV Camara"


def test_embed_job_main_space_clap_grava_na_tabela_do_registro(tmp_path, monkeypatch):
    """`--space clap` -> tabela `govexe_clap`, VECTOR(512), item de modalidade nativa."""
    jsonl, audio = _audio_native_jsonl(tmp_path)
    monkeypatch.setattr(embed_job, "embed_audio", _fake_embed_audio)
    monkeypatch.setattr(embed_job, "embed_images", _nao_deveria)
    monkeypatch.setattr(embed_job, "embed_texts", _nao_deveria)
    conn, capturado = _fake_pg(monkeypatch)

    assert embed_job.main(["--records", str(jsonl), "--space", "clap",
                           "--device", "cpu"]) == 0

    assert capturado["table"] == spaces.SPACES["clap"].table == "govexe_clap"
    ddl = conn.sql[1][0]
    assert "govexe_clap" in ddl and "VECTOR(512)" in ddl
    sql, params = conn.sql[4]
    assert "INSERT INTO govexe_clap" in sql
    assert params[0][:2] == ("audio:camara-1", embed_job.MODALITY_AUDIO_NATIVE)


def test_embed_job_main_space_default_mantem_o_fluxo_qwen3vl(tmp_path, monkeypatch):
    """Sem `--space`: mesmas 4 modalidades pelo mesmo checkpoint, na tabela do espaco A."""
    # no caminho pgvector a largura e SEMPRE a do registro: os fakes entregam 1024
    largura = spaces.SPACES["qwen3vl"].dim
    monkeypatch.setattr(embed_job, "embed_images",
                        lambda paths, **kw: np.eye(len(paths), largura, dtype="float32"))
    monkeypatch.setattr(embed_job, "embed_texts",
                        lambda textos, **kw: np.eye(len(textos), largura, dtype="float32"))
    monkeypatch.setattr(embed_job, "embed_audio", _nao_deveria)
    conn, capturado = _fake_pg(monkeypatch)
    jsonl = _records_jsonl(tmp_path)

    assert embed_job.main(["--records", str(jsonl), "--asr-backend", "fake",
                           "--device", "cpu"]) == 0

    assert capturado["table"] == spaces.SPACES["qwen3vl"].table == "govexe_qwen3vl"
    assert f"VECTOR({largura})" in conn.sql[1][0]        # a tabela nasce com a do registro
    assert "INSERT INTO govexe_qwen3vl" in conn.sql[4][0]
    assert {p[1] for p in conn.sql[4][1]} == {"doc_image", "image", "text", "audio_asr"}
    assert {p[8] for p in conn.sql[4][1]} == {spaces.SPACES["qwen3vl"].model_id}


def test_dim_divergente_falha_rapido(tmp_path, monkeypatch):
    """Embedder devolvendo outra largura nao chega ao upsert: a coluna VECTOR(n) e fixa."""
    jsonl, _ = _audio_native_jsonl(tmp_path)
    monkeypatch.setattr(embed_job, "embed_audio",
                        lambda paths, **kw: np.eye(len(paths), 256, dtype="float32"))
    store = embed_job.InMemoryVectorStore()

    with pytest.raises(ValueError, match="dim"):
        embed_job.main(["--records", str(jsonl), "--space", "clap", "--device", "cpu"],
                       store=store)

    assert store.count() == 0                            # nada gravado com largura errada


def test_space_clap_recusa_modalidade_que_nao_indexa(tmp_path):
    """O espaco do CLAP so aceita `audio_native` -- texto/imagem pertencem ao espaco A."""
    with pytest.raises(ValueError, match="clap"):
        embed_job.embed_records([{"modality": "text", "text": "meta de saude",
                                  "uri": "https://x/1"}], space="clap", device="cpu")


def test_space_qwen3vl_recusa_audio_native_antes_de_preparar_o_record(tmp_path):
    """E a recusa vem ANTES do `prepare_record`: nem o binario precisa existir."""
    t = asr.FakeTranscriber()

    with pytest.raises(ValueError, match="audio_native"):
        embed_job.embed_records(
            [{"item_id": "audio:camara-1", "modality": "audio_native",
              "audio_path": str(tmp_path / "nem-existe.m4a")}],
            space="qwen3vl", device="cpu", transcriber=t)

    assert t.calls == []                                 # nada de ASR num lote recusado


def test_space_clap_recusa_os_modos_de_entrada_de_pagina(tmp_path, monkeypatch):
    """`--images-dir`/`--local-bronze` produzem `doc_image`: o espaco do CLAP nao indexa."""
    pages = _pages(tmp_path, 1)
    monkeypatch.setattr(embed_job, "embed_images", _nao_deveria)
    monkeypatch.setattr(embed_job, "embed_audio", _nao_deveria)

    assert embed_job.main(["--images-dir", str(pages), "--space", "clap",
                           "--device", "cpu"]) == 1
    assert embed_job.main(["--local-bronze", str(tmp_path / "bucket"), "--space", "clap",
                           "--work-dir", str(tmp_path / "work"), "--device", "cpu"]) == 1


def test_dim_divergente_do_registro_e_recusado_no_caminho_pgvector(tmp_path, monkeypatch):
    """A tabela do espaco e `VECTOR(SPACES[...].dim)`: `--dim` outro nao cria nada."""
    jsonl = _records_jsonl(tmp_path)
    monkeypatch.setattr(embed_job, "embed_images", _nao_deveria)
    monkeypatch.setattr(embed_job, "embed_texts", _nao_deveria)
    conn, capturado = _fake_pg(monkeypatch)

    with pytest.raises(ValueError, match="dim"):
        embed_job.main(["--records", str(jsonl), "--asr-backend", "fake",
                        "--device", "cpu", "--dim", "4"])

    assert conn.sql == [] and capturado == {}            # nem conectou, nem criou schema


def test_dim_divergente_so_e_aceito_no_indice_local(tmp_path, monkeypatch):
    """O mesmo `--dim 4` passa no indice fake, que nao tem tabela canonica a corromper."""
    monkeypatch.setattr(embed_job, "embed_images", _fake_embed)
    monkeypatch.setattr(embed_job, "embed_texts", _fake_embed_texts)
    store = embed_job.InMemoryVectorStore()

    assert embed_job.main(["--records", str(_records_jsonl(tmp_path)), "--asr-backend",
                           "fake", "--device", "cpu", "--dim", "4"], store=store) == 0
    assert {len(r.embedding) for r in store.rows.values()} == {4}


def test_dim_divergente_no_clap_e_recusado_ate_no_indice_local(tmp_path, monkeypatch):
    """CLAP nao e Matryoshka: pedir outra largura nao e override, e vetor que nao chega."""
    jsonl, _ = _audio_native_jsonl(tmp_path)
    monkeypatch.setattr(embed_job, "embed_audio", _nao_deveria)
    store = embed_job.InMemoryVectorStore()

    with pytest.raises(ValueError, match="Matryoshka"):
        embed_job.main(["--records", str(jsonl), "--space", "clap", "--device", "cpu",
                        "--dim", "4"], store=store)

    assert store.count() == 0


def test_space_desconhecido_falha_com_a_lista_do_que_existe():
    with pytest.raises(ValueError, match="espaco"):
        embed_job.get_space("nao-existe")
    assert embed_job.get_space("clap") is spaces.SPACES["clap"]
