import json
from pathlib import Path

from data.govexe.build_planos_benchmark import build, make_cached_fetch

ROW_OK = {
    "tse_sequencial": "BR001",
    "resumo": "Plano de saude e educacao",
    "destaque": "Foco em saude publica",
    "eixos": json.dumps([{"tema": "saude", "titulo": "Saude para todos",
                          "propostas": "mais UBS"}]),
    "pdf_url": "http://x/plano.pdf",
}


def _explode(*_a, **_kw):
    raise AssertionError("fetcher/renderer nao deveria ser chamado")


def _renderer_2p(pdf_bytes, out_dir, stem):
    paths = []
    for i in range(2):
        p = Path(out_dir) / f"{stem}_p{i:02d}.png"
        p.write_bytes(b"png")
        paths.append(str(p))
    return paths


def _read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_build_makes_queries_qrels(tmp_path: Path):
    rows = [{
        "tse_sequencial": "BR001",
        "resumo": "Plano de saude e educacao",
        "destaque": "Foco em saude publica",
        "eixos": json.dumps([{"tema": "saude", "titulo": "Saude para todos",
                              "propostas": "mais UBS"}]),
        "pdf_url": "http://x/plano.pdf",
    }, {
        "tse_sequencial": "BR002", "resumo": "", "destaque": "", "eixos": "[]",
        "pdf_url": "http://x/corrompido.pdf",
    }]

    def fetcher(url):            # BR002 "corrompe"
        return b"%PDF-ok" if "corrompido" not in url else b"lixo"

    def renderer(pdf_bytes, out_dir, stem):     # 2 paginas se ok; erro se lixo
        if pdf_bytes == b"lixo":
            raise ValueError("PDF ilegivel")
        paths = []
        for i in range(2):
            p = Path(out_dir) / f"{stem}_p{i:02d}.png"
            p.write_bytes(b"png")
            paths.append(str(p))
        return paths

    build(rows, tmp_path, fetcher, renderer)
    corpus = [json.loads(line)
              for line in (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = [json.loads(line)
               for line in (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    qrels = (tmp_path / "qrels.tsv").read_text(encoding="utf-8").splitlines()
    assert len(corpus) == 2                       # so BR001 (2 paginas); BR002 pulado
    assert any(q["text"] == "Foco em saude publica" for q in queries)
    assert all(line.split("\t")[1].startswith("BR001") for line in qrels)


def test_reusa_paginas_ja_renderizadas_sem_chamar_fetcher(tmp_path: Path):
    pages = tmp_path / "pages"
    pages.mkdir(parents=True)
    for i in range(2):
        (pages / f"BR001_p{i:02d}.png").write_bytes(b"png")

    build([ROW_OK], tmp_path, _explode, _explode)

    corpus = [json.loads(ln) for ln in _read_lines(tmp_path / "corpus.jsonl")]
    assert sorted(c["id"] for c in corpus) == ["BR001_p00", "BR001_p01"]
    assert _read_lines(tmp_path / "qrels.tsv")


def test_force_refaz_mesmo_com_paginas_existentes(tmp_path: Path):
    pages = tmp_path / "pages"
    pages.mkdir(parents=True)
    (pages / "BR001_p00.png").write_bytes(b"png")
    chamadas = []

    def fetcher(url):
        chamadas.append(url)
        return b"%PDF-ok"

    build([ROW_OK], tmp_path, fetcher, _renderer_2p, force=True)
    assert chamadas == ["http://x/plano.pdf"]


def test_segunda_construcao_nao_refaz_nada(tmp_path: Path):
    build([ROW_OK], tmp_path, lambda url: b"%PDF-ok", _renderer_2p)
    antes = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8")

    build([ROW_OK], tmp_path, _explode, _explode)       # skip global: nem olha as rows

    assert (tmp_path / "corpus.jsonl").read_text(encoding="utf-8") == antes


def test_reconstrucao_forcada_nao_duplica_linhas(tmp_path: Path):
    build([ROW_OK], tmp_path, lambda url: b"%PDF-ok", _renderer_2p)
    build([ROW_OK], tmp_path, lambda url: b"%PDF-ok", _renderer_2p, force=True)

    corpus = [json.loads(ln) for ln in _read_lines(tmp_path / "corpus.jsonl")]
    queries = [json.loads(ln) for ln in _read_lines(tmp_path / "queries.jsonl")]
    qrels = _read_lines(tmp_path / "qrels.tsv")
    assert [c["id"] for c in corpus] == ["BR001_p00", "BR001_p01"]
    assert sorted(q["id"] for q in queries) == ["BR001_destaque", "BR001_eixo0", "BR001_resumo"]
    assert len(qrels) == len(set(qrels)) == 6           # 3 queries x 2 paginas


def test_cached_fetch_baixa_uma_vez_e_reusa_o_pdf(tmp_path: Path):
    chamadas = []

    def fetcher(url):
        chamadas.append(url)
        return b"%PDF-conteudo"

    cached = make_cached_fetch([ROW_OK], tmp_path, fetcher=fetcher)
    assert cached("http://x/plano.pdf") == b"%PDF-conteudo"
    assert (tmp_path / "pdf_cache" / "BR001.pdf").read_bytes() == b"%PDF-conteudo"
    assert cached("http://x/plano.pdf") == b"%PDF-conteudo"
    assert chamadas == ["http://x/plano.pdf"]           # segunda veio do cache


def test_cached_fetch_com_url_desconhecida_nao_quebra(tmp_path: Path):
    cached = make_cached_fetch([ROW_OK], tmp_path, fetcher=lambda url: b"%PDF-x")
    assert cached("http://x/outro.pdf") == b"%PDF-x"
    assert list((tmp_path / "pdf_cache").glob("*.pdf"))
