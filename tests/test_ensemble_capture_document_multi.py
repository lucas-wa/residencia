"""Eixo documento com VARIOS PDFs legislativos (corpus v4).

Sem rede e sem pdfium: o download e o render/extracao sao dublados (monkeypatch), como
nos demais testes de captura. O que se testa e a distribuicao das paginas entre os
documentos, a unicidade dos nomes locais / `page_id` e a robustez a PDF curto ou vazio.
"""
import urllib.request
from pathlib import Path

import pytest

from ensemble.capture import document as doc


class _Resp:
    """Dublê de resposta HTTP (context manager com .read())."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _setup(monkeypatch, paginas: dict[str, int]):
    """Dubla download + extractor + renderer; `paginas` mapeia URL -> nº de paginas."""

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", req)
        return _Resp(f"%PDF-FAKE pages={paginas[url]}\n".encode("utf-8"))

    def fake_extractor(pdf_path):
        total = int(Path(pdf_path).read_text(encoding="utf-8").split("pages=")[1])

        def extract(pid: str) -> str:
            idx = int(pid.rsplit("#p", 1)[1])
            if idx > total:
                raise IndexError(pid)
            return f"Pagina de orcamento {pid} ({Path(pdf_path).name})"

        return extract

    def fake_renderer(pdf_path):
        def render(page: int, destino: Path) -> None:
            destino.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        return render

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(doc, "_pdfium_extractor", fake_extractor)
    monkeypatch.setattr(doc, "_pdfium_renderer", fake_renderer)


# URL degenerada: Path(url).name == "pdf" para TODA materia do Congresso.
URL_MATERIA = "https://www.congressonacional.leg.br/materias/pesquisa/-/materia/168120/pdf"
URL_MATERIA_2 = "https://www.congressonacional.leg.br/materias/pesquisa/-/materia/999999/pdf"
URL_PLOA = "https://www.congressonacional.leg.br/documents/1/2/Volume+I+-+PLOA2026.pdf/abc"


def test_fetch_budget_pages_multi_distribui_n_entre_os_docs(tmp_path, monkeypatch):
    urls = ["https://e.gov.br/a/um.pdf", "https://e.gov.br/a/dois.pdf", "https://e.gov.br/a/tres.pdf"]
    _setup(monkeypatch, {u: 40 for u in urls})

    recs = doc.fetch_budget_pages_multi(urls, dest_dir=str(tmp_path), n=9)

    assert len(recs) == 9
    por_doc = {}
    for r in recs:
        por_doc[Path(r.meta["pdf"]).name] = por_doc.get(Path(r.meta["pdf"]).name, 0) + 1
    assert len(por_doc) == 3
    assert set(por_doc.values()) == {3}


def test_fetch_budget_pages_multi_completa_quando_um_doc_e_curto(tmp_path, monkeypatch):
    urls = ["https://e.gov.br/a/curto.pdf", "https://e.gov.br/a/longo1.pdf", "https://e.gov.br/a/longo2.pdf"]
    _setup(monkeypatch, {urls[0]: 1, urls[1]: 40, urls[2]: 40})

    recs = doc.fetch_budget_pages_multi(urls, dest_dir=str(tmp_path), n=9)

    assert len(recs) == 9                                   # o deficit do curto foi coberto
    curtos = [r for r in recs if "curto" in r.meta["pdf"]]
    assert len(curtos) == 1


def test_fetch_budget_pages_multi_doc_vazio_nao_derruba_os_demais(tmp_path, monkeypatch):
    urls = ["https://e.gov.br/a/vazio.pdf", "https://e.gov.br/a/cheio.pdf"]
    _setup(monkeypatch, {urls[0]: 0, urls[1]: 40})

    recs = doc.fetch_budget_pages_multi(urls, dest_dir=str(tmp_path), n=8)

    assert len(recs) == 8
    assert all("cheio" in r.meta["pdf"] for r in recs)


def test_fetch_budget_pages_multi_para_em_n(tmp_path, monkeypatch):
    urls = [f"https://e.gov.br/a/d{i}.pdf" for i in range(5)]
    _setup(monkeypatch, {u: 100 for u in urls})

    recs = doc.fetch_budget_pages_multi(urls, dest_dir=str(tmp_path), n=7)

    assert len(recs) == 7


def test_fetch_budget_pages_multi_esgota_todos_sem_travar(tmp_path, monkeypatch):
    urls = ["https://e.gov.br/a/p1.pdf", "https://e.gov.br/a/p2.pdf"]
    _setup(monkeypatch, {urls[0]: 2, urls[1]: 3})

    recs = doc.fetch_budget_pages_multi(urls, dest_dir=str(tmp_path), n=50)

    assert len(recs) == 5                                   # n > total disponivel: entrega o que ha


def test_fetch_budget_pages_multi_nomes_locais_e_page_ids_unicos(tmp_path, monkeypatch):
    # duas materias (nome da URL degenera para "pdf") + o PLOA
    urls = [URL_MATERIA, URL_MATERIA_2, URL_PLOA]
    _setup(monkeypatch, {u: 20 for u in urls})

    recs = doc.fetch_budget_pages_multi(urls, dest_dir=str(tmp_path), n=9)

    assert len(recs) == 9
    nomes = {Path(r.meta["pdf"]).name for r in recs}
    assert len(nomes) == 3                                  # um arquivo local por documento
    assert "pdf" not in nomes                               # nao degenerou
    assert len({r.id for r in recs}) == 9                   # documento-<arquivo>#pN unico
    assert len({r.pointer_uri for r in recs}) == 9
    assert len({r.meta["a_repr"] for r in recs}) == 9
    for p in nomes:
        assert (tmp_path / p).is_file()


def test_fetch_budget_pages_multi_mantem_contrato_do_record(tmp_path, monkeypatch):
    urls = [URL_MATERIA, "https://e.gov.br/a/um.pdf"]
    _setup(monkeypatch, {u: 10 for u in urls})

    recs = doc.fetch_budget_pages_multi(urls, dest_dir=str(tmp_path), n=4)

    for r in recs:
        assert r.modality == "documento" and r.source == "pdf"
        assert r.descriptor.strip()
        assert r.meta["a_repr"] == r.meta["path"]
        assert Path(r.meta["path"]).is_file()


def test_fetch_budget_pages_multi_recusa_lista_vazia(tmp_path):
    with pytest.raises(ValueError):
        doc.fetch_budget_pages_multi([], dest_dir=str(tmp_path), n=5)


def test_fetch_budget_pages_mantem_o_nome_da_url_por_default(tmp_path, monkeypatch):
    """Retrocompat v3: um unico PDF segue gravado com o nome vindo da URL."""
    url = "https://e.gov.br/a/Volume+I+-+PLOA2026.pdf"
    _setup(monkeypatch, {url: 5})

    recs = doc.fetch_budget_pages(url, dest_dir=str(tmp_path), n=2)

    assert len(recs) == 2
    assert Path(recs[0].meta["pdf"]).name == "Volume+I+-+PLOA2026.pdf"
