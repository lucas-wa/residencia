from pathlib import Path

import ensemble.capture.text as text_mod
from ensemble.capture.document import parse_pages
from ensemble.capture.text import fetch_budget_articles, parse_article
from ensemble.records import validate_record

FIX = Path(__file__).parent / "fixtures" / "ensemble"

FEED_URL = "https://example.gov.br/rss"


def _rss(links: list[tuple[str, str]]) -> str:
    itens = "".join(
        f"<item><title>{titulo}</title><link>{link}</link></item>" for titulo, link in links
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<rss version="2.0"><channel>{itens}</channel></rss>'
    )


def _artigo(titulo: str, corpo: str) -> str:
    return (
        "<!doctype html><html lang=\"pt-BR\"><head><meta charset=\"utf-8\">"
        f"<title>{titulo}</title></head><body><article>"
        f"<h1>{titulo}</h1><p>{corpo}</p>"
        "</article></body></html>"
    )


def _serve(monkeypatch, feed_xml: str, artigos: dict[str, str]) -> None:
    """Faz `_http_get` servir o feed e os artigos dubles -- sem rede."""

    def fake_get(url: str, timeout: int = 30) -> str:
        if url == FEED_URL:
            return feed_xml
        return artigos[url]

    monkeypatch.setattr(text_mod, "_http_get", fake_get)


def test_parse_article_extrai_record():
    html = (FIX / "artigo_orcamento.html").read_text(encoding="utf-8")
    url = "https://example.gov.br/noticia/orcamento-2026"
    r = parse_article(html, url)
    validate_record(r)
    assert r.modality == "texto"
    assert r.pointer_uri == url
    assert "orçamento" in r.descriptor.lower()
    assert "nao deve aparecer" not in r.descriptor


def test_parse_pages_gera_n_records():
    def extractor(page_ref: str) -> str:
        return f"Pagina de orçamento: {page_ref}"

    recs = parse_pages(str(FIX / "loa_sample.pdf"), extractor=extractor, n=3)
    assert len(recs) == 3
    for r in recs:
        validate_record(r)
        assert r.modality == "documento"
    assert len({r.pointer_uri for r in recs}) == 3
    assert len({r.id for r in recs}) == 3


def test_fetch_budget_articles_nunca_zera_sem_topicais(monkeypatch):
    """Nenhuma materia cita orcamento: devolve os distratores, NAO lista vazia."""
    links = [
        ("Plenario aprova marco das eolicas", "https://example.gov.br/n/1"),
        ("Comissao debate licenciamento ambiental", "https://example.gov.br/n/2"),
        ("Senado analisa reforma administrativa", "https://example.gov.br/n/3"),
    ]
    artigos = {
        url: _artigo(titulo, "Texto legislativo geral sem o tema fiscal.")
        for titulo, url in links
    }
    _serve(monkeypatch, _rss(links), artigos)

    recs = fetch_budget_articles([FEED_URL], n=3)

    assert recs != []
    assert len(recs) == 3
    assert [r.pointer_uri for r in recs] == [url for _, url in links]
    for r in recs:
        validate_record(r)
        assert r.modality == "texto"


def test_fetch_budget_articles_ordena_topical_primeiro(monkeypatch):
    """Topicalidade ORDENA: o record topical vem primeiro, distratores completam."""
    links = [
        ("Comissao debate licenciamento ambiental", "https://example.gov.br/n/1"),
        ("Relator entrega parecer do Orçamento 2027", "https://example.gov.br/n/2"),
        ("Senado analisa reforma administrativa", "https://example.gov.br/n/3"),
    ]
    artigos = {
        "https://example.gov.br/n/1": _artigo(links[0][0], "Audiencia sobre licenciamento."),
        "https://example.gov.br/n/2": _artigo(
            links[1][0], "O relatório do orçamento da União foi entregue nesta terça."
        ),
        "https://example.gov.br/n/3": _artigo(
            links[2][0], "Debate sobre carreiras do servico publico."
        ),
    }
    _serve(monkeypatch, _rss(links), artigos)

    recs = fetch_budget_articles([FEED_URL], n=3)

    assert len(recs) == 3
    assert recs[0].pointer_uri == "https://example.gov.br/n/2"
    assert "orçamento" in recs[0].descriptor.lower()
    assert {r.pointer_uri for r in recs[1:]} == {
        "https://example.gov.br/n/1",
        "https://example.gov.br/n/3",
    }


def test_fetch_budget_articles_respeita_n(monkeypatch):
    """`n` menor que o disponivel corta em `n`, com o topical ainda na frente."""
    links = [
        ("Comissao debate licenciamento ambiental", "https://example.gov.br/n/1"),
        ("Relator entrega parecer do Orçamento 2027", "https://example.gov.br/n/2"),
        ("Senado analisa reforma administrativa", "https://example.gov.br/n/3"),
    ]
    artigos = {
        "https://example.gov.br/n/1": _artigo(links[0][0], "Audiencia sobre licenciamento."),
        "https://example.gov.br/n/2": _artigo(
            links[1][0], "O relatório do orçamento da União foi entregue nesta terça."
        ),
        "https://example.gov.br/n/3": _artigo(
            links[2][0], "Debate sobre carreiras do servico publico."
        ),
    }
    _serve(monkeypatch, _rss(links), artigos)

    recs = fetch_budget_articles([FEED_URL], n=2)

    assert len(recs) == 2
    assert recs[0].pointer_uri == "https://example.gov.br/n/2"
