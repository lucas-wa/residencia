"""Eixo imagem do ensemble v2: `fetch_gallery` crawleia LISTAGENS da Agencia Camara.

Nao existe galeria semantica no governo; o <figure><figcaption> esta nas paginas de
ARTIGO. Estes testes cobrem (a) a descoberta pura dos links de artigo na listagem e
(b) o crawl de ponta a ponta com os getters HTTP dublados -- nada de rede.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ensemble.capture import image as img_mod
from ensemble.capture.image import article_links, fetch_gallery
from ensemble.records import validate_record

LISTAGEM_URL = "https://www.camara.leg.br/noticias/ultimas"


def _artigo_html(slug: str, *, topical: bool) -> str:
    """Pagina de artigo da Agencia Camara: 1 <figure class="materia__midia"> com legenda."""
    tema = "Comissao debate o Orçamento de 2026" if topical else "Comissao debate transporte"
    return (
        f"<html><head><title>{tema}</title></head><body>"
        f"<article><h1>{tema}</h1><p>{tema}.</p>"
        f'<figure class="materia__midia">'
        f'<img src="https://www.camara.leg.br/img/{slug}.jpg">'
        f"<figcaption>Legenda da foto de {slug}</figcaption>"
        f"</figure></article></body></html>"
    )


def _fake_http(paginas: dict[str, str]):
    def _get(url: str, timeout: int = 30) -> str:
        if url not in paginas:
            raise RuntimeError(f"404 {url}")
        return paginas[url]

    return _get


def _dubla_http(monkeypatch: pytest.MonkeyPatch, paginas: dict[str, str]) -> None:
    monkeypatch.setattr(img_mod, "_http_get", _fake_http(paginas))
    monkeypatch.setattr(img_mod, "_http_get_bytes", lambda url, timeout=60: b"\xff\xd8dummy")


def test_article_links_so_artigos_absolutos_dedup():
    """Da listagem saem so os /noticias/<digitos>-<slug>, absolutos, dedup, em ordem."""
    html = """
    <html><body>
      <a href="/noticias/1317452-comissao-analisa-orcamento">rel</a>
      <a href="/institucional/agencia">irrelevante</a>
      <a href="https://www.camara.leg.br/noticias/1317453-relator-apresenta-parecer">abs</a>
      <a href="/noticias/rss/ultimas-noticias">feed (sem digitos)</a>
      <a href="https://www.camara.leg.br/noticias/1317452-comissao-analisa-orcamento">dup abs</a>
      <a href="/noticias/1317452-comissao-analisa-orcamento">dup rel</a>
      <a>sem href</a>
    </body></html>
    """
    assert article_links(html, LISTAGEM_URL) == [
        "https://www.camara.leg.br/noticias/1317452-comissao-analisa-orcamento",
        "https://www.camara.leg.br/noticias/1317453-relator-apresenta-parecer",
    ]


def test_fetch_gallery_crawleia_listagem(monkeypatch, tmp_path):
    """fetch_gallery: listagem -> artigos -> parse_gallery, com o binario gravado em
    dest_dir e `pointer_uri` dedup (o mesmo artigo linkado 2x nao duplica o record)."""
    a1 = "https://www.camara.leg.br/noticias/111-um-orcamento"
    a2 = "https://www.camara.leg.br/noticias/222-dois-orcamento"
    listagem = (
        f'<a href="/noticias/111-um-orcamento">1</a>'
        f'<a href="/noticias/222-dois-orcamento">2</a>'
        f'<a href="/noticias/111-um-orcamento">1 de novo</a>'
        f'<a href="/noticias/333-sumido">404</a>'   # fetch falha -> pulado, nao aborta
    )
    _dubla_http(monkeypatch, {
        LISTAGEM_URL: listagem,
        a1: _artigo_html("um", topical=True),
        a2: _artigo_html("dois", topical=True),
    })

    recs = fetch_gallery([LISTAGEM_URL], n=5, dest_dir=tmp_path)

    assert len(recs) == 2
    assert [r.pointer_uri for r in recs] == [
        "https://www.camara.leg.br/img/um.jpg",
        "https://www.camara.leg.br/img/dois.jpg",
    ]
    for r in recs:
        validate_record(r)
        assert r.modality == "imagem"
        assert r.descriptor.startswith("Legenda da foto de ")
        p = Path(r.meta["path"])
        assert p.is_file() and p.parent == tmp_path
        assert r.meta["a_repr"] == str(p)
        assert p.read_bytes() == b"\xff\xd8dummy"


def test_fetch_gallery_prioriza_topical_e_corta_em_n(monkeypatch, tmp_path):
    """Com topicais >= n, so eles saem (o distrator fica de fora)."""
    topicais = {
        f"https://www.camara.leg.br/noticias/{i}0-orc": _artigo_html(f"t{i}", topical=True)
        for i in (1, 2)
    }
    distrator = "https://www.camara.leg.br/noticias/99-transporte"
    listagem = "".join(
        f'<a href="{u}">x</a>' for u in [distrator, *topicais]
    )
    _dubla_http(monkeypatch, {
        LISTAGEM_URL: listagem,
        distrator: _artigo_html("d", topical=False),
        **topicais,
    })

    recs = fetch_gallery([LISTAGEM_URL], n=2, dest_dir=tmp_path)

    assert [r.pointer_uri for r in recs] == [
        "https://www.camara.leg.br/img/t1.jpg",
        "https://www.camara.leg.br/img/t2.jpg",
    ]


def test_fetch_gallery_completa_com_distratores(monkeypatch, tmp_path):
    """Topical < n: o filtro so ORDENA -- os distratores completam os slots (nunca 0)."""
    topical = "https://www.camara.leg.br/noticias/10-orc"
    d1 = "https://www.camara.leg.br/noticias/91-transporte"
    d2 = "https://www.camara.leg.br/noticias/92-saude"
    listagem = "".join(f'<a href="{u}">x</a>' for u in [d1, topical, d2])
    _dubla_http(monkeypatch, {
        LISTAGEM_URL: listagem,
        topical: _artigo_html("t", topical=True),
        d1: _artigo_html("d1", topical=False),
        d2: _artigo_html("d2", topical=False),
    })

    recs = fetch_gallery([LISTAGEM_URL], n=3, dest_dir=tmp_path)

    # o topical vem PRIMEIRO, mesmo tendo sido crawleado depois do d1
    assert [r.pointer_uri for r in recs] == [
        "https://www.camara.leg.br/img/t.jpg",
        "https://www.camara.leg.br/img/d1.jpg",
        "https://www.camara.leg.br/img/d2.jpg",
    ]


def test_fetch_gallery_sem_topical_ainda_rende(monkeypatch, tmp_path):
    """Nenhum artigo cita orcamento: o yield continua sendo > 0."""
    d1 = "https://www.camara.leg.br/noticias/91-transporte"
    _dubla_http(monkeypatch, {
        LISTAGEM_URL: f'<a href="{d1}">x</a>',
        d1: _artigo_html("d1", topical=False),
    })

    recs = fetch_gallery([LISTAGEM_URL], n=4, dest_dir=tmp_path)
    assert len(recs) == 1
    assert recs[0].pointer_uri == "https://www.camara.leg.br/img/d1.jpg"


def test_fetch_gallery_listagem_inacessivel_nao_aborta(monkeypatch, tmp_path):
    """Listagem que falha e pulada; a seguinte ainda rende."""
    boa = "https://www.camara.leg.br/noticias/ultimas-2"
    art = "https://www.camara.leg.br/noticias/10-orc"
    _dubla_http(monkeypatch, {
        boa: f'<a href="{art}">x</a>',
        art: _artigo_html("t", topical=True),
    })

    recs = fetch_gallery([LISTAGEM_URL, boa], n=4, dest_dir=tmp_path)
    assert [r.pointer_uri for r in recs] == ["https://www.camara.leg.br/img/t.jpg"]
