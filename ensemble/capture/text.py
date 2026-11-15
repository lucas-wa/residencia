"""Captura de texto: artigos de orcamento (Agencia Brasil / Camara / Senado)."""
from __future__ import annotations

import hashlib
import urllib.request
import xml.etree.ElementTree as ET

from ensemble.descriptors import text_descriptor
from ensemble.records import PointerRecord, validate_record

_KEYWORD = "orçamento"


def _corpo_html(raw_html: str) -> tuple[str, str]:
    """Devolve (html_do_corpo, titulo): <article> se houver, senao <body>; sem script/style."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "html.parser")
    titulo = soup.title.get_text(" ").strip() if soup.title else ""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    corpo = soup.find("article") or soup.find("main") or soup.body or soup
    return str(corpo), titulo


def parse_article(raw_html: str, url: str) -> PointerRecord:
    corpo, titulo = _corpo_html(raw_html)
    descriptor = text_descriptor(corpo, html=True)
    rec = PointerRecord(
        id="texto-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12],
        modality="texto",
        descriptor=descriptor,
        pointer_uri=url,
        source="noticia",
        meta={"title": titulo, "a_repr": descriptor},
    )
    validate_record(rec)
    return rec


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ensemble-capture)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_budget_articles(
    fontes: list[str], n: int = 20, keyword: str = _KEYWORD
) -> list[PointerRecord]:
    """Wrapper fino de rede: le feeds RSS (`fontes`) e baixa ate `n` materias.

    Para cada feed: itera os `<item>`, dedup por `link`, baixa a materia e roda
    `parse_article`. Falha de fetch/parse de um item e pulada, nao aborta.

    Topicalidade: materias cujo titulo ou corpo citam `keyword` vem PRIMEIRO; se as
    topicais nao chegarem a `n`, as demais (distratores) completam os slots. O filtro
    so ORDENA -- nunca zera o yield.

    Nao coberto por teste de rede; roda na Task 9.
    """
    topicais: list[PointerRecord] = []
    distratores: list[PointerRecord] = []
    seen: set[str] = set()
    for feed in fontes:
        root = ET.fromstring(_http_get(feed))
        for item in root.iter("item"):
            link = (item.findtext("link") or "").strip()
            titulo = (item.findtext("title") or "")
            if not link or link in seen:
                continue
            seen.add(link)
            try:
                rec = parse_article(_http_get(link), link)
            except Exception:
                continue
            if keyword in (titulo + " " + rec.descriptor).lower():
                topicais.append(rec)
            else:
                distratores.append(rec)
            if len(topicais) >= n:
                return topicais[:n]
    return (topicais + distratores)[:n]
