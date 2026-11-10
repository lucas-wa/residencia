"""Captura de imagem: fotos de agencia (Camara / EBC / Senado) com legenda.

Nao existe galeria semantica no governo: quem traz o markup certo (<figure> com
<img> + <figcaption>) e a PAGINA DE ARTIGO da Agencia Camara. Por isso a captura
trata cada URL de entrada como LISTAGEM, descobre os links de artigo
(`article_links`) e roda `parse_gallery` em cada artigo.
"""
from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse

from ensemble.descriptors import caption_descriptor
from ensemble.records import PointerRecord, validate_record

_KEYWORD = "orçamento"

# Artigo da Agencia Camara: /noticias/<digitos>-<slug>
_ARTIGO_RE = re.compile(r"^/noticias/\d+-[^/]+/?$")


def parse_gallery(
    raw: str, base_url: str, *, downloader: Callable[[str], bytes] | None = None,
    bin_dir: str | Path | None = None,
) -> list[PointerRecord]:
    """Records 'imagem' a partir do HTML da galeria.

    Cada <figure> traz <img src> + <figcaption>; o descritor e a legenda que ACOMPANHA a foto
    (nao gerada). Foto sem legenda (ausente ou so espacos) nao vira record.
    Com `downloader(url)->bytes` + `bin_dir`, o binario e gravado em `bin_dir/<id><ext>` e
    registrado em meta["path"]/meta["a_repr"]; download que falha -> a foto e pulada.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    out: list[PointerRecord] = []
    seen: set[str] = set()
    for fig in soup.find_all("figure"):
        img = fig.find("img")
        cap = fig.find("figcaption")
        src = (img.get("src") or "").strip() if img else ""
        if not src or cap is None:
            continue
        desc = caption_descriptor(cap.get_text(" "))
        if not desc:
            continue
        url = urljoin(base_url, src)
        if url in seen:
            continue
        seen.add(url)
        rid = "imagem-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        meta: dict = {"gallery": base_url}
        if downloader is not None and bin_dir is not None:
            ext = Path(urlparse(url).path).suffix or ".jpg"
            dest = Path(bin_dir) / f"{rid}{ext}"
            try:
                data = downloader(url)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            except Exception:
                continue
            meta["path"] = str(dest)
            meta["a_repr"] = str(dest)
        rec = PointerRecord(
            id=rid,
            modality="imagem",
            descriptor=desc,
            pointer_uri=url,
            source="galeria",
            meta=meta,
        )
        validate_record(rec)
        out.append(rec)
    return out


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ensemble-capture)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_get_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ensemble-capture)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def article_links(listing_html: str, base_url: str) -> list[str]:
    """URLs ABSOLUTAS dos artigos da Agencia Camara citados no HTML de uma listagem.

    Reconhece ancoras cujo caminho casa com `/noticias/<digitos>-<slug>` (href relativo
    ou absoluto; `urljoin` resolve contra `base_url`). Dedup preservando a ordem.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(listing_html, "html.parser")
    out: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        url = urljoin(base_url, href)
        if not _ARTIGO_RE.match(urlparse(url).path):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def fetch_gallery(
    gallery_urls: list[str], n: int = 20, dest_dir: str | Path | None = None
) -> list[PointerRecord]:
    """Wrapper fino de rede: crawleia LISTAGENS da Agencia Camara e extrai ate `n` fotos.

    Para cada URL de listagem: baixa o HTML -> `article_links` -> baixa cada artigo ->
    `parse_gallery` (que grava o binario em `dest_dir`). Falha de fetch/download de um
    artigo e pulada, nao aborta. Dedup por `pointer_uri`.

    Topicalidade: artigos cujo HTML cita orcamento vem PRIMEIRO; se os topicais nao
    chegarem a `n`, os demais (distratores) completam os slots. O filtro so ORDENA --
    nunca zera o yield.

    Nao coberto por teste de rede; roda na Task 9.
    """
    topicais: list[PointerRecord] = []
    distratores: list[PointerRecord] = []
    seen: set[str] = set()
    for listing_url in gallery_urls:
        try:
            listing = _http_get(listing_url)
        except Exception:
            continue
        for art_url in article_links(listing, listing_url):
            try:
                html = _http_get(art_url)
                recs = parse_gallery(
                    html, art_url, downloader=_http_get_bytes, bin_dir=dest_dir)
            except Exception:
                continue
            alvo = topicais if _KEYWORD in html.lower() else distratores
            for r in recs:
                if r.pointer_uri in seen:
                    continue
                seen.add(r.pointer_uri)
                alvo.append(r)
            if len(topicais) >= n:
                return topicais[:n]
    return (topicais + distratores)[:n]
