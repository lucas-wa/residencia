"""Captura de imagem: fotos de galeria de agencia (Camara / EBC / Senado) com legenda."""
from __future__ import annotations

import hashlib
import urllib.request
from urllib.parse import urljoin

from ensemble.descriptors import caption_descriptor
from ensemble.records import PointerRecord, validate_record


def parse_gallery(raw: str, base_url: str) -> list[PointerRecord]:
    """Records 'imagem' a partir do HTML da galeria.

    Cada <figure> traz <img src> + <figcaption>; o descritor e a legenda que ACOMPANHA a foto
    (nao gerada). Foto sem legenda (ausente ou so espacos) nao vira record.
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
        rec = PointerRecord(
            id="imagem-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12],
            modality="imagem",
            descriptor=desc,
            pointer_uri=url,
            source="galeria",
            meta={"gallery": base_url},
        )
        validate_record(rec)
        out.append(rec)
    return out


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ensemble-capture)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_gallery(gallery_urls: list[str], n: int = 20) -> list[PointerRecord]:
    """Wrapper fino de rede: baixa paginas de galeria (Camara/EBC/Senado) e extrai ate `n` fotos.

    Nao coberto por teste de rede; roda na Task 9.
    """
    out: list[PointerRecord] = []
    seen: set[str] = set()
    for gurl in gallery_urls:
        try:
            recs = parse_gallery(_http_get(gurl), gurl)
        except Exception:
            continue
        for r in recs:
            if r.pointer_uri in seen:
                continue
            seen.add(r.pointer_uri)
            out.append(r)
            if len(out) >= n:
                return out
    return out
