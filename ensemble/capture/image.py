"""Captura de imagem: fotos de galeria de agencia (Camara / EBC / Senado) com legenda."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse

from ensemble.descriptors import caption_descriptor
from ensemble.records import PointerRecord, validate_record


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


def fetch_gallery(
    gallery_urls: list[str], n: int = 20, dest_dir: str | Path | None = None
) -> list[PointerRecord]:
    """Wrapper fino de rede: baixa paginas de galeria (Camara/EBC/Senado) e extrai ate `n` fotos.

    Nao coberto por teste de rede; roda na Task 9.
    """
    out: list[PointerRecord] = []
    seen: set[str] = set()
    for gurl in gallery_urls:
        try:
            recs = parse_gallery(_http_get(gurl), gurl, downloader=_http_get_bytes, bin_dir=dest_dir)
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
