"""Recaptura do eixo imagem: so fotos TOPICAS de orcamento, anexadas ao corpus.

`fetch_gallery` (ensemble/capture/image.py) usa o filtro de orcamento apenas para
ORDENAR -- quando os topicais nao enchem `n`, distratores completam os slots. Foi o que
aconteceu no corpus congelado: as 9 fotos do eixo imagem sao de sessao generica.

Aqui o filtro DESCARTA. E ele roda sobre o CABECALHO + CORPO do artigo, nao sobre o
HTML inteiro: o menu global da Agencia Camara cita "Orcamento da Uniao" em TODA
pagina, entao casar o HTML cru deixa passar 100% dos artigos (medido: 50/50).

So "mencionar orcamento no corpo" tambem nao basta: medido em 11 artigos reais, os
que citam de passagem (seguro rural, IR, SUS) casam 1-3x e rendem foto de audiencia
de outro tema. A topicalidade e, entao, o CABECALHO (<h1>/<title>) ou densidade alta
no corpo -- separacao limpa na sonda: topicais 6-21 mencoes, de passagem 1-3.

Append-only: os records ja presentes no JSONL nunca sao tocados; o dedup e por `id`
(sha1 do pointer_uri) e por `pointer_uri`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ensemble.capture import image as _image
from ensemble.records import PointerRecord, validate_record

# Topicalidade de orcamento: PLOA/LDO/LOA/CMO e as variacoes de "orcamento".
ORCAMENTO_RE = re.compile(
    r"or[çc]ament|ploa|\bldo\b|\bloa\b|\bcmo\b|comiss[aã]o mista de or[çc]amento|lei or[çc]ament",
    re.IGNORECASE,
)

# Listagens com rendimento confirmado por sonda. A busca do portal
# (`/busca-portal?q=...`) e renderizada no cliente -- o HTML servido nao traz nenhum
# link de artigo --, entao o que sobra e paginar as ultimas noticias e filtrar pelo
# texto: ~4% dos artigos sao topicais, ~1 foto cada. O RSS do Senado fica como piso,
# mas rende 0 fotos: a materia do Agencia Senado usa <img alt> solto, sem
# <figure><figcaption>, que e o que `parse_gallery` sabe ler.
LISTAGENS_PADRAO: list[str] = [
    *(f"https://www.camara.leg.br/noticias/ultimas?pagina={p}" for p in range(1, 56)),
    "https://www.camara.leg.br/noticias/rss/ultimas-noticias",
    "https://www12.senado.leg.br/noticias/rss",
]


# Mencoes no corpo que valem por um cabecalho (sonda: de passagem <= 3, topical >= 6).
MIN_MENCOES = 5


def article_parts(raw_html: str) -> tuple[str, str]:
    """(cabecalho, corpo) do artigo, sem o chrome do site (nav/header/footer/aside/script).

    Cabecalho = <h1> + <title>; corpo = <article> (ou <main>/<body>) em texto puro.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "html.parser")
    h1 = soup.find("h1")
    titulo = (h1.get_text(" ") if h1 else "") + " " + (
        soup.title.get_text(" ") if soup.title else "")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
        tag.decompose()
    corpo = soup.find("article") or soup.find("main") or soup.body or soup
    return " ".join(titulo.split()), " ".join(corpo.get_text(" ").split())


def article_text(raw_html: str) -> str:
    """Cabecalho + corpo, em uma string."""
    cab, corpo = article_parts(raw_html)
    return f"{cab} {corpo}".strip()


def is_budget_article(raw_html: str, *, min_mencoes: int = MIN_MENCOES) -> bool:
    """O artigo E SOBRE orcamento? (nao basta citar: ver o docstring do modulo).

    Topical = o cabecalho casa `ORCAMENTO_RE`, OU o corpo casa >= `min_mencoes` vezes.
    """
    cabecalho, corpo = article_parts(raw_html)
    if ORCAMENTO_RE.search(cabecalho):
        return True
    return len(ORCAMENTO_RE.findall(corpo)) >= min_mencoes


def listing_links(listing_html: str, listing_url: str) -> list[str]:
    """Links de artigo de uma listagem: HTML da Agencia Camara OU feed RSS.

    Tenta `article_links` (ancoras `/noticias/<digitos>-<slug>`); se a listagem nao
    tiver nenhuma -- caso dos feeds --, le `<item><link>` como RSS. Dedup na ordem.
    """
    links = _image.article_links(listing_html, listing_url)
    if links:
        return links
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(listing_html)
    except ET.ParseError:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip()
        if link and link not in seen:
            seen.add(link)
            out.append(link)
    return out


def recapture_images(
    listagens: Iterable[str],
    *,
    n: int = 12,
    bin_dir: str | Path,
    ids_existentes: Iterable[str] = (),
    uris_existentes: Iterable[str] = (),
) -> list[PointerRecord]:
    """Ate `n` records 'imagem' TOPICOS de orcamento, com o binario em `bin_dir`.

    Para cada listagem: `listing_links` -> baixa o artigo -> `is_budget_article`
    (artigo fora do tema e DESCARTADO, nao rebaixado) -> `parse_gallery`. Cada foto
    passa por dois filtros de dedup antes do download: `id` ja presente no corpus e
    `pointer_uri` ja visto. Falha de rede em um artigo e pulada, nao aborta.
    """
    vistos_id = set(ids_existentes)
    vistos_uri = set(uris_existentes)
    out: list[PointerRecord] = []
    artigos_vistos: set[str] = set()
    for listing_url in listagens:
        try:
            listing = _image._http_get(listing_url)
        except Exception:
            continue
        for art_url in listing_links(listing, listing_url):
            if art_url in artigos_vistos:
                continue
            artigos_vistos.add(art_url)
            try:
                html = _image._http_get(art_url)
            except Exception:
                continue
            if not is_budget_article(html):
                continue
            try:
                # 1a passada SEM downloader: decide o que baixar antes de gastar rede.
                candidatos = _image.parse_gallery(html, art_url)
            except Exception:
                continue
            querer = {
                c.id for c in candidatos
                if c.id not in vistos_id and c.pointer_uri not in vistos_uri
            }
            if not querer:
                continue
            try:
                recs = _image.parse_gallery(
                    html, art_url,
                    downloader=_image._http_get_bytes, bin_dir=bin_dir,
                )
            except Exception:
                continue
            for r in recs:
                if r.id not in querer or r.id in vistos_id or r.pointer_uri in vistos_uri:
                    continue
                vistos_id.add(r.id)
                vistos_uri.add(r.pointer_uri)
                validate_record(r)
                out.append(r)
                if len(out) >= n:
                    return out
    return out


def _normaliza_path(rec: PointerRecord) -> None:
    """meta['path']/['a_repr'] em caminho relativo ao cwd, com '/' (igual ao corpus)."""
    p = rec.meta.get("path")
    if not p:
        return
    caminho = Path(p)
    try:
        caminho = caminho.relative_to(Path.cwd())
    except ValueError:
        pass
    rec.meta["path"] = rec.meta["a_repr"] = caminho.as_posix()


def append_images(
    records_path: str | Path, *, n: int = 12, listagens: Iterable[str] | None = None,
) -> list[PointerRecord]:
    """Anexa ate `n` imagens topicas ao JSONL, preservando os records existentes."""
    from ensemble.records import load_records, save_records

    records_path = Path(records_path)
    antigos = load_records(records_path)
    bin_dir = records_path.parent / "bin" / "imagem"
    novos = recapture_images(
        list(listagens) if listagens else LISTAGENS_PADRAO,
        n=n,
        bin_dir=bin_dir,
        ids_existentes={r.id for r in antigos},
        uris_existentes={r.pointer_uri for r in antigos},
    )
    for r in novos:
        _normaliza_path(r)
        if not r.meta.get("path"):
            raise RuntimeError(f"record {r.id} sem binario retido")
        validate_record(r)
    if novos:
        save_records(antigos + novos, records_path)
    return novos


def _tabela(novos: list[PointerRecord]) -> str:
    linhas = [f"{'id':<20} | {'legenda':<90} | pointer_uri"]
    for r in novos:
        linhas.append(f"{r.id:<20} | {r.descriptor[:90]:<90} | {r.pointer_uri}")
    return "\n".join(linhas)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="data/govexe/ensemble_orcamento/records.jsonl")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--listings", nargs="*", default=None)
    args = ap.parse_args(argv)

    novos = append_images(args.records, n=args.n, listagens=args.listings)
    print(_tabela(novos))
    print(f"\n{len(novos)} imagens topicas anexadas a {args.records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
