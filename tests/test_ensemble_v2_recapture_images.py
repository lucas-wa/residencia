"""Recaptura do eixo imagem: so fotos TOPICAS de orcamento, anexadas ao corpus.

Diferenca para `fetch_gallery`: la o filtro de orcamento so ORDENA (distrator completa
os slots); aqui ele DESCARTA. E ele le o CABECALHO/CORPO do artigo, nao o HTML cru --
o menu global da Agencia Camara cita "Orcamento da Uniao" em toda pagina, e citar de
passagem no corpo tambem nao e topicalidade.

Tudo com dubles: nada de rede.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ensemble.capture import image as img_mod
from ensemble.recapture_images import (
    append_images,
    is_budget_article,
    listing_links,
    recapture_images,
)
from ensemble.records import PointerRecord, load_records, save_records, validate_record

LISTAGEM_URL = "https://www.camara.leg.br/noticias/ultimas"

# O menu global cita orcamento em TODA pagina da Agencia Camara: e o distrator do filtro.
_CHROME = '<nav><a href="/orcamento-da-uniao">Orçamento da União</a></nav>'


def _artigo_html(slug: str, *, topical: bool, corpo: str | None = None) -> str:
    tema = "CMO aprova o relatório preliminar da LDO" if topical else "Comissao debate transporte"
    return (
        f"<html><head><title>{tema}</title></head><body>{_CHROME}"
        f"<article><h1>{tema}</h1><p>{corpo or tema}.</p>"
        f'<figure><img src="https://www.camara.leg.br/img/{slug}.jpg">'
        f"<figcaption>Legenda da foto de {slug}</figcaption>"
        f"</figure></article></body></html>"
    )


def _dubla_http(monkeypatch: pytest.MonkeyPatch, paginas: dict[str, str]) -> None:
    def _get(url: str, timeout: int = 30) -> str:
        if url not in paginas:
            raise RuntimeError(f"404 {url}")
        return paginas[url]

    monkeypatch.setattr(img_mod, "_http_get", _get)
    monkeypatch.setattr(img_mod, "_http_get_bytes", lambda url, timeout=60: b"\xff\xd8dummy")


def test_is_budget_article_ignora_o_chrome_do_site():
    """O menu global cita orcamento em toda pagina -> so o cabecalho/corpo decide."""
    assert is_budget_article(_artigo_html("t", topical=True))
    assert not is_budget_article(_artigo_html("d", topical=False))


def test_is_budget_article_descarta_mencao_de_passagem():
    """Citar orcamento 1-3x num artigo de outro tema NAO e topicalidade; densidade alta e."""
    de_passagem = _artigo_html(
        "d", topical=False,
        corpo="O projeto de seguro rural tem impacto no orçamento da União em 2027",
    )
    assert not is_budget_article(de_passagem)

    dense = _artigo_html(
        "t", topical=False,
        corpo=" ".join(["A peça contempla o orçamento do exercício."] * 5),
    )
    assert is_budget_article(dense)
    assert not is_budget_article(dense, min_mencoes=99)


def test_listing_links_cai_para_rss_quando_nao_ha_ancora():
    """Listagem HTML -> article_links; feed sem ancora -> <item><link>."""
    rss = (
        "<rss><channel>"
        "<item><link>https://www12.senado.leg.br/noticias/materias/2026/11/06/cmo</link></item>"
        "<item><link>https://www12.senado.leg.br/noticias/materias/2026/11/06/cmo</link></item>"
        "<item><link>https://www12.senado.leg.br/noticias/materias/2026/11/07/ldo</link></item>"
        "</channel></rss>"
    )
    assert listing_links(rss, "https://www12.senado.leg.br/noticias/rss") == [
        "https://www12.senado.leg.br/noticias/materias/2026/11/06/cmo",
        "https://www12.senado.leg.br/noticias/materias/2026/11/07/ldo",
    ]
    html = '<a href="/noticias/111-um-orcamento">1</a>'
    assert listing_links(html, LISTAGEM_URL) == [
        "https://www.camara.leg.br/noticias/111-um-orcamento"
    ]


def test_recapture_descarta_nao_topico_e_baixa_o_binario(monkeypatch, tmp_path):
    """Artigo fora do tema e DESCARTADO (nao rebaixado): so a foto topica sai."""
    topico = "https://www.camara.leg.br/noticias/111-cmo-ldo"
    distrator = "https://www.camara.leg.br/noticias/222-transporte"
    _dubla_http(monkeypatch, {
        LISTAGEM_URL: f'<a href="{distrator}">d</a><a href="{topico}">t</a>',
        topico: _artigo_html("cmo", topical=True),
        distrator: _artigo_html("transporte", topical=False),
    })

    recs = recapture_images([LISTAGEM_URL], n=5, bin_dir=tmp_path)

    assert [r.pointer_uri for r in recs] == ["https://www.camara.leg.br/img/cmo.jpg"]
    r = recs[0]
    validate_record(r)
    assert r.modality == "imagem"
    assert r.descriptor == "Legenda da foto de cmo"
    p = Path(r.meta["path"])
    assert p.is_file() and p.parent == tmp_path
    assert r.meta["a_repr"] == str(p)
    assert p.read_bytes() == b"\xff\xd8dummy"


def test_recapture_dedup_por_id_ja_presente(monkeypatch, tmp_path):
    """Foto cujo id ja esta no corpus nao volta -- e nem chega a ser baixada."""
    a1 = "https://www.camara.leg.br/noticias/111-cmo-ldo"
    a2 = "https://www.camara.leg.br/noticias/222-ploa"
    _dubla_http(monkeypatch, {
        LISTAGEM_URL: f'<a href="{a1}">1</a><a href="{a2}">2</a>',
        a1: _artigo_html("cmo", topical=True),
        a2: _artigo_html("ploa", topical=True),
    })

    todos = recapture_images([LISTAGEM_URL], n=5, bin_dir=tmp_path)
    assert len(todos) == 2

    baixados: list[str] = []
    monkeypatch.setattr(
        img_mod, "_http_get_bytes",
        lambda url, timeout=60: (baixados.append(url), b"\xff\xd8dummy")[1],
    )
    resto = recapture_images(
        [LISTAGEM_URL], n=5, bin_dir=tmp_path, ids_existentes={todos[0].id})

    assert [r.id for r in resto] == [todos[1].id]
    assert baixados == [todos[1].pointer_uri]  # a foto ja presente nao foi rebaixada


def test_append_images_preserva_os_antigos(monkeypatch, tmp_path):
    """Append-only: os records anteriores continuam la, com os mesmos ids e na ordem."""
    destino = tmp_path / "records.jsonl"
    antigos = [
        PointerRecord(id="texto-aaa", modality="texto", descriptor="d", pointer_uri="u1",
                      source="noticia"),
        PointerRecord(id="imagem-bbb", modality="imagem", descriptor="legenda velha",
                      pointer_uri="https://www.camara.leg.br/img/velha.jpg", source="galeria"),
    ]
    save_records(antigos, destino)
    art = "https://www.camara.leg.br/noticias/111-cmo-ldo"
    _dubla_http(monkeypatch, {
        LISTAGEM_URL: f'<a href="{art}">1</a>',
        art: _artigo_html("cmo", topical=True),
    })

    novos = append_images(destino, n=3, listagens=[LISTAGEM_URL])

    assert len(novos) == 1
    depois = load_records(destino)
    assert [r.id for r in depois[:2]] == ["texto-aaa", "imagem-bbb"]
    assert len(depois) == 3
    assert depois[2].id == novos[0].id
    # binario sob <records>/bin/imagem, com caminho em estilo posix
    assert novos[0].meta["path"].endswith(f"bin/imagem/{novos[0].id}.jpg")
    assert Path(novos[0].meta["path"]).is_file()
