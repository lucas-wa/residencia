from pathlib import Path

from ensemble.capture.document import parse_pages
from ensemble.capture.text import parse_article
from ensemble.records import validate_record

FIX = Path(__file__).parent / "fixtures" / "ensemble"


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
