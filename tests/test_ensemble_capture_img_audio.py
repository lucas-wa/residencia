from pathlib import Path

from ensemble.capture.audio import record_from_clip
from ensemble.capture.image import parse_gallery
from ensemble.records import validate_record

FIX = Path(__file__).parent / "fixtures" / "ensemble"
BASE = "https://example.gov.br/galeria/orcamento"


def test_parse_gallery_usa_legenda():
    raw = (FIX / "galeria_orcamento.html").read_text(encoding="utf-8")
    recs = parse_gallery(raw, BASE)
    assert len(recs) == 2
    for r in recs:
        validate_record(r)
        assert r.modality == "imagem"
    assert recs[0].descriptor == "Plenário vota o orçamento da União para 2026"
    assert recs[0].pointer_uri == "https://example.gov.br/img/plenario-loa.jpg"
    assert recs[1].pointer_uri == "https://cdn.example.gov.br/img/cmo-relator.jpg"


def test_foto_sem_legenda_e_pulada():
    raw = '<figure><img src="/a.jpg"></figure><figure><img src="/b.jpg"><figcaption> </figcaption></figure>'
    assert parse_gallery(raw, BASE) == []


def test_record_from_clip_usa_asr():
    url = "https://example.gov.br/audio/orcamento.mp3"
    r = record_from_clip("clip.mp3", url, transcriber=lambda p: "  O orçamento foi aprovado.  ")
    validate_record(r)
    assert r.modality == "audio"
    assert r.pointer_uri == url
    assert r.descriptor == "O orçamento foi aprovado."


def test_clip_sem_fala_e_pulado():
    assert record_from_clip("clip.mp3", "https://example.gov.br/a.mp3", transcriber=lambda p: "") is None
