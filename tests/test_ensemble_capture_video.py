import json
from pathlib import Path

from ensemble.capture.video import record_from_session
from ensemble.records import validate_record

FIX = Path(__file__).parent / "fixtures" / "ensemble"


def _info():
    return json.loads((FIX / "yt_sessao_orcamento.json").read_text(encoding="utf-8"))


def test_record_from_session_monta_pointer_com_timestamp():
    r = record_from_session(
        _info(), "sessao.wav", transcriber=lambda p: "  O relator apresentou o parecer da LOA.  ", start=125
    )
    validate_record(r)
    assert r.modality == "video"
    assert r.descriptor == "O relator apresentou o parecer da LOA."
    assert r.pointer_uri == "https://www.youtube.com/watch?v=abc123XYZ_0#t=125"


def test_sessao_sem_fala_e_pulada():
    assert record_from_session(_info(), "sessao.wav", transcriber=lambda p: "") is None


def test_records_from_segments_usa_offsets_reais():
    from ensemble.capture.video import group_segments, records_from_segments

    segs = [
        {"start": 0.0, "end": 40.0, "text": "Abertura da sessão."},
        {"start": 40.0, "end": 95.5, "text": "Relator lê o parecer da LOA."},
        {"start": 95.5, "end": 200.0, "text": "Votação do destaque."},
    ]
    recs = records_from_segments(_info(), group_segments(segs, window=90))
    assert len(recs) == 2
    base = "https://www.youtube.com/watch?v=abc123XYZ_0"
    assert [r.pointer_uri for r in recs] == [base + "#t=0", base + "#t=95"]
    assert recs[1].meta["start"] == 95 and recs[1].meta["end"] == 200
    for r in recs:
        validate_record(r)
        assert r.modality == "video"


def test_janela_sem_texto_e_pulada():
    from ensemble.capture.video import records_from_segments

    segs = [{"start": 10, "end": 20, "text": "  "}, {"start": 130, "end": 150, "text": "Fala."}]
    recs = records_from_segments(_info(), segs)
    assert len(recs) == 1 and recs[0].pointer_uri.endswith("#t=130")
