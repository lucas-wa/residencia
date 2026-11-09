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
