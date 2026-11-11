import json
from pathlib import Path

import pytest

from ensemble.capture import video as V
from ensemble.capture.video import record_from_session
from ensemble.records import validate_record

FIX = Path(__file__).parent / "fixtures" / "ensemble"


def _info():
    return json.loads((FIX / "yt_sessao_orcamento.json").read_text(encoding="utf-8"))


def _track_js():
    return (FIX / "senado_track_159966.js").read_text(encoding="utf-8")


def _listing_html():
    return (FIX / "senado_eventos_listing.html").read_text(encoding="utf-8")


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


def test_source_default_e_senado_e_pode_ser_sobrescrito():
    r = record_from_session(_info(), "s.wav", transcriber=lambda p: "Fala.")
    assert r.source == "senado"
    r2 = record_from_session(_info(), "s.wav", transcriber=lambda p: "Fala.", source="outra")
    assert r2.source == "outra"


# --- descoberta na fonte nativa do Senado (sem rede: fixtures) ---


def test_event_ids_extrai_da_listagem_e_dedup():
    assert V.event_ids(_listing_html()) == ["159966", "159965", "159960"]


def test_event_ids_sem_evento_e_lista_vazia():
    assert V.event_ids("<html><a href='/multimidia/ao-vivo'>x</a></html>") == []


def test_parse_track_data_do_corpo_js_real():
    t = V.parse_track_data(_track_js())
    assert t["COD_EVENTO"] == "159966"
    assert t["DES_SIGLA"] == "CMO"
    assert t["DES_TITULO"].startswith("Continuação da 9ª Reunião")
    assert t["DAT_EVENTO"] == "2026-11-10T14:30:00-03:00"
    assert len(t["UDS"]) == 2
    assert t["UDS"][0]["END_VID_HIGH"].endswith("_2091505.MP4")


def test_parse_track_data_aceita_prefixo_e_ponto_e_virgula():
    t = V.parse_track_data('window.trackData = {"COD_EVENTO": "1", "UDS": []};')
    assert t == {"COD_EVENTO": "1", "UDS": []}


def test_parse_track_data_sem_trackdata_levanta():
    with pytest.raises(ValueError, match="window.trackData"):
        V.parse_track_data('window.portalURL = "/multimidia"')


def test_is_orcamento_aceita_cmo():
    assert V.is_orcamento({"DES_SIGLA": "CMO", "DES_TITULO": "Reunião", "DES_CLASSE": ""})


def test_is_orcamento_casa_termo_sem_acento_e_flexao():
    assert V.is_orcamento({"DES_SIGLA": "CCT", "DES_TITULO": "", "DES_CLASSE": "Orçamentos Públicos"})
    assert V.is_orcamento({"DES_SIGLA": "CCT", "DES_TITULO": "Debate sobre orcamento impositivo"})


def test_is_orcamento_recusa_evento_fora_do_tema():
    assert not V.is_orcamento(
        {"DES_SIGLA": "CCT", "DES_TITULO": "Audiência sobre IA", "DES_CLASSE": "Comissão de Ciência"}
    )


def test_media_url_prefere_high_e_cai_para_end_vid():
    base = "https://www12.senado.leg.br"
    assert V._media_url({"END_VID_HIGH": "/a.MP4", "END_VID": "/b.MP4"}, base) == base + "/a.MP4"
    assert V._media_url({"END_VID": "/b.MP4"}, base) == base + "/b.MP4"
    assert V._media_url({"END_VID_HIGH": "", "END_VID": ""}, base) is None


# --- fetch_sessions com HTTP e whisper mockados ---


class _Seg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FakeWhisper:
    def __init__(self):
        self.chamadas = []

    def transcribe(self, path, language=None):
        self.chamadas.append((path, language))
        return (
            [
                _Seg(0.0, 40.0, "Abertura da reunião."),
                _Seg(40.0, 95.5, "Relator lê o parecer da LOA."),
                _Seg(95.5, 200.0, "Votação do destaque."),
            ],
            None,
        )


@pytest.fixture
def senado_mock(monkeypatch, tmp_path):
    """Mocka a camada HTTP, o whisper e o ffmpeg: nada de rede/GPU."""
    modelo = _FakeWhisper()
    baixados: list[str] = []

    def http_get(url, timeout=30):
        if "/multimidia/eventos/" in url:
            return _listing_html()
        if url.endswith("/multimidia/data/159966"):
            return _track_js()
        raise RuntimeError(f"HTTP 404: {url}")

    def http_get_bytes(url, timeout=60):
        baixados.append(url)
        return b"\x00\x00\x00\x18ftypmp42"

    monkeypatch.setattr(V, "_http_get", http_get)
    monkeypatch.setattr(V, "_http_get_bytes", http_get_bytes)
    monkeypatch.setattr(V, "_whisper_model", lambda *a, **k: modelo)
    monkeypatch.setattr(V, "_ffmpeg_keyframe", lambda media, t, destino: str(destino))
    return {"modelo": modelo, "baixados": baixados, "dest": tmp_path / "video"}


def test_fetch_sessions_monta_records_a_partir_dos_uds(senado_mock):
    recs = V.fetch_sessions(
        ["https://www12.senado.leg.br"], n=3, dest_dir=str(senado_mock["dest"]), dias=1
    )
    assert len(recs) == 3
    for r in recs:
        validate_record(r)
        assert r.modality == "video" and r.source == "senado"
        assert r.meta["channel"] == "TV Senado"
        assert r.meta["title"].startswith("Continuação da 9ª Reunião")
    # 2 janelas do 1o UD + 1 do 2o; pointer carrega o UD (senao os `#t=0` colidiriam)
    base = "https://www12.senado.leg.br/multimidia/evento/159966"
    assert [r.pointer_uri for r in recs] == [
        f"{base}?ud=3917420#t=0",
        f"{base}?ud=3917420#t=95",
        f"{base}?ud=3917428#t=0",
    ]
    assert len({r.id for r in recs}) == 3


def test_fetch_sessions_usa_end_vid_high_e_fallback_e_retem_binario(senado_mock):
    recs = V.fetch_sessions(
        "https://www12.senado.leg.br", n=3, dest_dir=str(senado_mock["dest"]), dias=1
    )
    assert senado_mock["baixados"] == [
        "https://www12.senado.leg.br/portalcedoc/pcedoc/2026/20261110/20261110151202_2091505.MP4",
        "https://www12.senado.leg.br/portalcedoc/pcedoc/2026/20261110/20261110151202_2091533.MP4",
    ]
    paths = [Path(r.meta["path"]) for r in recs]
    assert {p.name for p in paths} == {
        "20261110151202_2091505.MP4", "20261110151202_2091533.MP4"}
    for p in paths:
        assert p.is_file() and p.parent == senado_mock["dest"]
    assert recs[0].meta["a_repr"].endswith("159966_3917420_0.jpg")


def test_fetch_sessions_pula_evento_com_falha_sem_abortar(monkeypatch, senado_mock):
    listing = (
        "<a href='/multimidia/evento/999001'>quebrado</a>"
        "<a href='/multimidia/evento/159966'>CMO</a>"
    )
    original = V._http_get

    def http_get(url, timeout=30):
        if "/multimidia/eventos/" in url:
            return listing
        if url.endswith("/multimidia/data/999001"):
            return "resposta sem trackData"
        return original(url, timeout)

    monkeypatch.setattr(V, "_http_get", http_get)
    recs = V.fetch_sessions(
        ["https://www12.senado.leg.br"], n=2, dest_dir=str(senado_mock["dest"]), dias=1
    )
    assert len(recs) == 2
    assert all("evento/159966" in r.pointer_uri for r in recs)


def test_fetch_sessions_ignora_evento_fora_do_tema(monkeypatch, senado_mock):
    track = json.loads(_track_js().split("window.trackData =", 1)[1])
    track["DES_SIGLA"] = "CCT"
    track["DES_TITULO"] = "Audiência sobre inteligência artificial"
    track["DES_CLASSE"] = "Comissão de Ciência e Tecnologia"
    corpo = "window.trackData = " + json.dumps(track, ensure_ascii=False)

    def http_get(url, timeout=30):
        if "/multimidia/eventos/" in url:
            return "<a href='/multimidia/evento/159966'>x</a>"
        return corpo

    monkeypatch.setattr(V, "_http_get", http_get)
    assert V.fetch_sessions(
        ["https://www12.senado.leg.br"], n=3, dest_dir=str(senado_mock["dest"]), dias=1
    ) == []
    assert senado_mock["baixados"] == []
