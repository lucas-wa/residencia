import pytest

from ensemble import build_corpus as bc
from ensemble.corpus_v4 import CorpusPathError
from ensemble.records import MODALITIES, PointerRecord, load_records, validate_record

EIXOS = {
    "texto": "fetch_budget_articles",
    "documento": "fetch_budget_pages",
    "imagem": "fetch_gallery",
    "audio": "fetch_clips",
    "video": "fetch_sessions",
}


def _duble(modality, n):
    return [
        PointerRecord(
            id=f"{modality}-{i}",
            modality=modality,
            descriptor=f"descritor de orcamento {modality} {i}",
            pointer_uri=f"https://exemplo.gov.br/{modality}/{i}",
            source="duble",
        )
        for i in range(n)
    ]


def _patch(monkeypatch, vazio=None, fontes_vistas=None):
    chamadas = {}
    for modality, fn in EIXOS.items():
        def fake(*args, _m=modality, **kwargs):
            chamadas[_m] = kwargs.get("n")
            if fontes_vistas is not None:
                fontes_vistas.append((_m, args[0]))
            return [] if _m == vazio else _duble(_m, kwargs["n"])
        monkeypatch.setattr(bc, fn, fake)
    # o eixo documento do v4 passa por `fetch_budget_pages_multi` (lista de PDFs)
    def fake_multi(*args, **kwargs):
        chamadas["documento"] = kwargs.get("n")
        if fontes_vistas is not None:
            fontes_vistas.append(("documento-multi", args[0]))
        return [] if vazio == "documento" else _duble("documento", kwargs["n"])
    monkeypatch.setattr(bc, "fetch_budget_pages_multi", fake_multi)
    return chamadas


def test_build_corpus_concatena_e_valida(tmp_path, monkeypatch):
    chamadas = _patch(monkeypatch)
    destino = tmp_path / "sub" / "records.jsonl"
    out = bc.build_corpus(destino, por_eixo=3)
    assert out == destino
    recs = load_records(out)
    assert len(recs) == 15
    assert {r.modality for r in recs} == set(MODALITIES)
    for r in recs:
        validate_record(r)
    assert set(chamadas.values()) == {3}


@pytest.mark.parametrize("eixo", list(EIXOS))
def test_build_corpus_eixo_vazio_nomeia_eixo(tmp_path, monkeypatch, eixo):
    _patch(monkeypatch, vazio=eixo)
    with pytest.raises(RuntimeError, match=f"eixo '{eixo}' retornou 0 itens"):
        bc.build_corpus(tmp_path / "r.jsonl", por_eixo=2)


# --- v4: captura maior + sanity de caminho antes de salvar -----------------------------

def _duble_v4(modality, n, binarios, *, keyframe_sumido=False):
    """Dubles com binario em disco: os eixos image-route (documento/imagem/video) tem
    `meta["path"]` real, como a captura de verdade entrega depois do `_reter`."""
    recs = _duble(modality, n)
    if modality not in ("documento", "imagem", "video"):
        return recs
    for i, r in enumerate(recs):
        src = binarios / f"{modality}-{i}.bin"
        src.write_bytes(b"binario de duble")
        r.meta["path"] = str(src)
        if modality == "video":
            # o a_repr do video e o keyframe da janela, nao o mp4 inteiro
            kf = binarios / f"{modality}-{i}-kf.png"
            if not keyframe_sumido:
                kf.write_bytes(b"keyframe de duble")
            r.meta["a_repr"] = str(kf)
    return recs


def _patch_v4(monkeypatch, binarios, fontes_vistas=None, **kw):
    chamadas = {}
    binarios.mkdir(parents=True, exist_ok=True)
    for modality, fn in EIXOS.items():
        def fake(*args, _m=modality, **kwargs):
            chamadas[_m] = kwargs.get("n")
            if fontes_vistas is not None:
                fontes_vistas.append((_m, args[0]))
            return _duble_v4(_m, kwargs["n"], binarios, **kw)
        monkeypatch.setattr(bc, fn, fake)
    def fake_multi(*args, **kwargs):
        chamadas["documento"] = kwargs.get("n")
        if fontes_vistas is not None:
            fontes_vistas.append(("documento-multi", args[0]))
        return _duble_v4("documento", kwargs["n"], binarios, **kw)
    monkeypatch.setattr(bc, "fetch_budget_pages_multi", fake_multi)
    return chamadas


def test_build_corpus_v4_captura_50_por_eixo_por_default(tmp_path, monkeypatch):
    chamadas = _patch_v4(monkeypatch, tmp_path / "origem")
    destino = tmp_path / "v4" / "records.jsonl"
    out = bc.build_corpus_v4(destino, bin_dir=tmp_path / "v4" / "bin")
    assert out == destino
    assert set(chamadas.values()) == {50}
    assert len(load_records(out)) == 250


def test_build_corpus_v4_por_eixo_e_configuravel(tmp_path, monkeypatch):
    chamadas = _patch_v4(monkeypatch, tmp_path / "origem")
    bc.build_corpus_v4(tmp_path / "v4" / "records.jsonl", por_eixo=4,
                       bin_dir=tmp_path / "v4" / "bin")
    assert set(chamadas.values()) == {4}


def test_build_corpus_v4_recusa_salvar_com_image_route_sem_caminho(tmp_path, monkeypatch):
    _patch_v4(monkeypatch, tmp_path / "origem", keyframe_sumido=True)
    destino = tmp_path / "v4" / "records.jsonl"
    with pytest.raises(CorpusPathError, match="video-0"):
        bc.build_corpus_v4(destino, por_eixo=3, bin_dir=tmp_path / "v4" / "bin")
    assert not destino.exists()          # a sanity roda ANTES de salvar


def test_build_corpus_v3_nao_herda_a_sanity_de_caminho(tmp_path, monkeypatch):
    # o v3 (build_corpus) segue sem a checagem: o default de `validar` e None
    _patch(monkeypatch)
    destino = tmp_path / "v3" / "records.jsonl"
    assert bc.build_corpus(destino, por_eixo=2) == destino
    assert destino.exists()


# --- v4: eixo documento = varios PDFs legislativos -------------------------------------

def test_documento_v4_tem_varios_pdfs_legislativos():
    assert isinstance(bc.DOCUMENTO_V4, list)
    assert len(bc.DOCUMENTO_V4) >= 5
    assert len(set(bc.DOCUMENTO_V4)) == len(bc.DOCUMENTO_V4)
    assert all(u.startswith("https://") for u in bc.DOCUMENTO_V4)
    # o PLOA (fonte unica do v3) continua entre elas
    assert bc.FONTES_PADRAO["documento"] in bc.DOCUMENTO_V4


def test_build_corpus_v4_manda_a_lista_de_pdfs_ao_eixo_documento(tmp_path, monkeypatch):
    vistas = []
    _patch_v4(monkeypatch, tmp_path / "origem", fontes_vistas=vistas)
    bc.build_corpus_v4(tmp_path / "v4" / "records.jsonl", por_eixo=3,
                       bin_dir=tmp_path / "v4" / "bin")
    doc = [f for nome, f in vistas if nome.startswith("documento")]
    assert len(doc) == 1
    assert doc[0] == bc.DOCUMENTO_V4
    assert ("documento-multi", bc.DOCUMENTO_V4) in vistas   # foi pelo fetch de lista


def test_build_corpus_v4_respeita_fontes_explicitas(tmp_path, monkeypatch):
    vistas = []
    _patch_v4(monkeypatch, tmp_path / "origem", fontes_vistas=vistas)
    minha = ["https://x.gov.br/a.pdf", "https://x.gov.br/b.pdf"]
    bc.build_corpus_v4(tmp_path / "v4" / "records.jsonl", por_eixo=3,
                       fontes={"documento": minha}, bin_dir=tmp_path / "v4" / "bin")
    assert ("documento-multi", minha) in vistas


def test_build_corpus_v3_segue_com_o_ploa_unico(tmp_path, monkeypatch):
    """Regressao: o corpus v3 (congelado) nao amplia o eixo documento."""
    vistas = []
    _patch(monkeypatch, fontes_vistas=vistas)
    bc.build_corpus(tmp_path / "v3" / "records.jsonl", por_eixo=2)
    doc = [(nome, f) for nome, f in vistas if nome.startswith("documento")]
    assert doc == [("documento", bc.FONTES_PADRAO["documento"])]
    assert isinstance(doc[0][1], str)                       # URL unica, nao lista
