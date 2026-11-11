from pathlib import Path

from ensemble import build_corpus as bc
from ensemble.capture.audio import record_from_clip
from ensemble.capture.document import parse_pages
from ensemble.capture.image import parse_gallery
from ensemble.capture.text import parse_article
from ensemble.capture.video import records_from_segments
from ensemble.records import MODALITIES, PointerRecord, load_records, validate_record

PNG = b"\x89PNG\r\n\x1a\nfixture"
HTML = (
    '<figure><img src="/a/foto1.jpg"><figcaption>Plenario vota orcamento</figcaption></figure>'
)


def test_parser_registra_path_do_binario(tmp_path):
    # imagem: downloader duble devolve bytes; o parser grava e aponta meta["path"].
    recs = parse_gallery(HTML, "https://x.gov.br/g", downloader=lambda u: PNG, bin_dir=tmp_path)
    assert len(recs) == 1
    p = Path(recs[0].meta["path"])
    assert p.read_bytes() == PNG and p.parent == tmp_path
    assert recs[0].meta["a_repr"] == str(p)
    validate_record(recs[0])

    # audio: clipe ja em disco; a_repr = ASR.
    clip = tmp_path / "c.mp3"
    clip.write_bytes(b"mp3")
    a = record_from_clip(str(clip), "https://x/c.mp3", transcriber=lambda _p: "fala sobre orcamento")
    assert a.meta["path"] == str(clip) and a.meta["a_repr"] == "fala sobre orcamento"

    # video: midia retida + keyframe como a_repr.
    kf = tmp_path / "kf.jpg"
    kf.write_bytes(b"jpg")
    vs = records_from_segments(
        {"webpage_url": "https://y/v"}, [{"start": 5, "end": 9, "text": "orcamento"}],
        media_path="v.mp4", keyframe_fn=lambda t: str(kf),
    )
    assert vs[0].meta["path"] == "v.mp4" and vs[0].meta["a_repr"] == str(kf)

    # documento: renderer duble grava a imagem da pagina.
    def render(page, dest):
        dest.write_bytes(PNG)

    docs = parse_pages(
        "loa.pdf", extractor=lambda pid: "texto" if pid.endswith("#p1") else (_ for _ in ()).throw(IndexError()),
        n=5, renderer=render, bin_dir=tmp_path / "pg",
    )
    assert Path(docs[0].meta["path"]).read_bytes() == PNG

    # texto: sem binario, a_repr = texto.
    t = parse_article("<html><body><article>orcamento anual</article></body></html>", "https://z/n")
    assert t.meta["a_repr"] == t.descriptor and "path" not in t.meta


def test_build_corpus_salva_binarios(tmp_path, monkeypatch):
    def duble(modality, n, dest_dir=None):
        out = []
        for i in range(n):
            meta = {}
            if modality != "texto":
                # dublê grava o binario FORA de bin_dir; build_corpus deve retê-lo sob bin_dir
                src = tmp_path / "origem" / f"{modality}{i}.bin"
                src.parent.mkdir(exist_ok=True)
                src.write_bytes(b"bin")
                meta["path"] = str(src)
            out.append(PointerRecord(id=f"{modality}-{i}", modality=modality,
                                     descriptor=f"desc {modality} {i}",
                                     pointer_uri=f"https://e/{modality}/{i}", source="duble", meta=meta))
        return out

    for modality, fn in {"texto": "fetch_budget_articles", "documento": "fetch_budget_pages",
                         "imagem": "fetch_gallery", "audio": "fetch_clips",
                         "video": "fetch_sessions"}.items():
        monkeypatch.setattr(bc, fn, lambda *a, _m=modality, **k: duble(_m, k["n"]))

    bin_dir = tmp_path / "bins"
    destino = bc.build_corpus(tmp_path / "records.jsonl", por_eixo=2, bin_dir=bin_dir)
    recs = load_records(destino)
    assert len(recs) == 10 and {r.modality for r in recs} == set(MODALITIES)
    for r in recs:
        if r.modality == "texto":
            continue
        p = Path(r.meta["path"])
        assert p.is_file() and bin_dir in p.parents
