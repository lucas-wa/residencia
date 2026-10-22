"""CLI `scripts/embed_qwen3vl.py`: coleta, saida (.npy + manifest) e modo consulta.

O encode e monkeypatchado -- o teste nao carrega modelo nem toca a rede.
"""

import json

import numpy as np

from scripts import embed_qwen3vl as cli


def _pages(tmp_path, n):
    pages = tmp_path / "pages"
    pages.mkdir()
    for i in range(n):
        (pages / f"plano_p{i:02d}.png").write_bytes(b"")
    (pages / "nota.txt").write_text("ignorar", encoding="utf-8")
    return pages


def test_collect_paths_ordena_filtra_e_limita(tmp_path):
    pages = _pages(tmp_path, 3)
    todos = cli.collect_paths(str(pages))
    assert [p.split("\\")[-1].split("/")[-1] for p in todos] == [
        "plano_p00.png", "plano_p01.png", "plano_p02.png"]
    assert cli.collect_paths(str(pages), limit=2) == todos[:2]


def test_build_manifest_usa_o_stem_como_id(tmp_path):
    paths = cli.collect_paths(str(_pages(tmp_path, 2)))
    man = cli.build_manifest(paths, "Qwen/X", dim=8)
    assert man["model_id"] == "Qwen/X" and man["dim"] == 8 and man["count"] == 2
    assert [i["id"] for i in man["items"]] == ["plano_p00", "plano_p01"]
    assert [i["row"] for i in man["items"]] == [0, 1]


def test_main_grava_vetores_e_manifest(tmp_path, monkeypatch, capsys):
    pages = _pages(tmp_path, 3)
    out = tmp_path / "emb" / "planos"

    def fake_images(paths, **kw):
        assert kw["model_id"] == "Qwen/Qwen3-VL-Embedding-2B" and kw["batch_size"] == 4
        return np.eye(len(paths), 3, dtype="float32")

    monkeypatch.setattr(cli, "embed_images", fake_images)
    monkeypatch.setattr(cli, "embed_texts",
                        lambda texts, **kw: np.array([[1.0, 0.0, 0.0]], dtype="float32"))

    rc = cli.main(["--images-dir", str(pages), "--out", str(out), "--batch-size", "4",
                   "--device", "cpu", "--query", "saude"])

    assert rc == 0
    vecs = np.load(str(out) + ".npy")
    assert vecs.shape == (3, 3)
    man = json.loads((tmp_path / "emb" / "planos.manifest.json").read_text(encoding="utf-8"))
    assert man["count"] == 3 and man["dim"] == 3
    assert [i["id"] for i in man["items"]] == ["plano_p00", "plano_p01", "plano_p02"]
    saida = capsys.readouterr().out
    assert "query dim=3" in saida
    assert "plano_p00" in saida            # ranking impresso para a consulta


def test_main_sem_imagens_encontradas_sai_1(tmp_path):
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    assert cli.main(["--images-dir", str(vazio), "--out", str(tmp_path / "x")]) == 1


def test_main_so_query_nao_varre_disco(tmp_path, monkeypatch, capsys):
    def nao_deveria(*a, **kw):
        raise AssertionError("collect_paths nao deveria ser chamado no modo consulta")

    monkeypatch.setattr(cli, "collect_paths", nao_deveria)
    monkeypatch.setattr(cli, "embed_texts",
                        lambda texts, **kw: np.array([[0.6, 0.8]], dtype="float32"))
    assert cli.main(["--query", "orcamento"]) == 0
    assert "query dim=2" in capsys.readouterr().out
