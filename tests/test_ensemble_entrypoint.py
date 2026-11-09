"""Entrypoint CLI do ensemble: le records+queries, roda a demo, grava o output anotado.

O embedder real (BGE-M3) e construido por import tardio em `_build_embedder`; aqui ele e
sempre INJETADO por um duble, de modo que nenhum teste baixa modelo nem toca a GPU.
"""

import numpy as np

from ensemble.records import PointerRecord, save_records
from ensemble.run_ensemble import main

VOCAB = ["orçamento", "saúde", "foto", "pônei", "áudio", "reunião", "clipe"]


def fake_embedder(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            if w in VOCAB:
                out[i, VOCAB.index(w)] += 1.0
    return out


def _rec(id_: str, modality: str, descriptor: str) -> PointerRecord:
    return PointerRecord(
        id=id_,
        modality=modality,
        descriptor=descriptor,
        pointer_uri=f"gs://b/{id_}",
        source="teste",
        meta={},
    )


def _corpus() -> list[PointerRecord]:
    return [
        _rec("d1", "documento", "Orçamento saúde"),
        _rec("i1", "imagem", "foto pônei"),
        _rec("a1", "audio", "áudio reunião orçamento"),
        _rec("t1", "texto", "orçamento saúde reunião"),
        _rec("v1", "video", "clipe orçamento"),
    ]


def test_main_grava_output(tmp_path):
    records_path = tmp_path / "records.jsonl"
    queries_path = tmp_path / "queries.txt"
    out_path = tmp_path / "ensemble_demo.txt"

    save_records(_corpus(), records_path)
    queries_path.write_text("orçamento saúde\nfoto pônei\n", encoding="utf-8")

    rc = main(
        ["--records", str(records_path), "--queries", str(queries_path),
         "--out", str(out_path)],
        embedder=fake_embedder,
    )

    assert rc == 0
    assert out_path.exists()
    conteudo = out_path.read_text(encoding="utf-8")
    # as tags de modalidade da demo aparecem no output anotado
    assert "[DOCUMENTO]" in conteudo
    assert "Consulta: orçamento saúde" in conteudo
    assert "Modalidades ausentes (corpus):" in conteudo


def test_main_records_inexistente_retorna_erro(tmp_path, capsys):
    records_path = tmp_path / "nao_existe.jsonl"
    queries_path = tmp_path / "queries.txt"
    out_path = tmp_path / "ensemble_demo.txt"
    queries_path.write_text("orçamento\n", encoding="utf-8")

    rc = main(
        ["--records", str(records_path), "--queries", str(queries_path),
         "--out", str(out_path)],
        embedder=fake_embedder,
    )

    assert rc != 0
    err = capsys.readouterr().err
    assert "nao_existe.jsonl" in err
    assert not out_path.exists()
