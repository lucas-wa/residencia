import json
from pathlib import Path

import pytest

from finetune.make_pairs import (
    PageContext,
    load_contexts,
    make_pairs,
    plan_id_of,
    template_generate_queries,
)


def _corpus(tmp_path: Path, planos=("BR001", "BR002"), paginas=2) -> Path:
    """Mini benchmark no layout de data/govexe/build_planos_benchmark.py."""
    root = tmp_path / "ptbr_planos"
    pages = root / "pages"
    pages.mkdir(parents=True)
    corpus, queries, qrels = [], [], []
    for seq in planos:
        page_ids = []
        for i in range(paginas):
            png = pages / f"{seq}_p{i:02d}.png"
            png.write_bytes(b"png")
            corpus.append({"id": png.stem, "image_path": str(png)})
            page_ids.append(png.stem)
        queries.append({"id": f"{seq}_resumo", "text": f"Plano {seq}: saude e educacao"})
        qrels += [f"{seq}_resumo\t{pid}\t1" for pid in page_ids]
    root.joinpath("corpus.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in corpus), encoding="utf-8")
    root.joinpath("queries.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in queries), encoding="utf-8")
    root.joinpath("qrels.tsv").write_text("\n".join(qrels), encoding="utf-8")
    return root


def _mock_gen(chamadas=None):
    """Gerador mockado: K consultas estaveis por pagina, registrando as chamadas."""

    def gen(ctx: PageContext) -> list[str]:
        if chamadas is not None:
            chamadas.append(ctx.page_id)
        return [f"consulta {i} sobre {ctx.page_id}" for i in range(ctx.k)]

    return gen


def _read_pairs(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_formato_dos_pares(tmp_path: Path):
    root = _corpus(tmp_path)
    out = tmp_path / "pairs.jsonl"

    linhas = make_pairs(root, out, generate_queries=_mock_gen(), k=2, neg=2)

    assert len(linhas) == 4 * 2                      # 4 paginas x 2 consultas
    assert _read_pairs(out) == linhas
    for r in linhas:
        assert set(r) == {"id", "query", "pos_id", "pos_image", "neg_ids", "neg_images",
                          "label", "generator"}
        assert r["label"] == 1
        assert r["query"].startswith("consulta ")
        assert Path(r["pos_image"]).name == f"{r['pos_id']}.png"
        assert len(r["neg_images"]) == len(r["neg_ids"]) == 2
        assert all(Path(p).exists() for p in [r["pos_image"], *r["neg_images"]])


def test_negativos_nunca_sao_a_pagina_positiva(tmp_path: Path):
    root = _corpus(tmp_path, planos=("BR001", "BR002", "BR003"), paginas=3)

    linhas = make_pairs(root, tmp_path / "pairs.jsonl", generate_queries=_mock_gen(),
                        k=2, neg=4)

    for r in linhas:
        assert r["pos_id"] not in r["neg_ids"]
        assert r["pos_image"] not in r["neg_images"]
        assert len(set(r["neg_ids"])) == len(r["neg_ids"])      # sem negativo repetido
        # preferencia por outro plano: nenhum negativo sai do plano da pagina positiva
        assert all(plan_id_of(n) != plan_id_of(r["pos_id"]) for n in r["neg_ids"])


def test_corpus_de_um_plano_so_cai_para_outras_paginas(tmp_path: Path):
    root = _corpus(tmp_path, planos=("BR001",), paginas=2)

    linhas = make_pairs(root, tmp_path / "pairs.jsonl", generate_queries=_mock_gen(),
                        k=1, neg=3)

    for r in linhas:
        assert r["neg_ids"] and r["pos_id"] not in r["neg_ids"]  # so sobra 1 candidata


def test_segunda_rodada_nao_duplica_nem_rechama_o_gerador(tmp_path: Path):
    root = _corpus(tmp_path)
    out = tmp_path / "pairs.jsonl"
    chamadas: list[str] = []

    primeira = make_pairs(root, out, generate_queries=_mock_gen(chamadas), k=2, neg=2)
    assert len(chamadas) == 4                        # uma chamada por pagina

    segunda = make_pairs(root, out, generate_queries=_mock_gen(chamadas), k=2, neg=2)

    assert len(chamadas) == 4                        # cache em disco: gerador nao rodou
    assert segunda == primeira
    ids = [r["id"] for r in _read_pairs(out)]
    assert len(ids) == len(set(ids)) == len(primeira)


def test_force_regera_mas_continua_sem_duplicar(tmp_path: Path):
    root = _corpus(tmp_path)
    out = tmp_path / "pairs.jsonl"
    chamadas: list[str] = []
    primeira = make_pairs(root, out, generate_queries=_mock_gen(chamadas), k=2, neg=2)

    segunda = make_pairs(root, out, generate_queries=_mock_gen(chamadas), k=2, neg=2,
                         force=True)

    assert len(chamadas) == 8                        # regerou as 4 paginas
    assert [r["id"] for r in segunda] == [r["id"] for r in primeira]


def test_limit_restringe_paginas_mas_negativos_vem_do_corpus_inteiro(tmp_path: Path):
    root = _corpus(tmp_path)

    linhas = make_pairs(root, tmp_path / "pairs.jsonl", generate_queries=_mock_gen(),
                        k=1, neg=2, limit=1)

    assert {r["pos_id"] for r in linhas} == {"BR001_p00"}
    assert linhas[0]["neg_ids"]                      # negativos do BR002, fora do recorte


def test_limit_depois_do_cache_so_acrescenta(tmp_path: Path):
    root = _corpus(tmp_path)
    out = tmp_path / "pairs.jsonl"
    make_pairs(root, out, generate_queries=_mock_gen(), k=1, neg=2, limit=1)

    linhas = make_pairs(root, out, generate_queries=_mock_gen(), k=1, neg=2)

    assert len(linhas) == 4
    assert len({r["id"] for r in linhas}) == 4


def test_contexto_traz_textos_do_plano_via_qrels(tmp_path: Path):
    root = _corpus(tmp_path)

    ctxs = load_contexts(root, k=3)

    assert [c.page_id for c in ctxs][:2] == ["BR001_p00", "BR001_p01"]
    assert ctxs[0].plan_id == "BR001"
    assert ctxs[0].k == 3
    assert ctxs[0].plan_texts == ("Plano BR001: saude e educacao",)


def test_gerador_default_e_deterministico_e_usa_o_contexto():
    ctx = PageContext(page_id="BR001_p00", image_path="p.png", plan_id="BR001", k=3,
                      plan_texts=("Saude publica e mais UBS",))

    qs = template_generate_queries(ctx)

    assert qs == template_generate_queries(ctx)
    assert len(qs) == 3 and len(set(qs)) == 3
    assert all("Saude publica e mais UBS" in q for q in qs)
    assert any("BR001" in q for q in qs)


def test_gerador_default_sem_texto_nenhum_ainda_gera_consulta():
    ctx = PageContext(page_id="BR001_p00", image_path="p.png", plan_id="BR001", k=2)

    qs = template_generate_queries(ctx)

    assert len(qs) == 2 and all(q.strip() for q in qs)


def test_corpus_ausente_falha_claro(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_contexts(tmp_path)
