import json
from pathlib import Path

import pytest

from ensemble.records import PointerRecord, save_records
from finetune.make_pairs import (
    GENERATORS,
    PageContext,
    build_query_prompt,
    contexts_from_records,
    filter_contexts,
    llm_generate_queries,
    load_contexts,
    load_doc_filter,
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
        assert set(r) == {"id", "query", "pos_id", "pos_modality", "pos_image", "pos_text",
                          "neg_ids", "neg_images", "label", "generator"}
        assert r["label"] == 1
        assert r["pos_modality"] == "imagem" and r["pos_text"] == ""
        assert r["query"].startswith("consulta ")
        assert Path(r["pos_image"]).name == f"{r['pos_id']}.png"
        assert len(r["neg_images"]) == len(r["neg_ids"]) == 2
        assert all(Path(p).exists() for p in [r["pos_image"], *r["neg_images"]])


def test_make_pairs_emite_texto_e_imagem(tmp_path: Path):
    """Corpus v4: o positivo pode ser a PAGINA (imagem) ou o texto do item (`a_repr`)."""
    root = tmp_path / "corpus_v4"
    root.mkdir()
    png = root / "BR001_p00.png"
    png.write_bytes(b"png")
    corpus = [
        {"id": "BR001_p00", "image_path": str(png)},
        {"id": "TX001", "a_repr": "o plano preve mais UBS no interior"},
    ]
    root.joinpath("corpus.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in corpus), encoding="utf-8")

    linhas = make_pairs(root, tmp_path / "pairs.jsonl", generate_queries=_mock_gen(),
                        k=1, neg=1)

    por_modalidade = {r["pos_modality"]: r for r in linhas}
    assert set(por_modalidade) == {"imagem", "texto"}
    imagem = por_modalidade["imagem"]
    assert imagem["pos_id"] == "BR001_p00" and imagem["pos_image"] == str(png)
    assert not imagem["pos_text"]
    texto = por_modalidade["texto"]
    assert texto["pos_id"] == "TX001"
    assert texto["pos_text"] == "o plano preve mais UBS no interior"
    assert not texto["pos_image"]
    # negativos continuam sendo caminhos de imagem: nenhum vazio escapa para o jsonl
    assert all(p for r in linhas for p in r["neg_images"])


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


def test_llm_generator_usa_chat_fn_injetado():
    """O backend e injetado: o gerador so monta o prompt e limpa a resposta."""
    prompts: list[str] = []

    def duble(prompt: str) -> str:
        prompts.append(prompt)
        return ("1. quais as propostas de saude do plano BR001?\n"
                "   \n"
                "- quais as propostas de saude do plano BR001?\n"
                "2) o plano preve creches em tempo integral?\n")

    ctx = PageContext(page_id="BR001_p00", image_path="p.png", plan_id="BR001", k=3,
                      text="o plano preve mais UBS no interior")

    qs = llm_generate_queries(ctx, chat_fn=duble)

    assert qs == ["quais as propostas de saude do plano BR001?",
                  "o plano preve creches em tempo integral?"]
    assert len(prompts) == 1
    assert "BR001" in prompts[0] and "mais UBS no interior" in prompts[0]
    assert "3" in prompts[0]                         # pede ctx.k consultas


def test_llm_generator_respeita_k():
    def duble(_prompt: str) -> str:
        return "\n".join(f"consulta {i}" for i in range(10))

    ctx = PageContext(page_id="BR001_p00", image_path="p.png", plan_id="BR001", k=2)

    assert llm_generate_queries(ctx, chat_fn=duble) == ["consulta 0", "consulta 1"]


def test_llm_generator_sem_backend_erra():
    ctx = PageContext(page_id="BR001_p00", image_path="p.png", plan_id="BR001", k=2)

    with pytest.raises((RuntimeError, ValueError)) as exc:
        llm_generate_queries(ctx)

    assert not isinstance(exc.value, NotImplementedError)
    assert "chat_fn" in str(exc.value)


def test_generators_llm_aponta_para_o_gerador_de_llm():
    assert GENERATORS["llm"] is llm_generate_queries


def _records_v4(image_path: str = "/corpus/bin/documento/documento-001_p00.png"):
    """Um record por modalidade do corpus v4 (`ensemble.records.PointerRecord`)."""
    return [
        PointerRecord(id="documento-001", modality="documento",
                      descriptor="OCR: metas de saude do plano plurianual",
                      pointer_uri="pdf:ppa#1", source="pdf",
                      meta={"path": image_path, "a_repr": image_path}),
        PointerRecord(id="imagem-002", modality="imagem",
                      descriptor="foto da inauguracao de uma UBS no interior",
                      pointer_uri="http://exemplo/ubs.jpg", source="galeria",
                      meta={"path": "/corpus/bin/imagem/ubs.jpg",
                            "a_repr": "/corpus/bin/imagem/ubs.jpg"}),
        PointerRecord(id="video-003", modality="video",
                      descriptor="trecho do debate sobre creches em tempo integral",
                      pointer_uri="http://exemplo/debate#t=30", source="youtube",
                      meta={"a_repr": "/corpus/bin/video/debate_t30.png"}),
        PointerRecord(id="texto-004", modality="texto",
                      descriptor="noticia sobre o orcamento da educacao",
                      pointer_uri="http://exemplo/noticia", source="noticia",
                      meta={"a_repr": "o orcamento da educacao cresce 7% em 2026"}),
        PointerRecord(id="audio-005", modality="audio",
                      descriptor="transcricao ASR do programa de radio",
                      pointer_uri="http://exemplo/radio.mp3", source="radio",
                      meta={"a_repr": "a prefeitura anunciou mais leitos de UTI"}),
    ]


def test_contexts_from_records_segue_a_rota_do_baseline_a():
    """Mesmo roteamento do A: documento/imagem/video -> imagem; texto/audio -> texto."""
    ctxs = contexts_from_records(_records_v4(), k=2)

    por_id = {c.page_id: c for c in ctxs}
    assert len(ctxs) == len(por_id) == 5
    assert all(c.k == 2 for c in ctxs)

    documento = por_id["documento-001"]
    assert documento.modality == "imagem"
    assert documento.image_path == "/corpus/bin/documento/documento-001_p00.png"
    assert documento.pos_value == documento.image_path       # positivo = a pagina
    assert documento.text == "OCR: metas de saude do plano plurianual"   # contexto = OCR
    assert documento.plan_id == "pdf"                        # `source` do record

    assert por_id["imagem-002"].modality == "imagem"
    assert por_id["imagem-002"].image_path == "/corpus/bin/imagem/ubs.jpg"
    assert por_id["video-003"].modality == "imagem"
    assert por_id["video-003"].image_path == "/corpus/bin/video/debate_t30.png"
    assert por_id["video-003"].text == "trecho do debate sobre creches em tempo integral"

    texto = por_id["texto-004"]
    assert texto.modality == "texto"
    assert texto.image_path == ""
    assert texto.text == "o orcamento da educacao cresce 7% em 2026"
    assert texto.pos_value == texto.text                     # positivo = o texto
    audio = por_id["audio-005"]
    assert audio.modality == "texto"
    assert audio.text == "a prefeitura anunciou mais leitos de UTI"
    assert audio.pos_value == audio.text


def test_item_de_imagem_leva_o_descriptor_para_o_prompt():
    """O gap que isso fecha: item image-route gerava prompt sem contexto nenhum."""
    ctx = contexts_from_records(_records_v4())[0]

    prompt = build_query_prompt(ctx)

    assert ctx.text and ctx.text.strip()
    assert "metas de saude do plano plurianual" in prompt
    assert "sem texto extraido" not in prompt


def test_contexts_from_records_sem_source_cai_para_v4():
    rec = PointerRecord(id="texto-009", modality="texto", descriptor="d",
                        pointer_uri="http://x", source="", meta={"a_repr": "texto do artigo"})

    assert contexts_from_records([rec])[0].plan_id == "v4"


def test_make_pairs_le_records_jsonl_do_corpus_v4(tmp_path: Path):
    """Entrada canonica do v4: `--records` em vez do corpus.jsonl do benchmark."""
    png = tmp_path / "documento-001_p00.png"
    png.write_bytes(b"png")
    records = tmp_path / "records.jsonl"
    save_records(_records_v4(image_path=str(png)), records)
    out = tmp_path / "pairs.jsonl"

    linhas = make_pairs(records=records, out=out, generate_queries=_mock_gen(), k=1, neg=2)

    assert _read_pairs(out) == linhas
    por_id = {r["pos_id"]: r for r in linhas}
    assert set(por_id) == {"documento-001", "imagem-002", "video-003",
                           "texto-004", "audio-005"}
    documento = por_id["documento-001"]
    assert documento["pos_modality"] == "imagem"
    assert documento["pos_image"] == str(png) and documento["pos_text"] == ""
    audio = por_id["audio-005"]
    assert audio["pos_modality"] == "texto"
    assert audio["pos_text"] == "a prefeitura anunciou mais leitos de UTI"
    assert audio["pos_image"] == ""
    assert all(p for r in linhas for p in r["neg_images"])   # negativo e sempre caminho
    # deterministico: refazer do zero devolve exatamente as mesmas linhas
    assert make_pairs(records=records, out=out, generate_queries=_mock_gen(), k=1, neg=2,
                      force=True) == linhas


def test_make_pairs_sem_corpus_nem_records_erra_claro():
    with pytest.raises(ValueError) as exc:
        make_pairs()

    assert "records" in str(exc.value)


# -- split por DOCUMENTO (--include-docs / --exclude-docs) -----------------------------
#
# O eval do v4 tem 7 queries humanas sobre 22 docs. Para que a historia de vazamento seja
# limpa, NENHUM desses 22 pode ser alvo positivo de treino -- e o conjunto sintetico de
# avaliacao e o recorte inverso (so eles). O que NAO se recorta e o pool de negativos: um
# distrator nao carrega rotulo, e tirar metade do corpus de la so enfraqueceria o InfoNCE.


def test_load_doc_filter_ignora_vazias_e_comentarios(tmp_path: Path):
    arquivo = tmp_path / "eval_docs.txt"
    arquivo.write_text("# docs do eval\ndocumento-001\n\n  video-003  \n", encoding="utf-8")

    assert load_doc_filter(arquivo) == {"documento-001", "video-003"}


def test_filter_contexts_sem_recorte_devolve_tudo():
    ctxs = contexts_from_records(_records_v4())

    assert [c.page_id for c in filter_contexts(ctxs)] == [c.page_id for c in ctxs]


def test_filter_contexts_aplica_include_e_exclude_juntos():
    ctxs = contexts_from_records(_records_v4())

    recorte = filter_contexts(ctxs, include_docs={"documento-001", "video-003"},
                              exclude_docs={"video-003"})

    assert [c.page_id for c in recorte] == ["documento-001"]


def test_make_pairs_exclude_docs_tira_o_doc_do_alvo_mas_nao_do_pool(tmp_path: Path):
    """Nenhum doc de eval vira positivo -- e todos continuam disponiveis como negativo."""
    records = tmp_path / "records.jsonl"
    save_records(_records_v4(), records)
    out = tmp_path / "pairs.jsonl"

    linhas = make_pairs(records=records, out=out, generate_queries=_mock_gen(), k=1, neg=4,
                        exclude_docs={"documento-001", "imagem-002"})

    assert {r["pos_id"] for r in linhas} == {"video-003", "texto-004", "audio-005"}
    # o pool NAO foi recortado: os dois banidos ainda aparecem como distratores
    negativos = {n for r in linhas for n in r["neg_ids"]}
    assert {"documento-001", "imagem-002"} <= negativos


def test_make_pairs_include_docs_gera_so_sobre_os_docs_de_eval(tmp_path: Path):
    """O conjunto SINTETICO de avaliacao: consultas ancoradas so nos docs julgados."""
    records = tmp_path / "records.jsonl"
    save_records(_records_v4(), records)

    linhas = make_pairs(records=records, out=tmp_path / "syn.jsonl",
                        generate_queries=_mock_gen(), k=2, neg=1,
                        include_docs={"documento-001", "audio-005"})

    assert {r["pos_id"] for r in linhas} == {"documento-001", "audio-005"}
    assert len(linhas) == 4                           # k=2 consultas por doc


def test_make_pairs_cli_le_os_arquivos_de_recorte(tmp_path: Path, monkeypatch):
    """`--exclude-docs FILE` chega ao `make_pairs` como conjunto (um doc_id por linha)."""
    import finetune.make_pairs as M

    records = tmp_path / "records.jsonl"
    save_records(_records_v4(), records)
    excluir = tmp_path / "eval_docs.txt"
    excluir.write_text("# derivado do qrels\ndocumento-001\n", encoding="utf-8")
    out = tmp_path / "pairs.jsonl"

    monkeypatch.setattr(M, "resolve_generator", lambda *a, **k: _mock_gen())
    monkeypatch.setattr("sys.argv", ["make_pairs", "--records", str(records),
                                     "--out", str(out), "--k", "1",
                                     "--exclude-docs", str(excluir)])
    M.main()

    assert "documento-001" not in {r["pos_id"] for r in _read_pairs(out)}


def test_make_pairs_injeta_o_chat_fn_no_gerador_llm(tmp_path: Path):
    root = _corpus(tmp_path, planos=("BR001", "BR002"), paginas=1)
    prompts: list[str] = []

    def duble(prompt: str) -> str:
        prompts.append(prompt)
        return "o que o plano diz sobre saude?\no que o plano diz sobre saude?\n\n"

    linhas = make_pairs(root, tmp_path / "pairs.jsonl", generator_name="llm",
                        chat_fn=duble, k=2, neg=1)

    assert len(prompts) == 2                         # uma chamada por pagina
    assert {r["query"] for r in linhas} == {"o que o plano diz sobre saude?"}
    assert {r["generator"] for r in linhas} == {"llm"}
    assert {r["pos_id"] for r in linhas} == {"BR001_p00", "BR002_p00"}
