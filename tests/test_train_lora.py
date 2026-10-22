"""Treinador LoRA/QLoRA (`finetune/train_lora.py`) -- tudo OFFLINE.

Nenhum teste carrega modelo, toca a rede ou dispara treino: o data loader e puro Python,
a resolucao de base/metodo e um dicionario e o `--dry-run` tolera torch/colpali/
bitsandbytes ausentes (o caso real deste venv) ou mockados.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from finetune import train_lora as T


def _pairs_file(tmp_path: Path, n: int = 4, negs: int = 2) -> Path:
    """pairs.jsonl no formato de finetune/make_pairs.py, com as imagens no disco."""
    pages = tmp_path / "pages"
    pages.mkdir(exist_ok=True)
    for i in range(n + negs + 1):
        (pages / f"BR_p{i:02d}.png").write_bytes(b"png")
    linhas = []
    for i in range(n):
        neg_ids = [f"BR_p{(i + 1 + j) % (n + negs):02d}" for j in range(negs)]
        linhas.append({
            "id": f"id{i:03d}",
            "query": f"o que o plano propoe sobre tema {i}?",
            "pos_id": f"BR_p{i:02d}",
            "pos_image": str(pages / f"BR_p{i:02d}.png"),
            "neg_ids": neg_ids,
            "neg_images": [str(pages / f"{p}.png") for p in neg_ids],
            "label": 1,
            "generator": "template",
        })
    out = tmp_path / "pairs.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in linhas),
                   encoding="utf-8")
    return out


# -- data loader ----------------------------------------------------------------------


def test_load_pairs_le_o_formato_de_make_pairs(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=3))

    assert [p.id for p in pares] == ["id000", "id001", "id002"]
    assert pares[0].label == 1 and pares[0].generator == "template"
    assert len(pares[0].neg_images) == len(pares[0].neg_ids) == 2
    assert Path(pares[0].pos_image).exists()


def test_load_pairs_limit_e_dedup_por_id(tmp_path):
    arq = _pairs_file(tmp_path, n=4)
    repetida = json.loads(arq.read_text(encoding="utf-8").splitlines()[0])
    with arq.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(repetida) + "\n")

    assert len(T.load_pairs(arq)) == 4                 # a linha repetida nao duplica
    assert len(T.load_pairs(arq, limit=2)) == 2


def test_load_pairs_sem_arquivo_falha_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_pairs"):
        T.load_pairs(tmp_path / "nao_existe.jsonl")


def test_load_pairs_arquivo_vazio_falha(tmp_path):
    vazio = tmp_path / "pairs.jsonl"
    vazio.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nenhum par"):
        T.load_pairs(vazio)


@pytest.mark.parametrize("obj,erro", [
    ({"id": "a", "query": "", "pos_id": "p", "pos_image": "x.png"}, "campos ausentes"),
    ({"id": "a", "query": "q", "pos_image": "x.png"}, "campos ausentes"),
    ({"id": "a", "query": "q", "pos_id": "p", "pos_image": "x.png",
      "neg_ids": ["n1", "n2"], "neg_images": ["n1.png"]}, "neg_ids"),
])
def test_parse_pair_valida_o_contrato(obj, erro):
    with pytest.raises(ValueError, match=erro):
        T.parse_pair(obj, lineno=7)


def test_load_pairs_json_invalido_aponta_a_linha(tmp_path):
    arq = tmp_path / "pairs.jsonl"
    arq.write_text('{"id": "a"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="linha 1"):
        T.load_pairs(arq)


def test_missing_images_reporta_o_que_falta(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=2))
    assert T.missing_images(pares) == []

    Path(pares[0].pos_image).unlink()
    assert T.missing_images(pares) == [pares[0].pos_image]


# -- batches --------------------------------------------------------------------------


def test_batch_poe_os_positivos_primeiro_e_depois_os_negativos(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=4))

    batch = T.make_batch(pares[:2], max_neg=2)

    assert batch.queries == tuple(p.query for p in pares[:2])
    assert batch.images[:2] == (pares[0].pos_image, pares[1].pos_image)
    assert batch.n_pos == 2 and batch.n_neg == len(batch.images) - 2
    assert len(set(batch.images)) == len(batch.images)      # sem imagem repetida


def test_batch_descarta_negativo_que_e_positivo_de_outro_exemplo(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=4))
    # id000 tem BR_p01 como negativo, e BR_p01 e o positivo de id001
    assert pares[1].pos_image in pares[0].neg_images

    batch = T.make_batch(pares[:2], max_neg=2)

    assert batch.images.count(pares[1].pos_image) == 1


def test_max_neg_zero_deixa_so_in_batch_negatives(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=3))

    batch = T.make_batch(pares, max_neg=0)

    assert batch.n_neg == 0
    assert batch.images == tuple(p.pos_image for p in pares)


def test_iter_batches_cobre_todos_os_pares_e_o_ultimo_batch_e_menor(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=5))

    batches = list(T.iter_batches(pares, batch_size=2, max_neg=1))

    assert [b.n_pos for b in batches] == [2, 2, 1]
    vistos = [p.id for b in batches for p in b.pairs]
    assert sorted(vistos) == sorted(p.id for p in pares)


def test_iter_batches_shuffle_e_deterministico_por_seed(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=5))

    def ordem(seed):
        return [p.id for b in T.iter_batches(pares, 2, shuffle=True, seed=seed)
                for p in b.pairs]

    assert ordem(1) == ordem(1)
    assert sorted(ordem(1)) == sorted(p.id for p in pares)


def test_iter_batches_rejeita_batch_size_invalido(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=2))
    with pytest.raises(ValueError, match="batch_size"):
        list(T.iter_batches(pares, batch_size=0))


# -- base / metodo / estimativas ------------------------------------------------------


def test_resolve_config_resolve_base_e_metodo():
    cfg = T.resolve_config("colqwen2.5", "lora")
    assert cfg.spec.model_id == "vidore/colqwen2.5-v0.2" and cfg.spec.kind == "colqwen2.5"

    cfg8b = T.resolve_config("colembed8b", "qlora")
    assert cfg8b.spec.params_b == 8.0 and cfg8b.spec.kind == "colembed"


@pytest.mark.parametrize("base,metodo,erro", [
    ("colqwen3", "lora", "base desconhecida"),
    ("colqwen2.5", "full", "metodo desconhecido"),
])
def test_resolve_config_rejeita_base_ou_metodo_invalido(base, metodo, erro):
    with pytest.raises(ValueError, match=erro):
        T.resolve_config(base, metodo)


@pytest.mark.parametrize("campo", ["batch_size", "epochs", "grad_accum"])
def test_resolve_config_rejeita_tamanho_zero(campo):
    with pytest.raises(ValueError, match=campo):
        T.resolve_config("colqwen2.5", "lora", **{campo: 0})


def test_warmup_cosine_sobe_e_desce_sem_cantos(tmp_path):
    f = T.warmup_cosine(total_steps=100, warmup_ratio=0.1)

    assert 0 < f(0) < f(5) <= 1.0                     # warmup linear
    assert f(10) == pytest.approx(1.0)                # pico ao fim do warmup
    assert f(99) < 0.05                               # cosseno chegando a zero
    assert all(0.0 <= f(p) <= 1.0 for p in range(120))

    um_passo = T.warmup_cosine(total_steps=1)         # smoke: nao pode estourar
    assert 0.0 <= um_passo(0) <= 1.0


def test_out_path_default_nomeia_por_base_e_metodo():
    assert T.resolve_config("colembed8b", "qlora").out_path.name == "colembed8b-qlora"
    assert T.resolve_config("colqwen2.5", "lora", output_dir="x/y").out_path == Path("x/y")


def test_count_steps_respeita_grad_accum_e_max_steps():
    cfg = T.resolve_config("colqwen2.5", "lora", batch_size=2, grad_accum=4, epochs=2)
    assert T.count_steps(10, cfg) == (5, 3)          # 5 batches/epoca, 10/4 -> 3 passos
    cfg2 = T.resolve_config("colqwen2.5", "lora", batch_size=2, grad_accum=4, epochs=2,
                            max_steps=2)
    assert T.count_steps(10, cfg2)[1] == 2


def test_estimate_lora_params_segue_a_formula_a_mais_b():
    spec = T.BASES["colqwen2.5"]
    alvos = ("q_proj",)

    esperado = 8 * (spec.hidden + spec.heads * spec.head_dim) * spec.layers

    assert T.estimate_lora_params(spec, r=8, target_modules=alvos) == esperado
    # dobrar o rank dobra os parametros; mais alvos so acrescentam
    assert T.estimate_lora_params(spec, 16, alvos) == 2 * esperado
    assert T.estimate_lora_params(spec, 8, ("q_proj", "o_proj")) > esperado


def test_estimate_lora_params_ignora_modulo_desconhecido():
    spec = T.BASES["colqwen2.5"]
    assert (T.estimate_lora_params(spec, 8, ("q_proj", "nao_existe"))
            == T.estimate_lora_params(spec, 8, ("q_proj",)))


def test_estimate_vram_qlora_pesa_menos_que_lora_e_8b_mais_que_3b():
    lora3b = T.estimate_vram_gb(T.resolve_config("colqwen2.5", "lora"), 60_000_000)
    qlora3b = T.estimate_vram_gb(T.resolve_config("colqwen2.5", "qlora"), 60_000_000)
    qlora8b = T.estimate_vram_gb(T.resolve_config("colembed8b", "qlora"), 87_000_000)

    assert qlora3b["pesos"] < lora3b["pesos"]
    assert qlora3b["total"] < qlora8b["total"] < 24.0        # go/no-go da L4 24 GB
    assert lora3b["total"] == pytest.approx(
        sum(lora3b[k] for k in ("pesos", "lora_otimizador", "ativacoes", "overhead")), abs=0.02)


def test_missing_deps_depende_do_metodo_e_da_base():
    todas = {"torch": "2.9", "transformers": "5.0", "peft": "0.11",
             "bitsandbytes": None, "colpali_engine": "0.3.10"}

    assert T.missing_deps(T.resolve_config("colqwen2.5", "lora"), todas) == []
    assert T.missing_deps(T.resolve_config("colqwen2.5", "qlora"), todas) == ["bitsandbytes"]
    # ColEmbed carrega por AutoModel(trust_remote_code): nao precisa do colpali-engine
    sem_colpali = {**todas, "colpali_engine": None}
    assert T.missing_deps(T.resolve_config("colembed8b", "lora"), sem_colpali) == []


# -- dry run --------------------------------------------------------------------------


def _deps_ok(monkeypatch):
    monkeypatch.setattr(T, "available_deps", lambda: {
        "torch": "2.9.1", "transformers": "5.0.0", "peft": "0.11.0",
        "bitsandbytes": "0.43.0", "colpali_engine": "0.3.10"})


def test_dry_run_valida_dados_e_estima_sem_nenhuma_lib(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "available_deps", lambda: dict.fromkeys(
        ("torch", "transformers", "peft", "bitsandbytes", "colpali_engine")))
    cfg = T.resolve_config("colqwen2.5", "lora", pairs=str(_pairs_file(tmp_path, n=4)),
                           batch_size=2, max_neg=2)

    rel = T.dry_run(cfg)

    assert rel.n_pairs == 4 and rel.n_batches == 2
    assert rel.batch_exemplo["queries"] == 2 and rel.batch_exemplo["negativos"] >= 1
    assert rel.trainable_estimado > 0 and rel.vram_gb["total"] > 0
    assert set(rel.deps_faltando) == {"torch", "transformers", "peft", "colpali_engine"}
    assert any("seria" in a and "vidore/colqwen2.5-v0.2" in a for a in rel.avisos)
    assert rel.trainable_real is None                 # nada foi carregado


def test_dry_run_com_deps_presentes_nao_reclama(tmp_path, monkeypatch):
    _deps_ok(monkeypatch)
    cfg = T.resolve_config("colembed8b", "qlora", pairs=str(_pairs_file(tmp_path, n=3)))

    rel = T.dry_run(cfg)

    assert rel.deps_faltando == [] and rel.avisos == []
    assert rel.model_id == "nvidia/nemotron-colembed-vl-8b-v2"


def test_dry_run_avisa_imagem_ausente(tmp_path, monkeypatch):
    _deps_ok(monkeypatch)
    arq = _pairs_file(tmp_path, n=2)
    Path(json.loads(arq.read_text(encoding="utf-8").splitlines()[0])["pos_image"]).unlink()
    cfg = T.resolve_config("colqwen2.5", "lora", pairs=str(arq))

    rel = T.dry_run(cfg)

    assert rel.imagens_faltando and any("imagens ausentes" in a for a in rel.avisos)


def test_dry_run_load_model_embrulha_o_lora_e_reporta_o_real(tmp_path, monkeypatch):
    _deps_ok(monkeypatch)
    chamadas = []

    class FakeModel:
        def parameters(self):
            return iter(())

    monkeypatch.setattr(T, "load_base_model", lambda cfg: (chamadas.append("load") or
                                                           (FakeModel(), object())))
    monkeypatch.setattr(T, "wrap_lora", lambda m, cfg: chamadas.append("lora") or m)
    monkeypatch.setattr(T, "trainable_parameters", lambda m: (59_000_000, 3_000_000_000))
    cfg = T.resolve_config("colqwen2.5", "lora", pairs=str(_pairs_file(tmp_path, n=2)))

    rel = T.dry_run(cfg, load_model=True)

    assert chamadas == ["load", "lora"]
    assert (rel.trainable_real, rel.total_real) == (59_000_000, 3_000_000_000)


def test_dry_run_load_model_sem_deps_so_avisa(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "available_deps", lambda: dict.fromkeys(
        ("torch", "transformers", "peft", "bitsandbytes", "colpali_engine")))
    monkeypatch.setattr(T, "load_base_model", lambda cfg: pytest.fail("nao pode carregar"))
    cfg = T.resolve_config("colqwen2.5", "lora", pairs=str(_pairs_file(tmp_path, n=2)))

    rel = T.dry_run(cfg, load_model=True)

    assert any("--load-model ignorado" in a for a in rel.avisos)


def test_format_report_mostra_o_essencial(tmp_path, monkeypatch):
    _deps_ok(monkeypatch)
    cfg = T.resolve_config("colqwen2.5", "lora", pairs=str(_pairs_file(tmp_path, n=2)))

    texto = T.format_report(T.dry_run(cfg))

    assert "DRY RUN" in texto and "SAINDO sem treinar" in texto
    assert "vidore/colqwen2.5-v0.2" in texto and "VRAM estimada" in texto


# -- CLI ------------------------------------------------------------------------------


def test_help_nao_importa_nada_pesado(capsys):
    with pytest.raises(SystemExit) as e:
        T.main(["--help"])
    assert e.value.code == 0
    saida = capsys.readouterr().out
    assert "--dry-run" in saida and "qlora" in saida


def test_main_dry_run_nunca_chama_train(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(T, "train", lambda *a, **kw: pytest.fail("treino disparado!"))
    monkeypatch.setattr(T, "run_benchmark_eval",
                        lambda *a, **kw: pytest.fail("avaliacao disparada!"))

    rc = T.main(["--dry-run", "--pairs", str(_pairs_file(tmp_path, n=4)),
                 "--base", "colembed8b", "--method", "qlora", "--batch-size", "2"])

    assert rc == 0
    saida = capsys.readouterr().out
    assert "DRY RUN" in saida and "nemotron-colembed-vl-8b-v2" in saida


def test_main_dry_run_json(tmp_path, capsys):
    rc = T.main(["--dry-run", "--json", "--pairs", str(_pairs_file(tmp_path, n=3))])

    assert rc == 0
    rel = json.loads(capsys.readouterr().out)
    assert rel["n_pairs"] == 3 and rel["config"]["method"] == "lora"
    assert rel["trainable_estimado"] > 0


def test_main_dry_run_sem_pairs_sai_1(tmp_path):
    assert T.main(["--dry-run", "--pairs", str(tmp_path / "nao_existe.jsonl")]) == 1


def test_main_base_invalida_sai_2(capsys):
    with pytest.raises(SystemExit):              # argparse barra antes, com codigo 2
        T.main(["--dry-run", "--base", "inexistente"])


def test_main_target_modules_vira_tupla(tmp_path, capsys):
    T.main(["--dry-run", "--json", "--pairs", str(_pairs_file(tmp_path, n=2)),
            "--target-modules", "q_proj, v_proj", "--lora-r", "8"])

    rel = json.loads(capsys.readouterr().out)
    assert rel["config"]["target_modules"] == ["q_proj", "v_proj"]
    assert rel["trainable_estimado"] == T.estimate_lora_params(
        T.BASES["colqwen2.5"], 8, ("q_proj", "v_proj"))


# -- hook de avaliacao ----------------------------------------------------------------


def test_run_benchmark_eval_chama_run_vidore_e_restaura_argv(tmp_path, monkeypatch):
    import benchmark

    visto = {}
    fake = types.ModuleType("benchmark.run_vidore")
    fake.main = lambda: visto.setdefault("argv", list(sys.argv))
    monkeypatch.setitem(sys.modules, "benchmark.run_vidore", fake)
    monkeypatch.setattr(sys.modules["benchmark"], "run_vidore", fake, raising=False)
    cfg_yaml = tmp_path / "eval.yaml"
    cfg_yaml.write_text("name: x\n", encoding="utf-8")
    argv_antes = list(sys.argv)

    assert T.run_benchmark_eval(cfg_yaml, limit=2, tag="antes") is True

    assert visto["argv"] == ["run_vidore", "--config", str(cfg_yaml), "--limit", "2"]
    assert sys.argv == argv_antes


def test_run_benchmark_eval_sem_config_so_avisa(tmp_path):
    assert T.run_benchmark_eval(tmp_path / "nao_existe.yaml") is False


def test_run_benchmark_eval_nao_propaga_erro_do_benchmark(tmp_path, monkeypatch):
    import benchmark

    def explode():
        raise RuntimeError("dataset indisponivel")

    fake = types.ModuleType("benchmark.run_vidore")
    fake.main = explode
    monkeypatch.setitem(sys.modules, "benchmark.run_vidore", fake)
    monkeypatch.setattr(sys.modules["benchmark"], "run_vidore", fake, raising=False)
    cfg_yaml = tmp_path / "eval.yaml"
    cfg_yaml.write_text("name: x\n", encoding="utf-8")

    assert T.run_benchmark_eval(cfg_yaml) is False       # avisa, nao derruba o treino


def test_write_after_config_aponta_para_o_adaptador(tmp_path):
    import yaml

    base = tmp_path / "v3_colqwen25.yaml"
    base.write_text("name: v3_colqwen25\nmodel_id: vidore/colqwen2.5-v0.2\n"
                    "model_kind: colqwen2.5\ndataset: vidore/vidore_v3_hr\n",
                    encoding="utf-8")
    adapter = tmp_path / "run"
    adapter.mkdir()
    cfg = T.resolve_config("colqwen2.5", "lora", eval_config=str(base))

    destino = T.write_after_config(cfg, adapter)

    dados = yaml.safe_load(destino.read_text(encoding="utf-8"))
    assert destino == adapter / "eval_after.yaml"
    assert dados["model_id"] == str(adapter)
    assert dados["name"] == "v3_colqwen25_ft_lora"
    assert dados["model_kind"] == "colqwen2.5"          # o resto do config nao muda


def test_write_after_config_sem_base_devolve_none(tmp_path):
    cfg = T.resolve_config("colqwen2.5", "lora", eval_config=str(tmp_path / "nao_existe.yaml"))
    assert T.write_after_config(cfg, tmp_path) is None
