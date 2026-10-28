"""Preflight de versoes: testado so com dicts (nunca importa torch/transformers de verdade)."""

import pytest

from benchmark.check_env import analyze, collect, main

OK = {
    # 4.57.6 e o que o requirements-slurm.txt (`transformers>=4.57,<5`) resolve hoje:
    # dentro da faixa do jina-embeddings-v4 (>=4.52,<5), entao o preflight fica limpo.
    "transformers": "4.57.6",
    "colpali_engine": "0.3.10",
    "torch": "2.5.1",
    "huggingface_hub": "0.34.0",
}


def test_colpali_engine_path_quando_versoes_casam():
    r = analyze(OK)
    assert r["status"] == "ok"
    assert r["colqwen_path"] == "colpali-engine"


def test_colpali_engine_path_com_transformers_4_47():
    # colpali-engine compativel + transformers >4.46.1 basta (nao precisa de 4.49).
    # O status global vira warn porque 4.47.1 fica abaixo de TRANSFORMERS_FOR_JINAV4_SPEC
    # (>=4.52): e o aviso do jina-embeddings-v4, nao do caminho do ColQwen2.
    r = analyze({**OK, "transformers": "4.47.1"})
    assert r["colqwen_path"] == "colpali-engine"
    assert any(m.startswith("[ok] ColQwen2/ColPali via colpali-engine")
               for m in r["messages"])


def test_warn_caminho_nativo_sem_colpali_engine():
    r = analyze({**OK, "colpali_engine": None})
    assert r["status"] == "warn"
    assert r["colqwen_path"] == "transformers-native"
    assert any("vidore/colqwen2-v1.0-hf" in m for m in r["messages"])


def test_warn_caminho_nativo_com_colpali_engine_antigo():
    r = analyze({**OK, "colpali_engine": "0.3.4"})          # 0.3.4 NAO satisfaz >0.3.4
    assert r["status"] == "warn"
    assert r["colqwen_path"] == "transformers-native"


def test_fail_sem_transformers():
    r = analyze({**OK, "transformers": None})
    assert r["status"] == "fail"
    assert r["colqwen_path"] == "none"


def test_fail_sem_colpali_engine_e_transformers_antigo():
    r = analyze({"transformers": "4.47.1", "colpali_engine": None, "torch": "2.5.1"})
    assert r["status"] == "fail"
    assert r["colqwen_path"] == "none"


def test_transformers_4_46_1_nao_satisfaz_maior_que_4_46_1():
    # parsing de versao: >4.46.1 e estrito; 4.46.1 fica de fora (e <4.49 => fail)
    r = analyze({**OK, "transformers": "4.46.1"})
    assert r["status"] == "fail"
    assert r["colqwen_path"] == "none"


def test_versao_invalida_nao_explode():
    r = analyze({**OK, "transformers": "nao-e-versao"})
    assert r["status"] == "fail"
    assert any("nao-e-versao" in m for m in r["messages"])


def test_mensagens_listam_versoes_observadas():
    msgs = " | ".join(analyze(OK)["messages"])
    for lib in ("transformers", "colpali_engine", "torch", "huggingface_hub"):
        assert lib in msgs
    assert "4.57.6" in msgs and "0.3.10" in msgs


def test_transformers_5_vira_warn_por_causa_do_jina_v4():
    # transformers 5.x remove ROPE_INIT_FUNCTIONS["default"] -> o custom code do
    # jina-embeddings-v4 estoura KeyError: 'default'. Avisa, nao falha: o ColQwen2
    # (que e o que este preflight decide) continua carregando.
    r = analyze({**OK, "transformers": "5.0.0"})
    assert r["status"] == "warn"
    assert r["colqwen_path"] == "colpali-engine"
    assert any("v3_jinav4" in m and "ROPE_INIT_FUNCTIONS" in m for m in r["messages"])


def test_sem_aviso_de_jina_v4_na_faixa_suportada():
    assert not any("v3_jinav4" in m for m in analyze(OK)["messages"])


def test_torch_ausente_vira_warn():
    r = analyze({**OK, "torch": None})
    assert r["status"] == "warn"
    assert r["colqwen_path"] == "colpali-engine"       # o caminho segue valido
    assert any("torch" in m for m in r["messages"])


def test_aceita_chave_com_hifen():
    r = analyze({"transformers": "4.49.0", "colpali-engine": "0.3.10", "torch": "2.5.1"})
    assert r["colqwen_path"] == "colpali-engine"


def test_dict_vazio_e_fail_sem_excecao():
    r = analyze({})
    assert r["status"] == "fail"
    assert r["colqwen_path"] == "none"


def test_collect_retorna_as_libs_esperadas():
    v = collect()
    for lib in ("transformers", "colpali_engine", "torch", "huggingface_hub",
                "sentence_transformers", "mteb", "open_clip_torch", "pypdfium2", "numpy"):
        assert lib in v
        assert v[lib] is None or isinstance(v[lib], str)


@pytest.mark.parametrize(("versions", "argv", "code"), [
    (OK, [], 0),
    (OK, ["--strict"], 0),
    ({**OK, "colpali_engine": None}, [], 0),            # warn sem --strict passa
    ({**OK, "colpali_engine": None}, ["--strict"], 1),  # warn com --strict falha
    ({**OK, "transformers": None}, [], 1),              # fail sempre falha
])
def test_main_exit_codes(monkeypatch, capsys, versions, argv, code):
    monkeypatch.setattr("benchmark.check_env.collect", lambda: versions)
    assert main(argv) == code
    assert capsys.readouterr().out.strip()               # imprimiu o relatorio
