"""Os YAMLs do trio SOTA 2026 carregam e batem com o dispatch do factory."""

from pathlib import Path

import pytest

from benchmark.common.config import ExperimentConfig
from benchmark.models.factory import is_multi_vector

CONFIGS = Path(__file__).resolve().parents[1] / "benchmark" / "configs"

V3 = ["v3_colembed8b.yaml", "v3_qwen3vl8b.yaml", "v3_colqwen25.yaml",
      # Semana 7: par de continuidade (v2 -> V3) + jina-v4.
      "v3_colpali.yaml", "v3_colqwen2.yaml", "v3_jinav4.yaml"]
GOVEXE = ["govexe_colembed8b.yaml", "govexe_qwen3vl8b.yaml", "govexe_colqwen25.yaml"]

EXPECTED_KIND = {
    "colembed8b": ("colembed", True),
    "qwen3vl8b": ("qwen3vl", False),
    "colqwen25": ("colqwen2.5", True),
    "colpali": ("colpali", True),
    "colqwen2": ("colqwen2", True),
    # jina-v4 roda denso neste wrapper (o multi-vetor fica para depois)
    "jinav4": ("jinav4", False),
}


@pytest.mark.parametrize("fname", V3 + GOVEXE)
def test_config_carrega_e_dispara_o_kind_certo(fname):
    cfg = ExperimentConfig.from_yaml(CONFIGS / fname)
    key = fname.replace(".yaml", "").split("_", 1)[1]
    kind, multi = EXPECTED_KIND[key]
    assert cfg.model_kind == kind
    assert is_multi_vector(cfg) is multi
    assert cfg.model_id
    assert cfg.device == "cuda"


@pytest.mark.parametrize("fname", V3)
def test_configs_v3_apontam_para_o_vidore_v3(fname):
    cfg = ExperimentConfig.from_yaml(CONFIGS / fname)
    assert cfg.dataset.startswith("vidore/vidore_v3")
    assert 10 in cfg.k_values


@pytest.mark.parametrize("v3,v2", [("v3_colpali.yaml", "vidore_colpali.yaml"),
                                   ("v3_colqwen2.yaml", "vidore_colqwen2.yaml")])
def test_par_de_continuidade_mantem_o_mesmo_checkpoint_do_v2(v3, v2):
    """O par de continuidade so muda de dataset (v2 -> V3): mesmo checkpoint e kind."""
    novo = ExperimentConfig.from_yaml(CONFIGS / v3)
    antigo = ExperimentConfig.from_yaml(CONFIGS / v2)
    assert novo.model_id == antigo.model_id
    assert novo.model_kind == antigo.model_kind


@pytest.mark.parametrize("fname", GOVEXE)
def test_configs_govexe_apontam_para_o_conjunto_ptbr(fname):
    cfg = ExperimentConfig.from_yaml(CONFIGS / fname)
    assert cfg.dataset == "govexe_ptbr_planos"


@pytest.mark.parametrize("fname", ["v3_colembed8b.yaml", "v3_qwen3vl8b.yaml",
                                   "govexe_colembed8b.yaml", "govexe_qwen3vl8b.yaml"])
def test_batch_size_pequeno_para_8b(fname):
    cfg = ExperimentConfig.from_yaml(CONFIGS / fname)
    assert cfg.batch_size <= 2
