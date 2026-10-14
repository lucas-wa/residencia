from benchmark.common.config import ExperimentConfig
from benchmark.models.factory import is_multi_vector


def test_is_multi_vector():
    base = dict(model_id="x", dataset="d")
    assert is_multi_vector(ExperimentConfig(name="a", model_kind="colqwen2", **base)) is True
    assert is_multi_vector(ExperimentConfig(name="b", model_kind="clip", **base)) is False
    assert is_multi_vector(ExperimentConfig(name="c", model_kind="text", **base)) is False


def test_is_multi_vector_modelos_2026():
    base = dict(model_id="x", dataset="d")
    # late interaction (MaxSim)
    assert is_multi_vector(ExperimentConfig(name="d", model_kind="colqwen2.5", **base)) is True
    assert is_multi_vector(ExperimentConfig(name="e", model_kind="colembed", **base)) is True
    # Qwen3-VL-Embedding e denso (um vetor por item)
    assert is_multi_vector(ExperimentConfig(name="f", model_kind="qwen3vl", **base)) is False


def test_build_retriever_kind_desconhecido():
    cfg = ExperimentConfig(name="z", model_id="x", dataset="d", model_kind="nope")
    try:
        from benchmark.models.factory import build_retriever

        build_retriever(cfg)
    except ValueError as exc:
        assert "nope" in str(exc)
        return
    raise AssertionError("esperava ValueError para model_kind desconhecido")
