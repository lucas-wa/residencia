from benchmark.common.config import ExperimentConfig
from benchmark.models.factory import is_multi_vector


def test_is_multi_vector():
    base = dict(model_id="x", dataset="d")
    assert is_multi_vector(ExperimentConfig(name="a", model_kind="colqwen2", **base)) is True
    assert is_multi_vector(ExperimentConfig(name="b", model_kind="clip", **base)) is False
    assert is_multi_vector(ExperimentConfig(name="c", model_kind="text", **base)) is False
