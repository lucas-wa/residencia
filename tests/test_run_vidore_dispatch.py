from benchmark.common.datasets import load_vidore_v2, load_vidore_v3
from benchmark.run_vidore import pick_loader


def test_v3_vai_para_o_loader_v3():
    assert pick_loader("vidore/vidore_v3_hr") is load_vidore_v3
    assert pick_loader("vidore/vidore_v3_hr_mteb_format") is load_vidore_v3


def test_v2_continua_no_loader_v2():
    assert pick_loader("vidore/biomedical_lectures_v2") is load_vidore_v2
    assert pick_loader("vidore/esg_reports_v2") is load_vidore_v2
