"""Registro de espacos (`gcp/slice/spaces.py`): a fonte declarativa de verdade.

O `embed_job --space` e a consulta multi-espaco leem daqui -- entao o que se testa e o
contrato do registro: a composicao de cada arquitetura, as invariantes globais (1 tabela
por espaco, dim dentro do teto do HNSW) e a separacao de modalidades do CLAP.
"""

from gcp.slice.spaces import architecture_spaces, SPACES


def test_A_tem_um_espaco_e_Aaudio_dois():
    assert [s.name for s in architecture_spaces("A")] == ["qwen3vl"]
    assert [s.name for s in architecture_spaces("A+audio")] == ["qwen3vl", "clap"]


def test_dim_dentro_do_teto_hnsw_e_tabelas_distintas():
    tabs = {s.table for s in SPACES.values()}
    assert len(tabs) == len(SPACES)                      # 1 tabela por espaco
    assert all(s.dim <= 2000 for s in SPACES.values())


def test_clap_so_indexa_audio_native():
    assert SPACES["clap"].modalities == ("audio_native",)
