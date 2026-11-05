"""Registro de espacos (`gcp/slice/spaces.py`): a fonte declarativa de verdade.

O `embed_job --space` e a consulta multi-espaco leem daqui -- entao o que se testa e o
contrato do registro: a composicao de cada arquitetura, as invariantes globais (1 tabela
por espaco, dim dentro do teto do HNSW) e a separacao de modalidades do CLAP.
"""

from gcp.slice.spaces import architecture_spaces, spaces_for_modalities, SPACES


def test_A_tem_um_espaco_e_Aaudio_dois():
    assert [s.name for s in architecture_spaces("A")] == ["qwen3vl"]
    assert [s.name for s in architecture_spaces("A+audio")] == ["qwen3vl", "clap"]


def test_dim_dentro_do_teto_hnsw_e_tabelas_distintas():
    tabs = {s.table for s in SPACES.values()}
    assert len(tabs) == len(SPACES)                      # 1 tabela por espaco
    assert all(s.dim <= 2000 for s in SPACES.values())


def test_clap_so_indexa_audio_native():
    assert SPACES["clap"].modalities == ("audio_native",)


def test_espacos_por_modalidade_saem_do_registro():
    """Quem indexa cada modalidade e o registro -- a avaliacao so pergunta."""
    assert [s.name for s in spaces_for_modalities("A+audio", {"doc_image"})] == ["qwen3vl"]
    assert [s.name for s in spaces_for_modalities("A+audio", {"audio_native"})] == ["clap"]
    assert [s.name for s in spaces_for_modalities("A+audio",
                                                  {"audio_asr", "audio_native"})] \
        == ["qwen3vl", "clap"]                       # na ordem da arquitetura


def test_espacos_por_modalidade_respeitam_a_arquitetura():
    assert [s.name for s in spaces_for_modalities("A", {"audio_asr"})] == ["qwen3vl"]
    assert spaces_for_modalities("A", {"audio_native"}) == []   # A nao tem o CLAP
