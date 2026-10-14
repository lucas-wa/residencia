"""Resolucao do metodo de encode dos checkpoints custom code (sem carregar modelo)."""

from benchmark.models._remote_api import call_batch, first_method


class _Model:
    def forward_queries(self, items, batch_size=8):
        return ("com_batch_size", list(items), batch_size)

    def outro(self, items):
        return ("sem_batch_size", list(items))


def test_first_method_pega_o_primeiro_existente():
    m = _Model()
    assert first_method(m, ("nao_existe", "forward_queries")).__name__ == "forward_queries"
    assert first_method(m, ("nao_existe", "tambem_nao")) is None


def test_first_method_ignora_atributo_nao_chamavel():
    m = _Model()
    m.encode_queries = "nao e funcao"
    assert first_method(m, ("encode_queries", "outro")).__name__ == "outro"


def test_call_batch_passa_batch_size_quando_a_assinatura_aceita():
    tag, items, bs = call_batch(_Model().forward_queries, ["a", "b", "c"])
    assert tag == "com_batch_size" and items == ["a", "b", "c"]
    assert bs == 3            # o lote ja veio batelado pelo wrapper


def test_call_batch_omite_batch_size_quando_nao_ha():
    assert call_batch(_Model().outro, ["a"]) == ("sem_batch_size", ["a"])


def test_call_batch_lote_vazio_nao_manda_batch_size_zero():
    _, _, bs = call_batch(_Model().forward_queries, [])
    assert bs == 1
