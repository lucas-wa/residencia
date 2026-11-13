"""Ferramental do corpus v4: piso por eixo e split treino/avaliacao disjunto."""
import pytest

from ensemble.corpus_v4 import AXES, BalanceError, SplitError, check_balance, tag_split


def _qrels_por_eixo(contagens: dict[str, int]) -> tuple[dict, dict]:
    """Monta (qrels, queries) com `contagens[eixo]` queries julgadas em cada eixo.

    Cada query recebe UM relevante do seu eixo, com o prefixo de modalidade do
    `item_id` (`documento-`, `imagem-`, ...) que o corpus v2/v4 ja usa.
    """
    qrels: dict[str, set[str]] = {}
    queries: dict[str, str] = {}
    for eixo, n in contagens.items():
        for i in range(n):
            qid = f"{eixo}-q{i}"
            qrels[qid] = {f"{eixo}-item{i}"}
            queries[qid] = f"consulta {eixo} {i}"
    return qrels, queries


def test_balance_acusa_eixo_abaixo_do_piso():
    qrels, queries = _qrels_por_eixo(
        {"documento": 8, "imagem": 2, "audio": 8, "texto": 9, "video": 8})
    with pytest.raises(BalanceError, match="imagem"):
        check_balance(qrels, queries, floor=8)


def test_balance_axis_vazio_conta_zero():
    # video sem nenhuma query julgada: entra no dict como 0 (sem KeyError) e reprova.
    qrels, queries = _qrels_por_eixo(
        {"documento": 8, "imagem": 8, "audio": 8, "texto": 8})
    with pytest.raises(BalanceError) as exc:
        check_balance(qrels, queries, floor=8)
    assert "video" in str(exc.value)

    contagens = check_balance(qrels, queries, floor=0)
    assert set(contagens) == set(AXES)
    assert contagens["video"] == 0
    assert contagens["documento"] == 8


def test_balance_passa_quando_todos_no_piso():
    qrels, queries = _qrels_por_eixo(
        {"documento": 8, "imagem": 10, "audio": 8, "texto": 12, "video": 8})
    assert check_balance(qrels, queries) == {
        "documento": 8, "imagem": 10, "audio": 8, "texto": 12, "video": 8,
    }


def test_balance_conta_query_uma_vez_por_eixo_e_ignora_qid_fora_de_queries():
    # query cross-modal conta nos DOIS eixos (criterio do run_eval), mas uma vez em cada;
    # qid presente so nas qrels nao e julgado (nao esta no conjunto de queries avaliadas).
    qrels = {
        "q1": {"imagem-a", "imagem-b", "texto-a"},
        "q2": {"imagem-c"},
        "orfa": {"imagem-z"},
    }
    queries = {"q1": "x", "q2": "y"}
    assert check_balance(qrels, queries, floor=0) == {
        "documento": 0, "imagem": 2, "audio": 0, "texto": 1, "video": 0,
    }


def test_split_disjunto():
    queries = {f"q{i}": f"texto {i}" for i in range(5)}
    split = tag_split(queries, eval_ids=["q1", "q3"])
    assert split["eval"] == ["q1", "q3"]
    assert split["train"] == ["q0", "q2", "q4"]
    assert not set(split["train"]) & set(split["eval"])
    assert set(split["train"]) | set(split["eval"]) == set(queries)


def test_split_eval_id_inexistente_levanta():
    queries = {"q0": "a", "q1": "b"}
    with pytest.raises(SplitError, match="q9"):
        tag_split(queries, eval_ids=["q1", "q9"])


def test_split_id_nos_dois_lados_levanta():
    queries = {"q0": "a", "q1": "b", "q2": "c"}
    with pytest.raises(SplitError, match="q1"):
        tag_split(queries, eval_ids=["q1"], train_ids=["q0", "q1"])


def test_split_train_id_inexistente_levanta():
    queries = {"q0": "a", "q1": "b"}
    with pytest.raises(SplitError, match="q7"):
        tag_split(queries, eval_ids=["q1"], train_ids=["q0", "q7"])


def test_split_erros_sao_valueerror():
    assert issubclass(SplitError, ValueError)
    assert issubclass(BalanceError, ValueError)
