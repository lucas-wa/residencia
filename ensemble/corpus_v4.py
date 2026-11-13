"""Ferramental do corpus v4 balanceado: piso de queries por eixo + split treino/avaliacao.

Duas garantias que o v3 nao tinha e que o fine-tune denso da Semana 10 exige:

- **piso por eixo** (`check_balance`): cada eixo precisa de pelo menos `floor` queries
  JULGADAS (default 8 na Semana 10). "Julgada naquele eixo" segue exatamente o criterio
  do `ensemble.eval.run_eval`: a query tem >=1 item relevante do eixo. Eixo sem nenhuma
  query julgada conta 0 -- nunca KeyError -- e, sendo 0 < piso, reprova.
- **split disjunto** (`tag_split`): treino e avaliacao nunca compartilham um qid. Treinar
  o adapter em query que depois aparece na avaliacao contamina o antes/depois, entao a
  disjuncao e invariante (erro, nao aviso).

O eixo de um item vem do PREFIXO do `item_id` (`documento-`, `imagem-`, `audio-`,
`texto-`, `video-`), que a captura ja escreve (`ensemble/capture/*.py`). Os eixos
canonicos sao os `AXES` do `run_eval` -- fonte unica, reexportada aqui.
"""
from __future__ import annotations

from collections.abc import Iterable

from ensemble.eval.run_eval import AXES

__all__ = ["AXES", "FLOOR", "BalanceError", "SplitError", "axis_of", "check_balance",
           "tag_split"]

# Piso de queries julgadas por eixo na Semana 10.
FLOOR = 8


class BalanceError(ValueError):
    """Algum eixo tem menos queries julgadas que o piso exigido."""


class SplitError(ValueError):
    """Split treino/avaliacao mal-formado (id inexistente ou id nos dois lados)."""


def axis_of(item_id: str) -> str | None:
    """Eixo do item pelo prefixo de modalidade do `item_id`; `None` se nao reconhecido."""
    prefixo, sep, _ = item_id.partition("-")
    return prefixo if sep and prefixo in AXES else None


def check_balance(qrels: dict[str, set[str]], queries: dict[str, str], *,
                  floor: int = FLOOR) -> dict[str, int]:
    """Conta as queries julgadas por eixo e exige o piso `floor` em TODOS os eixos.

    Julgada no eixo = a query tem >=1 relevante daquele eixo (criterio do `run_eval`);
    uma query cross-modal conta em cada eixo que toca, uma vez por eixo. So entram os
    qids de `queries` -- qid presente so nas qrels nao e avaliado, logo nao conta.

    Devolve `eixo -> contagem` com TODOS os `AXES` como chave (eixo vazio = 0). Levanta
    `BalanceError` nomeando os eixos abaixo do piso e suas contagens.
    """
    contagens: dict[str, int] = {axis: 0 for axis in AXES}
    for qid in queries:
        eixos_da_query = {axis_of(item) for item in qrels.get(qid, ())}
        for axis in eixos_da_query:
            if axis is not None:
                contagens[axis] += 1

    faltantes = [(axis, contagens[axis]) for axis in AXES if contagens[axis] < floor]
    if faltantes:
        detalhe = ", ".join(f"{axis}={n}" for axis, n in faltantes)
        raise BalanceError(
            f"eixo(s) abaixo do piso de {floor} queries julgadas: {detalhe}")
    return contagens


def tag_split(queries: dict[str, str], *, eval_ids: Iterable[str],
              train_ids: Iterable[str] | None = None) -> dict[str, list[str]]:
    """Marca cada query como treino ou avaliacao. Devolve `{"train": [...], "eval": [...]}`.

    Sem `train_ids`, o treino e o complemento: `train = queries - eval_ids` (a particao
    cobre todas as queries). Com `train_ids` explicito, os dois lados sao exatamente as
    listas dadas -- query fora das duas fica de fora do split, por conta do chamador.

    Invariantes (erro, nunca aviso): todo id citado existe em `queries`, e `train` e
    `eval` sao disjuntos. Ids ordenados na saida, para o split ser reproduzivel.
    """
    eval_set = set(eval_ids)
    _exigir_existencia(eval_set, queries, lado="eval_ids")

    if train_ids is None:
        train_set = set(queries) - eval_set
    else:
        train_set = set(train_ids)
        _exigir_existencia(train_set, queries, lado="train_ids")
        sobreposto = train_set & eval_set
        if sobreposto:
            raise SplitError(
                "id(s) em treino E avaliacao (o split precisa ser disjunto): "
                + ", ".join(sorted(sobreposto)))

    return {"train": sorted(train_set), "eval": sorted(eval_set)}


def _exigir_existencia(ids: set[str], queries: dict[str, str], *, lado: str) -> None:
    desconhecidos = ids - set(queries)
    if desconhecidos:
        raise SplitError(
            f"{lado} cita qid(s) que nao existem em queries: "
            + ", ".join(sorted(desconhecidos)))
