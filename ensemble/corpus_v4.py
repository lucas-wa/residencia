"""Ferramental do corpus v4 balanceado: piso de queries por eixo + split treino/avaliacao.

Duas garantias que o v3 nao tinha e que o fine-tune denso da Semana 10 exige:

- **piso por eixo** (`check_balance`): cada eixo precisa de pelo menos `floor` queries
  JULGADAS (default 8 na Semana 10). "Julgada naquele eixo" segue exatamente o criterio
  do `ensemble.eval.run_eval`: a query tem >=1 item relevante do eixo. Eixo sem nenhuma
  query julgada conta 0 -- nunca KeyError -- e, sendo 0 < piso, reprova.
- **split disjunto** (`tag_split`): treino e avaliacao nunca compartilham um qid. Treinar
  o adapter em query que depois aparece na avaliacao contamina o antes/depois, entao a
  disjuncao e invariante (erro, nao aviso).

- **sanity de caminho** (`check_corpus_paths`): item que entra pela rota de IMAGEM do
  baseline A (`documento`/`imagem`/`video`) precisa de um `a_repr` que seja arquivo
  EXISTENTE -- senao o `embed_images` quebraria no `open`, no meio do run, depois de ja
  ter carregado o modelo.

O eixo de um item vem do PREFIXO do `item_id` (`documento-`, `imagem-`, `audio-`,
`texto-`, `video-`), que a captura ja escreve (`ensemble/capture/*.py`). Os eixos
canonicos sao os `AXES` do `run_eval` -- fonte unica, reexportada aqui.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

from ensemble.eval.run_eval import AXES
from ensemble.native.unified_a import a_repr, a_route
from ensemble.records import PointerRecord

__all__ = ["AXES", "EVAL_FRAC", "FLOOR", "SEED", "BalanceError", "CorpusPathError",
           "SplitError", "axis_of", "check_balance", "check_corpus_paths",
           "finalize_split", "tag_split"]

# Piso de queries julgadas por eixo na Semana 10.
FLOOR = 8

# Fatia das queries julgadas de cada eixo reservada a AVALIACAO (o resto e treino).
EVAL_FRAC = 0.3

# Mesma semente do resto do fine-tune (`finetune/train_lora.py`), para o split ser
# reproduzivel entre maquinas.
SEED = 42

# Quantos ofensores a mensagem de erro lista antes de truncar (o total sempre aparece).
MAX_OFENSORES = 5


class BalanceError(ValueError):
    """Algum eixo tem menos queries julgadas que o piso exigido."""


class SplitError(ValueError):
    """Split treino/avaliacao mal-formado (id inexistente ou id nos dois lados)."""


class CorpusPathError(ValueError):
    """Item image-route cujo `a_repr` nao e um arquivo existente em disco."""


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


def check_corpus_paths(records: Iterable[PointerRecord]) -> None:
    """Exige arquivo em disco no `a_repr` de todo item que entra pela rota de imagem.

    `documento`/`imagem`/`video` sao embeddados por `embed_images`, que faz `open` do
    `a_repr` (ver `ensemble.native.unified_a`): caminho torto so apareceria la, com o
    modelo ja carregado. Itens `texto`/`audio` nao sao checados -- o `a_repr` deles e o
    texto real (artigo, transcricao ASR), nao um caminho.

    Levanta `CorpusPathError` nomeando os ofensores (ate `MAX_OFENSORES`) e o total.
    """
    ofensores: list[str] = []
    for rec in records:
        if a_route(rec) != "image":
            continue
        caminho = a_repr(rec)
        if not caminho or not Path(caminho).is_file():
            ofensores.append(f"{rec.id} ({rec.modality}): {caminho!r}")

    if ofensores:
        mostrados = ", ".join(ofensores[:MAX_OFENSORES])
        resto = len(ofensores) - MAX_OFENSORES
        raise CorpusPathError(
            f"{len(ofensores)} item(ns) image-route sem arquivo no a_repr "
            f"(o embed_images do baseline A quebraria no open): {mostrados}"
            + (f", ... e mais {resto}" if resto > 0 else ""))


def finalize_split(queries: dict[str, str], qrels: dict[str, set[str]], *,
                   floor: int = FLOOR, eval_frac: float = EVAL_FRAC,
                   seed: int = SEED) -> tuple[set[str], set[str]]:
    """Valida o piso por eixo e devolve `(train_ids, eval_ids)` disjuntos.

    Ordem importa: `check_balance` primeiro (propaga `BalanceError`), porque separar
    avaliacao de um eixo que nem tem queries suficientes seria ruido com cara de metrica.

    A reserva e POR EIXO -- ~`eval_frac` das queries julgadas de cada eixo vao para a
    avaliacao, para nenhum eixo ficar sem avaliacao ou sem treino. Eixo com >=2 julgadas
    sempre contribui >=1 query a avaliacao e >=1 ao treino. Query cross-modal (julgada em
    mais de um eixo) e decidida UMA vez, no primeiro eixo que a alcanca na ordem de
    `AXES`, e conta para a cota dos demais -- e a mesma query, nao da para treinar nela
    por um eixo e avaliar por outro.

    O sorteio e `random.Random(seed)`: mesmo seed, mesmo split. O treino e o complemento
    (queries nao-julgadas inclusive). A disjuncao e validada por `tag_split`, que propaga
    `SplitError`.
    """
    if not 0.0 < eval_frac < 1.0:
        raise ValueError(f"eval_frac precisa estar em (0, 1); recebi {eval_frac!r}")
    check_balance(qrels, queries, floor=floor)

    eval_ids = _selecionar_eval(queries, qrels, eval_frac=eval_frac, seed=seed)
    train_ids = set(queries) - eval_ids
    tag_split(queries, eval_ids=eval_ids, train_ids=train_ids)
    return train_ids, eval_ids


def _selecionar_eval(queries: dict[str, str], qrels: dict[str, set[str]], *,
                     eval_frac: float, seed: int) -> set[str]:
    """Sorteia, eixo a eixo e sem repetir query, os qids reservados a avaliacao."""
    rng = random.Random(seed)
    eval_ids: set[str] = set()
    decididas: set[str] = set()          # ja em eval OU ja fixada no treino

    for axis in AXES:
        julgadas = sorted(qid for qid in queries
                          if any(axis_of(item) == axis for item in qrels.get(qid, ())))
        if len(julgadas) < 2:
            # 0 ou 1 julgada: reservar a unica deixaria o eixo sem treino -- fica tudo
            # no treino e a avaliacao desse eixo e problema do piso, nao do sorteio.
            decididas.update(julgadas)
            continue

        alvo = max(1, min(int(eval_frac * len(julgadas) + 0.5), len(julgadas) - 1))
        ja_em_eval = sum(1 for qid in julgadas if qid in eval_ids)
        candidatos = [qid for qid in julgadas if qid not in decididas]
        faltam = min(max(alvo - ja_em_eval, 0), max(len(candidatos) - 1, 0))

        sorteadas = set(rng.sample(candidatos, faltam))
        eval_ids |= sorteadas
        decididas.update(julgadas)

    return eval_ids


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
