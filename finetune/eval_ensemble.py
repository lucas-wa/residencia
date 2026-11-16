"""Hook de avaliacao ANTES/DEPOIS do adaptador denso no benchmark do ENSEMBLE (v4).

O ViDoRe V3 (`finetune/train_lora.run_benchmark_eval`) mede as bases late-interaction em
paginas de documento; a base DENSA (`qwen3vl`) existe para o retriever unificado A, que e
cross-modal e e avaliado no corpus v4 do ensemble. Entao o ganho do finetune denso se mede
onde ele vai ser usado: o MESMO baseline A do `ensemble/eval/run_eval.py`, com e sem o
adaptador.

A metrica e o nDCG@10 do A na linha `conjunto` do driver: ranking GLOBAL (sem filtro de
eixo) e media sobre as queries JULGADAS -- o mesmo criterio do `run_eval.evaluate`
(`if rel_all:`), nao a media sobre todas as queries. Reusa `ensemble/eval/metrics.py` e
`ensemble/native/unified_a.py`: nada de metrica reimplementada aqui.

`unified_ndcg_at10` e PURA e injetavel (`embed_items`/`embed_query` entram como
callables), entao os testes exercitam a metrica inteira com um duble deterministico, sem
GPU. Quem carrega modelo e so o `run_ensemble_eval_before_after`, por
`ensemble.native.embedders.build_unified_a_embedder` (import tardio, como o resto do
ensemble).
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Corte do nDCG reportado no antes/depois (o maior corte do CSV do driver).
NDCG_K = 10

__all__ = ["NDCG_K", "unified_ndcg_at10", "ndcg_multi", "load_eval_data",
           "run_ensemble_eval_before_after", "run_eval_multi_before_after", "main",
           "main_multi"]


def unified_ndcg_at10(records, queries: dict[str, str], qrels: dict[str, set[str]], *,
                      embed_items, embed_query) -> float:
    """nDCG@10 medio do retriever A sobre as queries JULGADAS do conjunto.

    `records` sao os `PointerRecord` do corpus v4; `embed_items`/`embed_query` sao os
    mesmos callables que o `UnifiedRetrieverA` recebe no driver -- um duble
    deterministico roda isto inteiro sem GPU. Lembrando que o A e multimodal: o
    `embed_items` recebe pares `(rota, valor)` (ver `ensemble/native/unified_a.py`).

    Criterio de "julgada" copiado do `ensemble/eval/run_eval.py` (linha `conjunto`):
    entra na media a query cujo conjunto-relevante e NAO-VAZIO, e o denominador e o
    numero dessas queries. O `k` da busca e o corpus inteiro (ranking completo), como la.
    """
    from ensemble.native.unified_a import UnifiedRetrieverA

    corpus = list(records)
    retriever = UnifiedRetrieverA(corpus, embed_items=embed_items, embed_query=embed_query)
    return _ndcg_de_um_conjunto(retriever, max(len(corpus), 1), queries, qrels)


def _ndcg_de_um_conjunto(retriever, k_busca: int, queries: dict[str, str],
                         qrels: dict[str, set[str]]) -> float:
    """O laco da metrica, sobre um retriever JA construido (o corpus ja embeddado).

    Existe para que `unified_ndcg_at10` (um conjunto, um retriever) e `ndcg_multi` (N
    conjuntos, UM retriever) compartilhem a MESMA definicao de nDCG@10 -- quem muda entre
    as duas e so quantas vezes o corpus e embeddado, nunca o numero medido.
    """
    from ensemble.eval.metrics import ndcg_at_k

    soma, julgadas = 0.0, 0
    for qid, texto in queries.items():
        relevantes = set(qrels.get(qid, ()))
        if not relevantes:              # query nao-julgada: nao entra na media (padrao IR)
            continue
        julgadas += 1
        ranqueados = [item_id for item_id, _ in retriever.search(texto, k_busca)]
        soma += ndcg_at_k(ranqueados, relevantes, NDCG_K)

    if not julgadas:
        log.warning("nenhuma query julgada no conjunto: nDCG@%d sem denominador (0.0)",
                    NDCG_K)
        return 0.0
    return soma / julgadas


def ndcg_multi(records, conjuntos, *, embed_items, embed_query) -> dict[str, float]:
    """nDCG@10 de VARIOS conjuntos de queries sobre UM unico embedding do corpus.

    `conjuntos` e uma sequencia de `(nome, queries, qrels)`. O embedding dos ITENS nao
    depende das queries: para um mesmo estado do modelo (base OU adaptador) o corpus vale
    para todos os conjuntos, entao o `UnifiedRetrieverA` e montado UMA vez -- `embed_items`
    e chamado 1x, nao 1x por conjunto. Embeddar 150 itens na L4 custa ~30min; e esse o
    tempo que some aqui.

    O numero medido e identico ao de `unified_ndcg_at10` conjunto a conjunto (os dois
    chamam `_ndcg_de_um_conjunto`). Devolve `{nome: ndcg}`, na ordem recebida.
    """
    from ensemble.native.unified_a import UnifiedRetrieverA

    corpus = list(records)
    retriever = UnifiedRetrieverA(corpus, embed_items=embed_items, embed_query=embed_query)
    k_busca = max(len(corpus), 1)

    return {nome: _ndcg_de_um_conjunto(retriever, k_busca, queries, qrels)
            for nome, queries, qrels in conjuntos}


def load_eval_data(corpus_path, queries_path, qrels_path):
    """`(records, queries, qrels)` dos arquivos do v4 -- import tardio do pacote ensemble."""
    from ensemble.eval.qrels import load_qrels, load_queries
    from ensemble.records import load_records

    return (load_records(corpus_path), load_queries(queries_path), load_qrels(qrels_path))


def _ndcg_do_a(records, queries, qrels, adapter_path: str | None, tag: str) -> float:
    """Monta o A (com ou sem adaptador) e mede -- o par antes/depois chama duas vezes."""
    from ensemble.native.embedders import build_unified_a_embedder

    embed_items, embed_query = build_unified_a_embedder(adapter_path=adapter_path)
    valor = unified_ndcg_at10(records, queries, qrels,
                              embed_items=embed_items, embed_query=embed_query)
    log.info("[eval ensemble %s] A nDCG@%d = %.4f (adaptador: %s)",
             tag, NDCG_K, valor, adapter_path or "nenhum")
    return valor


def _liberar_memoria() -> None:
    """Solta a carga anterior do Qwen3-VL antes da proxima (VRAM, nao resultado).

    `gc.collect()` derruba as referencias que o embedder anterior ainda segurava e o
    `empty_cache()` devolve o cache do alocador do torch -- sem isto, a 2a carga depende
    do GC ter rodado na hora certa e, numa L4 24GB, as duas podem coexistir. Imports
    locais e guardados: os testes puros (dubles, sem GPU) nao tem torch e so pulam.
    """
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:                      # ambiente sem torch: nada a liberar
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_ensemble_eval_before_after(cfg, adapter_dir, records, queries,
                                   qrels) -> dict[str, float | int]:
    """nDCG@10 do A no v4 ANTES (base crua) e DEPOIS (com `adapter_path=adapter_dir`).

    Carrega o Qwen3-VL duas vezes -- de proposito: o "antes" tem de ser o checkpoint
    publico, sem nenhum peso do treino. Entre as duas cargas, `_liberar_memoria` solta a
    primeira (as duas na VRAM ao mesmo tempo apertariam uma L4 24GB); o numero medido nao
    muda com isso. Devolve `{"antes", "depois", "delta",
    "n_queries"}` e loga as duas linhas; quem chama (`train_lora.run_ensemble_eval`) e
    quem decide que uma falha aqui nao derruba o treino.
    """
    julgadas = sum(1 for qid in queries if qrels.get(qid))
    log.info("[eval ensemble] %s/%s em %d record(s), %d query(ies) julgada(s)",
             getattr(cfg, "base", "?"), getattr(cfg, "method", "?"), len(records), julgadas)

    antes = _ndcg_do_a(records, queries, qrels, None, "antes")
    _liberar_memoria()                       # a base crua sai da VRAM antes da finetunada
    depois = _ndcg_do_a(records, queries, qrels, str(Path(adapter_dir)), "depois")

    resultado = {"antes": antes, "depois": depois, "delta": depois - antes,
                 "n_queries": julgadas}
    log.info("[eval ensemble] nDCG@%d antes=%.4f depois=%.4f delta=%+.4f (%d queries)",
             NDCG_K, antes, depois, depois - antes, julgadas)
    return resultado


# -- N conjuntos, 2 cargas de modelo, 2 embeddings do corpus ---------------------------
#
# `run_ensemble_eval_before_after` mede UM conjunto e, para medir dois (a ancora-ouro
# humana e o sintetico de N maior), e chamada duas vezes: 4 cargas do Qwen3-VL e 4
# embeddings do corpus para responder a mesma pergunta sobre o mesmo adaptador. O
# embedding dos ITENS nao depende das queries, entao o par abaixo faz o corte obvio: por
# ESTADO do modelo (base, depois adaptador), carrega uma vez, embedda o corpus uma vez e
# pontua todos os conjuntos. 2 cargas e 2 embeddings para QUALQUER numero de conjuntos.


def _ndcg_multi_do_a(records, conjuntos, adapter_path: str | None,
                     tag: str) -> dict[str, float]:
    """Monta o A (com ou sem adaptador) UMA vez e mede todos os conjuntos nele."""
    from ensemble.native.embedders import build_unified_a_embedder

    embed_items, embed_query = build_unified_a_embedder(adapter_path=adapter_path)
    valores = ndcg_multi(records, conjuntos,
                         embed_items=embed_items, embed_query=embed_query)
    for nome, valor in valores.items():
        log.info("[eval ensemble %s] [%s] A nDCG@%d = %.4f (adaptador: %s)",
                 tag, nome, NDCG_K, valor, adapter_path or "nenhum")
    return valores


def run_eval_multi_before_after(cfg, adapter_dir, records,
                                conjuntos) -> dict[str, dict[str, float | int]]:
    """`run_ensemble_eval_before_after` para VARIOS conjuntos, sem repagar o embedding.

    `conjuntos` e uma sequencia de `(nome, queries, qrels)`. O "antes" continua sendo o
    checkpoint publico e o "depois" o mesmo diretorio de adaptador -- o que muda e que
    cada estado do modelo e carregado (e o corpus embeddado) UMA vez para todos os
    conjuntos, com `_liberar_memoria` entre os dois como no caminho de um conjunto so.

    Devolve `{nome: {"antes", "depois", "delta", "n_queries"}}`.
    """
    conjuntos = [(nome, queries, qrels) for nome, queries, qrels in conjuntos]
    julgadas = {nome: sum(1 for qid in queries if qrels.get(qid))
                for nome, queries, qrels in conjuntos}
    log.info("[eval ensemble] %s/%s em %d record(s), %d conjunto(s): %s",
             getattr(cfg, "base", "?"), getattr(cfg, "method", "?"), len(records),
             len(conjuntos),
             ", ".join(f"{nome}={n} julgada(s)" for nome, n in julgadas.items()))

    antes = _ndcg_multi_do_a(records, conjuntos, None, "antes")
    _liberar_memoria()                       # a base crua sai da VRAM antes da finetunada
    depois = _ndcg_multi_do_a(records, conjuntos, str(Path(adapter_dir)), "depois")

    resultado: dict[str, dict[str, float | int]] = {}
    for nome, _, _ in conjuntos:
        resultado[nome] = {"antes": antes[nome], "depois": depois[nome],
                           "delta": depois[nome] - antes[nome],
                           "n_queries": julgadas[nome]}
        log.info("[eval ensemble [%s]] nDCG@%d antes=%.4f depois=%.4f delta=%+.4f "
                 "(%d queries)", nome, NDCG_K, antes[nome], depois[nome],
                 depois[nome] - antes[nome], julgadas[nome])
    return resultado


# -- CLI: a MESMA medicao, fora do treino ----------------------------------------------
#
# `train_lora.py` ja dispara este hook no fim do treino, mas so para UM conjunto de
# avaliacao. A iteracao da Semana 10 mede dois (a ancora-ouro humana e o conjunto
# SINTETICO de N maior) sobre o MESMO adaptador: treinar duas vezes seria pagar a GPU
# duas vezes para responder a mesma pergunta. Entao o 2o conjunto entra por aqui, com o
# adaptador ja salvo em disco -- 1 treino, 2 medicoes.


def main(argv: list[str] | None = None) -> int:
    """`python -m finetune.eval_ensemble --corpus ... --queries ... --qrels ... --adapter ...`"""
    import argparse
    from types import SimpleNamespace

    ap = argparse.ArgumentParser(
        description="nDCG@10 do retriever A no v4, ANTES e DEPOIS de um adaptador ja salvo.")
    ap.add_argument("--corpus", required=True, help="records.jsonl do corpus v4")
    ap.add_argument("--queries", required=True, help="queries do conjunto (qid<TAB>texto)")
    ap.add_argument("--qrels", required=True, help="qrels TREC do conjunto (4 colunas)")
    ap.add_argument("--adapter", required=True,
                    help="diretorio do adaptador PEFT (o 'depois'; o 'antes' e a base crua)")
    ap.add_argument("--tag", default="",
                    help="rotulo do conjunto no log (ex.: sintetico) -- so apresentacao")
    ap.add_argument("--base", default="qwen3vl")
    ap.add_argument("--method", default="qlora")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records, queries, qrels = load_eval_data(a.corpus, a.queries, a.qrels)
    cfg = SimpleNamespace(base=a.base, method=a.method)
    res = run_ensemble_eval_before_after(cfg, a.adapter, records, queries, qrels)

    rotulo = f" [{a.tag}]" if a.tag else ""
    print(f"[eval ensemble{rotulo}] nDCG@{NDCG_K} antes={res['antes']:.4f} "
          f"depois={res['depois']:.4f} delta={res['delta']:+.4f} "
          f"({res['n_queries']} queries julgadas)")
    return 0


# -- CLI `multi`: N conjuntos num passo so -------------------------------------------
#
# Mesma medicao da CLI acima, mas com os conjuntos TODOS na mesma invocacao -- que e o que
# permite reusar o embedding do corpus. Rodar a CLI de um conjunto duas vezes nao reusa
# nada: sao dois processos, duas cargas do modelo cada.


def _partir_conjunto(spec: str) -> tuple[str, str, str]:
    """`NOME:queries:qrels` -> as tres partes, tolerando `C:/...` no Windows.

    O separador e `:`, e caminho do Windows tem `:` na letra de unidade: as partes que
    sobram como `['C', '/x/q.txt']` sao recoladas. O NOME (a 1a parte) nunca e recolado --
    senao um nome de uma letra engoliria o caminho seguinte.
    """
    partes = spec.split(":")
    juntas: list[str] = []
    for parte in partes:
        if (len(juntas) >= 2 and len(juntas[-1]) == 1 and juntas[-1].isalpha()
                and parte[:1] in ("/", "\\")):
            juntas[-1] = f"{juntas[-1]}:{parte}"
        else:
            juntas.append(parte)
    if len(juntas) != 3 or not all(juntas):
        raise SystemExit(f"--conjunto espera NOME:queries:qrels (recebi: {spec!r})")
    return juntas[0], juntas[1], juntas[2]


def main_multi(argv: list[str] | None = None) -> int:
    """`python -m finetune.eval_ensemble multi --corpus ... --conjunto ouro:q:qr --adapter ...`"""
    import argparse
    from types import SimpleNamespace

    ap = argparse.ArgumentParser(
        prog="python -m finetune.eval_ensemble multi",
        description="nDCG@10 do A antes/depois de um adaptador, em VARIOS conjuntos de "
                    "queries, com UMA carga de modelo e UM embedding do corpus por estado.")
    ap.add_argument("--corpus", required=True, help="records.jsonl do corpus v4")
    ap.add_argument("--conjunto", action="append", required=True,
                    metavar="NOME:queries:qrels",
                    help="conjunto a medir (repita a flag para medir varios)")
    ap.add_argument("--adapter", required=True,
                    help="diretorio do adaptador PEFT (o 'depois'; o 'antes' e a base crua)")
    ap.add_argument("--base", default="qwen3vl")
    ap.add_argument("--method", default="qlora")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records: list = []
    conjuntos: list[tuple[str, dict, dict]] = []
    for spec in a.conjunto:
        nome, caminho_queries, caminho_qrels = _partir_conjunto(spec)
        # O corpus e reparseado por conjunto: e leitura de jsonl, nao embedding -- o que
        # custa (o embed dos itens) acontece uma vez so, la dentro do `ndcg_multi`.
        records, queries, qrels = load_eval_data(a.corpus, caminho_queries, caminho_qrels)
        conjuntos.append((nome, queries, qrels))

    cfg = SimpleNamespace(base=a.base, method=a.method)
    res = run_eval_multi_before_after(cfg, a.adapter, records, conjuntos)

    for nome, r in res.items():
        print(f"[eval ensemble [{nome}]] nDCG@{NDCG_K} antes={r['antes']:.4f} "
              f"depois={r['depois']:.4f} delta={r['delta']:+.4f} "
              f"({r['n_queries']} queries julgadas)")
    return 0


if __name__ == "__main__":
    import sys

    # `multi` como subcomando: a CLI de UM conjunto segue sendo a invocacao padrao (e os
    # scripts que ja a chamam nao mudam).
    if sys.argv[1:2] == ["multi"]:
        raise SystemExit(main_multi(sys.argv[2:]))
    raise SystemExit(main())
