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

__all__ = ["NDCG_K", "unified_ndcg_at10", "load_eval_data", "run_ensemble_eval_before_after"]


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
    from ensemble.eval.metrics import ndcg_at_k
    from ensemble.native.unified_a import UnifiedRetrieverA

    corpus = list(records)
    retriever = UnifiedRetrieverA(corpus, embed_items=embed_items, embed_query=embed_query)
    k_busca = max(len(corpus), 1)

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


def run_ensemble_eval_before_after(cfg, adapter_dir, records, queries,
                                   qrels) -> dict[str, float | int]:
    """nDCG@10 do A no v4 ANTES (base crua) e DEPOIS (com `adapter_path=adapter_dir`).

    Carrega o Qwen3-VL duas vezes -- de proposito: o "antes" tem de ser o checkpoint
    publico, sem nenhum peso do treino. Devolve `{"antes", "depois", "delta",
    "n_queries"}` e loga as duas linhas; quem chama (`train_lora.run_ensemble_eval`) e
    quem decide que uma falha aqui nao derruba o treino.
    """
    julgadas = sum(1 for qid in queries if qrels.get(qid))
    log.info("[eval ensemble] %s/%s em %d record(s), %d query(ies) julgada(s)",
             getattr(cfg, "base", "?"), getattr(cfg, "method", "?"), len(records), julgadas)

    antes = _ndcg_do_a(records, queries, qrels, None, "antes")
    depois = _ndcg_do_a(records, queries, qrels, str(Path(adapter_dir)), "depois")

    resultado = {"antes": antes, "depois": depois, "delta": depois - antes,
                 "n_queries": julgadas}
    log.info("[eval ensemble] nDCG@%d antes=%.4f depois=%.4f delta=%+.4f (%d queries)",
             NDCG_K, antes, depois, depois - antes, julgadas)
    return resultado
