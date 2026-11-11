"""Driver de avaliacao do ensemble v2: roda os 5 especialistas nativos, funde por RRF, e
compara o **ensemble** com os baselines **A** (unificado, global) e **B** (so documento).

Espelha o `results/comparacao.csv` do v1: uma linha por **(sistema, eixo)** mais a linha
**conjunto**, com `recall@{1,5,10}`, `nDCG@{1,5,10}`, `MRR` e `latencia`. A semantica segue
a spec (`fusao/multimodal.md` secao 4.6):

- **ensemble** = `rrf_fuse` das listas de item_ids dos 5 especialistas. Por eixo = a lista
  fundida **restrita** aos itens daquele eixo; conjunto = a lista fundida inteira.
- **A** = `retriever_a.search` (ranking global cross-modal). Por eixo = o ranking global
  **filtrado** (nao re-ranqueado) ao eixo; conjunto = o ranking global inteiro.
- **B** = `retriever_b.search` (so documento). Reportado em `documento` e `conjunto`; nos
  eixos imagem/audio/texto/video as metricas sao **n/a** (`None` -- nao 0, nao excecao),
  porque B nao indexa esses itens.

Relevancia binaria e topical: uma query pode ser relevante a varios itens/eixos. A unidade
e pagina(doc)/janela(video)/arquivo(resto) -- os `item_id` JA sao unidades, entao o RRF e as
metricas deduplicam a unidade sem traducao. As metricas sao a media **sobre as queries
julgadas** (padrao IR): por eixo, so entram as queries cujo conjunto-relevante RESTRITO
aquele eixo e nao-vazio; no `conjunto`, as queries com qualquer relevante. Eixo sem nenhuma
query julgada -> a celula (sistema, eixo) e **n/a** (`None`), distinguindo "nao avaliado" de
"avaliado e falhou". `n_queries` reporta quantas queries julgadas sustentam cada media.

`evaluate` e puro e injetavel (dubles nos testes). `main` monta os sistemas reais via o
registry de embedders da Task 6 (`EMBEDDER_BUILDERS`) + o ranker ColQwen2.5 do eixo
documento, tudo por **import tardio**: importar este modulo nao baixa checkpoint nem toca
GPU. Os modelos reais so rodam no Slurm H100.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from time import perf_counter

from ensemble.eval.metrics import mrr, ndcg_at_k, recall_at_k
from ensemble.eval.qrels import load_qrels, load_queries, validate_qrels
from ensemble.eval.rrf import rrf_fuse
from ensemble.records import PointerRecord, load_records

__all__ = ["AXES", "CONJUNTO", "SYSTEMS", "B_AXES", "CSV_HEADER", "RRF_K",
           "RANKINGS_TOPN", "ensemble_search", "evaluate", "write_csv", "main"]

log = logging.getLogger(__name__)

# Ordem dos eixos (= ordem de iteracao dos especialistas no RRF, determinista).
AXES: tuple[str, ...] = ("documento", "imagem", "audio", "texto", "video")
CONJUNTO = "conjunto"
SYSTEMS: tuple[str, ...] = ("ensemble", "A", "B")
RRF_K = 60  # k do Reciprocal Rank Fusion (spec: RRF uniforme, k=60).
# Quantos ids por lista sobrevivem no dump de `--rankings-out`: @10 e o maior corte das
# metricas e a re-fusao offline so mexe no topo -- 50 cobre as duas sem JSON gigante.
RANKINGS_TOPN = 50

# Metricas por linha, na ordem do CSV do v1 (a `latencia` entra a parte).
_METRICS: tuple[str, ...] = (
    "recall@1", "nDCG@1", "recall@5", "nDCG@5", "recall@10", "nDCG@10", "MRR",
)
# B (ColQwen2.5) so indexa/pontua estes eixos; nos demais as metricas sao n/a (None).
# Fonte unica da regra "B so documento": o laco e o filtro de linha derivam daqui.
B_AXES: tuple[str, ...] = ("documento",)
_B_NA_AXES: tuple[str, ...] = tuple(a for a in AXES if a not in B_AXES)

# Cabecalho do CSV, espelhando `results/comparacao.csv` (sistema->arquitetura,
# eixo->tarefa, `n_queries` = queries julgadas da media, e os nomes de metrica em
# minusculas do v1).
CSV_HEADER: list[str] = [
    "arquitetura", "tarefa", "n_queries",
    "recall@1", "ndcg@1", "recall@5", "ndcg@5", "recall@10", "ndcg@10",
    "mrr", "latencia_s",
]
# row-key -> coluna do CSV, na ordem de CSV_HEADER.
_ROW_TO_CSV: list[tuple[str, str]] = [
    ("sistema", "arquitetura"), ("eixo", "tarefa"), ("n_queries", "n_queries"),
    ("recall@1", "recall@1"), ("nDCG@1", "ndcg@1"),
    ("recall@5", "recall@5"), ("nDCG@5", "ndcg@5"),
    ("recall@10", "recall@10"), ("nDCG@10", "ndcg@10"),
    ("MRR", "mrr"), ("latencia", "latencia_s"),
]

# Eixo documento do baseline B (ColQwen2.5), construido no `main` por import tardio.
DOC_MODEL = "vidore/colqwen2.5-v0.2"
DOC_KIND = "colqwen2.5"


def ensemble_search(specialists: dict, query: str, k: int,
                    *, rrf_k: int = RRF_K,
                    per_axis_out: dict | None = None) -> list[tuple[str, float]]:
    """O ensemble: funde por RRF as listas de item_ids dos especialistas.

    Cada especialista ranqueia so o seu eixo; o RRF usa apenas a ORDEM (os scores dos
    especialistas sao descartados -- escalas diferentes). A iteracao segue a ordem do dict
    (o `main` o monta em `AXES`), entao o desempate do RRF e determinista.

    `per_axis_out`, se dado, recebe eixo->lista de item_ids ANTES do RRF (as listas que
    acabaram de ser fundidas). E so um espelho para o dump de `--rankings-out`: evita
    repetir o `sp.search` (que e a parte cara, em GPU) so para registrar o ranking bruto.
    """
    listas = {axis: [item_id for item_id, _ in sp.search(query, k)]
              for axis, sp in specialists.items()}
    if per_axis_out is not None:
        per_axis_out.update(listas)
    return rrf_fuse(list(listas.values()), k=rrf_k)


def _metrics(ranked_ids: list[str], relevant: set[str]) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(ranked_ids, relevant, 1),
        "nDCG@1": ndcg_at_k(ranked_ids, relevant, 1),
        "recall@5": recall_at_k(ranked_ids, relevant, 5),
        "nDCG@5": ndcg_at_k(ranked_ids, relevant, 5),
        "recall@10": recall_at_k(ranked_ids, relevant, 10),
        "nDCG@10": ndcg_at_k(ranked_ids, relevant, 10),
        "MRR": mrr(ranked_ids, relevant),
    }


def evaluate(corpus: list[PointerRecord], queries: dict[str, str],
             qrels: dict[str, set[str]], *, specialists: dict,
             retriever_a, retriever_b, rankings_out: str | None = None) -> list[dict]:
    """Avalia ensemble/A/B por (sistema, eixo) + conjunto, media sobre as queries.

    `corpus` da a modalidade de cada id (filtro por eixo) e o `k` de busca (ranking
    completo). `specialists` e um dict eixo->Specialist; `retriever_a` ranqueia global;
    `retriever_b` so documento. Devolve uma linha-dict por (sistema, eixo), com `None`
    (n/a) nas metricas de B fora de documento.

    `rankings_out`, se dado, grava tambem um JSON com os rankings de cada query (por eixo
    antes do RRF, fundido, A, B e o conjunto-relevante), que sustenta experimentos de
    fusao OFFLINE (sweep de k do RRF, pesos por eixo) sem re-rodar GPU. A gravacao e
    best-effort: falhar nela NAO derruba a avaliacao -- o CSV e a saida principal.
    """
    modality_by_id = {r.id: r.modality for r in corpus}
    k_search = max(len(corpus), 1)
    n = len(queries)

    # somatorios[(sistema, eixo)][metrica]; latencia por sistema (independe do eixo);
    # `julgadas[eixo]` = queries com relevante naquele eixo (denominador da media IR).
    somas: dict[tuple[str, str], dict[str, float]] = {
        (s, axis): {m: 0.0 for m in _METRICS}
        for s in SYSTEMS for axis in (*AXES, CONJUNTO)
    }
    lat: dict[str, float] = {s: 0.0 for s in SYSTEMS}
    julgadas: dict[str, int] = {axis: 0 for axis in (*AXES, CONJUNTO)}
    dump: dict[str, dict] | None = {} if rankings_out else None

    for qid, qtext in queries.items():
        rel_all = set(qrels.get(qid, ()))
        rel_axis = {axis: {i for i in rel_all if modality_by_id.get(i) == axis}
                    for axis in AXES}

        per_axis: dict[str, list[str]] | None = {} if dump is not None else None
        t0 = perf_counter()
        fused_ids = [i for i, _ in ensemble_search(specialists, qtext, k_search,
                                                   per_axis_out=per_axis)]
        lat["ensemble"] += perf_counter() - t0

        t0 = perf_counter()
        a_ids = [i for i, _ in retriever_a.search(qtext, k_search)]
        lat["A"] += perf_counter() - t0

        t0 = perf_counter()
        b_ids = [i for i, _ in retriever_b.search(qtext, k_search)]
        lat["B"] += perf_counter() - t0

        rankings = {"ensemble": fused_ids, "A": a_ids, "B": b_ids}
        if dump is not None:
            dump[qid] = {
                "text": qtext,
                # `relevant` NAO e truncado: e o denominador das metricas offline.
                "relevant": sorted(rel_all),
                "ensemble_fused": fused_ids[:RANKINGS_TOPN],
                "per_axis": {axis: ids[:RANKINGS_TOPN]
                             for axis, ids in (per_axis or {}).items()},
                "A": a_ids[:RANKINGS_TOPN],
                "B": b_ids[:RANKINGS_TOPN],
            }

        for axis in AXES:
            rel = rel_axis[axis]
            if not rel:          # query nao-julgada neste eixo: nao conta na media
                continue
            julgadas[axis] += 1
            for sistema in ("ensemble", "A"):
                sub = [i for i in rankings[sistema] if modality_by_id.get(i) == axis]
                _accum(somas[(sistema, axis)], _metrics(sub, rel))
            if axis in B_AXES:   # B so pontua os eixos que indexa (documento)
                sub = [i for i in b_ids if modality_by_id.get(i) == axis]
                _accum(somas[("B", axis)], _metrics(sub, rel))

        if rel_all:
            julgadas[CONJUNTO] += 1
            for sistema in SYSTEMS:
                _accum(somas[(sistema, CONJUNTO)], _metrics(rankings[sistema], rel_all))

    if dump is not None:
        _dump_rankings(dump, rankings_out)

    rows: list[dict] = []
    for sistema in SYSTEMS:
        for axis in (*AXES, CONJUNTO):
            jc = julgadas[axis]
            if (sistema == "B" and axis in _B_NA_AXES) or jc == 0:
                rows.append(_na_row(sistema, axis))
            else:
                rows.append(_avg_row(sistema, axis, somas[(sistema, axis)],
                                     lat[sistema], jc, n))
    return rows


def _accum(dst: dict[str, float], src: dict[str, float]) -> None:
    for m, v in src.items():
        dst[m] += v


def _avg_row(sistema: str, eixo: str, somas: dict[str, float], lat: float,
             jc: int, n: int) -> dict:
    # metricas: media sobre as `jc` queries julgadas; latencia: media por query do
    # sistema (sobre todas as `n` queries -- e propriedade do sistema, nao do eixo).
    row: dict = {"sistema": sistema, "eixo": eixo, "n_queries": jc}
    for m in _METRICS:
        row[m] = somas[m] / jc
    row["latencia"] = lat / n if n else 0.0
    return row


def _na_row(sistema: str, eixo: str) -> dict:
    row: dict = {"sistema": sistema, "eixo": eixo, "n_queries": None}
    for m in _METRICS:
        row[m] = None
    row["latencia"] = None
    return row


def _dump_rankings(queries_dump: dict[str, dict], path) -> None:
    """Grava o JSON de rankings por query -- best-effort, NUNCA levanta.

    O artefato alimenta a re-fusao offline (`{"rrf_k": ..., "queries": {qid: {...}}}`);
    perde-lo e um aborrecimento, nao um motivo para perder a rodada de GPU, entao
    qualquer falha vira warning no log e a avaliacao segue com o CSV.
    """
    try:
        out = Path(path)
        if out.parent != Path(""):
            out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"rrf_k": RRF_K, "queries": queries_dump}, f,
                      ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001 -- saida secundaria: nao derruba a avaliacao
        log.warning("nao foi possivel gravar os rankings em %s: %s", path, e)
    else:
        log.info("rankings de %d query(ies) -> %s", len(queries_dump), path)


def write_csv(rows: list[dict], path) -> None:
    """Grava as linhas no CSV espelhando `results/comparacao.csv`. `None` vira `n/a`."""
    out = Path(path)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow(["n/a" if r[key] is None else r[key] for key, _ in _ROW_TO_CSV])


# --------------------------------------------------------------------------- #
# main: monta os sistemas reais por import tardio (builders dubles nos testes)
# --------------------------------------------------------------------------- #
def _build_doc_ranker():
    """Ranker ColQwen2.5 do eixo documento -- import TARDIO (torch/colpali ficam aqui)."""
    from benchmark.models.colvision import ColVisionRetriever

    log.info("carregando ranker de documento %s (kind=%s)", DOC_MODEL, DOC_KIND)
    return ColVisionRetriever(DOC_MODEL, kind=DOC_KIND, device="cuda")


def _default_builders() -> dict:
    """Registry da Task 6 (texto/imagem/audio/audio_asr/video/unified_a) + ranker de documento.

    Import tardio de `EMBEDDER_BUILDERS`: so a fiacao; os modelos reais carregam dentro de
    cada builder, no cluster.
    """
    from ensemble.native.embedders import EMBEDDER_BUILDERS

    builders = dict(EMBEDDER_BUILDERS)
    builders["documento"] = _build_doc_ranker
    return builders


def _build_systems(corpus: list[PointerRecord], builders: dict, *,
                   pages_dir=None, data_dir=None, audio_mode: str = "asr"):
    """Monta os 5 especialistas + retriever_a + retriever_b a partir dos builders.

    `builders["documento"]()` devolve um ranker (ColQwen-like); os demais devolvem o par
    `(embed_items, embed_query)`. retriever_b reusa o especialista de documento (mesmo
    indice ColQwen2.5), que ja ranqueia so paginas.

    `audio_mode="asr"` (padrao) ranqueia o eixo audio pela TRANSCRICAO (`descriptor` do
    record, posto la pelo Whisper na captura) com o BGE-M3 -- builder `"audio_asr"`;
    `audio_mode="clap"` mantem o CLAP sobre o arquivo de audio (builder `"audio"`,
    representacao padrao `meta["path"]`), como ablacao.
    """
    from ensemble.native.doc import DocSpecialist
    from ensemble.native.specialist import DenseSpecialist
    from ensemble.native.unified_a import UnifiedRetrieverA

    doc_spec = DocSpecialist(corpus, ranker=builders["documento"](),
                             pages_dir=pages_dir, data_dir=data_dir)
    specialists: dict = {"documento": doc_spec}
    for mod in ("imagem", "audio", "texto", "video"):
        kwargs: dict = {}
        builder = builders[mod]
        if mod == "audio" and audio_mode == "asr":
            builder = builders.get("audio_asr", builders["audio"])
            kwargs["repr_fn"] = lambda r: r.descriptor
        embed_items, embed_query = builder()
        specialists[mod] = DenseSpecialist(mod, corpus, embed_items=embed_items,
                                           embed_query=embed_query, **kwargs)

    a_items, a_query = builders["unified_a"]()
    retriever_a = UnifiedRetrieverA(corpus, embed_items=a_items, embed_query=a_query)

    # ordem AXES para o RRF ser determinista; retriever_b = o especialista de documento.
    specialists = {mod: specialists[mod] for mod in AXES}
    return specialists, retriever_a, doc_spec


def main(argv: list[str] | None = None, *, builders: dict | None = None) -> int:
    """Le corpus/queries/qrels, monta os sistemas, avalia e grava o CSV. 0 em sucesso.

    `builders` injetado pula o registry real (dubles nos testes, sem GPU/download). Insumo
    faltando ou qrels invalido -> retorno != 0 com mensagem clara no stderr.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ensemble.eval.run_eval",
        description="Driver de avaliacao do ensemble v2 (CSV comparativo por sistema/eixo).",
    )
    parser.add_argument("--corpus", required=True, help="records.jsonl (um por linha)")
    parser.add_argument("--queries", required=True, help="arquivo qid<TAB>texto")
    parser.add_argument("--qrels", required=True, help="qrels TREC `qid 0 item_id rel`")
    parser.add_argument("--out", required=True, help="CSV de saida")
    parser.add_argument("--rankings-out", default=None,
                        help="JSON opcional com os rankings de cada query (por eixo, "
                             "fundido, A e B) para re-fusao OFFLINE; best-effort: "
                             "falhar ao gravar nao derruba a avaliacao")
    parser.add_argument("--log", default=None, help="arquivo de log opcional")
    parser.add_argument("--pages-dir", default=None,
                        help="diretorio das paginas de documento (resolucao de caminho)")
    parser.add_argument("--data-dir", default=None, help="raiz alternativa do corpus")
    parser.add_argument(
        "--audio-mode", choices=["asr", "clap"], default="asr",
        help="eixo audio: 'asr' (padrao) embute a TRANSCRICAO (descriptor do record, "
             "vinda do Whisper) com o BGE-M3; 'clap' embute o arquivo de audio bruto "
             "com o CLAP (ablacao)",
    )
    args = parser.parse_args(argv)

    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, level=logging.INFO)

    for rotulo, caminho in (("corpus", args.corpus), ("queries", args.queries),
                            ("qrels", args.qrels)):
        if not Path(caminho).is_file():
            print(f"erro: {rotulo} nao encontrado: {caminho}", file=sys.stderr)
            return 2

    corpus = load_records(args.corpus)
    if not corpus:
        print(f"erro: nenhum record em {args.corpus}", file=sys.stderr)
        return 2
    queries = load_queries(args.queries)
    if not queries:
        print(f"erro: nenhuma query em {args.queries}", file=sys.stderr)
        return 2
    qrels = load_qrels(args.qrels)
    try:
        validate_qrels(qrels, queries, {r.id for r in corpus})
    except ValueError as e:
        print(f"erro: qrels invalido: {e}", file=sys.stderr)
        return 2

    if builders is None:
        builders = _default_builders()

    specialists, retriever_a, retriever_b = _build_systems(
        corpus, builders, pages_dir=args.pages_dir, data_dir=args.data_dir,
        audio_mode=args.audio_mode)

    rows = evaluate(corpus, queries, qrels, specialists=specialists,
                    retriever_a=retriever_a, retriever_b=retriever_b,
                    rankings_out=args.rankings_out)
    write_csv(rows, args.out)

    print(f"ok: {len(queries)} query(ies), {len(corpus)} record(s), "
          f"{len(rows)} linha(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
