"""Gera dados sinteticos de treino (consulta -> pagina relevante) para o finetuning pt-BR.

Entrada: o diretorio do benchmark pt-BR dos planos, no layout produzido por
`data/govexe/build_planos_benchmark.py` (e lido por `benchmark.common.datasets.load_local`):

    <corpus_dir>/corpus.jsonl    {"id": "<tse_seq>_p<NN>", "image_path": "..."}
    <corpus_dir>/queries.jsonl   {"id": "<tse_seq>_<destaque|resumo|eixoN>", "text": "..."}
    <corpus_dir>/qrels.tsv       <query_id>\t<page_id>\t1

`corpus.jsonl` e obrigatorio; `queries.jsonl`/`qrels.tsv` sao opcionais e, quando existem,
alimentam o contexto do gerador (destaque/resumo/eixos do plano daquela pagina).

Saida: `<out>/pairs.jsonl` -- uma linha JSON por par (query, pagina positiva, negativos):

    {"id": "<sha1 curto de query+pos_image>",   # chave de dedup/idempotencia
     "query": "quais as propostas do plano BR001 para saude publica?",
     "pos_id": "BR001_p00",                     # id da pagina no corpus.jsonl
     "pos_image": "...../pages/BR001_p00.png",  # pagina RELEVANTE (label=1)
     "neg_ids": ["BR002_p03", ...],             # len == --neg (ou menos, corpus pequeno)
     "neg_images": [".../BR002_p03.png", ...],  # amostradas de OUTRAS paginas
     "label": 1,
     "generator": "template"}

Contrato com o treinador (`finetune/train_lora.py`, tarefa C2 -- ainda nao existe): cada
linha e um exemplo contrastivo pronto: a query (texto) e o anchor, `pos_image` e o unico
positivo e `neg_images` sao os negativos daquele exemplo (hard negatives quando o gerador
real entrar; por ora, negativos aleatorios de outros planos). O treinador le o jsonl
linha a linha, nao precisa do corpus original e pode ignorar `neg_images` se usar apenas
in-batch negatives.

A geracao de consultas e plugavel: `generate_queries(context: PageContext) -> list[str]`.
O default (`template_generate_queries`) e deterministico e offline -- roda nos testes e
sem API nenhuma. Ver `GENERATORS` para plugar um backend de LLM real depois.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SEED = 42
DEFAULT_CORPUS_DIR = "data/govexe/ptbr_planos"
DEFAULT_OUT = "finetune/data/pairs.jsonl"
DEFAULT_K = 3
DEFAULT_NEG = 4
TOPIC_WORDS = 12          # palavras aproveitadas de cada texto do plano como "topico"


@dataclass(frozen=True)
class PageContext:
    """Tudo o que o gerador de consultas enxerga de uma pagina (sem IO proprio)."""

    page_id: str
    image_path: str
    plan_id: str
    k: int                                  # quantas consultas pedir para esta pagina
    text: str | None = None                 # texto da pagina, quando o corpus trouxer
    plan_texts: tuple[str, ...] = ()        # destaque/resumo/eixos do plano (queries.jsonl)


QueryGenerator = Callable[[PageContext], list[str]]

# Templates pt-BR no registro de quem procura um plano de governo (o alvo do finetuning).
TEMPLATES = (
    "o que o plano de governo propoe sobre {topico}?",
    "quais as propostas do plano {plano} para {topico}?",
    "em que pagina o plano {plano} trata de {topico}?",
    "diretrizes do plano de governo sobre {topico}",
    "{topico}: o que diz o plano {plano}?",
)


def _snippet(text: str, words: int = TOPIC_WORDS) -> str:
    return " ".join(text.split()[:words]).strip(" .;,:-")


def _topics(ctx: PageContext) -> list[str]:
    """Topicos deterministicos extraidos do contexto, do mais especifico ao mais generico."""
    raw = [t for t in (ctx.text, *ctx.plan_texts) if t and t.strip()]
    topics = list(dict.fromkeys(_snippet(t) for t in raw))
    topics = [t for t in topics if t]
    return topics or [f"as propostas do plano {ctx.plan_id}"]


def template_generate_queries(ctx: PageContext) -> list[str]:
    """Gerador DEFAULT: deterministico, offline, sem modelo nenhum.

    Combina os templates pt-BR com os topicos do plano (destaque/resumo/eixos, ou o texto
    da propria pagina quando houver). Serve de stub para fechar o pipeline C1->C2 e para
    os testes; a qualidade das consultas vem do backend de LLM (ver `GENERATORS`).
    """
    topicos = _topics(ctx)
    out: list[str] = []
    for i in range(max(ctx.k, 0)):
        tpl = TEMPLATES[i % len(TEMPLATES)]
        out.append(tpl.format(topico=topicos[i % len(topicos)], plano=ctx.plan_id))
    return list(dict.fromkeys(out))


def llm_generate_queries(ctx: PageContext) -> list[str]:
    """TODO(C1): backend de LLM real -- ponto de plug, NAO implementado de proposito.

    Aqui entra a chamada a um LLM (local via vLLM/transformers, ou API) pedindo `ctx.k`
    consultas pt-BR que um eleitor faria e que sao respondidas pela pagina
    `ctx.image_path` (modelo visual) ou por `ctx.text`/`ctx.plan_texts` (modelo textual).
    Requisitos para nao quebrar o resto do script: devolver `list[str]` nao vazia, sem
    duplicatas, e ser deterministico o bastante (temperatura baixa + seed) -- o cache em
    disco ja garante que uma pagina so e gerada uma vez.
    """
    raise NotImplementedError(
        "backend de LLM ainda nao plugado; use --generator template (offline)")


GENERATORS: dict[str, QueryGenerator] = {
    "template": template_generate_queries,
    "llm": llm_generate_queries,
}


# -- leitura do benchmark pt-BR -------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def plan_id_of(page_id: str) -> str:
    """`BR001_p07` -> `BR001` (layout do build_planos_benchmark); senao, o proprio id."""
    head, sep, tail = page_id.rpartition("_p")
    return head if sep and tail.isdigit() else page_id


def load_contexts(corpus_dir: str | Path, k: int = DEFAULT_K) -> list[PageContext]:
    """Monta um `PageContext` por pagina do corpus, anexando os textos do plano."""
    root = Path(corpus_dir)
    docs = _read_jsonl(root / "corpus.jsonl")
    if not docs:
        raise FileNotFoundError(f"corpus.jsonl ausente ou vazio em {root}")

    queries = {q["id"]: q.get("text", "") for q in _read_jsonl(root / "queries.jsonl")}
    per_page: dict[str, list[str]] = {}
    qrels_path = root / "qrels.tsv"
    if qrels_path.exists():
        for line in qrels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            qid, page_id, *_ = line.split("\t")
            text = queries.get(qid, "")
            if text:
                per_page.setdefault(page_id, []).append(text)

    ctxs = []
    for d in docs:
        page_id = d["id"]
        texts = per_page.get(page_id) or []
        if not texts:        # sem qrels: cai para os textos do mesmo plano, se houver
            plan = plan_id_of(page_id)
            texts = [t for qid, t in queries.items() if qid.startswith(f"{plan}_")]
        ctxs.append(PageContext(
            page_id=page_id,
            image_path=d.get("image_path") or "",
            plan_id=plan_id_of(page_id),
            k=k,
            text=d.get("text"),
            plan_texts=tuple(dict.fromkeys(texts)),
        ))
    return ctxs


# -- geracao dos pares ----------------------------------------------------------------


def pair_id(query: str, pos_image: str) -> str:
    """Chave de dedup: o mesmo (query, pagina positiva) nunca vira duas linhas."""
    return hashlib.sha1(f"{query}\x00{pos_image}".encode()).hexdigest()[:16]


def sample_negatives(ctx: PageContext, pool: list[PageContext], n: int,
                     rng: random.Random) -> list[PageContext]:
    """Amostra `n` paginas != `ctx`, preferindo paginas de OUTROS planos."""
    outros_planos = [c for c in pool if c.plan_id != ctx.plan_id]
    outras_paginas = [c for c in pool if c.page_id != ctx.page_id]
    candidatos = outros_planos or outras_paginas
    if not candidatos:
        return []
    return rng.sample(candidatos, min(n, len(candidatos)))


def _resolve_out(out: str | Path) -> Path:
    """`--out` aceita o arquivo (`.../pairs.jsonl`) ou o diretorio que vai conte-lo."""
    p = Path(out)
    return p if p.suffix == ".jsonl" else p / "pairs.jsonl"


def make_pairs(corpus_dir: str | Path, out: str | Path,
               generate_queries: QueryGenerator = template_generate_queries,
               k: int = DEFAULT_K, neg: int = DEFAULT_NEG, limit: int | None = None,
               force: bool = False, seed: int = SEED,
               generator_name: str = "template") -> list[dict]:
    """Escreve `pairs.jsonl` com K consultas por pagina + `neg` negativos, e devolve as linhas.

    E idempotente: o jsonl existente e relido como cache; paginas que ja tem pares nao
    voltam ao gerador (que no caminho real custa chamada de LLM) e as linhas antigas sao
    preservadas na ordem original. A dedup final e por `pair_id(query, pos_image)`, entao
    rodar duas vezes nao duplica nada. `force=True` ignora o cache e refaz tudo.

    Os negativos vem de um `random.Random` semeado por `(seed, page_id)`, logo a mesma
    pagina sempre recebe os mesmos negativos -- independente da ordem/limite da rodada.
    """
    out_path = _resolve_out(out)
    contexts = load_contexts(corpus_dir, k=k)
    pool = contexts                       # negativos saem do corpus inteiro, nao do recorte
    if limit:
        contexts = contexts[:limit]

    cached = [] if force else _read_jsonl(out_path)
    registros: dict[str, dict] = {r["id"]: r for r in cached if "id" in r}
    paginas_prontas = {r["pos_id"] for r in registros.values() if r.get("pos_id")}
    novas = 0

    for ctx in contexts:
        if ctx.page_id in paginas_prontas:
            log.debug("pagina %s ja tem pares em cache", ctx.page_id)
            continue
        rng = random.Random(f"{seed}:{ctx.page_id}")
        negativos = sample_negatives(ctx, pool, neg, rng)
        for query in generate_queries(ctx):
            query = (query or "").strip()
            if not query:
                continue
            rid = pair_id(query, ctx.image_path)
            if rid in registros:
                continue
            registros[rid] = {
                "id": rid,
                "query": query,
                "pos_id": ctx.page_id,
                "pos_image": ctx.image_path,
                "neg_ids": [c.page_id for c in negativos],
                "neg_images": [c.image_path for c in negativos],
                "label": 1,
                "generator": generator_name,
            }
            novas += 1

    linhas = list(registros.values())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in linhas), encoding="utf-8")
    log.info("pares=%d (novos=%d) de %d paginas -> %s",
             len(linhas), novas, len(contexts), out_path)
    return linhas


def main():
    ap = argparse.ArgumentParser(
        description="Gera pares sinteticos (consulta pt-BR -> pagina de plano) para o finetuning.")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS_DIR,
                    help="diretorio do benchmark pt-BR (corpus.jsonl/queries.jsonl/qrels.tsv)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="arquivo pairs.jsonl de saida (ou o diretorio que vai conte-lo)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="consultas por pagina")
    ap.add_argument("--neg", type=int, default=DEFAULT_NEG, help="negativos por par")
    ap.add_argument("--limit", type=int, default=None, help="usa as N primeiras paginas")
    ap.add_argument("--force", action="store_true",
                    help="ignora o pairs.jsonl existente e regera tudo")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--generator", choices=sorted(GENERATORS), default="template",
                    help="backend de geracao de consultas (template = offline/deterministico)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    linhas = make_pairs(a.corpus, a.out, generate_queries=GENERATORS[a.generator],
                        k=a.k, neg=a.neg, limit=a.limit, force=a.force, seed=a.seed,
                        generator_name=a.generator)
    print(f"{len(linhas)} pares em {_resolve_out(a.out)}")


if __name__ == "__main__":
    main()
