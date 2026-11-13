"""Gera dados sinteticos de treino (consulta -> pagina relevante) para o finetuning pt-BR.

Entrada: o diretorio do benchmark pt-BR dos planos, no layout produzido por
`data/govexe/build_planos_benchmark.py` (e lido por `benchmark.common.datasets.load_local`):

    <corpus_dir>/corpus.jsonl    {"id": "<tse_seq>_p<NN>", "image_path": "..."}
                                 {"id": "<item_id>", "text": "..."}   # item TEXTUAL (v4)
    <corpus_dir>/queries.jsonl   {"id": "<tse_seq>_<destaque|resumo|eixoN>", "text": "..."}
    <corpus_dir>/qrels.tsv       <query_id>\t<page_id>\t1

`corpus.jsonl` e obrigatorio; `queries.jsonl`/`qrels.tsv` sao opcionais e, quando existem,
alimentam o contexto do gerador (destaque/resumo/eixos do plano daquela pagina).

Um item do corpus e de UMA modalidade: `image_path` -> pagina (modalidade "imagem"), ou um
campo de texto (`text`, ou `a_repr` no layout do ensemble) -> modalidade "texto". E assim
que os pares cobrem TODOS os dominios que o unificado denso embedda (Semana 10).

Saida: `<out>/pairs.jsonl` -- uma linha JSON por par (query, item positivo, negativos):

    {"id": "<sha1 curto de query+positivo>",    # chave de dedup/idempotencia
     "query": "quais as propostas do plano BR001 para saude publica?",
     "pos_id": "BR001_p00",                     # id do item no corpus.jsonl
     "pos_modality": "imagem",                  # "imagem" | "texto"
     "pos_image": "...../pages/BR001_p00.png",  # pagina RELEVANTE (label=1); "" se texto
     "pos_text": "",                            # texto RELEVANTE (label=1); "" se imagem
     "neg_ids": ["BR002_p03", ...],             # len == --neg (ou menos, corpus pequeno)
     "neg_images": [".../BR002_p03.png", ...],  # amostradas de OUTRAS paginas
     "label": 1,
     "generator": "template"}

Contrato com o treinador (`finetune/train_lora.py`): cada linha e um exemplo contrastivo
pronto: a query (texto) e o anchor, o positivo e `pos_image` OU `pos_text` (conforme
`pos_modality`) e `neg_images` sao os negativos daquele exemplo (hard negatives quando o
gerador real entrar; por ora, negativos aleatorios de outros planos). O treinador le o
jsonl linha a linha, nao precisa do corpus original e pode ignorar `neg_images` se usar
apenas in-batch negatives. Linhas antigas (sem `pos_modality`/`pos_text`) continuam
valendo: o treinador assume "imagem".

A geracao de consultas e plugavel: `generate_queries(context: PageContext) -> list[str]`.
O default (`template_generate_queries`) e deterministico e offline -- roda nos testes e
sem API nenhuma. O outro backend (`llm_generate_queries`, `--generator llm`) delega a um
`chat_fn(prompt) -> str` injetado: este modulo monta o prompt e limpa a resposta, mas NAO
carrega modelo nem fala com a rede. Quem liga o LLM (local no H100, ou API) e o runner,
via `make_pairs(..., generator_name="llm", chat_fn=...)`.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import random
import re
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

MODALIDADE_IMAGEM = "imagem"
MODALIDADE_TEXTO = "texto"
MODALIDADES = (MODALIDADE_IMAGEM, MODALIDADE_TEXTO)

# Campos do corpus.jsonl que carregam o texto de um item textual, em ordem de preferencia
# (`text` e o layout do benchmark; `a_repr` e o do ensemble/corpus v4).
TEXT_FIELDS = ("text", "a_repr")


@dataclass(frozen=True)
class PageContext:
    """Tudo o que o gerador de consultas enxerga de um ITEM do corpus (sem IO proprio).

    O nome vem de quando todo item era uma pagina; no corpus v4 um item tambem pode ser
    textual (`modality == "texto"`, `image_path == ""`).
    """

    page_id: str
    image_path: str
    plan_id: str
    k: int                                  # quantas consultas pedir para este item
    text: str | None = None                 # texto do item, quando o corpus trouxer
    plan_texts: tuple[str, ...] = ()        # destaque/resumo/eixos do plano (queries.jsonl)
    modality: str = MODALIDADE_IMAGEM       # "imagem" | "texto"

    @property
    def pos_value(self) -> str:
        """O POSITIVO em si: o caminho da pagina, ou o texto do item."""
        return (self.text or "") if self.modality == MODALIDADE_TEXTO else self.image_path


QueryGenerator = Callable[[PageContext], list[str]]
ChatFn = Callable[[str], str]       # fronteira com o LLM: prompt -> resposta crua

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


LLM_PROMPT = """Voce ajuda a montar um buscador de planos de governo brasileiros.

Leia o trecho do plano {plano} abaixo e escreva {k} consultas em portugues do Brasil que um
ELEITOR digitaria num buscador e que sao respondidas por esse trecho.

Regras:
- uma consulta por linha, sem numeracao, sem marcadores, sem aspas, sem comentarios;
- linguagem de eleitor (curta, direta), nao copie frases inteiras do trecho;
- cada consulta precisa ser respondida pelo trecho; nao invente o que nao esta la;
- nao repita consultas.

Trecho do plano {plano}:
{contexto}
"""

NO_CHAT_FN = (
    "backend de LLM nao ligado: llm_generate_queries precisa de chat_fn=<callable(prompt)->str>"
    " (via make_pairs(..., generator_name='llm', chat_fn=...)). Para rodar offline, use"
    " --generator template."
)

# Marcadores de lista que o LLM costuma prefixar em cada linha ("1. ", "1) ", "- ", "* ").
MARKER_CHARS = "-*" + chr(0x2022) + chr(0x2013)      # hifen, asterisco, bullet, en dash
QUOTE_CHARS = "\"'" + chr(0x201C) + chr(0x201D)      # aspas retas e curvas, em volta da query
LIST_MARKERS = re.compile(
    rf"^\s*(?:[{re.escape(MARKER_CHARS)}]+|\(?\d{{1,2}}\s*[.)\]:{re.escape(MARKER_CHARS)}])\s*")
CONTEXT_CHARS = 1200      # quanto do contexto do item entra no prompt


def build_query_prompt(ctx: PageContext) -> str:
    """Prompt pt-BR pedindo `ctx.k` consultas respondidas pelo item `ctx`."""
    partes = [t.strip() for t in (ctx.text, *ctx.plan_texts) if t and t.strip()]
    contexto = "\n".join(dict.fromkeys(partes))[:CONTEXT_CHARS].strip()
    if not contexto:      # item so-imagem sem OCR nem destaque/resumo: sobra o id da pagina
        contexto = f"(sem texto extraido; pagina {ctx.page_id} do plano {ctx.plan_id})"
    return LLM_PROMPT.format(plano=ctx.plan_id, k=max(ctx.k, 1), contexto=contexto)


def parse_query_response(resposta: str, k: int) -> list[str]:
    """Uma consulta por linha -> lista limpa, sem vazias, sem duplicatas, com no maximo `k`."""
    limite = max(k, 0)
    vistas: dict[str, str] = {}
    for linha in (resposta or "").splitlines() if limite else ():
        query = LIST_MARKERS.sub("", linha).strip().strip(QUOTE_CHARS).strip()
        if not query:
            continue
        vistas.setdefault(query.casefold(), query)
        if len(vistas) >= limite:
            break
    return list(vistas.values())


def llm_generate_queries(ctx: PageContext, *, chat_fn: ChatFn | None = None) -> list[str]:
    """Backend de LLM: monta o prompt, chama `chat_fn` e limpa a resposta.

    `chat_fn(prompt) -> str` e a UNICA fronteira com o modelo -- este modulo nao carrega
    nem baixa nada. O runner (LLM local no H100, ou API) passa o callable via
    `make_pairs(..., generator_name="llm", chat_fn=...)`; os testes passam um duble.
    Sem backend ligado, erra alto em vez de cair escondido para o template.
    """
    if chat_fn is None:
        raise RuntimeError(NO_CHAT_FN)
    return parse_query_response(chat_fn(build_query_prompt(ctx)), ctx.k)


GENERATORS: dict[str, QueryGenerator] = {
    "template": template_generate_queries,
    "llm": llm_generate_queries,
}


def resolve_generator(generator_name: str = "template",
                      chat_fn: ChatFn | None = None) -> QueryGenerator:
    """Gerador nomeado, com o backend de LLM ja ligado quando ele for o escolhido.

    `GENERATORS` guarda funcoes de UM argumento (`ctx`), que e como `make_pairs` chama o
    gerador; o `chat_fn` entra aqui, preso por `functools.partial`.
    """
    gen = GENERATORS[generator_name]
    if gen is llm_generate_queries and chat_fn is not None:
        return functools.partial(llm_generate_queries, chat_fn=chat_fn)
    return gen


# -- leitura do benchmark pt-BR -------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def plan_id_of(page_id: str) -> str:
    """`BR001_p07` -> `BR001` (layout do build_planos_benchmark); senao, o proprio id."""
    head, sep, tail = page_id.rpartition("_p")
    return head if sep and tail.isdigit() else page_id


def _item_text(doc: dict) -> str:
    """Texto de um item do corpus (`text` ou `a_repr`), ja limpo -- "" quando nao ha."""
    for campo in TEXT_FIELDS:
        valor = doc.get(campo)
        if valor and str(valor).strip():
            return str(valor).strip()
    return ""


def item_modality(doc: dict) -> str:
    """Modalidade de um item do corpus: tem `image_path` -> "imagem"; senao texto.

    A imagem manda quando o item traz as duas coisas (uma pagina com OCR continua sendo
    uma pagina) -- e o que mantem o corpus antigo gerando exatamente os mesmos pares.
    """
    if (doc.get("image_path") or "").strip():
        return MODALIDADE_IMAGEM
    return MODALIDADE_TEXTO if _item_text(doc) else MODALIDADE_IMAGEM


def load_contexts(corpus_dir: str | Path, k: int = DEFAULT_K) -> list[PageContext]:
    """Monta um `PageContext` por item do corpus, anexando os textos do plano."""
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
            text=_item_text(d) or None,
            plan_texts=tuple(dict.fromkeys(texts)),
            modality=item_modality(d),
        ))
    return ctxs


# -- geracao dos pares ----------------------------------------------------------------


def pair_id(query: str, pos_value: str) -> str:
    """Chave de dedup: o mesmo (query, positivo) nunca vira duas linhas.

    `pos_value` e o caminho da pagina (modalidade imagem) ou o proprio texto do item
    (modalidade texto) -- os ids das linhas so-imagem ja escritas nao mudam.
    """
    return hashlib.sha1(f"{query}\x00{pos_value}".encode()).hexdigest()[:16]


def sample_negatives(ctx: PageContext, pool: list[PageContext], n: int,
                     rng: random.Random) -> list[PageContext]:
    """Amostra `n` paginas != `ctx`, preferindo paginas de OUTROS planos.

    So entram itens COM imagem: `neg_images` e uma lista de caminhos (contrato do
    treinador), e um item textual so renderia string vazia ali. Os negativos textuais
    chegam pelos in-batch negatives (os positivos dos outros exemplos do batch).
    """
    com_imagem = [c for c in pool if c.image_path]
    outros_planos = [c for c in com_imagem if c.plan_id != ctx.plan_id]
    outras_paginas = [c for c in com_imagem if c.page_id != ctx.page_id]
    candidatos = outros_planos or outras_paginas
    if not candidatos:
        return []
    return rng.sample(candidatos, min(n, len(candidatos)))


def _resolve_out(out: str | Path) -> Path:
    """`--out` aceita o arquivo (`.../pairs.jsonl`) ou o diretorio que vai conte-lo."""
    p = Path(out)
    return p if p.suffix == ".jsonl" else p / "pairs.jsonl"


def make_pairs(corpus_dir: str | Path, out: str | Path,
               generate_queries: QueryGenerator | None = None,
               k: int = DEFAULT_K, neg: int = DEFAULT_NEG, limit: int | None = None,
               force: bool = False, seed: int = SEED,
               generator_name: str = "template",
               chat_fn: ChatFn | None = None) -> list[dict]:
    """Escreve `pairs.jsonl` com K consultas por item + `neg` negativos, e devolve as linhas.

    Cada item vira um par da SUA modalidade (pagina -> `pos_image`, texto -> `pos_text`).

    O gerador vem de `generate_queries` (um callable de um argumento, usado nos testes) ou,
    quando nao informado, de `generator_name` via `resolve_generator` -- que e onde o
    `chat_fn` do backend de LLM e ligado.

    E idempotente: o jsonl existente e relido como cache; itens que ja tem pares nao
    voltam ao gerador (que no caminho real custa chamada de LLM) e as linhas antigas sao
    preservadas na ordem original. A dedup final e por `pair_id(query, positivo)`, entao
    rodar duas vezes nao duplica nada. `force=True` ignora o cache e refaz tudo.

    Os negativos vem de um `random.Random` semeado por `(seed, page_id)`, logo a mesma
    pagina sempre recebe os mesmos negativos -- independente da ordem/limite da rodada.
    """
    if generate_queries is None:
        generate_queries = resolve_generator(generator_name, chat_fn)
    elif chat_fn is not None and generate_queries is llm_generate_queries:
        generate_queries = functools.partial(llm_generate_queries, chat_fn=chat_fn)

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
            rid = pair_id(query, ctx.pos_value)
            if rid in registros:
                continue
            registros[rid] = {
                "id": rid,
                "query": query,
                "pos_id": ctx.page_id,
                "pos_modality": ctx.modality,
                "pos_image": ctx.image_path if ctx.modality == MODALIDADE_IMAGEM else "",
                "pos_text": ctx.pos_value if ctx.modality == MODALIDADE_TEXTO else "",
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
                    help="backend de geracao de consultas (template = offline/deterministico; "
                         "llm exige um chat_fn ligado por um runner, nao roda pela CLI pura)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    linhas = make_pairs(a.corpus, a.out, k=a.k, neg=a.neg, limit=a.limit, force=a.force,
                        seed=a.seed, generator_name=a.generator)
    print(f"{len(linhas)} pares em {_resolve_out(a.out)}")


if __name__ == "__main__":
    main()
