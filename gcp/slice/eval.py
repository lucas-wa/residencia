"""Driver de avaliacao da fatia GovExe: roda um retriever sobre um conjunto de avaliacao
e devolve QUALIDADE (recall@k, nDCG@k, MRR) e EFICIENCIA (latencia media por consulta).

E o que fecha a comparacao entre arquiteturas (`A` x `A+audio`, registro em
`gcp/slice/spaces.py`): cada execucao produz UMA linha -- uma arquitetura, uma tarefa --
e `write_comparison_csv` junta as linhas no CSV que vai para o relatorio.

O retriever e **injetado** (`fn(texto) -> list[item_id]`, melhor primeiro). Esse e o ponto
de desacoplamento: aqui nao entra checkpoint, GPU nem Cloud SQL, e o mesmo driver serve a
consulta multi-espaco de producao e a um duble no teste. Em producao quem o monta e o
`main()` abaixo, adaptando `search_architecture` (que devolve `[(item_id, score_rrf), ...]`)
para a lista de ids que as metricas esperam.

E e esse desacoplamento que deixa a arquitetura `B` entrar sem mexer no caminho pgvector:
B nao tem espaco pgvector nenhum (`spaces.uses_pgvector("B") is False`) -- o eixo documento
ranqueia as paginas por ColQwen2.5 + MaxSim em memoria (`gcp/slice/doc_specialist.py`) e
entrega o MESMO `retrieve(texto) -> ids`. `main()` so escolhe qual montar; daqui para baixo
as metricas, a latencia e a linha do CSV sao as mesmas, que e o que torna `B,ptbr_planos,...`
comparavel DIRETO com `A,ptbr_planos,...`.

Cada execucao mede UMA tarefa, e e o CORPUS que diz qual: paginas (`doc_image`) ou clipes
de audio. So os espacos que indexam a modalidade daquele corpus entram na fusao
(`spaces_for_corpus` -> `search_architecture(only_spaces=...)`) -- na tarefa doc-visual o
CLAP nao indexa nada do corpus, e fundi-lo entremearia clipes de audio entre as paginas
julgadas, fazendo `A+audio` diferir de `A` justamente onde a spec (secao 5) as declara
IDENTICAS.

As metricas NAO sao reimplementadas: vem de `benchmark/common/metrics.py`, que ja trata os
casos de borda (query sem relevante -> 0.0 e nao NaN; `k` maior que o ranking; relevante
fora do ranking). A lista fundida pode ser mais longa que `k` -- nao se corta antes, porque
cada metrica trunca no seu proprio `k` (`ranked_ids[:k]`).

Uso:
    export PGPASSWORD="$(gcloud secrets versions access latest --secret=govexe-pg-password)"
    CSV=results/comparacao.csv
    python -m gcp.slice.eval --arch A       --data-dir data/govexe/planos --out "$CSV"
    python -m gcp.slice.eval --arch A+audio --data-dir data/govexe/planos --out "$CSV"

    # B nao usa Cloud SQL: nem PGPASSWORD, nem proxy. So a GPU e as paginas.
    python -m gcp.slice.eval --arch B --data-dir data/govexe/ptbr_planos \
        --pages-dir ~/govexe-pages --out "$CSV"

A segunda rodada ACRESCENTA a sua linha ao mesmo CSV (a anterior fica), que e como a
comparacao entre arquiteturas se forma.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

# permite tanto `python gcp/slice/eval.py` quanto `python -m gcp.slice.eval`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.common.metrics import aggregate, mrr, ndcg_at_k, recall_at_k  # noqa: E402
from gcp.slice.spaces import (  # noqa: E402
    SPACES,
    spaces_for_modalities,
    specialists_for_modalities,
)

log = logging.getLogger(__name__)

__all__ = ["DEFAULT_K_VALUES", "corpus_modalities", "evaluate_architecture", "main",
           "specialist_for_corpus", "spaces_for_corpus", "write_comparison_csv"]

DEFAULT_K_VALUES = (1, 5, 10)

# -- modalidade-alvo do conjunto de avaliacao ---------------------------------------------
#
# Modalidades que o registro conhece -- a uniao do que os espacos declaram indexar.
MODALIDADES = frozenset(m for espaco in SPACES.values() for m in espaco.modalities)
# O MESMO clipe entra no indice duas vezes, com o MESMO `item_id` (e e disso que o RRF
# vive): como `audio_asr` (Whisper -> texto -> Qwen3-VL) e, em `A+audio`, como
# `audio_native` (forma de onda -> CLAP). Uma tarefa de audio cobra as duas: em `A` sobra
# so a trilha ASR (a baseline da spec secao 5), em `A+audio` as duas se fundem.
MODALIDADES_DE_AUDIO = ("audio_asr", "audio_native")
MODALIDADE_DE_IMAGEM = "doc_image"
MODALIDADE_DE_TEXTO = "text"

# Os nomes acima sao os MESMOS do registro: um erro de digitacao aqui nao casaria com
# espaco nenhum e viraria uma fusao silenciosamente errada. Falha no import, nao na rodada.
_FORA_DO_REGISTRO = ({*MODALIDADES_DE_AUDIO, MODALIDADE_DE_IMAGEM, MODALIDADE_DE_TEXTO}
                     - MODALIDADES)
if _FORA_DO_REGISTRO:
    raise ValueError(f"modalidade que nenhum espaco indexa: {sorted(_FORA_DO_REGISTRO)} "
                     f"(o registro gcp/slice/spaces.py conhece: {sorted(MODALIDADES)})")


def _campo(item, nome: str):
    """Le `nome` de uma query vinda como dict OU como `benchmark.common.types.Query`.

    `load_local` devolve o dataclass; os testes (e qualquer conjunto montado a mao) usam
    dicts. Aceitar os dois evita converter o conjunto so para avaliar.
    """
    if isinstance(item, Mapping):
        return item.get(nome)
    return getattr(item, nome, None)


def _relevantes(qrels, qid: str) -> dict[str, float]:
    """Julgamentos da query `qid` como `{item_id: ganho}`, no formato que o nDCG espera.

    Aceita as duas formas que circulam no repo: o `{qid: {docid: score}}` do `load_local`
    e o `{qid: {docid, ...}}` binario (conjunto). Julgamento <= 0 e NEGATIVO e cai fora --
    deixa-lo entrar inflaria o denominador do recall (`len(relevant_ids)`) com documentos
    que nao contam como acerto.
    """
    rel = qrels.get(qid) if qrels else None
    if not rel:
        return {}
    if isinstance(rel, Mapping):
        return {str(d): float(s) for d, s in rel.items() if float(s) > 0}
    return {str(d): 1.0 for d in rel}      # set/list/tuple -> binario


def _texto(query) -> str:
    """Texto da consulta, ou `ValueError` na hora se faltar.

    `Query.text` e opcional (ha query so-imagem no dataclass), e um `retrieve(None)` so
    estouraria la na frente, dentro do embedder, no meio de uma rodada ja paga.
    """
    texto = _campo(query, "text")
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError(f"query {_campo(query, 'id')!r} sem texto para consultar "
                         f"(text={texto!r}): este driver avalia consulta TEXTUAL")
    return texto


def _modalidade_do_doc(doc, audio_suffixes: tuple[str, ...]) -> str | None:
    """Modalidade de UM item do corpus, ou `None` quando nao da para dizer.

    Tres sinais, nesta ordem (do mais explicito ao mais inferido):

    1. o proprio `id`, quando ele carrega a modalidade -- `<modalidade>:<hash da uri>` e o
       id que `collector.default_item_id` gera e com que `audio_collector.py` emite os
       clipes (`audio_asr:552e77e1...`), entao e o mesmo id que o qrels referencia;
    2. um caminho de audio (campo proprio ou extensao conhecida) -> clipe;
    3. imagem -> `doc_image` (pagina de documento, o corpus desta fatia; uma foto seria
       `image`, mas as duas moram no mesmo espaco, entao a distincao nao mudaria a fusao),
       e texto nativo -> `text`.
    """
    doc_id = str(_campo(doc, "id") or "")
    prefixo, _, resto = doc_id.partition(":")
    if resto and prefixo in MODALIDADES:
        return prefixo
    if _campo(doc, "audio_path"):                 # campo proprio: qualquer extensao
        return MODALIDADES_DE_AUDIO[0]
    for campo in ("uri", "image_path", "path"):   # sem campo proprio: pela extensao
        caminho = _campo(doc, campo)
        if caminho and str(caminho).lower().endswith(audio_suffixes):
            return MODALIDADES_DE_AUDIO[0]
    if _campo(doc, "image_path"):
        return MODALIDADE_DE_IMAGEM
    if _campo(doc, "text"):
        return MODALIDADE_DE_TEXTO
    return None


def corpus_modalities(docs) -> set[str]:
    """Modalidades que o corpus de avaliacao cobra do indice (vazio = indecidivel).

    Um clipe conta pelas DUAS modalidades de audio (`MODALIDADES_DE_AUDIO`): e o mesmo
    item indexado pelo caminho ASR e pelo nativo, sob o mesmo `item_id`.

    Conjunto vazio nao e erro -- e "nao sei dizer", e quem chama cai de volta no
    comportamento antigo (fundir todos os espacos da arquitetura).
    """
    from gcp.slice.collector import AUDIO_SUFFIXES   # import tardio: so para as extensoes

    modalidades: set[str] = set()
    for doc in docs:
        inferida = _modalidade_do_doc(doc, AUDIO_SUFFIXES)
        if inferida in MODALIDADES_DE_AUDIO:
            modalidades.update(MODALIDADES_DE_AUDIO)
        elif inferida:
            modalidades.add(inferida)
    return modalidades


def spaces_for_corpus(arch: str, docs, *, tarefa: str = "") -> set[str] | None:
    """Nomes dos espacos de `arch` que indexam o corpus -- ou `None` para "todos".

    E a ponte entre o conjunto de avaliacao e `search_architecture(only_spaces=...)`:
    quem decide quais espacos indexam a modalidade do corpus e o registro
    (`spaces.spaces_for_modalities`), nunca uma lista escrita aqui.

    `ValueError` quando a arquitetura nao tem espaco algum para aquele corpus: avaliar
    assim devolveria 0.0 em tudo -- um numero que parece metrica e nao e.
    """
    alvo = corpus_modalities(docs)
    rotulo = f" ({tarefa})" if tarefa else ""
    if not alvo:
        log.warning("nao deu para inferir a modalidade do corpus%s: fundindo TODOS os "
                    "espacos de %s", rotulo, arch)
        return None
    espacos = spaces_for_modalities(arch, alvo)
    if not espacos:
        raise ValueError(
            f"a arquitetura {arch!r} nao indexa nada do corpus{rotulo} "
            f"(modalidades: {', '.join(sorted(alvo))}): nao ha o que avaliar")
    nomes = {espaco.name for espaco in espacos}
    log.info("corpus%s: modalidades %s -> espacos fundidos: %s", rotulo,
             ", ".join(sorted(alvo)), ", ".join(sorted(nomes)))
    return nomes


def specialist_for_corpus(arch: str, docs, *, tarefa: str = ""):
    """O `Specialist` (late-interaction) de `arch` que ranqueia ESTE corpus.

    Gemeo de `spaces_for_corpus` para o caminho sem pgvector. Mesma regra: quem diz quem
    ranqueia o que e o registro (`spaces.specialists_for_modalities`), nunca uma lista
    escrita aqui -- um corpus de audio simplesmente nao casa com o eixo documento de `B`,
    e isso tem de parar a rodada em vez de virar 0.0 em tudo.

    Devolve UM especialista. `B` hoje e o primeiro braco do ensemble; quando um segundo
    braco entrar, os rankings se fundem por RRF (como ja acontece entre espacos) e e aqui
    que a fusao vai morar -- ate la, dois especialistas para a mesma tarefa falham, em
    vez de um deles ser escolhido em silencio.
    """
    alvo = corpus_modalities(docs)
    rotulo = f" ({tarefa})" if tarefa else ""
    if not alvo:
        raise ValueError(f"nao deu para inferir a modalidade do corpus{rotulo}: o "
                         f"especialista de {arch!r} ranqueia por modalidade declarada")
    candidatos = specialists_for_modalities(arch, alvo)
    if not candidatos:
        raise ValueError(
            f"a arquitetura {arch!r} nao ranqueia nada do corpus{rotulo} "
            f"(modalidades: {', '.join(sorted(alvo))}): nao ha o que avaliar")
    if len(candidatos) > 1:
        raise ValueError(
            f"{len(candidatos)} especialistas de {arch!r} ranqueiam o corpus{rotulo} "
            f"({', '.join(e.name for e in candidatos)}): a fusao entre bracos do ensemble "
            f"ainda nao esta implementada")
    log.info("corpus%s: modalidades %s -> especialista: %s", rotulo,
             ", ".join(sorted(alvo)), candidatos[0].name)
    return candidatos[0]


def evaluate_architecture(arch: str, queries: list[dict], qrels,
                          retrieve: Callable[[str], list[str]], *,
                          k_values: Iterable[int] = DEFAULT_K_VALUES) -> dict:
    """Avalia `retrieve` sobre `queries`/`qrels` e devolve as metricas medias + latencia.

    Para cada query: cronometra UMA chamada a `retrieve(texto)` (so o retrieve entra no
    relogio -- montar as metricas nao e custo da arquitetura), calcula recall@k/nDCG@k/MRR
    contra os julgamentos daquela query e guarda a linha; no fim, `aggregate` tira a media
    de todas as chaves, latencia inclusa.

    `arch` nao muda o calculo: viaja junto para a linha sair auto-contida no CSV comparativo.
    Conjunto vazio devolve so `arquitetura`/`n_queries` -- metrica media de zero query seria
    um numero inventado.
    """
    k_values = tuple(k_values)
    por_query: list[dict[str, float]] = []
    for query in queries:
        qid = str(_campo(query, "id"))
        texto = _texto(query)
        relevante = _relevantes(qrels, qid)
        relevante_ids = set(relevante)

        inicio = time.perf_counter()
        ranked_ids = [str(i) for i in retrieve(texto)]
        latencia = time.perf_counter() - inicio

        linha: dict[str, float] = {}
        for k in k_values:
            linha[f"recall@{k}"] = recall_at_k(ranked_ids, relevante_ids, k)
            linha[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevante, k)
        linha["mrr"] = mrr(ranked_ids, relevante_ids)
        linha["latencia_s"] = latencia
        por_query.append(linha)
        log.debug("%s %s: %d hits em %.3fs", arch, qid, len(ranked_ids), latencia)

    if not por_query:
        log.warning("arquitetura %s: conjunto de avaliacao vazio", arch)
    return {"arquitetura": arch, "n_queries": len(por_query), **aggregate(por_query)}


def write_comparison_csv(path: str | Path, rows: list[dict]) -> None:
    """Escreve o CSV comparativo -- uma linha por (arquitetura, tarefa).

    As colunas sao a UNIAO das chaves das linhas, na ordem em que aparecem: duas
    arquiteturas avaliadas com `k_values` diferentes continuam cabendo no mesmo arquivo (o
    que faltar fica vazio) em vez de derrubar a escrita com `ValueError: dict contains
    fields not in fieldnames`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colunas: list[str] = []
    for row in rows:
        for coluna in row:
            if coluna not in colunas:
                colunas.append(coluna)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not colunas:
            log.warning("nenhuma linha para %s: arquivo vazio", path)
            return
        w = csv.DictWriter(f, fieldnames=colunas, restval="")
        w.writeheader()
        w.writerows(rows)


# -- producao: monta o retriever multi-espaco ----------------------------------------------


def _text_embedder(espaco, *, device: str, dtype: str) -> Callable[[list[str]], object]:
    """Embedder de TEXTO do espaco (`list[str] -> np.ndarray`), carregado sob demanda.

    Import tardio de proposito: e o que mantem este modulo (e os seus testes) livres de
    torch/transformers. A consulta de cada espaco tem de ser embutida pelo backbone DAQUELE
    espaco -- Qwen3-VL e CLAP nem tem a mesma dimensao.
    """
    if espaco.kind == "qwen3vl":
        from benchmark.models.qwen3vl_embed import embed_texts
        # `dim` = a largura declarada do espaco: a consulta sai na mesma dimensao da tabela.
        return lambda ts: embed_texts(ts, model_id=espaco.model_id, device=device,
                                      dtype=dtype, dim=espaco.dim)
    if espaco.kind == "clap":
        from benchmark.models.clap_embed import embed_texts as clap_embed_texts
        # o CLAP nao e Matryoshka: nada de `dim`/`dtype` (ver `embed_job._embed_clap`).
        return lambda ts: clap_embed_texts(ts, model_id=espaco.model_id, device=device)
    raise ValueError(f"sem embedder de texto para o espaco {espaco.name!r} "
                     f"(kind={espaco.kind!r})")


def _stores_e_embedders(arch: str, a: argparse.Namespace,
                        apenas: set[str] | None = None) -> tuple[dict, dict]:
    """Um store e um embedder por espaco da arquitetura, na chave que `search_architecture`
    espera (`Space.name`). Com `--local-store DIR`, cada espaco reabre o seu
    `DIR/<espaco>.json` (modo dev); sem ele, a tabela do espaco no Cloud SQL.

    `apenas` (os espacos que a tarefa vai consultar) evita abrir conexao e montar embedder
    de espaco que nao entra na fusao -- numa rodada paga, o checkpoint do CLAP nem precisa
    sair do disco para medir a tarefa doc-visual."""
    from gcp.slice.embed_job import InMemoryVectorStore, PgVectorStore
    from gcp.slice.spaces import architecture_spaces

    stores, embedders = {}, {}
    for espaco in architecture_spaces(arch):
        if apenas is not None and espaco.name not in apenas:
            continue
        if a.local_store:
            stores[espaco.name] = InMemoryVectorStore(
                path=str(Path(a.local_store) / f"{espaco.name}.json"))
        else:
            stores[espaco.name] = PgVectorStore.connect(
                host=a.pg_host, port=a.pg_port, dbname=a.pg_db, user=a.pg_user,
                password_env=a.password_env, table=espaco.table)
        embedders[espaco.name] = _text_embedder(espaco, device=a.device, dtype=a.dtype)
    return stores, embedders


def _linhas_anteriores(path: Path, arch: str, tarefa: str) -> list[dict]:
    """Linhas ja gravadas em `path`, menos a desta (arquitetura, tarefa).

    Cada execucao avalia UMA arquitetura, entao o CSV comparativo se forma ao longo de
    varias: reescrever do zero apagaria a linha da rodada anterior, e so acrescentar
    duplicaria a linha numa reexecucao.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)
                if (r.get("arquitetura"), r.get("tarefa")) != (arch, tarefa)]


def _retrieve_late_interaction(arch: str, docs, a: argparse.Namespace, *, tarefa: str,
                               retriever=None) -> Callable[[str], list[str]]:
    """O `retrieve` de uma arquitetura SEM pgvector (hoje: `B`, eixo documento).

    Import tardio de `doc_specialist` pelo mesmo motivo de `_text_embedder`: manter este
    modulo (e os seus testes) livres de torch. `retriever` injetado e o que permite ao
    teste exercitar o roteamento com um duble.
    """
    from gcp.slice.doc_specialist import build_doc_retriever

    especialista = specialist_for_corpus(arch, docs, tarefa=tarefa)
    return build_doc_retriever(docs, specialist=especialista, device=a.device,
                               dtype=a.dtype, data_dir=a.data_dir,
                               pages_dir=getattr(a, "pages_dir", None),
                               retriever=retriever)


def main(argv: list[str] | None = None, *, retriever=None) -> int:
    from gcp.slice.embed_job import add_pg_arguments
    from gcp.slice.query import MULTI_TOP_K, search_architecture
    from gcp.slice.spaces import ARCHITECTURES, uses_pgvector

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=sorted(ARCHITECTURES),
                    help="arquitetura a avaliar (registro em gcp/slice/spaces.py)")
    ap.add_argument("--data-dir", required=True,
                    help="pasta com queries.jsonl/corpus.jsonl/qrels.tsv")
    ap.add_argument("--out", required=True, help="CSV comparativo (acumulado entre rodadas)")
    ap.add_argument("--tarefa", default=None,
                    help="nome da tarefa na linha do CSV (default: o nome de --data-dir)")
    ap.add_argument("--k", type=int, default=MULTI_TOP_K,
                    help=f"profundidade pedida a cada espaco antes do RRF (default: "
                         f"{MULTI_TOP_K}); nao se aplica a arquitetura sem pgvector, que "
                         f"ranqueia o corpus inteiro em memoria")
    ap.add_argument("--pages-dir", default=None,
                    help="diretorio com as paginas do corpus (so para arquitetura sem "
                         "pgvector): na VM elas vem do Bronze, e nao do repo, que as "
                         "exclui do tarball por tamanho")
    ap.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES),
                    help="cortes das metricas (default: %(default)s)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    add_pg_arguments(ap)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Duas flags herdadas de `add_pg_arguments` mudam de sentido numa avaliacao MULTI-espaco
    # -- aqui nao ha uma tabela nem um indice, ha um por espaco. Falhar e melhor que aceitar
    # a flag e ignora-la em silencio.
    if a.table:
        ap.error("--table nao se aplica a avaliacao multi-espaco: a tabela de cada espaco "
                 "vem do registro (gcp/slice/spaces.py)")
    if a.local_store is True:
        ap.error("--local-store aqui pede um DIRETORIO (um indice por espaco, "
                 "<dir>/<espaco>.json), e nao um arquivo so")
    if a.pages_dir and uses_pgvector(a.arch):
        ap.error(f"--pages-dir nao se aplica a {a.arch}: ela ranqueia pelo indice "
                 "pgvector, cujos binarios ja estao no Bronze")

    from benchmark.common.datasets import load_local
    queries, docs, qrels = load_local(a.data_dir)
    tarefa = a.tarefa or Path(a.data_dir).name

    # O ROTEAMENTO da avaliacao, e a unica bifurcacao entre as arquiteturas: quem tem
    # espaco pgvector consulta o Cloud SQL e funde por RRF; quem nao tem (`B`) ranqueia
    # por late interaction em memoria. Quem decide e o registro, nao o nome da arquitetura.
    if uses_pgvector(a.arch):
        # A tarefa manda nos espacos: so os que indexam a modalidade DESTE corpus entram na
        # fusao. E o que mantem `A` e `A+audio` identicas na tarefa doc-visual (spec secao 5)
        # -- fundir o CLAP ali entremearia clipes de audio entre as paginas julgadas.
        espacos_da_tarefa = spaces_for_corpus(a.arch, docs, tarefa=tarefa)
        stores, embedders = _stores_e_embedders(a.arch, a, apenas=espacos_da_tarefa)

        # `search_architecture` devolve [(item_id, score_rrf), ...]; as metricas querem so
        # os ids (e na ordem). Nao se corta em `k` aqui: cada metrica trunca no seu corte.
        def retrieve(texto: str) -> list[str]:
            return [item_id for item_id, _ in search_architecture(
                a.arch, texto, k=a.k, stores=stores, embedders=embedders,
                only_spaces=espacos_da_tarefa)]
    else:
        retrieve = _retrieve_late_interaction(a.arch, docs, a, tarefa=tarefa,
                                              retriever=retriever)

    metricas = evaluate_architecture(a.arch, queries, qrels, retrieve,
                                     k_values=tuple(a.k_values))
    linha = {"arquitetura": a.arch, "tarefa": tarefa, **metricas}

    destino = Path(a.out)
    write_comparison_csv(destino, [*_linhas_anteriores(destino, a.arch, tarefa), linha])
    log.info("%s", "  ".join(f"{c}={v}" for c, v in linha.items()))
    print(f"linha gravada em {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
