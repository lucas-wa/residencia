"""Eixo DOCUMENTO da arquitetura B: ranqueamento late-interaction (ColQwen2.5 + MaxSim).

E o primeiro braco do ensemble de especialistas de `fusao/multimodal.md` secao 3. Ao
contrario de `A`/`A+audio`, aqui NAO ha pgvector: o ColQwen2.5 emite um vetor por TOKEN
(matriz [n_tokens, dim]) e a pontuacao e MaxSim --

    score(q, d) = sum_i max_j (q_i . d_j)

-- que uma coluna `vector(n)` ranqueada por cosseno nao sabe representar. Colapsar a
matriz num vetor so para caber na tabela destruiria exatamente a late interaction que se
quer medir. Com as 93 paginas de `ptbr_planos`, o indice cabe em memoria e o MaxSim e
instantaneo: a rodada de B dispensa Cloud SQL inteiro.

Nada aqui e reimplementado: o embedding vem de `benchmark.models.colvision`
(`ColVisionRetriever`, kind `"colqwen2.5"`) e o ranking de
`benchmark.common.index.maxsim_search`. Este modulo so faz a ponte com o driver de
avaliacao: ele devolve um `retrieve(texto) -> [item_id, ...]`, que e a unica coisa que
`gcp/slice/eval.py` sabe consumir -- e e por isso que a linha `B,ptbr_planos,...` sai
comparavel, metrica a metrica, com a linha `A,ptbr_planos,...`.

Os ids do ranking sao os `Doc.id` do `corpus.jsonl` -- os MESMOS que o `qrels.tsv`
referencia (`110002543953_p00`), sem traducao nenhuma pelo caminho.

CUSTO: embutir o corpus acontece UMA vez, em `build_doc_retriever`, de proposito. E a
indexacao (o analogo do `embed_job` de A) e nao pode entrar no cronometro por query do
`evaluate_architecture`; o que a latencia de B mede e embutir a consulta + MaxSim, o
equivalente ao `embed_texts` + `SELECT ... <=>` que a latencia de A mede.

Uso:
    from gcp.slice.doc_specialist import build_doc_retriever
    retrieve = build_doc_retriever(docs, specialist=SPECIALISTS["colqwen25_doc"])
    retrieve("metas para a saude publica")   # -> ['110002543953_p07', ...]
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path

__all__ = ["DEFAULT_BATCH_SIZE", "DocSpecialistRanker", "build_doc_retriever",
           "resolve_image_path"]

log = logging.getLogger(__name__)

# Paginas por lote no `encode_corpus`. E o knob de MEMORIA da rodada: o ColQwen2.5 guarda
# um vetor por token de imagem, entao um lote de paginas grandes e o que estoura os 24 GB
# da L4 (ja aconteceu na prova #2). 4 e o default do `ColVisionRetriever`; baixar para 2 ou
# 1 troca tempo por VRAM e nao muda ranking nenhum -- o MaxSim e por item.
DEFAULT_BATCH_SIZE = 4

# Raiz do repo: o `image_path` do `corpus.jsonl` e relativo a ela.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_image_path(image_path: str, *, data_dir: str | Path | None = None,
                       pages_dir: str | Path | None = None) -> Path:
    """Caminho REAL da pagina a partir do `image_path` do corpus.

    Tres problemas de uma vez, e por isso isto nao e um `Path(x)` solto:

    1. o `corpus.jsonl` foi gerado no Windows e grava `data\\govexe\\...\\p00.png`, com
       barra invertida -- no Linux da VM isso seria UM nome de arquivo, nao um caminho;
    2. o caminho e relativo a raiz do repo, e a avaliacao nem sempre roda de la;
    3. na VM as paginas NAO vem do repo (o tarball as exclui por tamanho): elas sao
       baixadas do Bronze para um diretorio proprio, e `pages_dir` aponta para ele. Como
       o objeto Bronze preserva o nome do arquivo (`collector.blob_name`), casar pelo
       BASENAME e exato -- e o mesmo stem que vira `item_id`.

    `pages_dir`, quando dado, tem precedencia: e o insumo explicito da rodada.
    """
    bruto = str(image_path).replace("\\", "/")
    nome = Path(bruto).name
    candidatos: list[Path] = []
    if pages_dir:
        candidatos.append(Path(pages_dir) / nome)
    candidatos.append(Path(bruto))
    if data_dir:
        candidatos += [Path(data_dir) / bruto, Path(data_dir) / "pages" / nome]
    candidatos.append(_REPO_ROOT / bruto)

    for candidato in candidatos:
        if candidato.is_file():
            return candidato
    raise FileNotFoundError(
        f"pagina do corpus nao encontrada: {image_path!r} (tentados: "
        f"{', '.join(str(c) for c in candidatos)}). Na VM passe --pages-dir apontando "
        f"para onde as paginas do Bronze foram baixadas.")


class DocSpecialistRanker:
    """Indice late-interaction em memoria: embute o corpus uma vez, ranqueia por MaxSim.

    `retriever` e injetado (qualquer objeto com `encode_corpus(docs)` e
    `encode_queries(queries)` devolvendo `list[np.ndarray]` de forma [n_tokens, dim]) --
    e o ponto que permite testar o roteamento e as metricas de B com um duble, sem GPU e
    sem baixar checkpoint nenhum. Em producao quem o monta e `build_doc_retriever`.
    """

    def __init__(self, doc_ids: list[str], doc_embeddings: list, retriever):
        if len(doc_ids) != len(doc_embeddings):
            raise ValueError(f"{len(doc_ids)} ids para {len(doc_embeddings)} embeddings: "
                             "o ranking nao teria como voltar para os ids do qrels")
        self.doc_ids = doc_ids
        self.doc_embeddings = doc_embeddings
        self._retriever = retriever

    def retrieve(self, text: str) -> list[str]:
        """Ranking COMPLETO do corpus para a consulta `text`, melhor primeiro.

        Devolve todos os itens, e nao um top-k: cada metrica de `eval.py` trunca no seu
        proprio `k` (`ranked_ids[:k]`), e com 93 paginas em memoria cortar antes so
        arriscaria zerar um `recall@10` sem economizar nada.
        """
        from benchmark.common.index import maxsim_search
        from benchmark.common.types import Query

        q_mv = self._retriever.encode_queries([Query(id="q", text=text)])
        ordem = maxsim_search(q_mv, self.doc_embeddings, len(self.doc_ids))[0]
        return [self.doc_ids[i] for i in ordem]


def _build_retriever(specialist, *, device: str, dtype: str, batch_size: int):
    """Constroi o `ColVisionRetriever` do especialista -- import tardio, como em `eval.py`.

    O import de `benchmark.models.colvision` arrasta torch/colpali-engine; mante-lo aqui
    dentro e o que permite importar este modulo (e testa-lo) numa maquina sem GPU.
    """
    from benchmark.models.colvision import ColVisionRetriever

    log.info("especialista %s: carregando %s (kind=%s, device=%s)", specialist.name,
             specialist.model_id, specialist.kind, device)
    return ColVisionRetriever(specialist.model_id, kind=specialist.kind, device=device,
                              dtype=dtype, batch_size=batch_size)


def build_doc_retriever(docs, *, specialist, device: str = "cuda", dtype: str = "bfloat16",
                        batch_size: int = DEFAULT_BATCH_SIZE,
                        data_dir: str | Path | None = None,
                        pages_dir: str | Path | None = None,
                        retriever=None) -> Callable[[str], list[str]]:
    """Embute o corpus com o especialista e devolve o `retrieve(texto) -> [item_id, ...]`.

    `docs` sao os `Doc` do `corpus.jsonl` (ou dicts equivalentes com `id`/`image_path`).
    `batch_size` e o knob de VRAM (ver `DEFAULT_BATCH_SIZE`); `retriever` injetado pula a
    construcao do modelo -- e o que os testes usam, e entao `batch_size` nao se aplica,
    porque quem o consome e o `ColVisionRetriever` que nao chegou a ser construido.

    Corpus vazio, ou pagina do corpus que nao existe em disco, falha AQUI: as duas coisas
    so apareceriam como metrica 0.0 la na frente, e um zero que parece metrica e pior que
    um erro.
    """
    itens = [(_campo(d, "id"), _campo(d, "image_path")) for d in docs]
    if not itens:
        raise ValueError("corpus vazio: nao ha o que ranquear")
    sem_imagem = [str(i) for i, p in itens if not p]
    if sem_imagem:
        raise ValueError(f"o eixo documento ranqueia a PAGINA renderizada, mas "
                         f"{len(sem_imagem)} item(ns) do corpus nao tem image_path: "
                         f"{', '.join(sem_imagem[:5])}")

    doc_ids = [str(i) for i, _ in itens]
    caminhos = [resolve_image_path(p, data_dir=data_dir, pages_dir=pages_dir)
                for _, p in itens]

    if retriever is None:
        retriever = _build_retriever(specialist, device=device, dtype=dtype,
                                     batch_size=batch_size)

    from benchmark.common.types import Doc

    log.info("especialista %s: embutindo %d paginas (uma vez; fora do cronometro das "
             "queries)", specialist.name, len(doc_ids))
    embeddings = retriever.encode_corpus(
        [Doc(id=i, image_path=str(c)) for i, c in zip(doc_ids, caminhos, strict=True)])
    log.info("especialista %s: indice late-interaction pronto (%d itens)",
             specialist.name, len(embeddings))
    return DocSpecialistRanker(doc_ids, list(embeddings), retriever).retrieve


def _campo(item, nome: str):
    """Le `nome` de um `Doc` (dataclass) ou de um dict -- como o `_campo` do `eval.py`.

    `load_local` devolve dataclass; teste e conjunto montado a mao usam dict.
    """
    if isinstance(item, Mapping):
        return item.get(nome)
    return getattr(item, nome, None)
