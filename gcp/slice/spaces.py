"""Registro declarativo dos espacos vetoriais e das arquiteturas da fatia GovExe.

Um **espaco** e um modelo de embedding com a sua propria geometria: checkpoint, dimensao
e tabela pgvector. Espacos nao sao comparaveis entre si -- a similaridade do Qwen3-VL e a
do CLAP vivem em escalas diferentes --, por isso cada um grava na sua tabela (indice HNSW
independente) e a consulta multi-espaco funde os *rankings* por RRF (`gcp/slice/fusion.py`),
nunca os vetores.

Uma **arquitetura** e so a lista de espacos que a compoem:

    A         = Qwen3-VL unificado (pagina, foto, texto e audio-via-ASR no mesmo indice)
    A+audio   = A + CLAP indexando o audio NATIVO, em paralelo ao caminho ASR->texto
    B         = ensemble de ESPECIALISTAS (`fusao/multimodal.md` secao 3); o primeiro
                braco e o eixo documento, ColQwen2.5 late-interaction -- ZERO espacos
                pgvector (ver `Specialist` abaixo)

O `item_id` e estavel entre tabelas: o mesmo item pode aparecer em mais de um espaco, e e
isso que permite ao RRF somar as contribuicoes.

Nem todo ranqueador cabe num `Space`: um modelo **late-interaction** (ColQwen2.5) emite um
vetor POR TOKEN e ranqueia por MaxSim, enquanto `Space` modela "1 vetor denso por item,
cosseno, 1 tabela pgvector". Forcar multi-vetor numa coluna `vector(n)` so destruiria a
late interaction. Por isso ha um segundo registro, `SPECIALISTS`: mesmo papel declarativo,
geometria outra. Uma arquitetura pode ter espacos, especialistas, ou os dois -- `B` hoje
so tem especialista, e `uses_pgvector("B")` e `False`.

Este modulo e a fonte unica de verdade lida pelo `embed_job --space` e pela consulta
multi-espaco. Mantenha-o **declarativo e sem imports pesados** (nada de torch/transformers):
construir o embedder e lazy e mora no modulo de cada backbone.

Uso:
    from gcp.slice.spaces import SPACES, architecture_spaces

    for espaco in architecture_spaces("A+audio"):
        print(espaco.name, espaco.table, espaco.dim)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "ARCHITECTURES",
    "ARCHITECTURE_SPECIALISTS",
    "HNSW_MAX_DIM",
    "SPACES",
    "SPECIALISTS",
    "Space",
    "Specialist",
    "architecture_spaces",
    "architecture_specialists",
    "spaces_for_modalities",
    "specialists_for_modalities",
    "uses_pgvector",
]

# Teto do indice do pgvector: uma coluna indexada por `hnsw` nao passa de 2000 dimensoes
# (espelha `embed_job.HNSW_MAX_DIM` -- repetido aqui de proposito para nao arrastar o
# import pesado do job para dentro do registro).
HNSW_MAX_DIM = 2000


@dataclass(frozen=True)
class Space:
    """Um espaco vetorial: um checkpoint, uma dimensao, uma tabela.

    `kind` identifica a familia do backbone (quem sabe construir o embedder); `modalities`
    sao as modalidades que ESTE espaco indexa -- a chave do roteamento no `embed_job`.
    """

    name: str
    table: str
    model_id: str
    kind: str
    dim: int
    modalities: tuple[str, ...]


SPACES: dict[str, Space] = {
    # Abordagem A: um unico checkpoint para pagina, foto, texto e audio colapsado em texto
    # pelo Whisper (`audio_asr`). 1024 e o corte Matryoshka/MRL da fatia: o checkpoint
    # entrega 2048 nativas, que nao cabem no HNSW (ver `embed_job.DEFAULT_DIM`).
    "qwen3vl": Space(
        name="qwen3vl",
        table="govexe_qwen3vl",
        model_id="Qwen/Qwen3-VL-Embedding-2B",
        kind="qwen3vl",
        dim=1024,
        modalities=("doc_image", "image", "text", "audio_asr"),
    ),
    # Audio NATIVO (a forma de onda, nao a transcricao) -- por isso a modalidade e
    # `audio_native`, distinta do `audio_asr` que o Qwen3-VL indexa.
    # ATENCAO: dim=512 e o valor do card do `laion/larger_clap_general`; confirmar no smoke
    # (um `embed_audio` real) antes de criar a tabela -- divergir aqui quebra o upsert.
    "clap": Space(
        name="clap",
        table="govexe_clap",
        model_id="laion/larger_clap_general",
        kind="clap",
        dim=512,
        modalities=("audio_native",),
    ),
}

@dataclass(frozen=True)
class Specialist:
    """Um ranqueador LATE-INTERACTION: multi-vetor, sem tabela pgvector.

    Nao tem `table` nem `dim` -- e o que o distingue de um `Space`. O item nao vira UM
    vetor denso comparado por cosseno numa coluna `vector(n)`, mas uma MATRIZ
    [n_tokens, dim] pontuada por MaxSim (`benchmark.common.index.maxsim_search`) contra a
    matriz da consulta. Com corpus pequeno (as 93 paginas de `ptbr_planos`) isso roda em
    memoria, instantaneo, e o Cloud SQL simplesmente nao entra na rodada.

    `kind` identifica a familia do backbone -- e a chave que `benchmark.models.colvision`
    resolve para o par (modelo, processor) do colpali-engine. `modalities` sao as
    modalidades que ESTE especialista ranqueia (mesmo vocabulario de `Space.modalities`).
    """

    name: str
    model_id: str
    kind: str
    modalities: tuple[str, ...]


SPECIALISTS: dict[str, Specialist] = {
    # Eixo documento da Abordagem B (`fusao/multimodal.md` secao 3): ColQwen2.5, Apache-2.0,
    # late interaction sobre a PAGINA renderizada -- sem OCR, sem pipeline de texto.
    # O wrapper e `benchmark.models.colvision.ColVisionRetriever(kind="colqwen2.5")`, que ja
    # roda neste repo; `vidore/colqwen2.5-v0.2` e o checkpoint da trilha ViDoRe.
    "colqwen25_doc": Specialist(
        name="colqwen25_doc",
        model_id="vidore/colqwen2.5-v0.2",
        kind="colqwen2.5",
        modalities=("doc_image",),
    ),
}

# Ordem importa: e a ordem em que os rankings chegam ao RRF (que e simetrico, mas a saida
# fica mais facil de ler/depurar com o espaco principal primeiro).
ARCHITECTURES: dict[str, tuple[str, ...]] = {
    "A": ("qwen3vl",),
    "A+audio": ("qwen3vl", "clap"),
    # B NAO tem espaco pgvector: a tupla vazia e a declaracao disso, nao um esquecimento.
    # O que B tem esta em `ARCHITECTURE_SPECIALISTS` -- e e por isso que a rodada de B
    # dispensa Cloud SQL inteiro (ver `run_slice.sh`).
    "B": (),
}

# Especialistas de cada arquitetura. Arquitetura ausente = nenhum (A e A+audio sao 100%
# pgvector). Mantido como dict proprio, e nao como um campo de `ARCHITECTURES`, para o
# caminho pgvector de A/A+audio continuar lendo exatamente o que sempre leu.
ARCHITECTURE_SPECIALISTS: dict[str, tuple[str, ...]] = {
    "B": ("colqwen25_doc",),
}

# Toda arquitetura do registro tem de ter ALGO que a ranqueie -- espaco ou especialista.
# Um nome so em `ARCHITECTURES`, com tupla vazia e sem especialista, seria uma arquitetura
# que avalia o nada e devolve 0.0 em tudo. Falha no import, nao na rodada paga.
_VAZIAS = sorted(nome for nome, espacos in ARCHITECTURES.items()
                 if not espacos and not ARCHITECTURE_SPECIALISTS.get(nome))
if _VAZIAS:
    raise ValueError(f"arquitetura sem espaco nem especialista: {_VAZIAS}")
_ORFAOS = sorted(set(ARCHITECTURE_SPECIALISTS) - set(ARCHITECTURES))
if _ORFAOS:
    raise ValueError(f"especialista declarado para arquitetura inexistente: {_ORFAOS}")


def uses_pgvector(name: str) -> bool:
    """`True` se `name` tem ao menos um espaco pgvector a consultar.

    E o teste que ROTEIA a avaliacao (`gcp/slice/eval.py`) e a orquestracao
    (`run_slice.sh`): `False` significa que a rodada nao precisa de Cloud SQL nenhum --
    nem instancia, nem proxy, nem indexacao -- porque quem ranqueia e especialista em
    memoria. Pergunta ao registro, e nao ao nome da arquitetura.
    """
    return bool(architecture_spaces(name))


def architecture_spaces(name: str) -> list[Space]:
    """Devolve os `Space` da arquitetura `name`, na ordem declarada em `ARCHITECTURES`.

    Lista VAZIA e uma resposta legitima: a arquitetura nao tem espaco pgvector (`B`), e
    quem ranqueia e `architecture_specialists`. Use `uses_pgvector` para rotear -- iterar
    o vazio aqui so renderia um ranking vazio la na frente.

    Levanta `ValueError` para arquitetura desconhecida -- um nome errado na CLI tem de
    falhar na hora, com a lista do que existe, e nao virar uma busca vazia la na frente.
    """
    try:
        nomes = ARCHITECTURES[name]
    except KeyError:
        conhecidas = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(
            f"arquitetura desconhecida: {name!r} (conhecidas: {conhecidas})"
        ) from None
    return [SPACES[espaco] for espaco in nomes]


def spaces_for_modalities(name: str, modalities: Iterable[str]) -> list[Space]:
    """Espacos de `name` que indexam ALGUMA das `modalities`, na ordem da arquitetura.

    E o que permite avaliar uma tarefa sem arrastar os espacos que nao tem o que dizer
    sobre ela: numa tarefa doc-visual (`doc_image`) o CLAP nao indexa nada do corpus, e
    fundi-lo so entremearia clipes de audio entre as paginas -- o que faria `A+audio`
    diferir de `A` numa tarefa em que a spec (secao 5) as declara IDENTICAS.

    Quem decide e o registro (`Space.modalities`), nunca o nome do espaco. Pode devolver
    lista vazia (nenhum espaco da arquitetura indexa aquela modalidade); cabe a quem chama
    decidir o que fazer com isso.
    """
    alvo = set(modalities)
    return [espaco for espaco in architecture_spaces(name)
            if alvo.intersection(espaco.modalities)]


def architecture_specialists(name: str) -> list[Specialist]:
    """Os `Specialist` (late-interaction) da arquitetura `name`, na ordem declarada.

    Lista vazia para `A`/`A+audio`, que sao inteiramente pgvector. Valida `name` contra
    `ARCHITECTURES` pelo mesmo motivo de `architecture_spaces`: nome errado falha na hora.
    """
    if name not in ARCHITECTURES:
        conhecidas = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"arquitetura desconhecida: {name!r} (conhecidas: {conhecidas})")
    return [SPECIALISTS[e] for e in ARCHITECTURE_SPECIALISTS.get(name, ())]


def specialists_for_modalities(name: str, modalities: Iterable[str]) -> list[Specialist]:
    """Especialistas de `name` que ranqueiam ALGUMA das `modalities`, na ordem declarada.

    Gemeo de `spaces_for_modalities` para o outro registro: a tarefa (= a modalidade do
    corpus) e quem decide quem entra, e quem decide e o registro, nunca o nome. O eixo
    documento de `B` so declara `doc_image` -- num corpus de audio a lista sai vazia, e e
    isso que impede B de "avaliar" uma tarefa sobre a qual nao tem nada a dizer.
    """
    alvo = set(modalities)
    return [esp for esp in architecture_specialists(name)
            if alvo.intersection(esp.modalities)]
