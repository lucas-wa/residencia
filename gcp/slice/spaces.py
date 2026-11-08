"""Registro declarativo dos espacos vetoriais e das arquiteturas da fatia GovExe.

Um **espaco** e um modelo de embedding com a sua propria geometria: checkpoint, dimensao
e tabela pgvector. Espacos nao sao comparaveis entre si -- a similaridade do Qwen3-VL e a
do CLAP vivem em escalas diferentes --, por isso cada um grava na sua tabela (indice HNSW
independente) e a consulta multi-espaco funde os *rankings* por RRF (`gcp/slice/fusion.py`),
nunca os vetores.

Uma **arquitetura** e so a lista de espacos que a compoem:

    A         = Qwen3-VL unificado (pagina, foto, texto e audio-via-ASR no mesmo indice)
    A+audio   = A + CLAP indexando o audio NATIVO, em paralelo ao caminho ASR->texto

O `item_id` e estavel entre tabelas: o mesmo item pode aparecer em mais de um espaco, e e
isso que permite ao RRF somar as contribuicoes.

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
    "HNSW_MAX_DIM",
    "SPACES",
    "Space",
    "architecture_spaces",
    "spaces_for_modalities",
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

# Ordem importa: e a ordem em que os rankings chegam ao RRF (que e simetrico, mas a saida
# fica mais facil de ler/depurar com o espaco principal primeiro).
ARCHITECTURES: dict[str, tuple[str, ...]] = {
    "A": ("qwen3vl",),
    "A+audio": ("qwen3vl", "clap"),
}


def architecture_spaces(name: str) -> list[Space]:
    """Devolve os `Space` da arquitetura `name`, na ordem declarada em `ARCHITECTURES`.

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
