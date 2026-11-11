"""Builders de embedder nativos do ensemble v2 -- IMPORT TARDIO.

Cada `build_*_embedder()` devolve `(embed_items, embed_query)`, ambos
`Callable[[list[str]], np.ndarray]` -> `[n, d]` float32. Imports de torch/transformers/
sentence-transformers ficam DENTRO das funcoes (como `_build_embedder` do v1 e o
`_build_retriever` do doc_specialist): importar este modulo nunca baixa checkpoint nem
toca em GPU, entao os testes cobrem so a fiacao. Os modelos reais rodam no Slurm H100.

Onde ja existe wrapper em `benchmark/models/` (CLAP, SigLIP, Qwen3-VL) ele e reusado.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

Embed = Callable[[list[str]], np.ndarray]
EmbedderPair = tuple[Embed, Embed]

TEXT_MODEL = "BAAI/bge-m3"
IMAGE_MODEL = "google/siglip2-so400m-patch14-384"
AUDIO_MODEL = "laion/larger_clap_general"
# o eixo video rankeia janelas pelo KEYFRAME (imagem), entao usa o mesmo SigLIP 2 do
# eixo imagem -- mesma familia de modelo, conjuntos de itens diferentes (eixos separados
# no RRF).
VIDEO_MODEL = IMAGE_MODEL
UNIFIED_A_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def build_text_embedder() -> EmbedderPair:
    """BGE-M3 (sentence-transformers); items e query sao texto."""
    from sentence_transformers import SentenceTransformer

    log.info("carregando %s", TEXT_MODEL)
    model = SentenceTransformer(TEXT_MODEL, device="cuda")
    model.half()

    def encode(texts: list[str]) -> np.ndarray:
        return _f32(model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False))

    return encode, encode


def build_audio_asr_embedder() -> EmbedderPair:
    """BGE-M3 sobre a TRANSCRICAO do audio (ASR->texto); items e query sao texto.

    O record de audio ja carrega a transcricao Whisper em `descriptor` (ver
    `ensemble/capture/audio.py`), entao o eixo audio pode ser ranqueado como texto,
    pelo mesmo modelo do eixo texto. Carrega o BGE-M3 uma segunda vez (um eixo por
    vez, instancias independentes); o modo CLAP fica em `build_audio_embedder`.
    """
    from sentence_transformers import SentenceTransformer

    log.info("carregando %s (eixo audio, ASR->texto)", TEXT_MODEL)
    model = SentenceTransformer(TEXT_MODEL, device="cuda")
    model.half()

    def encode(texts: list[str]) -> np.ndarray:
        return _f32(model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False))

    return encode, encode


def build_image_embedder() -> EmbedderPair:
    """SigLIP 2: items = caminhos de imagem, query = texto."""
    from benchmark.models.clip_siglip import ClipSiglipRetriever

    log.info("carregando %s", IMAGE_MODEL)
    m = ClipSiglipRetriever(IMAGE_MODEL, backend="hf", device="cuda")
    return (lambda paths: _f32(m._embed_image(list(paths)))), (
        lambda texts: _f32(m._embed_text(list(texts)))
    )


def build_audio_embedder() -> EmbedderPair:
    """CLAP: items = caminhos de audio, query = texto."""
    from benchmark.models.clap_embed import get_embedder

    log.info("carregando %s", AUDIO_MODEL)
    emb = get_embedder(model_id=AUDIO_MODEL, device="cuda")
    return (lambda paths: _f32(emb.embed_audio(list(paths)))), (
        lambda texts: _f32(emb.embed_texts(list(texts)))
    )


def build_video_embedder() -> EmbedderPair:
    """SigLIP 2: items = KEYFRAME POR JANELA (imagem), query = texto.

    O eixo de video rankeia janelas; o artefato discriminativo por janela e o
    keyframe (`meta["a_repr"]`), nao o video inteiro (`meta["path"]`, igual para
    todas as janelas). Logo `embed_items` embute a IMAGEM do keyframe de cada
    janela e `embed_query` embute o texto da consulta -- exatamente o que o
    SigLIP 2 faz. Mesmo modelo do eixo imagem, mas os dois eixos indexam
    conjuntos de itens diferentes e entram separados no RRF.

    O WeMM-Embedding-2B foi preterido: o checkpoint declara o backbone `qwen3_5`,
    so reconhecido em transformers >= 5.3, e o stack esta pinado em transformers
    4.57 (para casar com torch 2.9/CUDA e nao quebrar Qwen3-VL e ColQwen2.5).
    Fica como trabalho futuro, num ambiente separado.
    """
    from benchmark.models.clip_siglip import ClipSiglipRetriever

    log.info("carregando %s", VIDEO_MODEL)
    m = ClipSiglipRetriever(VIDEO_MODEL, backend="hf", device="cuda")
    return (lambda paths: _f32(m._embed_image(list(paths)))), (
        lambda texts: _f32(m._embed_text(list(texts)))
    )


def build_unified_a_embedder() -> EmbedderPair:
    """Qwen3-VL-Embedding (A): items = representacao-A (texto), query = texto."""
    from benchmark.models.qwen3vl_embed import Qwen3VLEmbedRetriever

    log.info("carregando %s", UNIFIED_A_MODEL)
    m = Qwen3VLEmbedRetriever(UNIFIED_A_MODEL, device="cuda")
    enc = lambda texts: _f32(m.embed_texts(list(texts)))  # noqa: E731
    return enc, enc


EMBEDDER_BUILDERS: dict[str, Callable[[], EmbedderPair]] = {
    "texto": build_text_embedder,
    "imagem": build_image_embedder,
    "audio": build_audio_embedder,
    "audio_asr": build_audio_asr_embedder,
    "video": build_video_embedder,
    "unified_a": build_unified_a_embedder,
}
