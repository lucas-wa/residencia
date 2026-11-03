"""ASR da fatia GCP: audio -> transcricao (passo Silver), para o MESMO embedder de texto.

E a materializacao do *pointer vector* de `fusao/multimodal.md` (secoes 3 e 4): nenhum
embedder unificado cobre audio, entao o audio NAO vira vetor proprio aqui -- ele vira
**texto** (transcricao Whisper) e e embutido por `embed_texts` do mesmo checkpoint
(`Qwen/Qwen3-VL-Embedding-2B`) que embute paginas e fotos. Um modelo, um espaco; o
binario original fica preservado em `pointer_uri` e o credito em `credit`.

Embedding NATIVO de audio (CLAP) e outro objetivo -- nao mora aqui.

Dois backends reais, ambos com **import lazy** (o modulo pode ser importado -- e e, pelos
testes offline -- sem nenhum dos dois instalado, sem baixar pesos e sem GPU):

    faster-whisper   DEFAULT. Reimplementacao do Whisper em CTranslate2: ~4x mais rapida
                     e com bem menos VRAM que a referencia (large-v3 em `float16` cabe em
                     ~3 GB), o que importa aqui porque na VM L4 (24 GB) o ASR divide a
                     placa com o Qwen3-VL-Embedding-2B. Mesmos pesos do `large-v3`.
    openai-whisper   referencia do card (`--asr-backend openai-whisper`), caso se queira
                     a implementacao original em PyTorch.

    fake             `FakeTranscriber`: transcricao canonica e deterministica, sem modelo
                     nenhum -- usado pelos testes e pelo modo dev do README.

Uso:
    from gcp.slice.asr import open_transcriber
    t = open_transcriber("faster-whisper", model="large-v3")   # nada carregado ainda
    texto = t.transcribe("sessao.mp3")                         # aqui sim carrega os pesos
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

# Whisper large-v3: o melhor pt-BR da familia (ver `fusao/multimodal.md` secao 4, FLEURS).
DEFAULT_ASR_MODEL = "large-v3"
DEFAULT_ASR_BACKEND = "faster-whisper"
ASR_BACKENDS = ("faster-whisper", "openai-whisper", "fake")
# Fala governamental em pt-BR: fixar o idioma evita o custo (e os erros) da deteccao.
DEFAULT_ASR_LANGUAGE = "pt"

# Prefixo da transcricao canonica do fake -- os testes o usam para provar que o texto
# embutido veio do Transcriber, e nao do record.
FAKE_PREFIX = "transcricao fake"


class Transcriber(Protocol):
    """Minimo que a fatia precisa de um ASR (Whisper real ou fake deterministico)."""

    def transcribe(self, audio_path: str | Path) -> str:
        """Audio -> transcricao em texto plano."""


class FakeTranscriber:
    """FAKE de `Transcriber`: transcricao canonica por arquivo. Testes e modo dev.

    Sem mapa, devolve `"<FAKE_PREFIX> do audio <stem>"` -- deterministico e distinto por
    arquivo, que e tudo o que o indice precisa para o lote misto rodar offline.
    """

    def __init__(self, transcripts: dict[str, str] | None = None):
        self.transcripts = dict(transcripts or {})
        self.calls: list[str] = []

    def transcribe(self, audio_path: str | Path) -> str:
        self.calls.append(str(audio_path))
        stem = Path(audio_path).stem
        texto = self.transcripts.get(str(audio_path)) or self.transcripts.get(stem)
        return texto or f"{FAKE_PREFIX} do audio {stem}"


class WhisperTranscriber:
    """Backend real: Whisper `large-v3`. Os pesos so carregam na primeira transcricao.

    O import do backend fica DENTRO de `_load()` de proposito (mesmo padrao dos wrappers
    de modelo do repo): importar `faster_whisper`/`whisper` no topo quebraria os testes
    offline e o modo dev, que nunca transcrevem de verdade.
    """

    def __init__(self, model: str = DEFAULT_ASR_MODEL, *,
                 backend: str = DEFAULT_ASR_BACKEND, device: str = "cuda",
                 compute_type: str | None = None,
                 language: str | None = DEFAULT_ASR_LANGUAGE):
        if backend not in ("faster-whisper", "openai-whisper"):
            raise ValueError(f"backend de ASR desconhecido: {backend!r} "
                             f"(use um de {ASR_BACKENDS})")
        self.model = model
        self.backend = backend
        self.device = device
        # int8 na CPU (unico jeito de o large-v3 rodar fora da GPU sem estourar RAM),
        # float16 na L4.
        self.compute_type = compute_type or ("float16" if device.startswith("cuda")
                                             else "int8")
        self.language = language
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        log.info("carregando Whisper %s (%s, %s)", self.model, self.backend, self.device)
        if self.backend == "faster-whisper":
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model, device=self.device,
                                       compute_type=self.compute_type)
        else:
            import whisper

            self._model = whisper.load_model(self.model, device=self.device)
        return self._model

    def transcribe(self, audio_path: str | Path) -> str:
        model = self._load()
        if self.backend == "faster-whisper":
            # `transcribe` do faster-whisper e preguicoso: devolve um gerador de segmentos
            # que so roda quando iterado.
            segments, _info = model.transcribe(str(audio_path), language=self.language,
                                               vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()
        out = model.transcribe(str(audio_path), language=self.language)
        return str(out.get("text", "")).strip()


def open_transcriber(backend: str = DEFAULT_ASR_BACKEND, *, model: str = DEFAULT_ASR_MODEL,
                     device: str = "cuda", language: str | None = DEFAULT_ASR_LANGUAGE,
                     transcripts: dict[str, str] | None = None) -> Transcriber:
    """`fake` -> transcritor canonico (dev/testes); senao, Whisper de verdade (lazy)."""
    if backend == "fake":
        return FakeTranscriber(transcripts)
    if backend not in ASR_BACKENDS:
        raise ValueError(f"backend de ASR desconhecido: {backend!r} (use um de {ASR_BACKENDS})")
    return WhisperTranscriber(model, backend=backend, device=device, language=language)


def add_asr_arguments(ap: argparse.ArgumentParser) -> None:
    """Flags do passo de ASR (so valem para records de modalidade `audio_asr`)."""
    ap.add_argument("--asr-backend", default=DEFAULT_ASR_BACKEND, choices=list(ASR_BACKENDS),
                    help=f"backend de ASR (default: {DEFAULT_ASR_BACKEND}; `fake` nao "
                         "carrega modelo nenhum -- modo dev e testes)")
    ap.add_argument("--asr-model", default=DEFAULT_ASR_MODEL,
                    help=f"checkpoint do Whisper (default: {DEFAULT_ASR_MODEL})")
    ap.add_argument("--asr-language", default=DEFAULT_ASR_LANGUAGE,
                    help=f"idioma da transcricao (default: {DEFAULT_ASR_LANGUAGE}; "
                         "vazio = deteccao automatica)")


def transcriber_from_args(args: argparse.Namespace) -> Transcriber:
    """Monta o `Transcriber` a partir das flags -- nada e carregado ate a 1a transcricao."""
    return open_transcriber(getattr(args, "asr_backend", DEFAULT_ASR_BACKEND),
                            model=getattr(args, "asr_model", DEFAULT_ASR_MODEL),
                            device=getattr(args, "device", "cuda"),
                            language=getattr(args, "asr_language", DEFAULT_ASR_LANGUAGE) or None)
