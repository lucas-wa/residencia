"""Captura de video: sessoes oficiais no YouTube (Camara / TV Senado / Canal Gov) sobre orcamento.

v1: video e *pointer* (descritor = ASR da fala do trecho), nao embedding de video (v2).
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Callable

from ensemble.descriptors import EmptyDescriptorError, asr_descriptor
from ensemble.records import PointerRecord, validate_record


def record_from_session(
    info: dict,
    audio_path: str,
    *,
    transcriber: Callable[[str], str],
    start: float | None = None,
    end: float | None = None,
) -> PointerRecord | None:
    """Record 'video' com descritor = ASR da fala; pointer = url#t=<start> (segundos).

    Sessao sem fala (ASR vazio) -> None (pulada).
    """
    try:
        desc = asr_descriptor(audio_path, transcriber=transcriber)
    except EmptyDescriptorError:
        return None
    url = info["webpage_url"]
    t0 = int(start or 0)
    meta: dict = {"title": info.get("title", ""), "channel": info.get("channel", ""), "start": t0}
    if end is not None:
        meta["end"] = int(end)
    rec = PointerRecord(
        id="video-" + hashlib.sha1(f"{url}#t={t0}".encode("utf-8")).hexdigest()[:12],
        modality="video",
        descriptor=desc,
        pointer_uri=f"{url}#t={t0}",
        source="youtube",
        meta=meta,
    )
    validate_record(rec)
    return rec


def _whisper_transcriber(model_size: str = "large-v3") -> Callable[[str], str]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size)

    def transcribe(path: str) -> str:
        segments, _ = model.transcribe(path, language="pt")
        return " ".join(s.text.strip() for s in segments)

    return transcribe


def fetch_sessions(
    canais: list[str], termo: str = "orçamento", n: int = 20, dest_dir: str | None = None
) -> list[PointerRecord]:
    """Wrapper fino de rede: busca `termo` nos canais oficiais com yt-dlp, baixa o audio e
    transcreve com faster-whisper. Etapa de preparacao (yt-dlp nao entra no requirements-slurm).

    Nao coberto por teste de rede.
    """
    import yt_dlp  # lazy: dependencia do lado de coleta/prep

    dest = Path(dest_dir or tempfile.mkdtemp(prefix="ensemble-video-"))
    dest.mkdir(parents=True, exist_ok=True)
    transcriber = _whisper_transcriber()
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest / "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "ignoreerrors": True,
    }
    out: list[PointerRecord] = []
    seen: set[str] = set()
    with yt_dlp.YoutubeDL(opts) as ydl:
        for canal in canais:
            base = canal.rstrip("/")
            search = f"{base}/search?query={termo}"
            listing = ydl.extract_info(search, download=False, process=False)
            for entry in (listing or {}).get("entries") or []:
                vid = (entry or {}).get("id")
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                try:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
                    path = ydl.prepare_filename(info)
                    rec = record_from_session(info, path, transcriber=transcriber)
                except Exception:
                    continue
                if rec is None:
                    continue
                out.append(rec)
                if len(out) >= n:
                    return out
    return out
