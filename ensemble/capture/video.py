"""Captura de video: sessoes oficiais no YouTube (Camara / TV Senado / Canal Gov) sobre orcamento.

v1: video e *pointer* (descritor = ASR da fala do trecho), nao embedding de video (v2).
"""
from __future__ import annotations

import hashlib
import logging
import tempfile
import urllib.parse
from pathlib import Path
from typing import Callable

from ensemble.capture.audio import _whisper_model
from ensemble.descriptors import EmptyDescriptorError, asr_descriptor
from ensemble.records import PointerRecord, validate_record

log = logging.getLogger(__name__)


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


def group_segments(segments: list[dict], window: float = 90.0) -> list[dict]:
    """Agrupa segmentos {"start","end","text"} em janelas de ~`window` segundos."""
    out: list[dict] = []
    cur: dict | None = None
    for seg in segments:
        if cur is None:
            cur = {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        else:
            cur["end"] = seg["end"]
            cur["text"] = (cur["text"] + " " + seg["text"].strip()).strip()
        if cur["end"] - cur["start"] >= window:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def records_from_segments(info: dict, segments: list[dict]) -> list[PointerRecord]:
    """Um record 'video' por janela, com pointer = url#t=<start real>. Janelas sem texto sao puladas."""
    out: list[PointerRecord] = []
    for seg in segments:
        text = seg.get("text", "")
        rec = record_from_session(
            info, info["webpage_url"], transcriber=lambda _p, t=text: t,
            start=seg["start"], end=seg.get("end"),
        )
        if rec is not None:
            out.append(rec)
    return out


def fetch_sessions(
    canais: list[str], termo: str = "orçamento", n: int = 20, dest_dir: str | None = None,
    window: float = 90.0,
) -> list[PointerRecord]:
    """Wrapper fino de rede: busca `termo` nos canais oficiais com yt-dlp, baixa o audio,
    transcreve com faster-whisper (segmentos com timestamp) e emite um record por janela.
    Etapa de preparacao (yt-dlp nao entra no requirements-slurm).

    Nao coberto por teste de rede.
    """
    import yt_dlp  # lazy: dependencia do lado de coleta/prep

    dest = Path(dest_dir or tempfile.mkdtemp(prefix="ensemble-video-"))
    dest.mkdir(parents=True, exist_ok=True)
    model = _whisper_model()
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
            search = f"{canal.rstrip('/')}/search?query={urllib.parse.quote(termo)}"
            listing = ydl.extract_info(search, download=False, process=False)
            for entry in (listing or {}).get("entries") or []:
                vid = (entry or {}).get("id")
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                try:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
                    path = ydl.prepare_filename(info)
                    raw, _ = model.transcribe(path, language="pt")
                    segs = [{"start": s.start, "end": s.end, "text": s.text} for s in raw]
                    recs = records_from_segments(info, group_segments(segs, window))
                except Exception as exc:
                    log.warning("sessao %s pulada: %s", vid, exc)
                    continue
                for rec in recs:
                    out.append(rec)
                    if len(out) >= n:
                        return out
    return out
