"""Captura de audio: clipes de orcamento (EBC / Radioagencia) transcritos por Whisper."""
from __future__ import annotations

import hashlib
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

from ensemble.descriptors import EmptyDescriptorError, asr_descriptor
from ensemble.records import PointerRecord, validate_record


def record_from_clip(
    clip_path: str, url: str, *, transcriber: Callable[[str], str]
) -> PointerRecord | None:
    """Record 'audio' com descritor = transcricao. Clipe sem fala (ASR vazio) -> None (pulado)."""
    try:
        desc = asr_descriptor(clip_path, transcriber=transcriber)
    except EmptyDescriptorError:
        return None
    rec = PointerRecord(
        id="audio-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12],
        modality="audio",
        descriptor=desc,
        pointer_uri=url,
        source="radio",
        meta={"clip": str(clip_path)},
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


def fetch_clips(feeds: list[str], dest_dir: str | None = None, n: int = 20) -> list[PointerRecord]:
    """Wrapper fino de rede: le feeds RSS com <enclosure> de audio (EBC/Radioagencia),
    baixa cada clipe e transcreve com faster-whisper.

    Nao coberto por teste de rede; roda na Task 9.
    """
    dest = Path(dest_dir or tempfile.mkdtemp(prefix="ensemble-audio-"))
    dest.mkdir(parents=True, exist_ok=True)
    transcriber = _whisper_transcriber()
    out: list[PointerRecord] = []
    seen: set[str] = set()
    for feed in feeds:
        req = urllib.request.Request(feed, headers={"User-Agent": "Mozilla/5.0 (ensemble-capture)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            root = ET.fromstring(resp.read())
        for item in root.iter("item"):
            enc = item.find("enclosure")
            url = (enc.get("url") if enc is not None else "") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            path = dest / Path(url.split("?")[0]).name
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ensemble-capture)"})
                with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                    f.write(r.read())
                rec = record_from_clip(str(path), url, transcriber=transcriber)
            except Exception:
                continue
            if rec is None:
                continue
            out.append(rec)
            if len(out) >= n:
                return out
    return out
