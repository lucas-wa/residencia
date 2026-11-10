"""Captura de video: sessoes oficiais do Portal Multimidia do Senado (TV Senado) sobre orcamento.

v1: video e *pointer* (descritor = ASR da fala do trecho), nao embedding de video (v2).

Fonte NATIVA (substitui o YouTube, que bloqueia yt-dlp em IP de datacenter):
  - `GET /multimidia/eventos/{aaaa}/{mm}/{dd}` -> HTML com links `/multimidia/evento/{id}`
  - `GET /multimidia/data/{id}` -> corpo JS `window.trackData = {...}` com os metadados do
    evento e a lista `UDS` (um segmento por orador), cada um com `END_VID`/`END_VID_HIGH`
  - `END_VID_HIGH` prefixado com a base e um MP4 progressivo direto: sem login, token,
    cookie ou Referer -- amigavel a datacenter, baixado por urllib (yt-dlp nao entra aqui;
    a URL da PAGINA do evento nao e suportada por ele).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from ensemble.capture.audio import _whisper_model
from ensemble.capture.image import _http_get, _http_get_bytes
from ensemble.descriptors import EmptyDescriptorError, asr_descriptor
from ensemble.records import PointerRecord, validate_record

log = logging.getLogger(__name__)

SENADO_BASE = "https://www12.senado.leg.br"

# Comissao Mista de Planos, Orcamentos Publicos e Fiscalizacao: eixo tematico do corpus.
_SIGLA_ORCAMENTO = "CMO"

_EVENTO_RE = re.compile(r"/multimidia/evento/(\d+)")


def record_from_session(
    info: dict,
    audio_path: str,
    *,
    transcriber: Callable[[str], str],
    start: float | None = None,
    end: float | None = None,
    media_path: str | None = None,
    keyframe_fn: Callable[[float], str | None] | None = None,
    source: str = "senado",
) -> PointerRecord | None:
    """Record 'video' com descritor = ASR da fala; pointer = url#t=<start> (segundos).

    Sessao sem fala (ASR vazio) -> None (pulada). Com `media_path`, o arquivo retido vai em
    meta["path"]; `keyframe_fn(start)` devolve o caminho do keyframe -> meta["a_repr"].
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
    if media_path:
        meta["path"] = str(media_path)
    if keyframe_fn is not None:
        kf = keyframe_fn(float(start or 0))
        if kf:
            meta["a_repr"] = str(kf)
    rec = PointerRecord(
        id="video-" + hashlib.sha1(f"{url}#t={t0}".encode("utf-8")).hexdigest()[:12],
        modality="video",
        descriptor=desc,
        pointer_uri=f"{url}#t={t0}",
        source=source,
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


def records_from_segments(
    info: dict, segments: list[dict], *, media_path: str | None = None,
    keyframe_fn: Callable[[float], str | None] | None = None,
    source: str = "senado",
) -> list[PointerRecord]:
    """Um record 'video' por janela, com pointer = url#t=<start real>. Janelas sem texto sao puladas."""
    out: list[PointerRecord] = []
    for seg in segments:
        text = seg.get("text", "")
        rec = record_from_session(
            info, info["webpage_url"], transcriber=lambda _p, t=text: t,
            start=seg["start"], end=seg.get("end"),
            media_path=media_path, keyframe_fn=keyframe_fn, source=source,
        )
        if rec is not None:
            out.append(rec)
    return out


def _ffmpeg_keyframe(media: str, t: float, destino: Path) -> str | None:
    """Extrai 1 frame em `t` segundos com ffmpeg; None se falhar."""
    import subprocess

    destino.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", media, "-frames:v", "1", str(destino)],
        capture_output=True,
    )
    return str(destino) if r.returncode == 0 and destino.exists() else None


def event_ids(listing_html: str) -> list[str]:
    """Ids dos eventos citados no HTML de `/multimidia/eventos/{aaaa}/{mm}/{dd}`.

    Dedup preservando a ordem de aparicao.
    """
    out: list[str] = []
    seen: set[str] = set()
    for ev in _EVENTO_RE.findall(listing_html or ""):
        if ev in seen:
            continue
        seen.add(ev)
        out.append(ev)
    return out


def parse_track_data(raw: str) -> dict:
    """Extrai o dict de `window.trackData = {...}` do corpo JS de `/multimidia/data/{id}`.

    O corpo traz outras atribuicoes (ex.: `window.portalURL`) e pode nao terminar em `;`;
    por isso o JSON e lido a partir do `=` com `raw_decode` (o que vier depois e ignorado).
    """
    i = (raw or "").find("window.trackData")
    if i < 0:
        raise ValueError("corpo sem 'window.trackData'")
    j = raw.find("=", i)
    if j < 0:
        raise ValueError("'window.trackData' sem atribuicao")
    obj, _ = json.JSONDecoder().raw_decode(raw[j + 1:].lstrip())
    if not isinstance(obj, dict):
        raise ValueError(f"trackData nao e objeto: {type(obj).__name__}")
    return obj


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))


def _radical(termo: str) -> str:
    """Radical do termo sem acento/caixa ('orçamento' -> 'orcament'), para casar flexoes."""
    s = _sem_acento(termo).lower().strip()
    while len(s) > 4 and s[-1] in "aeiou":
        s = s[:-1]
    return s


def is_orcamento(track: dict, termo: str = "orçamento") -> bool:
    """True para eventos da CMO ou cujo titulo/classe cite `termo` (sem acento, case-insensitive)."""
    if (track.get("DES_SIGLA") or "").strip().upper() == _SIGLA_ORCAMENTO:
        return True
    alvo = _sem_acento(f"{track.get('DES_TITULO', '')} {track.get('DES_CLASSE', '')}").lower()
    return _radical(termo) in alvo


def _media_url(ud: dict, base: str) -> str | None:
    """URL absoluta do MP4 do segmento: `END_VID_HIGH` com fallback `END_VID`."""
    for campo in ("END_VID_HIGH", "END_VID"):
        caminho = (ud.get(campo) or "").strip()
        if caminho:
            return caminho if caminho.startswith("http") else f"{base}{caminho}"
    return None


def fetch_sessions(
    fontes: list[str] | str = SENADO_BASE, termo: str = "orçamento", n: int = 20,
    dest_dir: str | None = None, window: float = 90.0, dias: int = 120,
) -> list[PointerRecord]:
    """Wrapper fino de rede: varre o Portal Multimidia do Senado de HOJE para tras (ate `dias`),
    seleciona eventos de orcamento (CMO / `termo` no titulo+classe), baixa o MP4 de cada
    segmento de orador (`UDS`), transcreve com faster-whisper e emite um record por janela.
    Etapa de preparacao (nao entra no requirements-slurm).

    Falha de listagem/evento/segmento e PULADA (log.warning), nao aborta. Para assim que
    juntar `n` records.

    Nao coberto por teste de rede.
    """
    bases = [fontes] if isinstance(fontes, str) else list(fontes)
    dest = Path(dest_dir or tempfile.mkdtemp(prefix="ensemble-video-"))
    dest.mkdir(parents=True, exist_ok=True)
    model = _whisper_model()
    hoje = date.today()
    out: list[PointerRecord] = []
    vistos: set[str] = set()
    for fonte in bases:
        base = (fonte or SENADO_BASE).rstrip("/")
        for delta in range(dias):
            d = hoje - timedelta(days=delta)
            try:
                listing = _http_get(f"{base}/multimidia/eventos/{d:%Y/%m/%d}")
            except Exception as exc:
                log.warning("listagem %s pulada: %s", d.isoformat(), exc)
                continue
            for ev in event_ids(listing):
                if ev in vistos:
                    continue
                vistos.add(ev)
                try:
                    track = parse_track_data(_http_get(f"{base}/multimidia/data/{ev}"))
                except Exception as exc:
                    log.warning("evento %s pulado: %s", ev, exc)
                    continue
                if not is_orcamento(track, termo):
                    continue
                out.extend(_records_from_event(
                    base, ev, track, model=model, dest=dest, window=window, faltam=n - len(out)))
                if len(out) >= n:
                    return out[:n]
    return out[:n]


def _records_from_event(
    base: str, ev: str, track: dict, *, model, dest: Path, window: float, faltam: int,
) -> list[PointerRecord]:
    """Baixa + transcreve os segmentos (`UDS`) de um evento ate juntar `faltam` records."""
    out: list[PointerRecord] = []
    for ud in track.get("UDS") or []:
        if len(out) >= faltam:
            break
        cod = str(ud.get("COD_UD") or len(out))
        url = _media_url(ud, base)
        if not url:
            continue
        # O pointer carrega o segmento (`?ud=`): varios UDS do mesmo evento reiniciam em
        # t=0 e colidiriam no id (sha1 de url#t) se apontassem so para a pagina do evento.
        info = {
            "webpage_url": f"{base}/multimidia/evento/{ev}?ud={cod}",
            "title": track.get("DES_TITULO", ""),
            "channel": "TV Senado",
        }
        try:
            path = dest / (Path(urlparse(url).path).name or f"{ev}_{cod}.mp4")
            path.write_bytes(_http_get_bytes(url))
            raw, _ = model.transcribe(str(path), language="pt")
            segs = [{"start": s.start, "end": s.end, "text": s.text} for s in raw]
            recs = records_from_segments(
                info, group_segments(segs, window), media_path=str(path),
                keyframe_fn=lambda t, p=str(path), e=ev, c=cod: _ffmpeg_keyframe(
                    p, t, dest / "keyframes" / f"{e}_{c}_{int(t)}.jpg"),
            )
        except Exception as exc:
            log.warning("segmento %s/%s pulado: %s", ev, cod, exc)
            continue
        out.extend(recs[: faltam - len(out)])
    return out
