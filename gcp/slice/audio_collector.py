"""Captura de audio de fontes publicas -> records `audio_asr` para o fluxo `--records`.

E a plumbing que faltava na ponta da modalidade de audio (`fusao/coleta-gcp.md` secao 6):
dada a URL de uma sessao do YouTube (SenadoTV, TV Camara) ou de uma midia direta da
RadioAgencia / Senado Multimidia, baixa o audio para um diretorio local e emite um record
no formato EXATO que `collector.py --records` e `embed_job.py --records` ja consomem.

Cada etapa continua no seu lugar -- este modulo NAO sobe nada e NAO transcreve nada:

    audio_collector.py   URL publica  -> audio local + record JSONL   (maquina local)
    collector.py         record       -> binario no Bronze + record com blob/uri
    embed_job.py         record       -> ASR (asr.py) -> embed_texts -> govexe_items

O `item_id` sai de `collector.default_item_id("audio_asr", <URL de origem>)`: a identidade
segue a FONTE, nao o caminho do arquivo baixado nem o objeto no Bronze -- recapturar a
mesma URL cai no mesmo item (upsert, nao duplicata). A URL tambem vai em `pointer_uri`,
que e o pointer auditavel ao lado do `credit` obrigatorio.

**Escopo e licenca**: so fontes publicas oficiais, pelo endereco publico. Nada de login,
cookies de sessao, DRM ou paywall -- por isso o downloader nao expoe nenhuma opcao de
credencial. Captura de SINAL de TV/radio (que exige oficio) esta fora desta fatia.

**ffmpeg**: o backend real (`yt-dlp`) precisa do **ffmpeg instalado no sistema** para
extrair a trilha de audio e para o corte de `--max-seconds`. O repo nao o instala; ver a
nota em `gcp/slice/README-slice.md`.

Uso:
    python -m gcp.slice.audio_collector \\
        --url "https://www.youtube.com/watch?v=<id>" \\
        --source tv_camara --credit "TV Camara" \\
        --audio-dir data/govexe/audio --out-records audios.jsonl --max-seconds 600

    # offline (sem rede, sem yt-dlp): gera audio canonico so para ensaiar o fluxo
    python -m gcp.slice.audio_collector --url "https://exemplo/x.mp3" --downloader fake \\
        --source tv_camara --credit "TV Camara" --out-records audios.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Protocol

# permite tanto `python gcp/slice/audio_collector.py` quanto `python -m ...`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gcp.slice.collector import (  # noqa: E402
    MODALITY_AUDIO,
    default_item_id,
    write_records,
)

log = logging.getLogger(__name__)

DEFAULT_AUDIO_DIR = "data/govexe/audio"
# wav PCM: formato que o faster-whisper (e o ffmpeg por tras dele) le sem surpresa, e o
# que o `FakeDownloader` consegue gerar byte a byte sem nenhuma dependencia.
DEFAULT_AUDIO_FORMAT = "wav"
DEFAULT_AUDIO_SUFFIX = f".{DEFAULT_AUDIO_FORMAT}"
DOWNLOADERS = ("yt-dlp", "fake")
# Clipe do fake: 8 kHz, mono, 8 bits -- o menor wav valido que serve de amostra.
FAKE_SAMPLE_RATE = 8000
FAKE_DEFAULT_SECONDS = 1


def audio_filename(url: str, suffix: str = DEFAULT_AUDIO_SUFFIX) -> str:
    """Nome de arquivo estavel e seguro para a URL (ultimo trecho legivel + hash curto).

    O hash e o que garante unicidade (duas sessoes podem terminar em `video.mp3`); o
    trecho legivel e so para o diretorio continuar inspecionavel a olho nu.
    """
    bruto = re.split(r"[?#]", url)[0].rstrip("/").split("/")[-1]
    legivel = re.sub(r"[^A-Za-z0-9._-]+", "-", bruto).strip("-.")[:40] or "audio"
    legivel = Path(legivel).stem or "audio"
    return f"{legivel}-{hashlib.sha256(url.encode('utf-8')).hexdigest()[:8]}{suffix}"


# -- interface de download ---------------------------------------------------------------


class AudioDownloader(Protocol):
    """Minimo que a captura precisa de um downloader (yt-dlp real ou fake offline)."""

    def download(self, url: str, dest_dir: str | Path,
                 max_seconds: int | None = None) -> Path:
        """Baixa o audio de `url` em `dest_dir` e devolve o caminho local."""


class YtDlpDownloader:
    """Backend real: `yt-dlp` (YouTube das casas legislativas e midia direta).

    O import de `yt_dlp` fica DENTRO de `download()` -- mesmo padrao do
    `WhisperTranscriber` e dos wrappers de modelo do repo: a dependencia e OPCIONAL e so
    o lado de coleta a instala (`gcp/slice/requirements-collect.txt`), entao o modulo tem
    de importar sem ela (e e assim que os testes offline rodam).

    Pede `bestaudio` e extrai com o postprocessor `FFmpegExtractAudio` -- por isso o
    **ffmpeg** precisa existir no sistema. Arquivo ja baixado nao e rebaixado (o proprio
    yt-dlp pula), entao recapturar o mesmo lote e barato.
    """

    def __init__(self, audio_format: str = DEFAULT_AUDIO_FORMAT, quiet: bool = True):
        self.audio_format = audio_format
        self.quiet = quiet

    def _options(self, dest_dir: str | Path) -> dict:
        """Opcoes do yt-dlp que NAO dependem do pacote (testaveis sem ele instalado)."""
        return {
            "format": "bestaudio/best",
            "outtmpl": str(Path(dest_dir) / "%(id)s.%(ext)s"),
            "noplaylist": True,          # uma sessao por URL, nunca o canal inteiro
            "quiet": self.quiet,
            "no_warnings": self.quiet,
            "postprocessors": [{"key": "FFmpegExtractAudio",
                                "preferredcodec": self.audio_format}],
        }

    def download(self, url: str, dest_dir: str | Path,
                 max_seconds: int | None = None) -> Path:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import download_range_func

        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        opts = self._options(dest_dir)
        if max_seconds:
            # corta na origem: so os `max_seconds` iniciais sao baixados/remuxados
            # (sessao legislativa passa facil de uma hora -- isto e banda e disco).
            opts["download_ranges"] = download_range_func(None, [(0, max_seconds)])
            opts["force_keyframes_at_cuts"] = True
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        caminho = Path(ydl.prepare_filename(info)).with_suffix(f".{self.audio_format}")
        log.info("audio baixado: %s -> %s", url, caminho)
        return caminho


class FakeDownloader:
    """FAKE de `AudioDownloader`: grava um wav canonico, sem rede e sem yt-dlp.

    Usado pelos testes e pelo modo dev do README -- e o que permite ensaiar
    captura -> records -> indice sem tocar fonte publica nenhuma. O clipe e silencio PCM
    deterministico cuja duracao respeita `max_seconds` (assim o contrato do parametro e
    observavel ate no tamanho do arquivo).
    """

    def __init__(self, suffix: str = DEFAULT_AUDIO_SUFFIX, source_file: str | Path | None = None):
        self.suffix = suffix
        self.source_file = Path(source_file) if source_file else None
        self.calls: list[tuple[str, str, int | None]] = []

    @staticmethod
    def _wav(seconds: int) -> bytes:
        """Wav PCM mono 8 bits de `seconds` segundos (cabecalho RIFF de 44 bytes)."""
        amostras = max(1, int(seconds)) * FAKE_SAMPLE_RATE
        dados = b"\x80" * amostras                       # silencio em PCM unsigned 8 bits
        return (b"RIFF" + struct.pack("<I", 36 + len(dados)) + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 1, 1, FAKE_SAMPLE_RATE, FAKE_SAMPLE_RATE, 1, 8)
                + b"data" + struct.pack("<I", len(dados)) + dados)

    def download(self, url: str, dest_dir: str | Path,
                 max_seconds: int | None = None) -> Path:
        self.calls.append((url, str(dest_dir), max_seconds))
        destino = Path(dest_dir) / audio_filename(url, self.suffix)
        destino.parent.mkdir(parents=True, exist_ok=True)
        if self.source_file is not None:
            shutil.copyfile(self.source_file, destino)
        else:
            destino.write_bytes(self._wav(max_seconds or FAKE_DEFAULT_SECONDS))
        return destino


def open_downloader(backend: str = "yt-dlp", *,
                    audio_format: str = DEFAULT_AUDIO_FORMAT) -> AudioDownloader:
    """`fake` -> downloader canonico offline; `yt-dlp` -> o real (import lazy)."""
    if backend == "fake":
        return FakeDownloader(f".{audio_format}")
    if backend != "yt-dlp":
        raise ValueError(f"downloader desconhecido: {backend!r} (use um de {DOWNLOADERS})")
    return YtDlpDownloader(audio_format)


# -- captura -------------------------------------------------------------------------------


def collect_audio(urls: list[str], *, source: str, credit: str,
                  downloader: AudioDownloader,
                  dest_dir: str | Path = DEFAULT_AUDIO_DIR,
                  max_seconds: int | None = None) -> list[dict]:
    """Baixa cada URL e devolve os records `audio_asr` prontos para o fluxo `--records`.

    URL repetida no lote e baixada uma vez so (mesma URL == mesmo `item_id`). Uma URL que
    falha (fonte fora do ar, formato recusado) e registrada no log e PULADA: um lote de
    captura nao deve morrer inteiro por causa de uma sessao indisponivel.
    """
    registros: list[dict] = []
    vistas: set[str] = set()
    for url in urls:
        if url in vistas:
            log.debug("url repetida no lote, pulando: %s", url)
            continue
        vistas.add(url)
        try:
            caminho = downloader.download(url, dest_dir, max_seconds=max_seconds)
        except Exception as erro:                    # fonte fora do ar, formato recusado...
            log.error("falha ao baixar %s: %s", url, erro)
            continue
        registros.append({
            "item_id": default_item_id(MODALITY_AUDIO, url),
            "modality": MODALITY_AUDIO,
            "audio_path": str(caminho),
            "pointer_uri": url,                      # a fonte publica, preservada
            "credit": credit,                        # exigencia de licenca das fontes
            "source": source,
        })
    log.info("captura: %d de %d urls baixadas para %s", len(registros), len(vistas), dest_dir)
    return registros


# -- CLI --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None, *,
         downloader: AudioDownloader | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", action="append", required=True, dest="urls", metavar="URL",
                    help="URL publica do audio (repita a flag para varias)")
    ap.add_argument("--source", required=True,
                    help="fonte gravada no record (ex.: tv_camara, senado_multimidia)")
    ap.add_argument("--credit", required=True,
                    help="credito (autor/fonte) -- obrigatorio: e a licenca da captura")
    ap.add_argument("--out-records", required=True,
                    help="JSONL de saida (entrada do `collector.py --records`)")
    ap.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR,
                    help=f"diretorio local dos audios (default: {DEFAULT_AUDIO_DIR})")
    ap.add_argument("--max-seconds", type=int, default=None,
                    help="baixa so os N primeiros segundos (sessao legislativa e longa)")
    ap.add_argument("--downloader", default="yt-dlp", choices=list(DOWNLOADERS),
                    help="backend de download (default: yt-dlp; `fake` nao usa rede)")
    ap.add_argument("--audio-format", default=DEFAULT_AUDIO_FORMAT,
                    help=f"formato extraido (default: {DEFAULT_AUDIO_FORMAT})")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    baixador = downloader or open_downloader(a.downloader, audio_format=a.audio_format)
    registros = collect_audio(a.urls, source=a.source, credit=a.credit,
                              downloader=baixador, dest_dir=a.audio_dir,
                              max_seconds=a.max_seconds)
    if not registros:
        log.error("nenhum audio baixado de %d url(s)", len(a.urls))
        return 1
    destino = write_records(registros, a.out_records)
    print(f"{len(registros)} records -> {destino}")
    for r in registros[:5]:
        print(f"  {r['item_id']}  {r['audio_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
