"""Coletor Bronze: sobe paginas, fotos e audios para o GCS, particionados por fonte/data.

Primeiro passo da fatia vertical (`gcp/slice/README-slice.md`): as paginas que
`data/govexe/build_planos_benchmark.py` rasteriza em `data/govexe/ptbr_planos/pages`
viram objetos imutaveis na camada **Bronze**, no layout que `fusao/coleta-gcp.md` ja
descreve:

    bronze/fonte=<source>/dt=<YYYY-MM-DD>/<arquivo>.png

O Bronze tambem recebe **audio** (sessoes de YouTube/RadioAgencia, secao 6 daquele doc) -- o
binario fica preservado como *pointer* auditavel; quem vira vetor e a transcricao, no
`embed_job.py`. Alem do modo "diretorio de paginas", ha o modo **`--records`**: um JSONL
com o lote misto (foto + legenda + credito, texto nativo, audio), cujos binarios sobem
para o Bronze e que sai enriquecido com `blob`/`uri` pronto para o
`embed_job.py --records` (formato documentado no README-slice.md).

A camada de storage fica atras da interface `ObjectStore` (5 metodos). Ha duas
implementacoes:

    GCSObjectStore    backend real (google-cloud-storage), usado na nuvem;
    LocalObjectStore  FAKE de filesystem -- modo dev e os testes offline, que nunca
                      tocam o GCS.

`collect()` e idempotente: objeto que ja existe no bucket nao e reenviado (use
`--force` para sobrescrever). Rodar o coletor duas vezes nao duplica nada -- o nome do
objeto e funcao de (fonte, data, arquivo).

Uso:
    python -m gcp.slice.collector --bucket gs://govexe-bronze \
        --images-dir data/govexe/ptbr_planos/pages
    python -m gcp.slice.collector --local-root /tmp/bronze --limit 4   # dev, sem GCS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import date as _date
from pathlib import Path
from typing import Protocol

# permite tanto `python gcp/slice/collector.py` quanto `python -m gcp.slice.collector`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.embed_qwen3vl import collect_paths  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = "data/govexe/ptbr_planos/pages"
DEFAULT_SOURCE = "tse_planos"
DEFAULT_PREFIX = "bronze"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
# Audio tambem entra no Bronze (sessoes de YouTube/RadioAgencia, `fusao/coleta-gcp.md` secao 6).
# O binario fica aqui como *pointer* auditavel; quem vira vetor e a transcricao (ASR).
AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".ogg", ".opus", ".flac")
MEDIA_SUFFIXES = IMAGE_SUFFIXES + AUDIO_SUFFIXES

# Modalidades do indice (`gcp/slice/embed_job.py` e quem as consome). Repetidas aqui como
# literais de proposito: o coletor nao importa o job (a dependencia corre no outro sentido).
MODALITY_DOC_IMAGE = "doc_image"
MODALITY_AUDIO = "audio_asr"


def default_item_id(modality: str, uri: str) -> str:
    """Id estavel de foto/texto/audio: `<modalidade>:<hash curto da uri>`.

    Paginas de documento usam o stem do arquivo (o id do `corpus.jsonl`); para as outras
    modalidades nao ha id natural garantido, entao a chave sai da URI -- deterministica,
    legivel na origem pela modalidade e estavel entre execucoes, que e o que o upsert do
    `embed_job.py` precisa. Um `item_id` explicito no record sempre vence este default.

    Mora aqui (e nao no job de embedding) porque quem emite record -- este coletor e o
    `audio_collector.py` -- precisa do mesmo id ANTES de o indice existir; o
    `embed_job.py` reexporta o nome.
    """
    return f"{modality}:{hashlib.sha256(uri.encode('utf-8')).hexdigest()[:16]}"


def modality_for(path: str | Path, default: str = MODALITY_DOC_IMAGE) -> str:
    """Modalidade a partir da extensao: audio -> `audio_asr`; o resto fica no `default`.

    Imagem nao se resolve so pela extensao -- uma pagina de plano (`doc_image`) e uma foto
    de agencia (`image`) sao ambas PNG/JPG -- entao quem decide e o `--modality` do CLI
    (ou o proprio record, no modo `--records`).
    """
    return (MODALITY_AUDIO if str(path).lower().endswith(AUDIO_SUFFIXES) else default)


# -- interface de storage --------------------------------------------------------------


class ObjectStore(Protocol):
    """Minimo que a fatia precisa de um object store (GCS real ou filesystem fake)."""

    def uri(self, name: str) -> str:
        """URI canonica do objeto (`gs://bucket/name` no backend real)."""

    def exists(self, name: str) -> bool:
        ...

    def upload(self, local_path: str | Path, name: str) -> str:
        """Sobe o arquivo e devolve a URI do objeto."""

    def list(self, prefix: str = "") -> list[str]:
        """Nomes dos objetos sob `prefix`, ordenados."""

    def download(self, name: str, dest_dir: str | Path) -> Path:
        """Baixa o objeto para `dest_dir/<basename>` e devolve o caminho local."""


class LocalObjectStore:
    """FAKE de `ObjectStore` sobre o filesystem -- modo dev e testes (nao toca a nuvem)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, name: str) -> Path:
        return self.root / name

    def uri(self, name: str) -> str:
        return self._path(name).as_posix()

    def exists(self, name: str) -> bool:
        return self._path(name).is_file()

    def upload(self, local_path: str | Path, name: str) -> str:
        dest = self._path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dest)
        return self.uri(name)

    def list(self, prefix: str = "") -> list[str]:
        if not self.root.is_dir():
            return []
        names = (p.relative_to(self.root).as_posix()
                 for p in self.root.rglob("*") if p.is_file())
        return sorted(n for n in names if n.startswith(prefix))

    def download(self, name: str, dest_dir: str | Path) -> Path:
        dest = Path(dest_dir) / Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(name), dest)
        return dest


class GCSObjectStore:
    """Backend real: Cloud Storage. `google-cloud-storage` so e importado aqui dentro."""

    def __init__(self, bucket: str, client=None):
        if client is None:
            from google.cloud import storage

            client = storage.Client()
        # aceita "gs://bucket", "gs://bucket/ignorado" ou so "bucket"
        self.bucket_name = bucket.removeprefix("gs://").split("/")[0]
        self._bucket = client.bucket(self.bucket_name)

    def uri(self, name: str) -> str:
        return f"gs://{self.bucket_name}/{name}"

    def exists(self, name: str) -> bool:
        return self._bucket.blob(name).exists()

    def upload(self, local_path: str | Path, name: str) -> str:
        self._bucket.blob(name).upload_from_filename(str(local_path))
        return self.uri(name)

    def list(self, prefix: str = "") -> list[str]:
        return sorted(b.name for b in self._bucket.list_blobs(prefix=prefix))

    def download(self, name: str, dest_dir: str | Path) -> Path:
        dest = Path(dest_dir) / Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._bucket.blob(name).download_to_filename(str(dest))
        return dest


def open_store(bucket: str | None = None, local_root: str | Path | None = None) -> ObjectStore:
    """`--bucket` -> GCS real; `--local-root` -> fake de filesystem. Um dos dois."""
    if bool(bucket) == bool(local_root):
        raise ValueError("informe exatamente um entre bucket (GCS) e local_root (dev/fake)")
    return GCSObjectStore(bucket) if bucket else LocalObjectStore(local_root)  # type: ignore


def store_from_location(location: str | Path) -> ObjectStore:
    """Uma flag so: `gs://...` -> GCS real; qualquer outro caminho -> fake de filesystem.

    Usado por `--records-store` do `embed_job.py`, que precisa de UM destino (o Bronze de
    onde os `blob` dos records serao baixados) sem repetir o par bucket/local-root.
    """
    texto = str(location)
    return GCSObjectStore(texto) if texto.startswith("gs://") else LocalObjectStore(texto)


# -- coleta ----------------------------------------------------------------------------


def blob_name(path: str | Path, source: str = DEFAULT_SOURCE, day: str | None = None,
              prefix: str = DEFAULT_PREFIX) -> str:
    """Nome do objeto Bronze: particionado por fonte e data, terminando no arquivo original.

    Deterministico de proposito -- e ele que torna o upload idempotente: a mesma pagina
    da mesma fonte, na mesma data de coleta, sempre cai no mesmo objeto.
    """
    day = day or _date.today().isoformat()
    return f"{prefix}/fonte={source}/dt={day}/{Path(path).name}"


def collect(paths: list[str], store: ObjectStore, *, source: str = DEFAULT_SOURCE,
            day: str | None = None, prefix: str = DEFAULT_PREFIX,
            force: bool = False, modality: str = MODALITY_DOC_IMAGE) -> list[dict]:
    """Sobe cada arquivo (imagem ou audio) para o Bronze e devolve um record por arquivo.

    Pula o que ja existe no destino (a menos de `force`), entao reprocessar uma fonte
    nao reescreve o bucket inteiro nem duplica objetos. O record ja sai no formato que o
    `embed_job.py --records` consome (`item_id`, `modality`, `blob`, `uri`).
    """
    day = day or _date.today().isoformat()
    registros: list[dict] = []
    for path in paths:
        name = blob_name(path, source, day, prefix)
        rec = {"item_id": Path(path).stem, "modality": modality_for(path, modality),
               "source": source, "blob": name, "uri": store.uri(name), "uploaded": False}
        if force or not store.exists(name):
            rec["uri"] = store.upload(path, name)
            rec["uploaded"] = True
        else:
            log.debug("objeto ja existe, pulando: %s", name)
        registros.append(rec)
    enviados = sum(r["uploaded"] for r in registros)
    log.info("bronze: %d arquivos (%d enviados, %d ja existiam) em fonte=%s dt=%s",
             len(registros), enviados, len(registros) - enviados, source, day)
    return registros


# -- records: lote misto (foto, texto nativo, audio) -------------------------------------


def collect_records(records: list[dict], store: ObjectStore, *, source: str = DEFAULT_SOURCE,
                    day: str | None = None, prefix: str = DEFAULT_PREFIX,
                    force: bool = False) -> list[dict]:
    """Sobe os BINARIOS dos records (foto/audio) para o Bronze e devolve os records prontos.

    Cada record ganha `blob` (objeto no Bronze), `uri` (a URI canonica desse objeto) e,
    quando o record nao traz um, `pointer_uri` apontando o binario original preservado --
    o *pointer* auditavel de `fusao/coleta-gcp.md` secao 6, ao lado do `credit`, que o coletor
    nunca descarta. Records de texto nativo nao tem binario e passam intactos.
    """
    day = day or _date.today().isoformat()
    saida: list[dict] = []
    for rec in records:
        novo = dict(rec)
        path = rec.get("image_path") or rec.get("audio_path") or rec.get("path")
        if not path:
            saida.append(novo)                       # texto nativo: nada a subir
            continue
        name = blob_name(path, novo.get("source", source), day, prefix)
        uri = store.uri(name)
        if force or not store.exists(name):
            uri = store.upload(path, name)
            novo["uploaded"] = True
        else:
            log.debug("objeto ja existe, pulando: %s", name)
            novo["uploaded"] = False
        novo.update(blob=name, uri=uri, modality=novo.get("modality")
                    or modality_for(path, MODALITY_DOC_IMAGE),
                    source=novo.get("source", source))
        novo.setdefault("pointer_uri", uri)
        saida.append(novo)
    binarios = sum(1 for r in saida if r.get("blob"))
    fontes = sorted({r.get("source", source) for r in saida})
    log.info("bronze: %d records (%d com binario) em dt=%s, fonte(s): %s",
             len(saida), binarios, day, ", ".join(fontes))
    return saida


def write_records(records: list[dict], path: str | Path) -> Path:
    """Grava os records em JSONL (a entrada do `embed_job.py --records`)."""
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                       encoding="utf-8")
    return destino


def load_records(path: str | Path) -> list[dict]:
    """Le um JSONL de records (um objeto por linha; linhas em branco ignoradas)."""
    linhas = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in linhas if ln.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    destino = ap.add_mutually_exclusive_group(required=True)
    destino.add_argument("--bucket", help="bucket GCS de destino (ex.: gs://govexe-bronze)")
    destino.add_argument("--local-root",
                         help="diretorio que simula o bucket (modo dev; nao toca a nuvem)")
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR,
                    help=f"diretorio das paginas (default: {DEFAULT_IMAGES_DIR})")
    ap.add_argument("--glob", default=None,
                    help="padrao dos arquivos dentro de --images-dir (use, por exemplo, "
                         "'**/*.mp3' para subir audio)")
    ap.add_argument("--records", default=None,
                    help="JSONL de records (foto/texto/audio): sobe os binarios para o "
                         "Bronze e reescreve os records com `blob`/`uri`")
    ap.add_argument("--out-records", default=None,
                    help="onde gravar o JSONL de records enriquecido (entrada do "
                         "`embed_job.py --records`)")
    ap.add_argument("--modality", default=MODALITY_DOC_IMAGE,
                    help=f"modalidade dos arquivos de --images-dir (default: "
                         f"{MODALITY_DOC_IMAGE}; audio e detectado pela extensao)")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help=f"fonte, usada na particao fonte=... (default: {DEFAULT_SOURCE})")
    ap.add_argument("--date", default=None,
                    help="data da particao dt=YYYY-MM-DD (default: hoje)")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX,
                    help=f"prefixo da camada no bucket (default: {DEFAULT_PREFIX})")
    ap.add_argument("--limit", type=int, default=None, help="sobe so as N primeiras paginas")
    ap.add_argument("--force", action="store_true",
                    help="reenvia mesmo quando o objeto ja existe no destino")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    store = open_store(a.bucket, a.local_root)

    if a.records:
        records = load_records(a.records)[:a.limit] if a.limit else load_records(a.records)
        if not records:
            log.error("nenhum record em %s", a.records)
            return 1
        registros = collect_records(records, store, source=a.source, day=a.date,
                                    prefix=a.prefix, force=a.force)
        if a.out_records:
            destino = write_records(registros, a.out_records)
            print(f"{len(registros)} records -> {destino}")
        else:
            log.warning("sem --out-records: os records enriquecidos saem so no stdout")
            for r in registros:
                print(json.dumps(r, ensure_ascii=False))
        return 0

    paths = collect_paths(a.images_dir, a.glob or "**/*.png", a.limit)
    if not paths:
        log.error("nenhum arquivo encontrado em %s", a.images_dir)
        return 1

    registros = collect(paths, store, source=a.source, day=a.date, prefix=a.prefix,
                        force=a.force, modality=a.modality)
    if a.out_records:
        print(f"{len(registros)} records -> {write_records(registros, a.out_records)}")
    for r in registros[:5]:
        print(f"{'up ' if r['uploaded'] else 'skip'} {r['uri']}")
    if len(registros) > 5:
        print(f"... e mais {len(registros) - 5} arquivos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
