"""Coletor Bronze: sobe as paginas-imagem dos planos para o GCS, particionadas por fonte/data.

Primeiro passo da fatia vertical (`gcp/slice/README-slice.md`): as paginas que
`data/govexe/build_planos_benchmark.py` rasteriza em `data/govexe/ptbr_planos/pages`
viram objetos imutaveis na camada **Bronze**, no layout que `fusao/coleta-gcp.md` ja
descreve:

    bronze/fonte=<source>/dt=<YYYY-MM-DD>/<pagina>.png

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
            force: bool = False) -> list[dict]:
    """Sobe cada pagina para o Bronze e devolve um registro por arquivo.

    Pula o que ja existe no destino (a menos de `force`), entao reprocessar uma fonte
    nao reescreve o bucket inteiro nem duplica objetos.
    """
    day = day or _date.today().isoformat()
    registros: list[dict] = []
    for path in paths:
        name = blob_name(path, source, day, prefix)
        if not force and store.exists(name):
            log.debug("objeto ja existe, pulando: %s", name)
            registros.append({"page_id": Path(path).stem, "blob": name,
                              "uri": store.uri(name), "uploaded": False})
            continue
        uri = store.upload(path, name)
        registros.append({"page_id": Path(path).stem, "blob": name,
                          "uri": uri, "uploaded": True})
    enviados = sum(r["uploaded"] for r in registros)
    log.info("bronze: %d paginas (%d enviadas, %d ja existiam) em fonte=%s dt=%s",
             len(registros), enviados, len(registros) - enviados, source, day)
    return registros


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    destino = ap.add_mutually_exclusive_group(required=True)
    destino.add_argument("--bucket", help="bucket GCS de destino (ex.: gs://govexe-bronze)")
    destino.add_argument("--local-root",
                         help="diretorio que simula o bucket (modo dev; nao toca a nuvem)")
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR,
                    help=f"diretorio das paginas (default: {DEFAULT_IMAGES_DIR})")
    ap.add_argument("--glob", default=None, help="padrao das imagens dentro de --images-dir")
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

    paths = collect_paths(a.images_dir, a.glob or "**/*.png", a.limit)
    if not paths:
        log.error("nenhuma pagina encontrada em %s", a.images_dir)
        return 1

    store = open_store(a.bucket, a.local_root)
    registros = collect(paths, store, source=a.source, day=a.date, prefix=a.prefix,
                        force=a.force)
    for r in registros[:5]:
        print(f"{'up ' if r['uploaded'] else 'skip'} {r['uri']}")
    if len(registros) > 5:
        print(f"... e mais {len(registros) - 5} paginas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
