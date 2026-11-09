"""Monta o corpus de demonstracao (tema Orcamento, ~20 itens por eixo) em JSONL.

Passo de dados: chama os 5 `fetch_*` (rede). Os valores de FONTES_PADRAO sao
PLACEHOLDERS a confirmar com o Lucas antes do run real.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ensemble.capture.audio import fetch_clips
from ensemble.capture.document import fetch_budget_pages
from ensemble.capture.image import fetch_gallery
from ensemble.capture.text import fetch_budget_articles
from ensemble.capture.video import fetch_sessions
from ensemble.records import PointerRecord, save_records, validate_record

DESTINO_PADRAO = "data/govexe/ensemble_orcamento/records.jsonl"

# TODO(confirmar com Lucas): fontes reais de cada eixo.
FONTES_PADRAO: dict[str, list[str] | str] = {
    "texto": [],       # feeds RSS de noticias (Agencia Camara/Senado/EBC)
    "documento": "",   # URL do PDF (LOA + relatorio da CMO)
    "imagem": [],      # paginas de galeria (Camara/EBC/Senado)
    "audio": [],       # feeds RSS com <enclosure> (Radioagencia)
    "video": [],       # handles de canais oficiais (ex. Canal Gov)
}


def _reter(rec: PointerRecord, eixo_dir: Path) -> None:
    """Garante o binario sob `eixo_dir`: copia meta["path"] de fora e reaponta o caminho."""
    p = rec.meta.get("path")
    if not p:
        return
    src = Path(p)
    if not src.is_file():
        raise FileNotFoundError(f"record {rec.id}: meta['path'] inexistente: {p}")
    if eixo_dir.resolve() in src.resolve().parents:
        return
    eixo_dir.mkdir(parents=True, exist_ok=True)
    dst = eixo_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    rec.meta["path"] = str(dst)


def build_corpus(
    destino: str | Path = DESTINO_PADRAO, *, por_eixo: int = 20, fontes: dict | None = None,
    bin_dir: str | Path | None = None,
) -> Path:
    """Salva `records.jsonl` e os binarios retidos sob `bin_dir/<eixo>/` (default: <destino>/../bin)."""
    f = {**FONTES_PADRAO, **(fontes or {})}
    destino = Path(destino)
    bin_dir = Path(bin_dir) if bin_dir is not None else destino.parent / "bin"
    d = {e: str(bin_dir / e) for e in ("documento", "imagem", "audio", "video")}
    eixos = [
        ("texto", lambda: fetch_budget_articles(f["texto"], n=por_eixo)),
        ("documento", lambda: fetch_budget_pages(f["documento"], dest_dir=d["documento"], n=por_eixo)),
        ("imagem", lambda: fetch_gallery(f["imagem"], n=por_eixo, dest_dir=d["imagem"])),
        ("audio", lambda: fetch_clips(f["audio"], dest_dir=d["audio"], n=por_eixo)),
        ("video", lambda: fetch_sessions(f["video"], n=por_eixo, dest_dir=d["video"])),
    ]
    todos: list[PointerRecord] = []
    for nome, fetch in eixos:
        recs = fetch()
        if not recs:
            raise RuntimeError(f"eixo '{nome}' retornou 0 itens")
        for r in recs:
            _reter(r, bin_dir / nome)
        todos.extend(recs)
    for r in todos:
        validate_record(r)
    destino.parent.mkdir(parents=True, exist_ok=True)
    save_records(todos, destino)
    return destino


if __name__ == "__main__":
    print(build_corpus(sys.argv[1] if len(sys.argv) > 1 else DESTINO_PADRAO))
