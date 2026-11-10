"""Monta o corpus de demonstracao (tema Orcamento, ~20 itens por eixo) em JSONL.

Passo de dados: chama os 5 `fetch_*` (rede). Os valores de FONTES_PADRAO sao fontes
REAIS, confirmadas (formato e alcancabilidade) antes do run.
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

# Fontes reais de cada eixo (NAO sao mais placeholders): formato e alcancabilidade
# conferidos por curl.
#   texto     -> RSS da Camara + do Senado, com <item><link> (lido por fetch_budget_articles)
#   documento -> PDF real do PLOA 2026 (Volume I), %PDF-1.6
#   imagem    -> listing da Agencia Camara, crawleado por fetch_gallery ate as paginas de
#                artigo (a galeria semantica nao existe; o <figure><figcaption> esta la)
#   audio     -> feeds podcast do Radio Senado, com <enclosure> (lido por fetch_clips)
#   video     -> canal oficial @canalgov
FONTES_PADRAO: dict[str, list[str] | str] = {
    "texto": [
        "https://www.camara.leg.br/noticias/rss/ultimas-noticias",
        "https://www12.senado.leg.br/noticias/rss",
    ],
    "documento": "https://www.congressonacional.leg.br/documents/150599582/152777548/Volume+I+-+PLOA2026.pdf/27b3dd01-42d6-4e0a-aff7-0f2cba33b67b",
    "imagem": [
        "https://www.camara.leg.br/noticias/ultimas",
    ],
    "audio": [
        "https://www12.senado.leg.br/radio/1/boletim.leg/podcast.xml",
        "https://www12.senado.leg.br/radio/1/voz-do-brasil/podcast.xml",
    ],
    "video": [
        "https://www.youtube.com/@canalgov",
    ],
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
