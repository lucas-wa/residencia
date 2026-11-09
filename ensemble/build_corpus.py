"""Monta o corpus de demonstracao (tema Orcamento, ~20 itens por eixo) em JSONL.

Passo de dados: chama os 5 `fetch_*` (rede). Os valores de FONTES_PADRAO sao
PLACEHOLDERS a confirmar com o Lucas antes do run real.
"""
from __future__ import annotations

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


def build_corpus(
    destino: str | Path = DESTINO_PADRAO, *, por_eixo: int = 20, fontes: dict | None = None
) -> Path:
    f = {**FONTES_PADRAO, **(fontes or {})}
    eixos = [
        ("texto", lambda: fetch_budget_articles(f["texto"], n=por_eixo)),
        ("documento", lambda: fetch_budget_pages(f["documento"], n=por_eixo)),
        ("imagem", lambda: fetch_gallery(f["imagem"], n=por_eixo)),
        ("audio", lambda: fetch_clips(f["audio"], n=por_eixo)),
        ("video", lambda: fetch_sessions(f["video"], n=por_eixo)),
    ]
    todos: list[PointerRecord] = []
    for nome, fetch in eixos:
        recs = fetch()
        if not recs:
            raise RuntimeError(f"eixo '{nome}' retornou 0 itens")
        todos.extend(recs)
    for r in todos:
        validate_record(r)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    save_records(todos, destino)
    return destino


if __name__ == "__main__":
    print(build_corpus(sys.argv[1] if len(sys.argv) > 1 else DESTINO_PADRAO))
