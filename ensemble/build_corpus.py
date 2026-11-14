"""Monta o corpus de demonstracao (tema Orcamento, ~20 itens por eixo) em JSONL.

Passo de dados: chama os 5 `fetch_*` (rede). Os valores de FONTES_PADRAO sao fontes
REAIS, confirmadas (formato e alcancabilidade) antes do run.
"""
from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from ensemble.capture.audio import fetch_clips
from ensemble.capture.document import fetch_budget_pages, fetch_budget_pages_multi
from ensemble.capture.image import fetch_gallery
from ensemble.capture.text import fetch_budget_articles
from ensemble.capture.video import fetch_sessions
from ensemble.corpus_v4 import check_corpus_paths
from ensemble.records import PointerRecord, save_records, validate_record

DESTINO_PADRAO = "data/govexe/ensemble_orcamento/records.jsonl"

# Corpus v4 (Semana 10): captura maior, em diretorio proprio para nao sobrescrever o v3.
DESTINO_V4 = "data/govexe/ensemble_orcamento_v4/records.jsonl"
POR_EIXO_V4 = 50

# Fontes reais de cada eixo (NAO sao mais placeholders): formato e alcancabilidade
# conferidos por curl.
#   texto     -> RSS da Camara + do Senado, com <item><link> (lido por fetch_budget_articles)
#   documento -> PDF real do PLOA 2026 (Volume I), %PDF-1.6
#   imagem    -> listing da Agencia Camara, crawleado por fetch_gallery ate as paginas de
#                artigo (a galeria semantica nao existe; o <figure><figcaption> esta la)
#   audio     -> feeds podcast do Radio Senado, com <enclosure> (lido por fetch_clips)
#   video     -> Portal Multimidia do Senado (TV Senado): /multimidia/eventos/{a}/{m}/{d}
#                lista os eventos, /multimidia/data/{id} traz os MP4 diretos (sem login/
#                token, amigavel a datacenter -- o YouTube bloqueia yt-dlp em IP de nuvem)
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
        "https://www12.senado.leg.br",
    ],
}

# Eixo documento do corpus v4: "legislativo geral" em vez do PLOA sozinho -- orcamento
# (PLOA/LDO/informativo), lei ordinaria e relatorio de atividade legislativa, para o
# fine-tune denso nao aprender so o layout de uma peca orcamentaria.
# As 5 URLs foram conferidas em 2026-11-14 (HTTP 200 + corpo comecando com %PDF, sem
# login/token). O v3 (FONTES_PADRAO["documento"]) segue com o PLOA unico.
DOCUMENTO_V4: list[str] = [
    # PLOA 2026, Volume I
    FONTES_PADRAO["documento"],
    # LDO 2026 (PLN 2/2025) -- atencao: o nome vindo da URL degenera para "pdf"
    "https://www.congressonacional.leg.br/materias/pesquisa/-/materia/168120/pdf",
    # Lei 15.321/2025 (anexo do projeto de lei)
    "https://www2.camara.leg.br/legin/fed/lei/2025/lei-15321-31-dezembro-2025-798617-anexo-pl.pdf",
    # Informativo do PLDO 2026 (Consultoria de Orcamento do Senado)
    "https://www12.senado.leg.br/orcamento/documentos/estudos/tipos-de-estudos/notas-tecnicas-e-informativos/informativo_pldo-2026.pdf",
    # Estatisticas da atividade legislativa da Camara, 2021
    "https://www2.camara.leg.br/atividade-legislativa/plenario/relatorios-da-atividade-legislativa/sessoes-legislativas/2021/ESTATISTICA%202021.pdf",
]


def _fetch_documento(fonte: str | list[str], *, dest_dir: str, n: int) -> list[PointerRecord]:
    """Eixo documento: uma URL (v3) ou uma lista de PDFs (v4)."""
    if isinstance(fonte, str):
        return fetch_budget_pages(fonte, dest_dir=dest_dir, n=n)
    return fetch_budget_pages_multi(fonte, dest_dir=dest_dir, n=n)


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
    validar: Callable[[list[PointerRecord]], None] | None = None,
) -> Path:
    """Salva `records.jsonl` e os binarios retidos sob `bin_dir/<eixo>/` (default: <destino>/../bin).

    `validar` e uma checagem EXTRA rodada depois do `_reter`/`validate_record` e ANTES de
    salvar -- corpus reprovado nao vira arquivo (ver `build_corpus_v4`).
    """
    f = {**FONTES_PADRAO, **(fontes or {})}
    destino = Path(destino)
    bin_dir = Path(bin_dir) if bin_dir is not None else destino.parent / "bin"
    d = {e: str(bin_dir / e) for e in ("documento", "imagem", "audio", "video")}
    eixos = [
        ("texto", lambda: fetch_budget_articles(f["texto"], n=por_eixo)),
        ("documento", lambda: _fetch_documento(f["documento"], dest_dir=d["documento"], n=por_eixo)),
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
    if validar is not None:
        validar(todos)
    destino.parent.mkdir(parents=True, exist_ok=True)
    save_records(todos, destino)
    return destino


def build_corpus_v4(
    destino: str | Path = DESTINO_V4, *, por_eixo: int = POR_EIXO_V4,
    fontes: dict | None = None, bin_dir: str | Path | None = None,
) -> Path:
    """Corpus v4: mesma captura, maior (50/eixo) e com sanity de caminho antes de salvar.

    O v4 alimenta o fine-tune denso da Semana 10, que precisa de mais itens por eixo
    (piso de 8 queries julgadas por eixo so fecha com corpus maior) e de todo item
    image-route com imagem de verdade em disco -- `check_corpus_paths` reprova o corpus
    ANTES de escrever o `records.jsonl`, em vez de deixar o `embed_images` do baseline A
    quebrar no meio do run.

    O eixo documento do v4 vem de `DOCUMENTO_V4` (varios PDFs legislativos), nao do PLOA
    sozinho; `fontes` explicitas do chamador ainda ganham.
    """
    return build_corpus(destino, por_eixo=por_eixo,
                        fontes={"documento": DOCUMENTO_V4, **(fontes or {})},
                        bin_dir=bin_dir, validar=check_corpus_paths)


if __name__ == "__main__":
    print(build_corpus(sys.argv[1] if len(sys.argv) > 1 else DESTINO_PADRAO))
