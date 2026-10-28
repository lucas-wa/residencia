"""Build the pt-BR visual-document benchmark from the GovExe `planos` config.

Output layout (consumed by `benchmark.common.datasets.load_local`):

    <out_dir>/pages/<tse_seq>_p<NN>.png
    <out_dir>/corpus.jsonl     {"id": "<tse_seq>_p<NN>", "image_path": "..."}
    <out_dir>/queries.jsonl    {"id": "<tse_seq>_<destaque|resumo|eixoN>", "text": "..."}
    <out_dir>/qrels.tsv        <query_id>\t<page_id>\t1
    <out_dir>/pdf_cache/<tse_seq>.pdf   (cache dos PDFs crus; so no caminho real)

Relevance is document-level: every query derived from a plan is relevant to every
page of that same plan (see spec section 6).

`build()` takes the PDF fetcher and the renderer as arguments so the assembly logic is
testable without network or real PDFs; `main()` wires the real ones (parquet + HTTP +
pypdfium2).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

SEED = 42
# Espelho PUBLICO dos planos no Hub. O repo original (`GovExe/gov-br-linguagem-simples`)
# e privado -- `load_dataset` nele responde 401/404 sem token.
HUB_DATASET = "lucas-wa/gov-br-linguagem-simples"
DEFAULT_PARQUET = r"D:\projects\govexe\dataset\planos\train.parquet"
DEFAULT_OUT_DIR = "data/govexe/ptbr_planos"
RENDER_DPI = 150
HTTP_TIMEOUT = 60
USER_AGENT = "residencia-benchmark/1.0 (+academic use)"


def _is_built(out: Path) -> bool:
    """True quando corpus/queries/qrels ja existem e nao estao vazios."""
    return all((out / name).exists() and (out / name).stat().st_size > 0
               for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv"))


def build(rows, out_dir, pdf_fetcher: Callable[[str], bytes],
          renderer: Callable[[bytes, str, str], list[str]], force: bool = False) -> None:
    """Assemble corpus/queries/qrels from `rows`, skipping plans whose PDF fails.

    E resumivel e idempotente (a construcao real baixa 27 PDFs e rasteriza centenas de
    paginas, entao um job interrompido nao pode comecar do zero):

    * se os tres arquivos de saida ja existem e nao estao vazios, retorna cedo;
    * por plano, se `<out_dir>/pages/<seq>_p*.png` ja existe, reusa os PNGs e nao toca
      na rede nem no renderer;
    * corpus/queries/qrels sao indexados por id, entao reconstruir nao duplica linhas.

    `force=True` ignora os dois caches e refaz tudo.
    """
    out = Path(out_dir)
    pages_dir = out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    if not force and _is_built(out):
        log.info("benchmark ja construido (use --force para refazer)")
        return

    corpus: dict[str, dict] = {}
    queries: dict[str, dict] = {}
    qrels: dict[str, list[str]] = {}
    for row in rows:
        seq = str(row["tse_sequencial"])
        cached_pages = [] if force else sorted(pages_dir.glob(f"{seq}_p*.png"))
        if cached_pages:
            page_paths = [str(p) for p in cached_pages]
            log.info("plano %s: reusando %d paginas ja rasterizadas", seq, len(page_paths))
        else:
            try:
                pdf = pdf_fetcher(row["pdf_url"])
                page_paths = renderer(pdf, str(pages_dir), seq)
            except Exception as e:                   # PDF corrompido/ilegivel
                log.warning("plano %s pulado: %s", seq, e)
                continue
        page_ids = []
        for pth in page_paths:
            pid = Path(pth).stem
            corpus[pid] = {"id": pid, "image_path": str(pth)}
            page_ids.append(pid)

        # queries derivadas dos alvos ja existentes
        def add_q(suffix, text, page_ids=page_ids, seq=seq):
            if text and text.strip():
                qid = f"{seq}_{suffix}"
                queries[qid] = {"id": qid, "text": text.strip()}
                qrels[qid] = list(dict.fromkeys(page_ids))

        add_q("destaque", row.get("destaque"))
        add_q("resumo", row.get("resumo"))
        try:
            for i, eixo in enumerate(json.loads(row.get("eixos") or "[]")):
                add_q(f"eixo{i}", f'{eixo.get("titulo", "")} {eixo.get("propostas", "")}')
        except json.JSONDecodeError:
            log.warning("eixos invalidos em %s", seq)

    qrel_lines = [f"{qid}\t{pid}\t1" for qid, pids in qrels.items() for pid in pids]
    (out / "corpus.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in corpus.values()), encoding="utf-8")
    (out / "queries.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in queries.values()), encoding="utf-8")
    (out / "qrels.tsv").write_text("\n".join(qrel_lines), encoding="utf-8")
    log.info("corpus=%d paginas, queries=%d, qrels=%d",
             len(corpus), len(queries), len(qrel_lines))


# -- real IO wiring ------------------------------------------------------------------


def load_planos_rows(parquet_path: str | None = None) -> list[dict]:
    """Read the `planos` rows from the local GovExe parquet, or from the Hub."""
    if parquet_path:
        import pandas as pd

        return pd.read_parquet(parquet_path).to_dict("records")
    from datasets import load_dataset

    ds = load_dataset(HUB_DATASET, "planos", split="train")
    return [dict(r) for r in ds]


def http_fetch(url: str) -> bytes:
    """GET the plan PDF (the GovExe-served original)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:  # noqa: S310
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"resposta nao parece PDF ({len(data)} bytes)")
    return data


def make_cached_fetch(rows, out_dir, fetcher: Callable[[str], bytes] = http_fetch
                      ) -> Callable[[str], bytes]:
    """Wrap `fetcher` with an on-disk PDF cache at `<out_dir>/pdf_cache/<seq>.pdf`.

    Mantem a assinatura `(url) -> bytes` esperada por `build()`: o `tse_sequencial` vem
    de um mapa url->seq capturado no closure. E a segunda linha de defesa -- o caso comum
    ja e coberto pelo reuso das paginas renderizadas --, util quando o job morre no meio
    da rasterizacao de um plano.
    """
    url_to_seq = {str(r["pdf_url"]): str(r["tse_sequencial"]) for r in rows}
    cache_dir = Path(out_dir) / "pdf_cache"

    def cached_fetch(url: str) -> bytes:
        import hashlib

        seq = url_to_seq.get(url) or hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        target = cache_dir / f"{seq}.pdf"
        if target.exists() and target.stat().st_size > 0:
            log.info("plano %s: PDF vindo do cache (%s)", seq, target)
            return target.read_bytes()
        data = fetcher(url)
        cache_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return data

    return cached_fetch


def pdfium_renderer(pdf_bytes: bytes, out_dir: str, stem: str,
                    dpi: int = RENDER_DPI) -> list[str]:
    """Rasterise every page to PNG at `dpi` and return the written paths."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_bytes)
    paths: list[str] = []
    try:
        scale = dpi / 72.0
        for i in range(len(doc)):
            target = Path(out_dir) / f"{stem}_p{i:02d}.png"
            if not target.exists():
                page = doc[i]
                bitmap = page.render(scale=scale)
                bitmap.to_pil().save(str(target))
                page.close()
            paths.append(str(target))
    finally:
        doc.close()
    if not paths:
        raise ValueError("PDF sem paginas")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=DEFAULT_PARQUET,
                    help="parquet local dos planos; vazio usa o dataset do HuggingFace")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--dpi", type=int, default=RENDER_DPI)
    ap.add_argument("--limit", type=int, default=None, help="amostra os N primeiros planos")
    ap.add_argument("--force", action="store_true",
                    help="refaz tudo, ignorando paginas/saidas ja existentes")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import random

    random.seed(SEED)
    rows = load_planos_rows(a.parquet or None)
    if a.limit:
        rows = rows[: a.limit]
    log.info("planos carregados: %d (seed=%d)", len(rows), SEED)

    def renderer(pdf_bytes, out_dir, stem):
        return pdfium_renderer(pdf_bytes, out_dir, stem, dpi=a.dpi)

    build(rows, a.out_dir, make_cached_fetch(rows, a.out_dir), renderer, force=a.force)


if __name__ == "__main__":
    main()
