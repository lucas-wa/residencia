"""Captura de documento: paginas do PDF orcamentario (LOA + relatorio da CMO, ano corrente).

A fonte e escolha a confirmar com o Lucas antes do fetch real (Task 9).
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Callable

from ensemble.descriptors import EmptyDescriptorError, ocr_descriptor
from ensemble.records import PointerRecord, validate_record


def page_id(pdf_path: str, page: int) -> str:
    """Identificador unico e estavel da pagina (1-based): '<arquivo>#p<N>'."""
    return f"{Path(pdf_path).name}#p{page}"


def parse_pages(
    pdf_path: str, *, extractor: Callable[[str], str], n: int = 20,
    renderer: Callable[[int, Path], None] | None = None, bin_dir: str | Path | None = None,
) -> list[PointerRecord]:
    """Gera ate `n` records 'documento'; o extractor recebe o page_id ('<arquivo>#pN').

    Paginas sem texto sao puladas; IndexError do extractor sinaliza fim do documento.
    Com `renderer(page, destino)` + `bin_dir`, a imagem da pagina e gravada em
    `bin_dir/<arquivo>_p<N>.png` e registrada em meta["path"]/meta["a_repr"] (binario retido).
    """
    out: list[PointerRecord] = []
    page = 0
    while len(out) < n:
        page += 1
        pid = page_id(pdf_path, page)
        try:
            desc = ocr_descriptor(os.fspath(pid), extractor=extractor)
        except EmptyDescriptorError:
            if page > n * 5:
                break
            continue
        except IndexError:
            break
        meta = {"pdf": str(pdf_path), "page": page}
        if renderer is not None and bin_dir is not None:
            img = Path(bin_dir) / f"{Path(pdf_path).name}_p{page}.png"
            img.parent.mkdir(parents=True, exist_ok=True)
            renderer(page, img)
            meta["path"] = str(img)
            meta["a_repr"] = str(img)
        rec = PointerRecord(
            id=f"documento-{pid}",
            modality="documento",
            descriptor=desc,
            pointer_uri=pid,
            source="pdf",
            meta=meta,
        )
        validate_record(rec)
        out.append(rec)
    return out


def _pdfium_extractor(pdf_path: str) -> Callable[[str], str]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)

    def extract(pid: str) -> str:
        idx = int(pid.rsplit("#p", 1)[1]) - 1
        if idx >= len(pdf):
            raise IndexError(pid)
        return pdf[idx].get_textpage().get_text_range()

    return extract


def _pdfium_renderer(pdf_path: str) -> Callable[[int, Path], None]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)

    def render(page: int, destino: Path) -> None:
        pdf[page - 1].render(scale=1.5).to_pil().save(destino)

    return render


def fetch_budget_pages(pdf_url: str, dest_dir: str = "data/ensemble", n: int = 20) -> list[PointerRecord]:
    """Wrapper fino de rede: baixa o PDF (LOA / relatorio CMO) e extrai texto via pypdfium2.

    Nao coberto por teste de rede; roda na Task 9.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    dest = str(Path(dest_dir) / Path(pdf_url.split("?")[0]).name)
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0 (ensemble-capture)"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    return parse_pages(dest, extractor=_pdfium_extractor(dest), n=n,
                       renderer=_pdfium_renderer(dest), bin_dir=Path(dest_dir) / "pages")
