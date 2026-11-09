"""Adaptadores de descritor textual por modalidade (ASR / legenda / OCR / texto)."""
from typing import Callable


class EmptyDescriptorError(ValueError):
    """Descritor vazio: o chamador deve descartar o item."""


def _limpa(s: str) -> str:
    return " ".join((s or "").split())


def asr_descriptor(media_path: str, *, transcriber: Callable[[str], str]) -> str:
    out = _limpa(transcriber(media_path))
    if not out:
        raise EmptyDescriptorError(f"ASR vazio: {media_path}")
    return out


def caption_descriptor(caption: str) -> str:
    return _limpa(caption)


def ocr_descriptor(doc_path: str, *, extractor: Callable[[str], str]) -> str:
    out = _limpa(extractor(doc_path))
    if not out:
        raise EmptyDescriptorError(f"OCR vazio: {doc_path}")
    return out


def text_descriptor(html_or_text: str, *, html: bool = False) -> str:
    if html:
        from bs4 import BeautifulSoup

        html_or_text = BeautifulSoup(html_or_text, "html.parser").get_text(" ")
    return _limpa(html_or_text)
