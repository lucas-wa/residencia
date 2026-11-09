import pytest

from ensemble.descriptors import (
    EmptyDescriptorError,
    asr_descriptor,
    caption_descriptor,
    text_descriptor,
)


def test_asr_usa_transcriber_injetado():
    out = asr_descriptor("x.wav", transcriber=lambda p: "fala sobre orçamento")
    assert out == "fala sobre orçamento"


def test_asr_vazio_levanta_EmptyDescriptorError():
    with pytest.raises(EmptyDescriptorError):
        asr_descriptor("x.wav", transcriber=lambda p: "   ")


def test_text_descriptor_extrai_de_html():
    assert text_descriptor("<p>Orçamento <b>2026</b></p>", html=True) == "Orçamento 2026"


def test_caption_normaliza_espacos():
    assert caption_descriptor("  foto da  votação  ") == "foto da votação"
