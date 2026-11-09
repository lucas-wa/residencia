"""Fiacao dos builders de embedder nativos -- sem baixar modelo."""
import sys


def test_modulo_importa_sem_baixar_modelo():
    for m in [k for k in sys.modules if k.startswith("ensemble.native.embedders")]:
        del sys.modules[m]
    before = {m for m in ("torch", "sentence_transformers", "transformers") if m in sys.modules}
    import ensemble.native.embedders as emb

    after = {m for m in ("torch", "sentence_transformers", "transformers") if m in sys.modules}
    assert after == before  # o import do modulo nao puxa libs de modelo
    assert emb.TEXT_MODEL == "BAAI/bge-m3"
    assert emb.IMAGE_MODEL == "google/siglip2-so400m-patch14-384"
    assert emb.AUDIO_MODEL == "laion/larger_clap_general"
    assert emb.VIDEO_MODEL == "WeMM-Embedding-2B"
    assert emb.UNIFIED_A_MODEL == "Qwen/Qwen3-VL-Embedding-2B"


def test_registry_cobre_os_5_builders():
    from ensemble.native.embedders import EMBEDDER_BUILDERS

    assert set(EMBEDDER_BUILDERS) == {"texto", "imagem", "audio", "video", "unified_a"}
    assert all(callable(b) for b in EMBEDDER_BUILDERS.values())
