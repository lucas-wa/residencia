"""Fiacao dos builders de embedder nativos -- sem baixar modelo."""
import sys
import types

import numpy as np


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
    assert emb.VIDEO_MODEL == "tencent/WeMM-Embedding-2B"
    assert emb.UNIFIED_A_MODEL == "Qwen/Qwen3-VL-Embedding-2B"


def test_registry_cobre_os_5_builders():
    from ensemble.native.embedders import EMBEDDER_BUILDERS

    assert set(EMBEDDER_BUILDERS) == {"texto", "imagem", "audio", "video", "unified_a"}
    assert all(callable(b) for b in EMBEDDER_BUILDERS.values())


class _FakeST:
    """Duble do `SentenceTransformer` do card do WeMM -- nenhum checkpoint e tocado."""

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.calls = []

    def encode_document(self, items):
        self.calls.append(("encode_document", list(items)))
        return np.ones((len(items), 2048), dtype="float64")

    def encode_query(self, texts):
        self.calls.append(("encode_query", list(texts)))
        return np.ones((len(texts), 2048), dtype="float64")


def _fake_sentence_transformers(monkeypatch, feitos):
    modulo = types.ModuleType("sentence_transformers")

    def fabrica(model_id, **kwargs):
        m = _FakeST(model_id, **kwargs)
        feitos.append(m)
        return m

    modulo.SentenceTransformer = fabrica
    monkeypatch.setitem(sys.modules, "sentence_transformers", modulo)


def test_video_usa_a_api_do_card_do_wemm(monkeypatch):
    """`encode_document([{"image": keyframe}])` / `encode_query(textos)`, saida float32."""
    from ensemble.native.embedders import VIDEO_MODEL, build_video_embedder

    feitos: list[_FakeST] = []
    _fake_sentence_transformers(monkeypatch, feitos)

    embed_items, embed_query = build_video_embedder()
    modelo = feitos[0]
    assert modelo.model_id == VIDEO_MODEL
    assert modelo.kwargs == {"trust_remote_code": True, "device": "cuda"}

    itens = embed_items(["kf/a.jpg", "kf/b.jpg"])
    assert itens.shape == (2, 2048) and itens.dtype == np.float32
    # o item e o KEYFRAME da janela, declarado como imagem -- nao o arquivo de video
    assert modelo.calls[0] == ("encode_document", [{"image": "kf/a.jpg"},
                                                   {"image": "kf/b.jpg"}])

    consultas = embed_query(["orcamento da saude"])
    assert consultas.shape == (1, 2048) and consultas.dtype == np.float32
    assert modelo.calls[1] == ("encode_query", ["orcamento da saude"])
