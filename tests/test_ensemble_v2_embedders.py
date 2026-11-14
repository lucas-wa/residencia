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
    # o eixo video embute o keyframe da janela -- mesmo SigLIP 2 do eixo imagem
    assert emb.VIDEO_MODEL == "google/siglip2-so400m-patch14-384"
    assert emb.UNIFIED_A_MODEL == "Qwen/Qwen3-VL-Embedding-2B"


def test_registry_cobre_os_5_builders():
    from ensemble.native.embedders import EMBEDDER_BUILDERS

    # os 5 eixos + `audio_asr` (modo ASR->texto do eixo audio, selecionavel no eval)
    assert set(EMBEDDER_BUILDERS) == {
        "texto", "imagem", "audio", "audio_asr", "video", "unified_a"}
    assert all(callable(b) for b in EMBEDDER_BUILDERS.values())


class _FakeSentenceTransformer:
    """Duble do `SentenceTransformer` -- nenhum checkpoint e tocado."""

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.calls = []
        self.halved = False

    def half(self):
        self.halved = True
        return self

    def encode(self, texts, **kwargs):
        self.calls.append(("encode", list(texts), kwargs))
        return np.ones((len(texts), 1024), dtype="float64")


def _fake_sentence_transformers(monkeypatch, feitos):
    modulo = types.ModuleType("sentence_transformers")

    def fabrica(model_id, **kwargs):
        m = _FakeSentenceTransformer(model_id, **kwargs)
        feitos.append(m)
        return m

    modulo.SentenceTransformer = fabrica
    monkeypatch.setitem(sys.modules, "sentence_transformers", modulo)


def test_audio_asr_embute_a_transcricao_com_bge_m3(monkeypatch):
    """Modo ASR do eixo audio: mesmo BGE-M3 do eixo texto, items E query sao texto."""
    from ensemble.native.embedders import TEXT_MODEL, build_audio_asr_embedder

    feitos: list[_FakeSentenceTransformer] = []
    _fake_sentence_transformers(monkeypatch, feitos)

    embed_items, embed_query = build_audio_asr_embedder()
    modelo = feitos[0]
    assert modelo.model_id == TEXT_MODEL
    assert modelo.kwargs == {"device": "cuda"}
    assert modelo.halved

    itens = embed_items(["transcricao do clipe"])
    assert itens.shape == (1, 1024) and itens.dtype == np.float32
    assert modelo.calls[0][:2] == ("encode", ["transcricao do clipe"])

    consultas = embed_query(["orcamento da saude"])
    assert consultas.shape == (1, 1024) and consultas.dtype == np.float32
    assert modelo.calls[1][:2] == ("encode", ["orcamento da saude"])
    # items e query passam pelo MESMO encoder (espaco unico, como no eixo texto)
    assert embed_items is embed_query


def test_audio_asr_e_clap_sao_builders_distintos():
    """O modo CLAP (audio bruto) continua disponivel como ablacao."""
    from ensemble.native.embedders import (
        EMBEDDER_BUILDERS,
        build_audio_asr_embedder,
        build_audio_embedder,
    )

    assert EMBEDDER_BUILDERS["audio"] is build_audio_embedder
    assert EMBEDDER_BUILDERS["audio_asr"] is build_audio_asr_embedder


class _FakeSiglip:
    """Duble do `ClipSiglipRetriever` -- nenhum checkpoint e tocado."""

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.calls = []

    def _embed_image(self, paths):
        self.calls.append(("_embed_image", list(paths)))
        return np.ones((len(paths), 1152), dtype="float64")

    def _embed_text(self, texts):
        self.calls.append(("_embed_text", list(texts)))
        return np.ones((len(texts), 1152), dtype="float64")


def _fake_clip_siglip(monkeypatch, feitos):
    modulo = types.ModuleType("benchmark.models.clip_siglip")

    def fabrica(model_id, **kwargs):
        m = _FakeSiglip(model_id, **kwargs)
        feitos.append(m)
        return m

    modulo.ClipSiglipRetriever = fabrica
    monkeypatch.setitem(sys.modules, "benchmark.models.clip_siglip", modulo)


def test_video_embute_o_keyframe_com_siglip2(monkeypatch):
    """`_embed_image(keyframes)` / `_embed_text(textos)`, saida float32."""
    from ensemble.native.embedders import VIDEO_MODEL, build_video_embedder

    feitos: list[_FakeSiglip] = []
    _fake_clip_siglip(monkeypatch, feitos)

    embed_items, embed_query = build_video_embedder()
    modelo = feitos[0]
    assert modelo.model_id == VIDEO_MODEL
    assert modelo.kwargs == {"backend": "hf", "device": "cuda"}

    itens = embed_items(["kf/a.jpg", "kf/b.jpg"])
    assert itens.shape == (2, 1152) and itens.dtype == np.float32
    # o item e o KEYFRAME da janela (imagem) -- nao o arquivo de video compartilhado
    assert modelo.calls[0] == ("_embed_image", ["kf/a.jpg", "kf/b.jpg"])

    consultas = embed_query(["orcamento da saude"])
    assert consultas.shape == (1, 1152) and consultas.dtype == np.float32
    assert modelo.calls[1] == ("_embed_text", ["orcamento da saude"])


DIM_A = 8


class _FakeQwen3VL:
    """Duble do `Qwen3VLEmbedRetriever` -- nenhum checkpoint e tocado.

    Os vetores sao MARCADOS para o teste reconhecer a origem de cada linha: coluna 0 =
    1.0 (rota imagem) / 2.0 (rota texto); coluna 1 = posicao dentro do grupo. Mesmo
    `dim` nos dois lados (espaco unico), como no modelo real.
    """

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.calls = []

    def _marcados(self, marca, n):
        v = np.zeros((n, DIM_A), dtype="float64")
        v[:, 0] = marca
        v[:, 1] = np.arange(n, dtype="float64")
        return v

    def embed_images(self, paths):
        paths = list(paths)
        self.calls.append(("embed_images", paths))
        return self._marcados(1.0, len(paths))

    def embed_texts(self, texts):
        texts = list(texts)
        self.calls.append(("embed_texts", texts))
        return self._marcados(2.0, len(texts))


def _fake_qwen3vl(monkeypatch, feitos):
    modulo = types.ModuleType("benchmark.models.qwen3vl_embed")

    def fabrica(model_id, **kwargs):
        m = _FakeQwen3VL(model_id, **kwargs)
        feitos.append(m)
        return m

    modulo.Qwen3VLEmbedRetriever = fabrica
    monkeypatch.setitem(sys.modules, "benchmark.models.qwen3vl_embed", modulo)


def test_unified_a_repassa_o_adapter_path(monkeypatch):
    """O A avaliado DEPOIS do finetune carrega o adaptador LoRA treinado."""
    from ensemble.native.embedders import UNIFIED_A_MODEL, build_unified_a_embedder

    feitos: list[_FakeQwen3VL] = []
    _fake_qwen3vl(monkeypatch, feitos)

    embed_items, embed_query = build_unified_a_embedder(adapter_path="finetune/runs/x")

    assert feitos[0].model_id == UNIFIED_A_MODEL
    assert feitos[0].kwargs == {"device": "cuda", "adapter_path": "finetune/runs/x"}
    # mesmo modelo (espaco unico) para itens e query, mas encoders distintos: o de itens
    # roteia por modalidade, o de query e sempre texto
    assert embed_items is not embed_query
    assert embed_items([("text", "orcamento")]).dtype == np.float32
    assert embed_query(["orcamento"]).dtype == np.float32


def test_unified_a_sem_adapter_path_segue_na_base(monkeypatch):
    from ensemble.native.embedders import build_unified_a_embedder

    feitos: list[_FakeQwen3VL] = []
    _fake_qwen3vl(monkeypatch, feitos)

    build_unified_a_embedder()

    assert feitos[0].kwargs == {"device": "cuda", "adapter_path": None}


def test_unified_a_roteia_itens_por_modalidade_e_remonta_na_ordem(monkeypatch):
    """Batch misto: imagens por `embed_images`, textos por `embed_texts`, ordem preservada."""
    from ensemble.native.embedders import build_unified_a_embedder

    feitos: list[_FakeQwen3VL] = []
    _fake_qwen3vl(monkeypatch, feitos)

    embed_items, _ = build_unified_a_embedder()
    saida = embed_items([("image", "a.jpg"), ("text", "oi"), ("image", "b.png")])

    modelo = feitos[0]
    assert modelo.calls == [
        ("embed_images", ["a.jpg", "b.png"]),   # um batch por rota, nao um por item
        ("embed_texts", ["oi"]),
    ]
    assert saida.shape == (3, DIM_A) and saida.dtype == np.float32
    assert [float(linha[0]) for linha in saida] == [1.0, 2.0, 1.0]  # rota de cada linha
    assert [float(linha[1]) for linha in saida] == [0.0, 0.0, 1.0]  # posicao no grupo


def test_unified_a_nao_chama_o_modelo_para_grupo_vazio(monkeypatch):
    from ensemble.native.embedders import build_unified_a_embedder

    feitos: list[_FakeQwen3VL] = []
    _fake_qwen3vl(monkeypatch, feitos)

    so_texto, _ = build_unified_a_embedder()
    so_texto([("text", "oi"), ("text", "tchau")])
    assert feitos[0].calls == [("embed_texts", ["oi", "tchau"])]

    so_imagem, _ = build_unified_a_embedder()
    so_imagem([("image", "a.jpg")])
    assert feitos[1].calls == [("embed_images", ["a.jpg"])]


def test_unified_a_query_e_sempre_texto(monkeypatch):
    """Recuperacao cross-modal: a consulta entra como texto no mesmo espaco."""
    from ensemble.native.embedders import build_unified_a_embedder

    feitos: list[_FakeQwen3VL] = []
    _fake_qwen3vl(monkeypatch, feitos)

    _, embed_query = build_unified_a_embedder()
    consultas = embed_query(["orçamento da saúde"])

    assert feitos[0].calls == [("embed_texts", ["orçamento da saúde"])]
    assert consultas.shape == (1, DIM_A) and consultas.dtype == np.float32


def test_imagem_e_video_compartilham_o_modelo_mas_sao_eixos_separados(monkeypatch):
    """Mesma familia de modelo nos dois eixos; cada um indexa seu proprio conjunto."""
    from ensemble.native.embedders import (
        IMAGE_MODEL,
        VIDEO_MODEL,
        build_image_embedder,
        build_video_embedder,
    )

    assert VIDEO_MODEL == IMAGE_MODEL

    feitos: list[_FakeSiglip] = []
    _fake_clip_siglip(monkeypatch, feitos)

    img_items, _ = build_image_embedder()
    vid_items, _ = build_video_embedder()
    img_items(["pag/1.png"])
    vid_items(["kf/a.jpg"])

    assert len(feitos) == 2  # instancias independentes, um eixo por vez
    assert feitos[0].calls == [("_embed_image", ["pag/1.png"])]
    assert feitos[1].calls == [("_embed_image", ["kf/a.jpg"])]
