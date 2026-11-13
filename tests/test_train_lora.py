"""Treinador LoRA/QLoRA (`finetune/train_lora.py`) -- tudo OFFLINE.

Nenhum teste carrega modelo, toca a rede ou dispara treino: o data loader e puro Python,
a resolucao de base/metodo e um dicionario e o `--dry-run` tolera torch/colpali/
bitsandbytes ausentes (o caso real deste venv) ou mockados.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from finetune import train_lora as T


def _pairs_file(tmp_path: Path, n: int = 4, negs: int = 2) -> Path:
    """pairs.jsonl no formato de finetune/make_pairs.py, com as imagens no disco."""
    pages = tmp_path / "pages"
    pages.mkdir(exist_ok=True)
    for i in range(n + negs + 1):
        (pages / f"BR_p{i:02d}.png").write_bytes(b"png")
    linhas = []
    for i in range(n):
        neg_ids = [f"BR_p{(i + 1 + j) % (n + negs):02d}" for j in range(negs)]
        linhas.append({
            "id": f"id{i:03d}",
            "query": f"o que o plano propoe sobre tema {i}?",
            "pos_id": f"BR_p{i:02d}",
            "pos_image": str(pages / f"BR_p{i:02d}.png"),
            "neg_ids": neg_ids,
            "neg_images": [str(pages / f"{p}.png") for p in neg_ids],
            "label": 1,
            "generator": "template",
        })
    out = tmp_path / "pairs.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in linhas),
                   encoding="utf-8")
    return out


# -- data loader ----------------------------------------------------------------------


def test_load_pairs_le_o_formato_de_make_pairs(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=3))

    assert [p.id for p in pares] == ["id000", "id001", "id002"]
    assert pares[0].label == 1 and pares[0].generator == "template"
    assert len(pares[0].neg_images) == len(pares[0].neg_ids) == 2
    assert Path(pares[0].pos_image).exists()


def test_load_pairs_limit_e_dedup_por_id(tmp_path):
    arq = _pairs_file(tmp_path, n=4)
    repetida = json.loads(arq.read_text(encoding="utf-8").splitlines()[0])
    with arq.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(repetida) + "\n")

    assert len(T.load_pairs(arq)) == 4                 # a linha repetida nao duplica
    assert len(T.load_pairs(arq, limit=2)) == 2


def test_load_pairs_sem_arquivo_falha_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_pairs"):
        T.load_pairs(tmp_path / "nao_existe.jsonl")


def test_load_pairs_arquivo_vazio_falha(tmp_path):
    vazio = tmp_path / "pairs.jsonl"
    vazio.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nenhum par"):
        T.load_pairs(vazio)


@pytest.mark.parametrize("obj,erro", [
    ({"id": "a", "query": "", "pos_id": "p", "pos_image": "x.png"}, "campos ausentes"),
    ({"id": "a", "query": "q", "pos_image": "x.png"}, "campos ausentes"),
    ({"id": "a", "query": "q", "pos_id": "p", "pos_image": "x.png",
      "neg_ids": ["n1", "n2"], "neg_images": ["n1.png"]}, "neg_ids"),
])
def test_parse_pair_valida_o_contrato(obj, erro):
    with pytest.raises(ValueError, match=erro):
        T.parse_pair(obj, lineno=7)


def test_load_pairs_json_invalido_aponta_a_linha(tmp_path):
    arq = tmp_path / "pairs.jsonl"
    arq.write_text('{"id": "a"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="linha 1"):
        T.load_pairs(arq)


def test_missing_images_reporta_o_que_falta(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=2))
    assert T.missing_images(pares) == []

    Path(pares[0].pos_image).unlink()
    assert T.missing_images(pares) == [pares[0].pos_image]


# -- batches --------------------------------------------------------------------------


def test_batch_poe_os_positivos_primeiro_e_depois_os_negativos(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=4))

    batch = T.make_batch(pares[:2], max_neg=2)

    assert batch.queries == tuple(p.query for p in pares[:2])
    assert batch.images[:2] == (pares[0].pos_image, pares[1].pos_image)
    assert batch.n_pos == 2 and batch.n_neg == len(batch.images) - 2
    assert len(set(batch.images)) == len(batch.images)      # sem imagem repetida


def test_batch_descarta_negativo_que_e_positivo_de_outro_exemplo(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=4))
    # id000 tem BR_p01 como negativo, e BR_p01 e o positivo de id001
    assert pares[1].pos_image in pares[0].neg_images

    batch = T.make_batch(pares[:2], max_neg=2)

    assert batch.images.count(pares[1].pos_image) == 1


def test_max_neg_zero_deixa_so_in_batch_negatives(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=3))

    batch = T.make_batch(pares, max_neg=0)

    assert batch.n_neg == 0
    assert batch.images == tuple(p.pos_image for p in pares)


def test_pairbatch_dense_items_modalidade_mista(tmp_path):
    """Batch misto: `.dense_items` leva a modalidade (em ingles) ao `DenseEncoder`."""
    pagina = str(tmp_path / "BR_p99.png")
    texto = T.Pair(id="t0", query="o que o plano diz sobre creches?", pos_id="TX001",
                   pos_image="", pos_modality="texto", pos_text="trecho do plano")
    imagem = T.Pair(id="i0", query="qual a meta de saude?", pos_id="BR_p99",
                    pos_image=pagina)

    batch = T.make_batch([texto, imagem], max_neg=0)

    assert list(batch.dense_items) == [("text", "trecho do plano"), ("image", pagina)]
    assert batch.images == (pagina,)          # so o caminho de imagem; o texto nao vaza
    assert batch.n_pos == 2 and batch.n_neg == 0

    # caso so-imagem: `.images` continua identico ao de antes (caminho MaxSim/colqwen)
    pares = T.load_pairs(_pairs_file(tmp_path, n=2))
    so_imagem = T.make_batch(pares, max_neg=1)
    assert so_imagem.images[:2] == (pares[0].pos_image, pares[1].pos_image)
    assert list(so_imagem.dense_items)[:2] == [("image", pares[0].pos_image),
                                               ("image", pares[1].pos_image)]
    assert len(so_imagem.dense_items) == len(so_imagem.images)


def test_modalidade_dense_mapeia_pt_para_o_encoder():
    """O par fala pt-BR ("imagem"/"texto"); o `DenseEncoder`, ingles ("image"/"text")."""
    assert T.MODALIDADE_DENSE == {"imagem": "image", "texto": "text"}
    assert set(T.MODALIDADE_DENSE.values()) == set(T.DENSE_MODALITIES)


def test_parse_pair_le_modalidade_e_texto_com_retrocompat():
    antigo = T.parse_pair({"id": "a", "query": "q", "pos_id": "p", "pos_image": "x.png"})
    assert antigo.pos_modality == "imagem" and antigo.pos_text == ""

    novo = T.parse_pair({"id": "b", "query": "q", "pos_id": "t", "pos_modality": "texto",
                         "pos_text": "trecho", "pos_image": ""})
    assert novo.pos_modality == "texto" and novo.pos_text == "trecho"

    with pytest.raises(ValueError, match="campos ausentes"):   # texto sem pos_text
        T.parse_pair({"id": "c", "query": "q", "pos_id": "t", "pos_modality": "texto",
                      "pos_image": "x.png"}, lineno=3)
    with pytest.raises(ValueError, match="modalidade"):
        T.parse_pair({"id": "d", "query": "q", "pos_id": "t", "pos_modality": "audio",
                      "pos_image": "x.png"}, lineno=4)


def test_iter_batches_cobre_todos_os_pares_e_o_ultimo_batch_e_menor(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=5))

    batches = list(T.iter_batches(pares, batch_size=2, max_neg=1))

    assert [b.n_pos for b in batches] == [2, 2, 1]
    vistos = [p.id for b in batches for p in b.pairs]
    assert sorted(vistos) == sorted(p.id for p in pares)


def test_iter_batches_shuffle_e_deterministico_por_seed(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=5))

    def ordem(seed):
        return [p.id for b in T.iter_batches(pares, 2, shuffle=True, seed=seed)
                for p in b.pairs]

    assert ordem(1) == ordem(1)
    assert sorted(ordem(1)) == sorted(p.id for p in pares)


def test_iter_batches_rejeita_batch_size_invalido(tmp_path):
    pares = T.load_pairs(_pairs_file(tmp_path, n=2))
    with pytest.raises(ValueError, match="batch_size"):
        list(T.iter_batches(pares, batch_size=0))


# -- base / metodo / estimativas ------------------------------------------------------


def test_resolve_config_resolve_base_e_metodo():
    cfg = T.resolve_config("colqwen2.5", "lora")
    assert cfg.spec.model_id == "vidore/colqwen2.5-v0.2" and cfg.spec.kind == "colqwen2.5"

    cfg8b = T.resolve_config("colembed8b", "qlora")
    assert cfg8b.spec.params_b == 8.0 and cfg8b.spec.kind == "colembed"


def test_base_qwen3vl_resolve():
    """A base DENSA (Qwen3-VL-Embedding-2B) resolve e atravessa as estimativas."""
    cfg = T.resolve_config("qwen3vl", "lora")

    assert cfg.spec.kind == "qwen3vl"
    assert "Qwen3-VL" in cfg.spec.model_id

    estimado = T.estimate_lora_params(cfg.spec, cfg.lora_r)
    assert estimado > 0

    vram = T.estimate_vram_gb(cfg, estimado)
    assert "total" in vram and vram["total"] > 0


@pytest.mark.parametrize("base,metodo,erro", [
    ("colqwen3", "lora", "base desconhecida"),
    ("colqwen2.5", "full", "metodo desconhecido"),
])
def test_resolve_config_rejeita_base_ou_metodo_invalido(base, metodo, erro):
    with pytest.raises(ValueError, match=erro):
        T.resolve_config(base, metodo)


@pytest.mark.parametrize("campo", ["batch_size", "epochs", "grad_accum"])
def test_resolve_config_rejeita_tamanho_zero(campo):
    with pytest.raises(ValueError, match=campo):
        T.resolve_config("colqwen2.5", "lora", **{campo: 0})


def test_warmup_cosine_sobe_e_desce_sem_cantos(tmp_path):
    f = T.warmup_cosine(total_steps=100, warmup_ratio=0.1)

    assert 0 < f(0) < f(5) <= 1.0                     # warmup linear
    assert f(10) == pytest.approx(1.0)                # pico ao fim do warmup
    assert f(99) < 0.05                               # cosseno chegando a zero
    assert all(0.0 <= f(p) <= 1.0 for p in range(120))

    um_passo = T.warmup_cosine(total_steps=1)         # smoke: nao pode estourar
    assert 0.0 <= um_passo(0) <= 1.0


def test_out_path_default_nomeia_por_base_e_metodo():
    assert T.resolve_config("colembed8b", "qlora").out_path.name == "colembed8b-qlora"
    assert T.resolve_config("colqwen2.5", "lora", output_dir="x/y").out_path == Path("x/y")


def test_count_steps_respeita_grad_accum_e_max_steps():
    cfg = T.resolve_config("colqwen2.5", "lora", batch_size=2, grad_accum=4, epochs=2)
    assert T.count_steps(10, cfg) == (5, 3)          # 5 batches/epoca, 10/4 -> 3 passos
    cfg2 = T.resolve_config("colqwen2.5", "lora", batch_size=2, grad_accum=4, epochs=2,
                            max_steps=2)
    assert T.count_steps(10, cfg2)[1] == 2


def test_estimate_lora_params_segue_a_formula_a_mais_b():
    spec = T.BASES["colqwen2.5"]
    alvos = ("q_proj",)

    esperado = 8 * (spec.hidden + spec.heads * spec.head_dim) * spec.layers

    assert T.estimate_lora_params(spec, r=8, target_modules=alvos) == esperado
    # dobrar o rank dobra os parametros; mais alvos so acrescentam
    assert T.estimate_lora_params(spec, 16, alvos) == 2 * esperado
    assert T.estimate_lora_params(spec, 8, ("q_proj", "o_proj")) > esperado


def test_estimate_lora_params_ignora_modulo_desconhecido():
    spec = T.BASES["colqwen2.5"]
    assert (T.estimate_lora_params(spec, 8, ("q_proj", "nao_existe"))
            == T.estimate_lora_params(spec, 8, ("q_proj",)))


def test_estimate_vram_qlora_pesa_menos_que_lora_e_8b_mais_que_3b():
    lora3b = T.estimate_vram_gb(T.resolve_config("colqwen2.5", "lora"), 60_000_000)
    qlora3b = T.estimate_vram_gb(T.resolve_config("colqwen2.5", "qlora"), 60_000_000)
    qlora8b = T.estimate_vram_gb(T.resolve_config("colembed8b", "qlora"), 87_000_000)

    assert qlora3b["pesos"] < lora3b["pesos"]
    assert qlora3b["total"] < qlora8b["total"] < 24.0        # go/no-go da L4 24 GB
    assert lora3b["total"] == pytest.approx(
        sum(lora3b[k] for k in ("pesos", "lora_otimizador", "ativacoes", "overhead")), abs=0.02)


def test_missing_deps_depende_do_metodo_e_da_base():
    todas = {"torch": "2.9", "transformers": "5.0", "peft": "0.11",
             "bitsandbytes": None, "colpali_engine": "0.3.10"}

    assert T.missing_deps(T.resolve_config("colqwen2.5", "lora"), todas) == []
    assert T.missing_deps(T.resolve_config("colqwen2.5", "qlora"), todas) == ["bitsandbytes"]
    # ColEmbed carrega por AutoModel(trust_remote_code): nao precisa do colpali-engine
    sem_colpali = {**todas, "colpali_engine": None}
    assert T.missing_deps(T.resolve_config("colembed8b", "lora"), sem_colpali) == []
    # idem a base densa Qwen3-VL (transformers puro), mas QLoRA ainda exige bitsandbytes
    assert T.missing_deps(T.resolve_config("qwen3vl", "lora"), sem_colpali) == []
    assert T.missing_deps(T.resolve_config("qwen3vl", "qlora"), sem_colpali) == ["bitsandbytes"]


# -- carregamento da base + LoRA restrito ---------------------------------------------


@pytest.fixture
def _transformers_duble(monkeypatch):
    """Substitui as classes de `transformers` usadas por `load_base_model` por dubles.

    Nada e baixado: o dry-run/teste so precisa saber QUAL classe foi chamada e com que
    kwargs. `AutoModel` vira uma armadilha de proposito -- carregar o Qwen3-VL por ela
    reinicializa 100% dos pesos em silencio (ver a docstring de
    `benchmark/models/qwen3vl_embed.py`).
    """
    pytest.importorskip("transformers")
    # Tocar um atributo LAZY do `transformers` (`AutoProcessor` & cia.) troca o objeto de
    # modulo em `sys.modules` e levaria junto os patches ja aplicados: resolvemos tudo
    # ANTES e so entao pegamos o modulo definitivo.
    import transformers

    for nome in ("AutoModel", "AutoProcessor", "AutoModelForImageTextToText"):
        getattr(transformers, nome, None)
    transformers = sys.modules["transformers"]
    visto: dict = {}

    class _FullDuble:
        """O wrapper `...ForConditionalGeneration`: o que interessa e o `.model`."""

        def __init__(self):
            self.model = "QWEN3VL_MODEL_INTERNO"

    class _ForCondGenDuble:
        @staticmethod
        def from_pretrained(model_id, **kw):
            visto["model_id"] = model_id
            visto["kwargs"] = kw
            return _FullDuble()

    class _AutoModelProibido:
        @staticmethod
        def from_pretrained(*a, **kw):
            pytest.fail("AutoModel reinicializa os pesos do Qwen3-VL em silencio")

    class _ProcDuble:
        tokenizer = types.SimpleNamespace(padding_side="left")

        @classmethod
        def from_pretrained(cls, model_id, **kw):
            visto["proc_model_id"] = model_id
            visto["proc_kwargs"] = kw
            return cls()

    monkeypatch.setattr(transformers, "Qwen3VLForConditionalGeneration", _ForCondGenDuble,
                        raising=False)
    monkeypatch.setattr(transformers, "AutoModel", _AutoModelProibido)
    monkeypatch.setattr(transformers, "AutoProcessor", _ProcDuble)
    visto["proc_cls"] = _ProcDuble
    return visto


def test_qwen3vl_carrega_forcondgen_e_usa_model(monkeypatch, _transformers_duble):
    """A base densa carrega por `Qwen3VLForConditionalGeneration` e devolve o `.model`.

    `AutoModel` instanciaria o `Qwen3VLModel` base, cujo state_dict NAO casa com o
    checkpoint (prefixo `model.`): o treino rodaria sobre pesos aleatorios sem erro.
    """
    pytest.importorskip("torch")
    cfg = T.resolve_config("qwen3vl", "lora", device="cpu")

    model, proc = T.load_base_model(cfg)

    assert _transformers_duble["model_id"] == "Qwen/Qwen3-VL-Embedding-2B"
    assert model == "QWEN3VL_MODEL_INTERNO"        # o Qwen3VLModel interno, COM pesos
    # `eager` materializa a atencao inteira e estoura OOM numa L4 com uma pagina densa
    assert _transformers_duble["kwargs"]["attn_implementation"] == "sdpa"
    assert "quantization_config" not in _transformers_duble["kwargs"]   # lora bf16
    assert _transformers_duble["proc_model_id"] == cfg.spec.model_id
    assert isinstance(proc, _transformers_duble["proc_cls"])

    # QLoRA: o mesmo caminho, agora com a config NF4 dos outros ramos
    nf4 = object()
    monkeypatch.setattr(T, "quantization_config", lambda cfg: nf4)
    T.load_base_model(T.resolve_config("qwen3vl", "qlora", device="cpu"))

    assert _transformers_duble["kwargs"]["quantization_config"] is nf4


def test_lora_target_so_language_model():
    """O LoRA do `qwen3vl` nao pode tocar a torre de visao (`visual.*`).

    As duas torres tem as MESMAS projecoes (`q_proj`, ..., `down_proj`), entao a lista de
    sufixos do PEFT (match por `key.endswith`) adaptaria a visao junto.
    """
    import re

    from peft.tuners.tuners_utils import check_target_module_exists

    alvo = T.lora_target_modules(T.resolve_config("qwen3vl", "lora"))

    assert isinstance(alvo, str)                   # regex: a lista de sufixos nao serve
    assert re.fullmatch(alvo, "model.language_model.layers.0.self_attn.q_proj")
    assert re.fullmatch(alvo, "language_model.layers.27.mlp.down_proj")
    assert not re.fullmatch(alvo, "model.visual.blocks.0.attn.q_proj")
    assert not re.fullmatch(alvo, "visual.blocks.0.mlp.down_proj")
    assert not re.fullmatch(alvo, "model.language_model.layers.0.self_attn.rotary")

    # o PEFT instalado aplica o MESMO criterio (`re.fullmatch` quando e string)
    peft_cfg = types.SimpleNamespace(target_modules=alvo, exclude_modules=None)
    assert check_target_module_exists(peft_cfg, "model.language_model.layers.0.mlp.up_proj")
    assert not check_target_module_exists(peft_cfg, "model.visual.blocks.5.attn.v_proj")

    # as bases late-interaction continuam na lista simples de sufixos
    assert (T.lora_target_modules(T.resolve_config("colqwen2.5", "lora"))
            == list(T.DEFAULT_TARGET_MODULES))
    assert (T.lora_target_modules(T.resolve_config("colembed8b", "qlora",
                                                   target_modules=("q_proj", "v_proj")))
            == ["q_proj", "v_proj"])


# -- encoder denso + InfoNCE ----------------------------------------------------------


class _ProcessadorDuble:
    """Processor minimo para o caminho denso: chat template -> string, batch -> ids+mascara.

    Fica no formato que `Qwen3VLEmbedRetriever._build_inputs` espera (ele so chama
    `apply_chat_template` e depois o proprio processor), sem baixar pesos nem tokenizer.
    """

    def apply_chat_template(self, conv, **kw):
        return " ".join(parte.get("text", "") for msg in conv for parte in msg["content"])

    def __call__(self, text=None, **kw):
        import torch

        n = len(list(text))
        return {"input_ids": torch.ones((n, 4), dtype=torch.long),
                "attention_mask": torch.ones((n, 4), dtype=torch.long)}


def _modelo_duble(hidden: int = 8, vocab: int = 16):
    """nn.Module treinavel cujo forward devolve `last_hidden_state` [B, L, H] com grad."""
    import torch

    class Duble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, hidden)

        def forward(self, input_ids=None, attention_mask=None, **kw):
            return types.SimpleNamespace(last_hidden_state=self.emb(input_ids))

    return Duble()


def test_dense_encoder_preserva_grad():
    torch = pytest.importorskip("torch")
    enc = T.DenseEncoder(_modelo_duble(hidden=8), _ProcessadorDuble())

    q = enc.encode_queries(["o que o plano propoe?", "qual a meta de saude?"])

    assert tuple(q.shape) == (2, 8)
    assert q.requires_grad is True                      # nada de no_grad/detach/numpy
    assert torch.allclose(q.norm(dim=-1), torch.ones(2), atol=1e-5)   # L2-normalizado

    q.sum().backward()                                  # o gradiente chega ao modelo
    assert enc.model.emb.weight.grad is not None


def test_dense_encoder_encode_items_aceita_imagem_e_texto(tmp_path):
    """O lado do corpus: caminho de imagem (default) e item ja anotado como texto."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from PIL import Image

    pagina = tmp_path / "BR_p00.png"
    Image.new("RGB", (40, 50), "white").save(pagina)
    enc = T.DenseEncoder(_modelo_duble(hidden=8), _ProcessadorDuble())

    d = enc.encode_items([str(pagina), ("text", "trecho do plano")])

    assert tuple(d.shape) == (2, 8)
    assert d.requires_grad is True
    assert torch.allclose(d.norm(dim=-1), torch.ones(2), atol=1e-5)

    with pytest.raises(ValueError, match="modalidade desconhecida"):
        enc.encode_items([("audio", "x.wav")], modality="audio")


def test_dense_infonce_diagonal():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    torch.manual_seed(0)
    q = F.normalize(torch.randn(3, 8), dim=-1)
    # os B positivos PRIMEIRO (identicos as queries) e depois um negativo -- o contrato
    # de `PairBatch.images`, logo o alvo e a diagonal
    d = torch.cat([q.clone(), F.normalize(torch.randn(1, 8), dim=-1)])

    scores = q @ d.T
    assert torch.equal(scores.argmax(dim=1), torch.arange(3))

    perda = T.dense_infonce_loss(q, d, temperature=0.05)

    assert torch.isfinite(perda) and float(perda) > 0
    assert float(perda) == pytest.approx(float(T.contrastive_loss(scores, 0.05)), rel=1e-5)


def test_build_encoder_dispatch_por_kind():
    """`qwen3vl` -> encoder DENSO; as bases late-interaction seguem no MaxSim."""
    pytest.importorskip("torch")
    denso = T.build_encoder(T.resolve_config("qwen3vl", "lora"),
                            _modelo_duble(hidden=8), _ProcessadorDuble())
    assert isinstance(denso, T.DenseEncoder)

    tardio = T.build_encoder(T.resolve_config("colqwen2.5", "lora"), object(), object())
    assert isinstance(tardio, T.LateInteractionEncoder) and tardio.kind == "colqwen2.5"


class _EncoderDenosDuble:
    """Dublê do `DenseEncoder`: registra o que recebeu e devolve vetores normalizados."""

    def __init__(self, dim: int = 4):
        import torch
        import torch.nn.functional as F

        self.dim = dim
        self.vistos: dict = {}
        self._F, self._torch = F, torch

    def _vecs(self, n: int):
        self._torch.manual_seed(n)
        return self._F.normalize(self._torch.randn(n, self.dim), dim=-1)

    def encode_queries(self, texts):
        self.vistos["queries"] = list(texts)
        return self._vecs(len(self.vistos["queries"]))

    def encode_items(self, items):
        self.vistos["items"] = list(items)
        return self._vecs(len(self.vistos["items"]))


class _EncoderTardioDuble:
    """Dublê do `LateInteractionEncoder`: devolve multi-vetores [N, L, D] + mascara."""

    def __init__(self, dim: int = 4):
        import torch

        self.vistos: dict = {}
        self.dim, self._torch = dim, torch

    def queries(self, texts):
        self.vistos["queries"] = list(texts)
        self._torch.manual_seed(1)
        return self._torch.randn(len(self.vistos["queries"]), 3, self.dim), None

    def images(self, paths):
        self.vistos["images"] = list(paths)
        self._torch.manual_seed(2)
        return self._torch.randn(len(self.vistos["images"]), 5, self.dim), None


def test_batch_loss_dispatch_por_kind(tmp_path):
    """O kind escolhe encoder+perda: denso (InfoNCE cosseno) vs late-interaction (MaxSim)."""
    torch = pytest.importorskip("torch")
    pares = T.load_pairs(_pairs_file(tmp_path, n=2))
    batch = T.make_batch(pares, max_neg=1)

    denso = _EncoderDenosDuble()
    perda = T.batch_loss(T.resolve_config("qwen3vl", "lora"), denso, batch)

    assert perda.ndim == 0 and torch.isfinite(perda)
    assert denso.vistos["queries"] == list(batch.queries)
    assert denso.vistos["items"] == list(batch.dense_items)      # itens COM modalidade

    tardio = _EncoderTardioDuble()
    perda2 = T.batch_loss(T.resolve_config("colqwen2.5", "lora"), tardio, batch)

    assert perda2.ndim == 0 and torch.isfinite(perda2)
    assert tardio.vistos["queries"] == list(batch.queries)
    assert tardio.vistos["images"] == list(batch.images)         # caminhos de imagem


# -- dry run --------------------------------------------------------------------------


def _deps_ok(monkeypatch):
    monkeypatch.setattr(T, "available_deps", lambda: {
        "torch": "2.9.1", "transformers": "5.0.0", "peft": "0.11.0",
        "bitsandbytes": "0.43.0", "colpali_engine": "0.3.10"})


def test_dry_run_valida_dados_e_estima_sem_nenhuma_lib(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "available_deps", lambda: dict.fromkeys(
        ("torch", "transformers", "peft", "bitsandbytes", "colpali_engine")))
    cfg = T.resolve_config("colqwen2.5", "lora", pairs=str(_pairs_file(tmp_path, n=4)),
                           batch_size=2, max_neg=2)

    rel = T.dry_run(cfg)

    assert rel.n_pairs == 4 and rel.n_batches == 2
    assert rel.batch_exemplo["queries"] == 2 and rel.batch_exemplo["negativos"] >= 1
    assert rel.trainable_estimado > 0 and rel.vram_gb["total"] > 0
    assert set(rel.deps_faltando) == {"torch", "transformers", "peft", "colpali_engine"}
    assert any("seria" in a and "vidore/colqwen2.5-v0.2" in a for a in rel.avisos)
    assert rel.trainable_real is None                 # nada foi carregado


def test_dry_run_com_deps_presentes_nao_reclama(tmp_path, monkeypatch):
    _deps_ok(monkeypatch)
    cfg = T.resolve_config("colembed8b", "qlora", pairs=str(_pairs_file(tmp_path, n=3)))

    rel = T.dry_run(cfg)

    assert rel.deps_faltando == [] and rel.avisos == []
    assert rel.model_id == "nvidia/nemotron-colembed-vl-8b-v2"


def test_dry_run_avisa_imagem_ausente(tmp_path, monkeypatch):
    _deps_ok(monkeypatch)
    arq = _pairs_file(tmp_path, n=2)
    Path(json.loads(arq.read_text(encoding="utf-8").splitlines()[0])["pos_image"]).unlink()
    cfg = T.resolve_config("colqwen2.5", "lora", pairs=str(arq))

    rel = T.dry_run(cfg)

    assert rel.imagens_faltando and any("imagens ausentes" in a for a in rel.avisos)


def test_dry_run_load_model_embrulha_o_lora_e_reporta_o_real(tmp_path, monkeypatch):
    _deps_ok(monkeypatch)
    chamadas = []

    class FakeModel:
        def parameters(self):
            return iter(())

    monkeypatch.setattr(T, "load_base_model", lambda cfg: (chamadas.append("load") or
                                                           (FakeModel(), object())))
    monkeypatch.setattr(T, "wrap_lora", lambda m, cfg: chamadas.append("lora") or m)
    monkeypatch.setattr(T, "trainable_parameters", lambda m: (59_000_000, 3_000_000_000))
    cfg = T.resolve_config("colqwen2.5", "lora", pairs=str(_pairs_file(tmp_path, n=2)))

    rel = T.dry_run(cfg, load_model=True)

    assert chamadas == ["load", "lora"]
    assert (rel.trainable_real, rel.total_real) == (59_000_000, 3_000_000_000)


def test_dry_run_load_model_sem_deps_so_avisa(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "available_deps", lambda: dict.fromkeys(
        ("torch", "transformers", "peft", "bitsandbytes", "colpali_engine")))
    monkeypatch.setattr(T, "load_base_model", lambda cfg: pytest.fail("nao pode carregar"))
    cfg = T.resolve_config("colqwen2.5", "lora", pairs=str(_pairs_file(tmp_path, n=2)))

    rel = T.dry_run(cfg, load_model=True)

    assert any("--load-model ignorado" in a for a in rel.avisos)


def test_format_report_mostra_o_essencial(tmp_path, monkeypatch):
    _deps_ok(monkeypatch)
    cfg = T.resolve_config("colqwen2.5", "lora", pairs=str(_pairs_file(tmp_path, n=2)))

    texto = T.format_report(T.dry_run(cfg))

    assert "DRY RUN" in texto and "SAINDO sem treinar" in texto
    assert "vidore/colqwen2.5-v0.2" in texto and "VRAM estimada" in texto


# -- CLI ------------------------------------------------------------------------------


def test_help_nao_importa_nada_pesado(capsys):
    with pytest.raises(SystemExit) as e:
        T.main(["--help"])
    assert e.value.code == 0
    saida = capsys.readouterr().out
    assert "--dry-run" in saida and "qlora" in saida


def test_main_dry_run_nunca_chama_train(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(T, "train", lambda *a, **kw: pytest.fail("treino disparado!"))
    monkeypatch.setattr(T, "run_benchmark_eval",
                        lambda *a, **kw: pytest.fail("avaliacao disparada!"))

    rc = T.main(["--dry-run", "--pairs", str(_pairs_file(tmp_path, n=4)),
                 "--base", "colembed8b", "--method", "qlora", "--batch-size", "2"])

    assert rc == 0
    saida = capsys.readouterr().out
    assert "DRY RUN" in saida and "nemotron-colembed-vl-8b-v2" in saida


def test_main_dry_run_json(tmp_path, capsys):
    rc = T.main(["--dry-run", "--json", "--pairs", str(_pairs_file(tmp_path, n=3))])

    assert rc == 0
    rel = json.loads(capsys.readouterr().out)
    assert rel["n_pairs"] == 3 and rel["config"]["method"] == "lora"
    assert rel["trainable_estimado"] > 0


def test_main_dry_run_sem_pairs_sai_1(tmp_path):
    assert T.main(["--dry-run", "--pairs", str(tmp_path / "nao_existe.jsonl")]) == 1


def test_main_base_invalida_sai_2(capsys):
    with pytest.raises(SystemExit):              # argparse barra antes, com codigo 2
        T.main(["--dry-run", "--base", "inexistente"])


def test_main_target_modules_vira_tupla(tmp_path, capsys):
    T.main(["--dry-run", "--json", "--pairs", str(_pairs_file(tmp_path, n=2)),
            "--target-modules", "q_proj, v_proj", "--lora-r", "8"])

    rel = json.loads(capsys.readouterr().out)
    assert rel["config"]["target_modules"] == ["q_proj", "v_proj"]
    assert rel["trainable_estimado"] == T.estimate_lora_params(
        T.BASES["colqwen2.5"], 8, ("q_proj", "v_proj"))


# -- hook de avaliacao ----------------------------------------------------------------


def test_run_benchmark_eval_chama_run_vidore_e_restaura_argv(tmp_path, monkeypatch):
    import benchmark

    visto = {}
    fake = types.ModuleType("benchmark.run_vidore")
    fake.main = lambda: visto.setdefault("argv", list(sys.argv))
    monkeypatch.setitem(sys.modules, "benchmark.run_vidore", fake)
    monkeypatch.setattr(sys.modules["benchmark"], "run_vidore", fake, raising=False)
    cfg_yaml = tmp_path / "eval.yaml"
    cfg_yaml.write_text("name: x\n", encoding="utf-8")
    argv_antes = list(sys.argv)

    assert T.run_benchmark_eval(cfg_yaml, limit=2, tag="antes") is True

    assert visto["argv"] == ["run_vidore", "--config", str(cfg_yaml), "--limit", "2"]
    assert sys.argv == argv_antes


def test_run_benchmark_eval_sem_config_so_avisa(tmp_path):
    assert T.run_benchmark_eval(tmp_path / "nao_existe.yaml") is False


def test_run_benchmark_eval_nao_propaga_erro_do_benchmark(tmp_path, monkeypatch):
    import benchmark

    def explode():
        raise RuntimeError("dataset indisponivel")

    fake = types.ModuleType("benchmark.run_vidore")
    fake.main = explode
    monkeypatch.setitem(sys.modules, "benchmark.run_vidore", fake)
    monkeypatch.setattr(sys.modules["benchmark"], "run_vidore", fake, raising=False)
    cfg_yaml = tmp_path / "eval.yaml"
    cfg_yaml.write_text("name: x\n", encoding="utf-8")

    assert T.run_benchmark_eval(cfg_yaml) is False       # avisa, nao derruba o treino


def test_write_after_config_aponta_para_o_adaptador(tmp_path):
    import yaml

    base = tmp_path / "v3_colqwen25.yaml"
    base.write_text("name: v3_colqwen25\nmodel_id: vidore/colqwen2.5-v0.2\n"
                    "model_kind: colqwen2.5\ndataset: vidore/vidore_v3_hr\n",
                    encoding="utf-8")
    adapter = tmp_path / "run"
    adapter.mkdir()
    cfg = T.resolve_config("colqwen2.5", "lora", eval_config=str(base))

    destino = T.write_after_config(cfg, adapter)

    dados = yaml.safe_load(destino.read_text(encoding="utf-8"))
    assert destino == adapter / "eval_after.yaml"
    assert dados["model_id"] == str(adapter)
    assert dados["name"] == "v3_colqwen25_ft_lora"
    assert dados["model_kind"] == "colqwen2.5"          # o resto do config nao muda


def test_write_after_config_sem_base_devolve_none(tmp_path):
    cfg = T.resolve_config("colqwen2.5", "lora", eval_config=str(tmp_path / "nao_existe.yaml"))
    assert T.write_after_config(cfg, tmp_path) is None


# -- hook de avaliacao da base DENSA: ensemble/v4 em vez do ViDoRe ---------------------


def _v4_fixture(tmp_path: Path):
    """corpus/queries/qrels minimos do v4 -- os arquivos que o hook do ensemble le."""
    from ensemble.records import PointerRecord, save_records

    corpus = tmp_path / "records.jsonl"
    save_records([PointerRecord("doc1", "documento", "d", "file://doc1", "t",
                                {"a_repr": "orcamento"})], corpus)
    queries = tmp_path / "queries_v4.txt"
    queries.write_text("q1\torcamento\n", encoding="utf-8")
    qrels = tmp_path / "qrels_v4.tsv"
    qrels.write_text("q1 0 doc1 1\n", encoding="utf-8")
    return corpus, queries, qrels


def _argv_qwen3vl(tmp_path, corpus, queries, qrels, *extra):
    return ["--base", "qwen3vl", "--pairs", str(_pairs_file(tmp_path, n=2)),
            "--eval-corpus", str(corpus), "--eval-queries", str(queries),
            "--eval-qrels", str(qrels), *extra]


def test_eval_hook_qwen3vl_usa_ensemble_nao_vidore(tmp_path, monkeypatch):
    """Base densa: o antes/depois sai do benchmark do ENSEMBLE, nao do ViDoRe V3."""
    from finetune import eval_ensemble

    corpus, queries, qrels = _v4_fixture(tmp_path)
    adapter = tmp_path / "runs" / "qwen3vl-lora"
    adapter.mkdir(parents=True)
    chamadas: list[dict] = []

    monkeypatch.setattr(T, "train", lambda cfg, pairs=None: adapter)
    monkeypatch.setattr(T, "run_benchmark_eval",
                        lambda *a, **kw: pytest.fail("ViDoRe nao avalia a base densa"))
    monkeypatch.setattr(T, "write_after_config",
                        lambda *a, **kw: pytest.fail("YAML do ViDoRe nao entra aqui"))

    def fake_hook(cfg, adapter_dir, records, qs, qr):
        chamadas.append({"base": cfg.base, "adapter": str(adapter_dir),
                         "ids": [r.id for r in records], "queries": qs, "qrels": qr})
        return {"antes": 0.4, "depois": 0.6, "delta": 0.2, "n_queries": 1}

    monkeypatch.setattr(eval_ensemble, "run_ensemble_eval_before_after", fake_hook)

    rc = T.main(_argv_qwen3vl(tmp_path, corpus, queries, qrels))

    assert rc == 0
    assert len(chamadas) == 1
    assert chamadas[0]["base"] == "qwen3vl"
    assert chamadas[0]["adapter"] == str(adapter)
    assert chamadas[0]["ids"] == ["doc1"]              # corpus v4 lido do disco
    assert chamadas[0]["queries"] == {"q1": "orcamento"}
    assert chamadas[0]["qrels"] == {"q1": {"doc1"}}


def test_eval_hook_qwen3vl_desligado_sem_corpus(tmp_path, monkeypatch):
    """Opt-in: sem `--eval-corpus` nao se gasta GPU avaliando o v4."""
    from finetune import eval_ensemble

    adapter = tmp_path / "runs"
    adapter.mkdir()
    monkeypatch.setattr(T, "train", lambda cfg, pairs=None: adapter)
    monkeypatch.setattr(eval_ensemble, "run_ensemble_eval_before_after",
                        lambda *a, **kw: pytest.fail("avaliacao disparada sem opt-in"))

    assert T.main(["--base", "qwen3vl", "--pairs", str(_pairs_file(tmp_path, n=2))]) == 0


def test_eval_hook_qwen3vl_respeita_no_eval(tmp_path, monkeypatch):
    from finetune import eval_ensemble

    corpus, queries, qrels = _v4_fixture(tmp_path)
    adapter = tmp_path / "runs"
    adapter.mkdir()
    monkeypatch.setattr(T, "train", lambda cfg, pairs=None: adapter)
    monkeypatch.setattr(eval_ensemble, "run_ensemble_eval_before_after",
                        lambda *a, **kw: pytest.fail("--no-eval ignorado"))

    assert T.main(_argv_qwen3vl(tmp_path, corpus, queries, qrels, "--no-eval")) == 0


def test_eval_hook_late_interaction_continua_no_vidore(tmp_path, monkeypatch):
    """colqwen2.5/colembed nao mudam: ViDoRe antes e depois, com o YAML derivado."""
    from finetune import eval_ensemble

    base_yaml = tmp_path / "v3.yaml"
    base_yaml.write_text("name: v3\nmodel_id: vidore/colqwen2.5-v0.2\n", encoding="utf-8")
    adapter = tmp_path / "runs"
    adapter.mkdir()
    tags: list[str] = []

    monkeypatch.setattr(T, "train", lambda cfg, pairs=None: adapter)
    monkeypatch.setattr(T, "run_benchmark_eval",
                        lambda config_path, limit=None, tag="": tags.append(tag) or True)
    monkeypatch.setattr(eval_ensemble, "run_ensemble_eval_before_after",
                        lambda *a, **kw: pytest.fail("hook do ensemble no caminho ViDoRe"))

    rc = T.main(["--base", "colqwen2.5", "--pairs", str(_pairs_file(tmp_path, n=2)),
                 "--eval-config", str(base_yaml)])

    assert rc == 0
    assert tags == ["antes", "depois"]
    assert (adapter / "eval_after.yaml").exists()
