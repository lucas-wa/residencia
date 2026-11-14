"""Runner do LLM gerador de consultas (`finetune/gen_queries_llm.py`) -- tudo OFFLINE.

`transformers` entra por `sys.modules` como duble: nenhum teste baixa pesos, carrega
modelo ou toca GPU. O que se testa aqui e a FIACAO -- que `build_chat_fn` devolve um
`chat_fn(prompt) -> str` ligado no `generate` deterministico do modelo, e que esse
callable serve o backend `llm` de `finetune/make_pairs.py`.
"""

import sys
import types

import pytest

from finetune.gen_queries_llm import DEFAULT_MODEL_ID, build_chat_fn
from finetune.make_pairs import PageContext, resolve_generator

RESPOSTA = ("1. quais as propostas de saude do plano?\n"
            "2) o plano preve creches em tempo integral?\n")


class _Entrada(dict):
    """Minimo de uma BatchEncoding: dict de tensores + `.to(device)`."""

    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _FakeTokenizer:
    criados: list[tuple[str, dict]] = []
    instancias: list["_FakeTokenizer"] = []

    def __init__(self):
        self.chats: list[tuple[list[dict], dict]] = []
        self.decodificados: list[tuple[list[int], dict]] = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.criados.append((model_id, kwargs))
        tok = cls()
        cls.instancias.append(tok)
        return tok

    def apply_chat_template(self, mensagens, **kwargs):
        self.chats.append((mensagens, kwargs))
        return "<chat>" + mensagens[-1]["content"]

    def __call__(self, textos, **kwargs):
        self.textos = textos
        self.call_kwargs = kwargs
        return _Entrada([[1, 2, 3]])

    def decode(self, tokens, **kwargs):
        self.decodificados.append((list(tokens), kwargs))
        return RESPOSTA + "\n"


class _FakeModel:
    criados: list[tuple[str, dict]] = []
    instancias: list["_FakeModel"] = []
    device = "cuda:0"

    def __init__(self):
        self.chamadas: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.criados.append((model_id, kwargs))
        m = cls()
        cls.instancias.append(m)
        return m

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.chamadas.append(kwargs)
        # prompt (3 tokens) + 2 tokens novos: so os novos voltam pelo decode
        return [[1, 2, 3, 41, 42]]


@pytest.fixture
def fake_transformers(monkeypatch):
    _FakeTokenizer.criados = []
    _FakeTokenizer.instancias = []
    _FakeModel.criados = []
    _FakeModel.instancias = []
    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = _FakeTokenizer
    mod.AutoModelForCausalLM = _FakeModel
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return mod


def test_build_chat_fn_devolve_fronteira_str_para_str(fake_transformers):
    chat_fn = build_chat_fn()

    resposta = chat_fn("gere consultas sobre o trecho X")

    assert isinstance(resposta, str)
    assert resposta == RESPOSTA.strip()               # so o texto gerado, ja limpo
    assert _FakeModel.criados and _FakeModel.criados[0][0] == DEFAULT_MODEL_ID
    assert _FakeTokenizer.criados[0][0] == DEFAULT_MODEL_ID
    # o prompt do make_pairs vai inteiro para o chat template, como mensagem do usuario
    (mensagens, kwargs), = _FakeTokenizer.instancias[0].chats
    assert mensagens[-1]["role"] == "user"
    assert mensagens[-1]["content"] == "gere consultas sobre o trecho X"
    assert kwargs["add_generation_prompt"] is True and kwargs["tokenize"] is False
    # o decode ve SO os tokens novos (o prompt fica fora da resposta)
    (novos, _), = _FakeTokenizer.instancias[0].decodificados
    assert novos == [41, 42]


def test_build_chat_fn_gera_deterministico_a_partir_do_prompt(fake_transformers):
    chat_fn = build_chat_fn(max_new_tokens=32)

    chat_fn("meu prompt")

    (kwargs,) = _FakeModel.instancias[0].chamadas
    assert kwargs["do_sample"] is False               # geracao deterministica
    assert kwargs["max_new_tokens"] == 32
    assert kwargs["input_ids"] == [[1, 2, 3]]


def test_build_chat_fn_pluma_device_e_dtype(fake_transformers):
    build_chat_fn("Qwen/Outro-7B", device="cpu", dtype="float16")

    model_id, kwargs = _FakeModel.criados[0]
    assert model_id == "Qwen/Outro-7B"
    assert kwargs["device_map"] == "cpu"
    assert "float16" in str(kwargs.get("torch_dtype") or kwargs.get("dtype"))


def test_chat_fn_do_runner_alimenta_o_backend_llm(fake_transformers):
    """Integracao leve: o runner liga o gerador `llm` ja testado do make_pairs."""
    gen = resolve_generator("llm", chat_fn=build_chat_fn())
    ctx = PageContext(page_id="documento-001", image_path="p.png", plan_id="pdf", k=3,
                      text="OCR: metas de saude do plano")

    assert gen(ctx) == ["quais as propostas de saude do plano?",
                        "o plano preve creches em tempo integral?"]
