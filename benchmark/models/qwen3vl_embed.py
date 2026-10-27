"""Qwen3-VL-Embedding -- retriever DENSO (um vetor por item), licenca Apache-2.0.

Query = texto; documento = imagem da pagina (`Doc.image_path`). Os dois lados saem
L2-normalizados, entao `score` e um produto interno -- `runner.evaluate(...,
multi_vector=False)` indexa com FAISS/numpy.

Checkpoint default: `Qwen/Qwen3-VL-Embedding-2B` (cabe folgado numa L4 24 GB); ha
4B/8B na mesma familia -- troque por `model_id`. Repo de referencia:
github.com/QwenLM/Qwen3-VL-Embedding.

`dim` ativa Matryoshka (MRL): o vetor e truncado para `dim` e renormalizado. `None`
mantem a dimensao nativa do checkpoint.

`_embed_queries` / `_embed_images` sao metodos de instancia de proposito: o teste de
contrato os substitui para exercitar o wrapper sem baixar pesos.

Alem do protocolo `Retriever` (usado pelo benchmark), o modulo expoe o mesmo encode
como API de inferencia fora do benchmark -- e o embedder da fatia GCP:

    from benchmark.models.qwen3vl_embed import embed_images, embed_texts

    vecs = embed_images(["pagina0.png", "pagina1.png"])   # [N, dim] float32, L2=1
    q = embed_texts(["metas de saude"])                   # [1, dim] float32, L2=1
    scores = q @ vecs.T                                   # produto interno = cosseno

`get_embedder` guarda o modelo carregado num cache por (model_id, device, dtype,
batch_size, dim), entao chamadas repetidas em um mesmo processo nao recarregam pesos.

API CONFIRMADA (card huggingface.co/Qwen/Qwen3-VL-Embedding-2B + modeling de referencia
github.com/QwenLM/Qwen3-VL-Embedding, `src/models/qwen3_vl_embedding.py`)
---------------------------------------------------------------------------------------
O `config.json` do checkpoint NAO tem `auto_map` e declara
`architectures: ["Qwen3VLForConditionalGeneration"]`: `trust_remote_code=True` nao
carrega codigo nenhum do repo e `AutoModel` instancia o `Qwen3VLModel` do proprio
transformers -- um VLM comum, SEM metodo de encode. O `Qwen3VLEmbedder` do card e um
script do repo (`scripts/qwen3_vl_embedding.py`), nao um metodo do modelo. Logo o
caminho abaixo e o caminho REAL (nao um fallback):

    1. messages = [system(instrucao), user([{type:image|text, ...}])]
    2. text  = processor.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=False)
    3. imgs  = qwen_vl_utils.process_vision_info(messages)[0]   # smart-resize, patch 16
    4. inputs = processor(text=[...], images=imgs, padding=True, truncation=True,
                          max_length=8192, do_resize=False, return_tensors="pt")
    5. emb   = last_hidden_state[ultimo token NAO-pad]          # 1_Pooling: "lasttoken"
    6. emb   = emb / ||emb||                                    # modules.json: Normalize

Passar so `pixel_values` (`processor(images=...)` cru) estoura
`ValueError: You must specify exactly one of input_ids or inputs_embeds` -- o VLM precisa
dos `input_ids` do chat template, onde moram os placeholders de imagem.

Instrucao: o system prompt default e `DEFAULT_INSTRUCTION` ("Represent the user's
input."), igual ao modeling de referencia. O card diz que instrucoes especificas de tarefa
rendem +1~5%; `instruction=` no construtor troca a dos DOIS lados (uma instrucao por lado
seria a proxima volta, depois de medir).
"""

from __future__ import annotations

import inspect
import unicodedata

import numpy as np

from benchmark.common.types import Doc, Query
from benchmark.models._remote_api import call_batch, first_method

# Amortecedor: nenhuma release do card expoe metodo de encode no modelo (ver docstring),
# mas um repack futuro (ou o wrapper sentence-transformers) pode expor -- se existir, ele
# ganha do caminho generico.
_QUERY_FNS = ("encode_query", "embed_query", "get_query_embeddings", "encode_queries")
_IMAGE_FNS = ("encode_document", "embed_document", "get_doc_embeddings", "encode_images",
              "encode_corpus")

# 2B e o default da inferencia fora do benchmark (fatia GCP, VM L4 24 GB).
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"

# System prompt default do modeling de referencia.
DEFAULT_INSTRUCTION = "Represent the user's input."
# `max_length` do processor no modeling de referencia.
MAX_LENGTH = 8192
# Patch do encoder visual do Qwen3-VL (o Qwen2-VL usava 14): o smart-resize precisa dele
# para gerar um grid que case com o processor rodando com `do_resize=False`.
IMAGE_PATCH_SIZE = 16
# Conteudo quando o item vem vazio (mesma convencao do modeling de referencia).
_EMPTY_TEXT = "NULL"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-9)


def _instruction_text(instruction: str | None) -> str:
    """Instrucao do system prompt, terminada em pontuacao (regra do modeling oficial)."""
    instr = (instruction or DEFAULT_INSTRUCTION).strip()
    if instr and not unicodedata.category(instr[-1]).startswith("P"):
        instr += "."
    return instr


def _messages(instruction: str | None, *, text: str | None = None, image=None) -> list[dict]:
    """Conversa de um item: system(instrucao) + user([imagem e/ou texto])."""
    content: list[dict] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": _EMPTY_TEXT})
    return [
        {"role": "system",
         "content": [{"type": "text", "text": _instruction_text(instruction)}]},
        {"role": "user", "content": content},
    ]


def _field(out, name: str):
    """Campo de um ModelOutput/dataclass/dict de saida (None quando ausente)."""
    if isinstance(out, dict):
        return out.get(name)
    return getattr(out, name, None)


def _vision_inputs(convs: list[list[dict]]):
    """Imagens prontas para o processor via helper oficial -- `None` se ele faltar.

    `qwen_vl_utils.process_vision_info` faz o smart-resize que o card assume (e por isso
    o processor roda com `do_resize=False`). Sem o pacote instalado devolvemos `None` e o
    chamador manda as PIL cruas, deixando o resize por conta do processor.
    """
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        return None
    try:
        params = inspect.signature(process_vision_info).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs = {"image_patch_size": IMAGE_PATCH_SIZE} if "image_patch_size" in params else {}
    vision = process_vision_info(convs, **kwargs)
    images = vision[0] if isinstance(vision, tuple) else vision
    return images or None


class Qwen3VLEmbedRetriever:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda",
                 dtype: str = "bfloat16", batch_size: int = 8, dim: int | None = None,
                 instruction: str | None = None, max_length: int = MAX_LENGTH):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.name = f"qwen3vl:{model_id}"
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.dim = dim
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self.max_length = max_length
        self.torch_dtype = getattr(torch, dtype)
        try:
            model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=self.torch_dtype,
                device_map=device, attn_implementation="flash_attention_2",
            )
        except Exception:  # flash-attn ausente -> fallback obrigatorio
            model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=self.torch_dtype,
                device_map=device, attn_implementation="eager",
            )
        self._model = model.eval()
        # padding a DIREITA: o pooling pega o ultimo token nao-pad por `soma(mask) - 1`,
        # que so vale com essa convencao (a mesma do modeling de referencia).
        self._proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                                   padding_side="right")
        tok = getattr(self._proc, "tokenizer", None)
        if tok is not None:                 # o kwarg acima nem sempre desce ao tokenizer
            tok.padding_side = "right"

    # -- normalizacao da saida ---------------------------------------------------

    @staticmethod
    def _as_numpy(x) -> np.ndarray:
        """Tensor (torch) ou array -> np.float32, sem exigir torch importado."""
        if hasattr(x, "detach"):
            x = x.detach().to("cpu").float().numpy()
        return np.asarray(x, dtype="float32")

    @classmethod
    def _stack(cls, out) -> np.ndarray:
        """Saida de um batch -> [n, dim] float32 (aceita lista de vetores ou matriz)."""
        if isinstance(out, (list, tuple)):
            rows = [np.atleast_2d(cls._as_numpy(t)) for t in out]
            return np.ascontiguousarray(np.vstack(rows), dtype="float32")
        return np.ascontiguousarray(np.atleast_2d(cls._as_numpy(out)), dtype="float32")

    def _postprocess(self, emb: np.ndarray) -> np.ndarray:
        """Trunca para `self.dim` (MRL) e L2-normaliza."""
        emb = np.atleast_2d(np.asarray(emb, dtype="float32"))
        if self.dim is not None:
            emb = emb[:, : self.dim]
        return _l2_normalize(emb)

    # -- embedding ---------------------------------------------------------------

    @staticmethod
    def _pool(hidden, attention_mask=None):
        """Hidden state do ULTIMO token NAO-pad (`1_Pooling/config.json`: "lasttoken").

        Com padding a direita (como o processor e carregado), o indice e
        `soma(mask) - 1`; sem mascara, o ultimo da sequencia.
        """
        if attention_mask is None:
            return hidden[:, -1]
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[list(range(hidden.shape[0])), lengths]

    def _build_inputs(self, convs: list[list[dict]], images: list | None = None):
        """Conversas -> batch do processor COM `input_ids` (e `pixel_values` se houver imagem).

        E aqui que mora o bug historico: sem o chat template o VLM recebe so
        `pixel_values` e estoura `You must specify exactly one of input_ids or
        inputs_embeds`. Os placeholders de imagem entram no texto pelo template.
        """
        texts = [self._proc.apply_chat_template(conv, add_generation_prompt=True,
                                                tokenize=False)
                 for conv in convs]
        kwargs: dict = {"text": texts, "return_tensors": "pt", "padding": True,
                        "truncation": True, "max_length": self.max_length}
        if images:
            prepared = _vision_inputs(convs)
            kwargs["images"] = prepared if prepared is not None else list(images)
            if prepared is not None:
                kwargs["do_resize"] = False   # o smart-resize ja saiu do qwen_vl_utils
        return self._proc(**kwargs)

    def _forward_generic(self, inputs) -> np.ndarray:
        """Forward do VLM + pooling do ultimo token (o caminho real -- ver docstring)."""
        import torch

        device = getattr(self._model, "device", None)
        batch = {k: (v.to(device) if device is not None and hasattr(v, "to") else v)
                 for k, v in dict(inputs).items()}
        with torch.no_grad():
            out = self._model(**batch)
        emb = _field(out, "embeddings")        # repacks que ja devolvem o vetor pronto
        if emb is None:
            hidden = _field(out, "last_hidden_state")
            if hidden is None:
                hidden_states = _field(out, "hidden_states")
                hidden = hidden_states[-1] if hidden_states else out
            mask = _field(out, "attention_mask")
            if mask is None:
                mask = batch.get("attention_mask")
            emb = self._pool(hidden, mask)
        return self._stack(emb)

    def _embed_queries(self, texts: list[str]) -> np.ndarray:
        import torch

        fn = first_method(self._model, _QUERY_FNS)
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            if fn is not None:
                with torch.no_grad():
                    out.append(self._stack(call_batch(fn, batch)))
            else:
                convs = [_messages(self.instruction, text=t) for t in batch]
                out.append(self._forward_generic(self._build_inputs(convs)))
        return np.vstack(out) if out else np.zeros((0, 1), dtype="float32")

    def _embed_images(self, paths: list[str]) -> np.ndarray:
        import torch
        from PIL import Image

        fn = first_method(self._model, _IMAGE_FNS)
        out: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[start:start + self.batch_size]]
            try:
                if fn is not None:
                    with torch.no_grad():
                        out.append(self._stack(call_batch(fn, images)))
                else:
                    convs = [_messages(self.instruction, image=im) for im in images]
                    out.append(self._forward_generic(self._build_inputs(convs, images=images)))
            finally:
                for im in images:
                    im.close()
        return np.vstack(out) if out else np.zeros((0, 1), dtype="float32")

    # -- API de inferencia (fora do benchmark) -----------------------------------

    def embed_images(self, paths: list[str]) -> np.ndarray:
        """Caminhos de imagem -> `[N, dim]` float32 L2-normalizado (um vetor por pagina)."""
        return self._postprocess(self._embed_images(list(paths)))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Textos -> `[N, dim]` float32 L2-normalizado (mesmo espaco das imagens)."""
        return self._postprocess(self._embed_queries(list(texts)))

    # -- Retriever protocol ------------------------------------------------------

    def encode_queries(self, queries: list[Query]) -> np.ndarray:
        return self.embed_texts([q.text or "" for q in queries])

    def encode_corpus(self, docs: list[Doc]) -> np.ndarray:
        return self.embed_images([d.image_path or "" for d in docs])

    def score(self, q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
        return q_emb @ d_emb.T


# -- API de modulo: embedding reutilizavel fora do benchmark ----------------------------

_EMBEDDERS: dict[tuple, Qwen3VLEmbedRetriever] = {}


def get_embedder(model_id: str = DEFAULT_MODEL_ID, device: str = "cuda",
                 dtype: str = "bfloat16", batch_size: int = 8,
                 dim: int | None = None) -> Qwen3VLEmbedRetriever:
    """Embedder carregado (cache por configuracao) -- evita recarregar pesos no processo."""
    key = (model_id, device, dtype, batch_size, dim)
    emb = _EMBEDDERS.get(key)
    if emb is None:
        emb = _EMBEDDERS[key] = Qwen3VLEmbedRetriever(
            model_id, device=device, dtype=dtype, batch_size=batch_size, dim=dim)
    return emb


def embed_images(paths: list[str], *, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda",
                 dtype: str = "bfloat16", batch_size: int = 8, dim: int | None = None,
                 embedder: Qwen3VLEmbedRetriever | None = None) -> np.ndarray:
    """Embute imagens de pagina -> `np.ndarray [N, dim]` float32 L2-normalizado.

    `embedder` injeta um wrapper ja carregado (ou um duble, nos testes); sem ele o
    modelo vem de `get_embedder(...)`.
    """
    emb = embedder or get_embedder(model_id=model_id, device=device, dtype=dtype,
                                   batch_size=batch_size, dim=dim)
    return emb.embed_images(list(paths))


def embed_texts(texts: list[str], *, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda",
                dtype: str = "bfloat16", batch_size: int = 8, dim: int | None = None,
                embedder: Qwen3VLEmbedRetriever | None = None) -> np.ndarray:
    """Embute textos (consultas) -> `np.ndarray [N, dim]` float32 L2-normalizado."""
    emb = embedder or get_embedder(model_id=model_id, device=device, dtype=dtype,
                                   batch_size=batch_size, dim=dim)
    return emb.embed_texts(list(texts))
