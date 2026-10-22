"""Finetuning LoRA/QLoRA de recuperacao visual pt-BR (tarefa C2 do plano da Semana 7).

Entrada: o `pairs.jsonl` produzido por `finetune/make_pairs.py` -- uma linha por exemplo
contrastivo ja montado:

    {"id", "query", "pos_id", "pos_image", "neg_ids", "neg_images", "label", "generator"}

A query (texto) e o anchor, `pos_image` e o unico positivo e `neg_images` sao os negativos
daquele exemplo. Com `--max-neg 0` os negativos do arquivo sao ignorados e o treino usa so
in-batch negatives (os positivos dos OUTROS exemplos do batch).

Bases (todas late-interaction/MaxSim, o mesmo contrato do `benchmark/models/colvision.py`
e `benchmark/models/colembed.py`):

    colqwen2.5  vidore/colqwen2.5-v0.2             ~3B   PRIMARIA
    colembed8b  nvidia/nemotron-colembed-vl-8b-v2  ~8B   opcao (licenca nao comercial)

Metodos:

    lora   pesos da base em bf16         -> H100 (`finetune/slurm_finetune.sh`)
    qlora  pesos da base em NF4 (4-bit)  -> L4 24 GB (`finetune/run_gcp_finetune.sh`)

NADA e treinado por acidente: o caminho seguro e `--dry-run`, que valida config + pares +
batches, estima parametros treinaveis e VRAM, imprime o resumo e SAI. O dry-run roda sem
GPU e tolera a ausencia de torch/colpali-engine/peft/bitsandbytes -- nesse caso valida o
que da (parsing + leitura dos pares + contas analiticas) e avisa o que seria carregado.
Com `--load-model` o dry-run tambem carrega a base de verdade e embrulha o LoRA (ai sim
precisa das libs e do checkpoint em cache), reportando o numero REAL de parametros.

Uso:

    python finetune/train_lora.py --dry-run --pairs finetune/data/pairs.jsonl
    python finetune/train_lora.py --dry-run --base colembed8b --method qlora
    python finetune/train_lora.py --base colqwen2.5 --method lora --epochs 1   # treina

Hook de avaliacao (OPT-IN, antes/depois): `--eval-config benchmark/configs/v3_colqwen25.yaml`
chama `benchmark.run_vidore` por import, antes do treino (baseline) e de novo ao fim, com
o YAML derivado que aponta para o adaptador salvo. As duas linhas caem no mesmo
`results/vidore.csv`. Sem a flag nao se avalia nada -- o ViDoRe V3 inteiro custa GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_PAIRS = "finetune/data/pairs.jsonl"
DEFAULT_OUT = "finetune/runs"
SEED = 42

# Campos obrigatorios de cada linha do pairs.jsonl (contrato de make_pairs.py).
REQUIRED_FIELDS = ("id", "query", "pos_id", "pos_image")

# Tokens visuais por pagina assumidos nas estimativas de VRAM (resolucao dinamica do
# Qwen-VL: uma pagina A4 rasterizada costuma cair nesta ordem de grandeza).
TOKENS_POR_PAGINA = 1280

# Projecoes alvo do LoRA nos backbones Qwen2.5-VL / Qwen3-VL (atencao + MLP).
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj")


# -- bases e metodos ------------------------------------------------------------------


@dataclass(frozen=True)
class BaseSpec:
    """Uma base treinavel + as dimensoes usadas nas ESTIMATIVAS (params/VRAM).

    As dimensoes sao as do backbone de texto publicado no card; servem so para a conta
    analitica do dry-run. O numero real de parametros treinaveis sai do
    `--load-model` (ou do inicio do treino), que le o modelo de verdade.
    """

    key: str
    model_id: str
    kind: str                       # "colqwen2.5" | "colembed" (dispatch do encoder)
    params_b: float                 # tamanho da base, em bilhoes de parametros
    hidden: int
    layers: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    license_note: str = ""
    # Config do benchmark a reavaliar depois do treino (hook --eval-config).
    eval_config: str = ""


BASES: dict[str, BaseSpec] = {
    # Backbone Qwen2.5-VL-3B-Instruct (hidden 2048, 36 camadas, MLP 11008, 16 heads /
    # 2 kv heads). TODO confirmar em runtime: `model.config` do checkpoint carregado.
    "colqwen2.5": BaseSpec(
        key="colqwen2.5", model_id="vidore/colqwen2.5-v0.2", kind="colqwen2.5",
        params_b=3.0, hidden=2048, layers=36, intermediate=11008,
        heads=16, kv_heads=2, head_dim=128,
        license_note="Apache-2.0 (confirmar no card)",
        eval_config="benchmark/configs/v3_colqwen25.yaml",
    ),
    # Nemotron ColEmbed V2 8B (backbone Eagle2/Qwen3-VL 8B: hidden 4096, 36 camadas,
    # MLP 12288, 32 heads / 8 kv heads). TODO confirmar em runtime: idem acima -- o id
    # do 8B e o unico da familia ainda nao conferido no Hub (ver benchmark/models/
    # colembed.py).
    "colembed8b": BaseSpec(
        key="colembed8b", model_id="nvidia/nemotron-colembed-vl-8b-v2", kind="colembed",
        params_b=8.0, hidden=4096, layers=36, intermediate=12288,
        heads=32, kv_heads=8, head_dim=128,
        license_note="NAO comercial (uso academico)",
        eval_config="benchmark/configs/v3_colembed8b.yaml",
    ),
}

METHODS = ("lora", "qlora")


# -- configuracao ---------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Tudo o que define uma rodada (o dry-run imprime isto inteiro)."""

    pairs: str = DEFAULT_PAIRS
    base: str = "colqwen2.5"
    method: str = "lora"
    output_dir: str = ""
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 1e-4
    warmup_ratio: float = 0.03
    max_neg: int = 2                 # negativos do arquivo usados por exemplo (0 = in-batch)
    max_steps: int | None = None
    temperature: float = 0.02
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    gradient_checkpointing: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    limit: int | None = None
    seed: int = SEED
    eval_config: str = ""
    eval_limit: int | None = None

    @property
    def spec(self) -> BaseSpec:
        return BASES[self.base]

    @property
    def out_path(self) -> Path:
        return Path(self.output_dir or f"{DEFAULT_OUT}/{self.base}-{self.method}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_modules"] = list(self.target_modules)
        return d


def resolve_config(base: str, method: str, **kw) -> TrainConfig:
    """Valida base/metodo e devolve a config (erro claro antes de qualquer import caro)."""
    if base not in BASES:
        raise ValueError(f"base desconhecida: {base!r} (use {'/'.join(sorted(BASES))})")
    if method not in METHODS:
        raise ValueError(f"metodo desconhecido: {method!r} (use {'/'.join(METHODS)})")
    cfg = TrainConfig(base=base, method=method, **kw)
    for campo in ("batch_size", "epochs", "grad_accum"):
        if getattr(cfg, campo) < 1:
            raise ValueError(f"{campo} deve ser >= 1 (recebido: {getattr(cfg, campo)})")
    return cfg


# -- dados ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """Uma linha do pairs.jsonl ja validada."""

    id: str
    query: str
    pos_id: str
    pos_image: str
    neg_ids: tuple[str, ...] = ()
    neg_images: tuple[str, ...] = ()
    label: int = 1
    generator: str = ""


def parse_pair(obj: dict, lineno: int = 0) -> Pair:
    """dict -> Pair, com erro apontando a linha (o jsonl vem de um gerador externo)."""
    faltando = [f for f in REQUIRED_FIELDS if not str(obj.get(f) or "").strip()]
    if faltando:
        raise ValueError(f"linha {lineno}: campos ausentes/vazios {faltando} em {obj!r}")
    negs = [str(p) for p in obj.get("neg_images") or []]
    neg_ids = [str(i) for i in obj.get("neg_ids") or []]
    if neg_ids and len(neg_ids) != len(negs):
        raise ValueError(f"linha {lineno}: neg_ids ({len(neg_ids)}) != neg_images ({len(negs)})")
    return Pair(
        id=str(obj["id"]), query=str(obj["query"]).strip(),
        pos_id=str(obj["pos_id"]), pos_image=str(obj["pos_image"]),
        neg_ids=tuple(neg_ids), neg_images=tuple(negs),
        label=int(obj.get("label", 1)), generator=str(obj.get("generator") or ""),
    )


def load_pairs(path: str | Path, limit: int | None = None) -> list[Pair]:
    """Le o pairs.jsonl inteiro (os shards aqui cabem em memoria), deduplicando por `id`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"pairs.jsonl nao encontrado em {p} -- gere com `python finetune/make_pairs.py`")
    vistos: dict[str, Pair] = {}
    for lineno, linha in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            obj = json.loads(linha)
        except json.JSONDecodeError as e:
            raise ValueError(f"linha {lineno}: JSON invalido ({e})") from e
        par = parse_pair(obj, lineno)
        vistos.setdefault(par.id, par)          # dedup: make_pairs ja chaveia por id
        if limit and len(vistos) >= limit:
            break
    if not vistos:
        raise ValueError(f"nenhum par valido em {p}")
    return list(vistos.values())


def missing_images(pairs: list[Pair], max_report: int = 5) -> list[str]:
    """Caminhos de imagem que nao existem no disco (dry-run reporta, nao aborta)."""
    faltando: list[str] = []
    for par in pairs:
        for caminho in (par.pos_image, *par.neg_images):
            if caminho and not Path(caminho).exists():
                faltando.append(caminho)
                if len(faltando) >= max_report:
                    return faltando
    return faltando


@dataclass(frozen=True)
class PairBatch:
    """Um passo contrastivo: B queries x (B positivos + negativos explicitos).

    `images` = os B positivos PRIMEIRO (na ordem das queries) e depois os negativos, logo
    o alvo da cross-entropy e simplesmente `arange(B)` -- ver `contrastive_loss`.
    """

    pairs: tuple[Pair, ...]
    queries: tuple[str, ...]
    images: tuple[str, ...]

    @property
    def n_pos(self) -> int:
        return len(self.pairs)

    @property
    def n_neg(self) -> int:
        return len(self.images) - len(self.pairs)


def make_batch(pairs: list[Pair], max_neg: int = 0) -> PairBatch:
    """Monta um PairBatch, cortando `max_neg` negativos por exemplo e deduplicando.

    Um negativo que ja e positivo de outro exemplo do batch e descartado: ele nao pode
    entrar como "errado" se a linha dele diz que e certo.
    """
    positivos = [p.pos_image for p in pairs]
    ja = set(positivos)
    negativos: list[str] = []
    for par in pairs:
        for caminho in par.neg_images[:max(max_neg, 0)]:
            if caminho not in ja:
                ja.add(caminho)
                negativos.append(caminho)
    return PairBatch(pairs=tuple(pairs), queries=tuple(p.query for p in pairs),
                     images=tuple(positivos + negativos))


def iter_batches(pairs: list[Pair], batch_size: int, max_neg: int = 0,
                 shuffle: bool = False, seed: int = SEED):
    """Gera os PairBatch de uma epoca (o ultimo batch pode ser menor)."""
    if batch_size < 1:
        raise ValueError(f"batch_size deve ser >= 1 (recebido: {batch_size})")
    ordem = list(pairs)
    if shuffle:
        random.Random(seed).shuffle(ordem)
    for start in range(0, len(ordem), batch_size):
        yield make_batch(ordem[start:start + batch_size], max_neg=max_neg)


def count_steps(n_pairs: int, cfg: TrainConfig) -> tuple[int, int]:
    """(batches por epoca, passos de otimizador no total), respeitando `--max-steps`."""
    por_epoca = math.ceil(n_pairs / cfg.batch_size) if n_pairs else 0
    passos = math.ceil(por_epoca * cfg.epochs / max(cfg.grad_accum, 1))
    if cfg.max_steps:
        passos = min(passos, cfg.max_steps)
    return por_epoca, passos


# -- estimativas (rodam sem torch) ----------------------------------------------------


def module_shapes(spec: BaseSpec) -> dict[str, tuple[int, int]]:
    """(fan_in, fan_out) de cada projecao alvo, por camada do backbone."""
    q_out = spec.heads * spec.head_dim
    kv_out = spec.kv_heads * spec.head_dim
    return {
        "q_proj": (spec.hidden, q_out),
        "k_proj": (spec.hidden, kv_out),
        "v_proj": (spec.hidden, kv_out),
        "o_proj": (q_out, spec.hidden),
        "gate_proj": (spec.hidden, spec.intermediate),
        "up_proj": (spec.hidden, spec.intermediate),
        "down_proj": (spec.intermediate, spec.hidden),
    }


def estimate_lora_params(spec: BaseSpec, r: int,
                         target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES) -> int:
    """Parametros treinaveis do LoRA: por modulo alvo, A[r, fan_in] + B[fan_out, r].

    ESTIMATIVA analitica (so o backbone de texto, que e onde o LoRA entra nestes
    checkpoints). O numero real vem de `peft.get_peft_model(...).num_parameters()` --
    use `--load-model` no dry-run para ve-lo.
    """
    shapes = module_shapes(spec)
    desconhecidos = [m for m in target_modules if m not in shapes]
    if desconhecidos:
        log.warning("modulos sem shape conhecido (fora da estimativa): %s", desconhecidos)
    por_camada = sum(r * (shapes[m][0] + shapes[m][1]) for m in target_modules if m in shapes)
    return por_camada * spec.layers


def estimate_vram_gb(cfg: TrainConfig, trainable: int) -> dict[str, float]:
    """Orcamento GROSSEIRO de VRAM, em GB, para a nota go/no-go dos runners.

    Hipoteses (documentadas de proposito -- confirmar na 1a execucao com
    `torch.cuda.max_memory_allocated()`):
      - pesos: 2 bytes/param em bf16; 0.5 byte/param em NF4 + ~6% de constantes de
        quantizacao (double quant);
      - LoRA: pesos bf16 + gradientes bf16 + 2 estados AdamW em fp32 = 12 bytes/param;
      - ativacoes: ~1280 tokens visuais por pagina (resolucao dinamica do Qwen-VL; uma
        pagina densa gera mais) e, COM gradient checkpointing, uma ativacao de
        [tokens, hidden] em bf16 guardada por camada -- sem checkpointing, ~10x isso.
        As queries (dezenas de tokens) e a matriz do MaxSim sao desprezaveis ao lado
        das paginas;
      - CUDA context + fragmentacao do allocator: 1.5 GB fixos.
    """
    gb = 1024 ** 3
    params = cfg.spec.params_b * 1e9
    pesos = params * (0.5 * 1.06 if cfg.method == "qlora" else 2.0) / gb
    lora = trainable * 12 / gb
    imagens_por_passo = cfg.batch_size * (1 + max(cfg.max_neg, 0))
    por_imagem = TOKENS_POR_PAGINA * cfg.spec.hidden * 2 * cfg.spec.layers
    if not cfg.gradient_checkpointing:
        por_imagem *= 10
    ativacoes = imagens_por_passo * por_imagem / gb
    overhead = 1.5
    total = pesos + lora + ativacoes + overhead
    return {"pesos": round(pesos, 2), "lora_otimizador": round(lora, 2),
            "ativacoes": round(ativacoes, 2), "overhead": round(overhead, 2),
            "total": round(total, 2)}


def available_deps() -> dict[str, str | None]:
    """Versao de cada dep pesada, ou None quando ausente (o dry-run so reporta)."""
    import importlib

    versoes: dict[str, str | None] = {}
    for nome in ("torch", "transformers", "peft", "bitsandbytes", "colpali_engine"):
        try:
            mod = importlib.import_module(nome)
            versoes[nome] = str(getattr(mod, "__version__", "?"))
        except Exception:                      # ausente ou quebrado (ABI do torchaudio)
            versoes[nome] = None
    return versoes


def missing_deps(cfg: TrainConfig, deps: dict[str, str | None]) -> list[str]:
    """Deps que FALTAM para esta config treinar de fato."""
    preciso = ["torch", "transformers", "peft"]
    if cfg.spec.kind != "colembed":
        preciso.append("colpali_engine")
    if cfg.method == "qlora":
        preciso.append("bitsandbytes")
    return [d for d in preciso if not deps.get(d)]


# -- modelo (imports pesados, so no caminho de treino/--load-model) -------------------


def quantization_config(cfg: TrainConfig):
    """BitsAndBytesConfig NF4 para o QLoRA; None no LoRA bf16."""
    if cfg.method != "qlora":
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, cfg.dtype),
    )


def load_base_model(cfg: TrainConfig):
    """(model, processor) da base, ja quantizada quando `--method qlora`.

    Reaproveita o dispatch de classes do benchmark (`colvision.resolve_classes`) para as
    bases ColQwen2.5 e o caminho `AutoModel(trust_remote_code=True)` do ColEmbed -- os
    mesmos de `benchmark/models/`.
    """
    import torch

    spec = cfg.spec
    td = getattr(torch, cfg.dtype)
    kw = dict(torch_dtype=td, device_map=cfg.device)
    qcfg = quantization_config(cfg)
    if qcfg is not None:
        kw["quantization_config"] = qcfg

    if spec.kind == "colembed":
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(spec.model_id, trust_remote_code=True,
                                          attn_implementation="eager", **kw)
        proc = AutoProcessor.from_pretrained(spec.model_id, trust_remote_code=True)
    else:
        from benchmark.models.colvision import resolve_classes

        model_cls, proc_cls = resolve_classes(spec.kind)
        model = model_cls.from_pretrained(spec.model_id, attn_implementation="eager", **kw)
        proc = proc_cls.from_pretrained(spec.model_id)
    return model, proc


def wrap_lora(model, cfg: TrainConfig):
    """Embrulha a base no adaptador LoRA (e prepara o k-bit training no QLoRA)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.method == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    peft_model = get_peft_model(model, lora)
    if cfg.gradient_checkpointing and hasattr(peft_model, "gradient_checkpointing_enable"):
        peft_model.gradient_checkpointing_enable()
    return peft_model


def trainable_parameters(model) -> tuple[int, int]:
    """(treinaveis, total) do modelo ja embrulhado."""
    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return treinaveis, total


# -- encoder + perda ------------------------------------------------------------------

_QUERY_FNS = ("forward_queries", "get_query_embeddings", "embed_queries", "encode_queries")
_PASSAGE_FNS = ("forward_passages", "get_doc_embeddings", "embed_passages", "forward_images",
                "encode_passages")
_PROC_QUERY_FNS = ("process_queries", "process_texts")
_PROC_IMAGE_FNS = ("process_images", "process_passages")


class LateInteractionEncoder:
    """Encode COM gradiente (os wrappers de `benchmark/models/` rodam em `no_grad`).

    Devolve `(embeddings [B, L, D], mask [B, L] ou None)`; a mascara vem do processor e
    e usada para descartar padding no MaxSim.
    """

    def __init__(self, model, processor, kind: str):
        self.model = model
        self.proc = processor
        self.kind = kind

    def _device(self):
        return getattr(self.model, "device", None) or next(self.model.parameters()).device

    def _run(self, items: list, model_fns, proc_fns):
        """Mesma ordem de tentativas do `benchmark/models/colembed.py`: a API propria do
        modelo primeiro (custom code do ColEmbed), senao processor + forward (o caminho
        do colpali-engine, onde o modelo nao expoe metodo de encode)."""
        from benchmark.models._remote_api import call_batch, first_method

        fn = first_method(self.model, model_fns)
        if fn is not None:
            return call_batch(fn, items), None
        proc_fn = first_method(self.proc, proc_fns)
        if proc_fn is None:
            raise AttributeError(
                f"{self.kind}: nenhuma API de encode conhecida (modelo: {model_fns}; "
                f"processor: {proc_fns}) -- conferir o card do checkpoint")
        batch = {k: v.to(self._device()) for k, v in dict(proc_fn(items)).items()}
        out = self.model(**batch)
        emb = getattr(out, "embeddings", None)
        if emb is None:
            emb = getattr(out, "last_hidden_state", out)
        return emb, batch.get("attention_mask")

    def queries(self, texts: list[str]):
        return self._run(list(texts), _QUERY_FNS, _PROC_QUERY_FNS)

    def images(self, paths: list[str]):
        from PIL import Image

        imagens = [Image.open(p).convert("RGB") for p in paths]
        try:
            return self._run(imagens, _PASSAGE_FNS, _PROC_IMAGE_FNS)
        finally:
            for im in imagens:
                im.close()


def maxsim_scores(q, d, q_mask=None, d_mask=None):
    """MaxSim (late interaction): [B, Lq, D] x [N, Ld, D] -> [B, N].

    Mesmo criterio do `benchmark.common.index.maxsim_search`, mas em torch e diferenciavel.
    """
    import torch

    sim = torch.einsum("bqh,nkh->bnqk", q.float(), d.float())
    if d_mask is not None:
        sim = sim.masked_fill(~d_mask.bool()[None, :, None, :], float("-inf"))
    melhor = sim.max(dim=-1).values                      # [B, N, Lq]
    if q_mask is not None:
        melhor = melhor * q_mask.bool()[:, None, :].to(melhor.dtype)
    return melhor.sum(dim=-1)


def contrastive_loss(scores, temperature: float = 0.02):
    """InfoNCE com os positivos na diagonal (ver `PairBatch.images`)."""
    import torch
    import torch.nn.functional as F

    alvos = torch.arange(scores.shape[0], device=scores.device)
    return F.cross_entropy(scores / max(temperature, 1e-6), alvos)


# -- treino ---------------------------------------------------------------------------


def warmup_cosine(total_steps: int, warmup_ratio: float = 0.03):
    """Fator de LR para `LambdaLR`: warmup linear e depois decaimento cosseno ate 0.

    Escrito a mao (em vez de OneCycleLR/transformers) porque aqui `total_steps` pode ser
    1-2 num smoke test, e os schedulers prontos tem cantos ruins nesse regime.
    """
    total = max(total_steps, 1)
    aquecimento = max(int(total * max(warmup_ratio, 0.0)), 1 if total > 1 else 0)

    def fator(passo: int) -> float:
        if passo < aquecimento:
            return (passo + 1) / max(aquecimento, 1)
        progresso = (passo - aquecimento) / max(total - aquecimento, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progresso, 1.0)))

    return fator


def train(cfg: TrainConfig, pairs: list[Pair] | None = None) -> Path:
    """Roda o finetuning de verdade e salva o adaptador em `cfg.out_path`.

    NUNCA e chamada no `--dry-run`. Exige torch + peft + (colpali_engine|transformers) e,
    no QLoRA, bitsandbytes.
    """
    faltando = missing_deps(cfg, available_deps())
    if faltando:
        raise RuntimeError(f"deps ausentes para treinar: {faltando} "
                           f"(instale requirements-slurm.txt ou use --dry-run)")
    import torch

    pares = pairs if pairs is not None else load_pairs(cfg.pairs, limit=cfg.limit)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    model, proc = load_base_model(cfg)
    model = wrap_lora(model, cfg)
    model.train()
    enc = LateInteractionEncoder(model, proc, cfg.spec.kind)

    por_epoca, total_passos = count_steps(len(pares), cfg)
    otim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(otim, lr_lambda=warmup_cosine(total_passos,
                                                                           cfg.warmup_ratio))
    treinaveis, total = trainable_parameters(model)
    log.info("treino: %d pares | %d batches/epoca | %d passos | treinaveis=%s/%s",
             len(pares), por_epoca, total_passos, f"{treinaveis:,}", f"{total:,}")

    passo, acumulados = 0, 0
    for epoca in range(cfg.epochs):
        for batch in iter_batches(pares, cfg.batch_size, max_neg=cfg.max_neg,
                                  shuffle=True, seed=cfg.seed + epoca):
            q, q_mask = enc.queries(list(batch.queries))
            d, d_mask = enc.images(list(batch.images))
            perda = contrastive_loss(maxsim_scores(q, d, q_mask, d_mask), cfg.temperature)
            (perda / cfg.grad_accum).backward()
            acumulados += 1
            if acumulados % cfg.grad_accum:
                continue
            otim.step()
            sched.step()
            otim.zero_grad(set_to_none=True)
            passo += 1
            log.info("epoca %d passo %d/%d loss=%.4f", epoca + 1, passo, total_passos,
                     float(perda))
            if cfg.max_steps and passo >= cfg.max_steps:
                break
        if cfg.max_steps and passo >= cfg.max_steps:
            break

    out = cfg.out_path
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    if hasattr(proc, "save_pretrained"):
        proc.save_pretrained(str(out))
    (out / "train_config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("adaptador salvo em %s", out)
    return out


# -- hook de avaliacao (benchmark.run_vidore) -----------------------------------------


def run_benchmark_eval(config_path: str | Path, limit: int | None = None,
                       tag: str = "") -> bool:
    """Chama `benchmark.run_vidore` por import (opcional: so avisa se nao der).

    O runner acrescenta uma linha a `results/vidore.csv` por chamada -- e assim que o
    antes/depois fica comparavel. Devolve True quando a avaliacao rodou.
    """
    try:
        from benchmark import run_vidore
    except Exception as e:                      # benchmark ausente ou deps do loader
        log.warning("hook de avaliacao indisponivel (%s: %s)", type(e).__name__, e)
        return False
    if not Path(config_path).exists():
        log.warning("config de avaliacao nao encontrado: %s", config_path)
        return False

    argv = ["run_vidore", "--config", str(config_path)]
    if limit:
        argv += ["--limit", str(limit)]
    log.info("[eval %s] %s", tag or "", " ".join(argv[1:]))
    anterior, sys.argv = sys.argv, argv
    try:
        run_vidore.main()
        return True
    except Exception as e:                      # avaliacao NAO pode derrubar o treino
        log.warning("[eval %s] falhou (%s: %s)", tag or "", type(e).__name__, e)
        return False
    finally:
        sys.argv = anterior


def write_after_config(cfg: TrainConfig, adapter_dir: Path) -> Path | None:
    """YAML derivado do `--eval-config`, apontando para o adaptador recem-salvo.

    TODO confirmar em runtime: o `from_pretrained` do colpali-engine resolve um diretorio
    de adaptador PEFT pelo `base_model_name_or_path` do `adapter_config.json` (e assim que
    os proprios checkpoints `vidore/colpali-*` carregam). Se a release instalada nao
    fizer isso, faca o merge (`model.merge_and_unload().save_pretrained(...)`) e aponte o
    YAML para o diretorio do merge.
    """
    import yaml

    origem = Path(cfg.eval_config or cfg.spec.eval_config)
    if not origem.exists():
        log.warning("sem config de avaliacao base (%s): nao gerei o YAML do depois", origem)
        return None
    dados = yaml.safe_load(origem.read_text(encoding="utf-8"))
    dados["name"] = f"{dados.get('name', cfg.base)}_ft_{cfg.method}"
    dados["model_id"] = str(adapter_dir)
    destino = adapter_dir / "eval_after.yaml"
    destino.write_text(yaml.safe_dump(dados, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")
    return destino


# -- dry run --------------------------------------------------------------------------


@dataclass
class DryRunReport:
    """O que o `--dry-run` apurou (serializavel: o runner pode logar como JSON)."""

    config: dict
    model_id: str
    n_pairs: int
    n_batches: int
    n_steps: int
    batch_exemplo: dict
    trainable_estimado: int
    vram_gb: dict
    deps: dict
    deps_faltando: list[str]
    imagens_faltando: list[str] = field(default_factory=list)
    trainable_real: int | None = None
    total_real: int | None = None
    avisos: list[str] = field(default_factory=list)


def dry_run(cfg: TrainConfig, load_model: bool = False) -> DryRunReport:
    """Valida config + pares + batches e estima params/VRAM, SEM treinar nada.

    Funciona sem GPU e sem torch/colpali/bitsandbytes: o que depende dessas libs vira
    aviso no relatorio. Com `load_model=True`, tambem carrega a base e embrulha o LoRA
    para reportar o numero REAL de parametros treinaveis (ai as libs sao obrigatorias).
    """
    avisos: list[str] = []
    pares = load_pairs(cfg.pairs, limit=cfg.limit)
    por_epoca, passos = count_steps(len(pares), cfg)
    exemplo = make_batch(pares[:cfg.batch_size], max_neg=cfg.max_neg)
    if cfg.max_neg and not exemplo.n_neg:
        avisos.append("--max-neg > 0, mas os pares nao trazem neg_images: so in-batch negatives")

    faltando_img = missing_images(pares)
    if faltando_img:
        avisos.append(f"imagens ausentes no disco (ex.: {faltando_img[0]}) -- o treino "
                      f"quebraria ao abrir a pagina")

    deps = available_deps()
    faltando_dep = missing_deps(cfg, deps)
    if faltando_dep:
        avisos.append(f"sem {', '.join(faltando_dep)}: validei so config+dados; seria "
                      f"carregado {cfg.spec.model_id} ({cfg.method})")

    estimado = estimate_lora_params(cfg.spec, cfg.lora_r, cfg.target_modules)
    rel = DryRunReport(
        config=cfg.to_dict(), model_id=cfg.spec.model_id, n_pairs=len(pares),
        n_batches=por_epoca, n_steps=passos,
        batch_exemplo={"queries": exemplo.n_pos, "positivos": exemplo.n_pos,
                       "negativos": exemplo.n_neg, "imagens": len(exemplo.images)},
        trainable_estimado=estimado, vram_gb=estimate_vram_gb(cfg, estimado),
        deps=deps, deps_faltando=faltando_dep, imagens_faltando=faltando_img,
        avisos=avisos,
    )

    if load_model:
        if faltando_dep:
            rel.avisos.append("--load-model ignorado: deps ausentes")
        else:
            try:
                model, _ = load_base_model(cfg)
                model = wrap_lora(model, cfg)
                rel.trainable_real, rel.total_real = trainable_parameters(model)
            except Exception as e:
                rel.avisos.append(f"--load-model falhou ({type(e).__name__}: {e})")
    return rel


def format_report(rel: DryRunReport) -> str:
    """Resumo legivel do dry-run (o que vai para o log do Slurm/VM)."""
    c = rel.config
    linhas = [
        "=== DRY RUN (nenhum passo de treino sera executado) ===",
        f"  base            {c['base']}  ->  {rel.model_id}",
        f"  metodo          {c['method']}  (dtype={c['dtype']}, device={c['device']})",
        f"  pares           {rel.n_pairs} de {c['pairs']}",
        f"  batches/epoca   {rel.n_batches} (batch_size={c['batch_size']}, "
        f"max_neg={c['max_neg']})",
        f"  passos totais   {rel.n_steps} (epocas={c['epochs']}, "
        f"grad_accum={c['grad_accum']}, max_steps={c['max_steps']})",
        f"  batch exemplo   {rel.batch_exemplo['queries']} queries x "
        f"{rel.batch_exemplo['imagens']} imagens "
        f"({rel.batch_exemplo['positivos']} pos + {rel.batch_exemplo['negativos']} neg)",
        f"  LoRA            r={c['lora_r']} alpha={c['lora_alpha']} "
        f"dropout={c['lora_dropout']} alvos={','.join(c['target_modules'])}",
        f"  treinaveis      ~{rel.trainable_estimado:,} (estimativa analitica)",
    ]
    if rel.trainable_real is not None:
        linhas.append(f"  treinaveis REAL {rel.trainable_real:,} de {rel.total_real:,} "
                      f"({100 * rel.trainable_real / max(rel.total_real or 1, 1):.2f}%)")
    v = rel.vram_gb
    linhas += [
        f"  VRAM estimada   {v['total']} GB = pesos {v['pesos']} + lora/otim "
        f"{v['lora_otimizador']} + ativacoes {v['ativacoes']} + overhead {v['overhead']}",
        "  deps            " + ", ".join(
            f"{k}={val or 'AUSENTE'}" for k, val in rel.deps.items()),
    ]
    linhas += [f"  AVISO           {a}" for a in rel.avisos]
    linhas.append("=== fim do dry run: SAINDO sem treinar ===")
    return "\n".join(linhas)


# -- CLI ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="train_lora.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=DEFAULT_PAIRS,
                    help=f"pairs.jsonl de finetune/make_pairs.py (default: {DEFAULT_PAIRS})")
    ap.add_argument("--base", choices=sorted(BASES), default="colqwen2.5",
                    help="base a treinar (colqwen2.5 = primaria, ~3B)")
    ap.add_argument("--method", choices=METHODS, default="lora",
                    help="lora (bf16, H100) ou qlora (NF4 4-bit, L4 24GB)")
    ap.add_argument("--output-dir", default="",
                    help=f"onde salvar o adaptador (default: {DEFAULT_OUT}/<base>-<metodo>)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2, help="queries por passo")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-neg", type=int, default=2,
                    help="negativos do arquivo por exemplo (0 = so in-batch negatives)")
    ap.add_argument("--max-steps", type=int, default=None, help="corta o treino em N passos")
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES),
                    help="projecoes alvo do LoRA, separadas por virgula")
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false",
                    help="desliga o checkpointing (mais rapido, MUITO mais VRAM)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None, help="usa so os N primeiros pares")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eval-config", default="",
                    help="YAML do benchmark: LIGA o hook de avaliacao antes/depois "
                         "(ex.: benchmark/configs/v3_colqwen25.yaml). Sem ele, nao avalia.")
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="--limit repassado ao benchmark.run_vidore")
    ap.add_argument("--no-eval", action="store_true",
                    help="desliga o hook de avaliacao mesmo com --eval-config")
    ap.add_argument("--dry-run", action="store_true",
                    help="valida config/dados/LoRA, imprime o resumo e SAI sem treinar")
    ap.add_argument("--load-model", action="store_true",
                    help="no --dry-run, carrega a base e embrulha o LoRA de verdade "
                         "(exige as libs e o checkpoint; sem isso o dry-run e offline)")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="no --dry-run, imprime o relatorio como JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        cfg = resolve_config(
            a.base, a.method, pairs=a.pairs, output_dir=a.output_dir, epochs=a.epochs,
            batch_size=a.batch_size, grad_accum=a.grad_accum, lr=a.lr,
            warmup_ratio=a.warmup_ratio, max_neg=a.max_neg, max_steps=a.max_steps,
            temperature=a.temperature, lora_r=a.lora_r, lora_alpha=a.lora_alpha,
            lora_dropout=a.lora_dropout,
            target_modules=tuple(m.strip() for m in a.target_modules.split(",") if m.strip()),
            gradient_checkpointing=a.gradient_checkpointing, device=a.device, dtype=a.dtype,
            limit=a.limit, seed=a.seed, eval_config=a.eval_config, eval_limit=a.eval_limit)
    except ValueError as e:
        log.error("%s", e)
        return 2

    if a.dry_run:
        try:
            rel = dry_run(cfg, load_model=a.load_model)
        except (FileNotFoundError, ValueError) as e:
            log.error("%s", e)
            return 1
        print(json.dumps(asdict(rel), ensure_ascii=False, indent=2) if a.as_json
              else format_report(rel))
        return 0

    # O hook e OPT-IN: sem --eval-config nao se gasta GPU rodando o ViDoRe V3 inteiro
    # por acidente antes e depois do treino.
    avalia = bool(cfg.eval_config) and not a.no_eval
    if avalia:
        run_benchmark_eval(cfg.eval_config, limit=cfg.eval_limit, tag="antes")

    adapter = train(cfg)

    if avalia:
        depois = write_after_config(cfg, adapter)
        if depois:
            run_benchmark_eval(depois, limit=cfg.eval_limit, tag="depois")
    print(f"adaptador: {adapter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
