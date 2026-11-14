"""Runner do gerador de consultas por LLM local: liga um chat_fn e chama `make_pairs`.

`finetune/make_pairs.py` monta o prompt e limpa a resposta, mas nao carrega modelo nenhum:
a fronteira com o LLM e um `chat_fn(prompt) -> str` injetado. Este modulo e o unico lugar
do subsistema [2] que toca `transformers` -- e so dentro de `build_chat_fn`, para que
importar o modulo (nos testes, por exemplo) nao exija a lib nem baixe pesos.

    python -m finetune.gen_queries_llm \
        --records data/govexe/ensemble_orcamento_v4/records.jsonl \
        --out finetune/data/pairs_v4.jsonl --k 3

Ruling da Semana 10: o gerador e o **Qwen2.5-7B-Instruct** (texto, pt-BR), rodando sobre o
`descriptor` do record -- que `contexts_from_records` ja poe em `ctx.text` tambem para os
itens de imagem/documento/video (OCR, legenda, transcricao). Isso cobre TODOS os eixos do
v4 com um modelo de texto; usar um VLM multimodal (gerar a consulta olhando a pagina em
vez do OCR) fica como trabalho futuro -- a fronteira `chat_fn` nao muda, so o runner.

A geracao e deterministica (`do_sample=False`): rodar duas vezes o mesmo corpus devolve
as mesmas consultas, e `make_pairs` ainda deduplica por `pair_id`.
"""

from __future__ import annotations

import argparse
import logging

from finetune.make_pairs import (
    DEFAULT_K,
    DEFAULT_NEG,
    DEFAULT_OUT,
    DEFAULT_RECORDS,
    SEED,
    ChatFn,
    _resolve_out,
    make_pairs,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 256

# O prompt de tarefa inteiro vem do `make_pairs` (mensagem do usuario); o system so fixa
# a lingua e o formato, que e o que o parser (`parse_query_response`) espera.
SYSTEM_PROMPT = ("Voce escreve consultas de busca em portugues do Brasil. "
                 "Responda APENAS com as consultas, uma por linha.")


def build_chat_fn(model_id: str = DEFAULT_MODEL_ID, *, device: str = "cuda",
                  dtype: str = "bfloat16",
                  max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> ChatFn:
    """Carrega o LLM local e devolve a fronteira `chat_fn(prompt) -> str`.

    O import de `transformers` fica AQUI dentro de proposito: o modulo e importavel (e
    testavel, com a lib dublada em `sys.modules`) sem transformers instalado.

    `dtype` vai como STRING para o `from_pretrained` (que aceita "bfloat16"/"float16"/
    "auto"), o que evita importar torch so para resolver o atributo. `generate` ja roda
    sob `no_grad` por decorador do proprio transformers.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("carregando o gerador %s (device=%s, dtype=%s)", model_id, device, dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device).eval()

    def chat_fn(prompt: str) -> str:
        mensagens = [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}]
        texto = tok.apply_chat_template(mensagens, tokenize=False,
                                        add_generation_prompt=True)
        entrada = tok([texto], return_tensors="pt").to(model.device)
        saida = model.generate(**entrada, max_new_tokens=max_new_tokens, do_sample=False)
        # `generate` devolve prompt + continuacao: o decode ve so os tokens NOVOS.
        novos = saida[0][len(entrada["input_ids"][0]):]
        return tok.decode(novos, skip_special_tokens=True).strip()

    return chat_fn


def main():
    ap = argparse.ArgumentParser(
        description="Gera pares de treino com as consultas escritas por um LLM local.")
    ap.add_argument("--records", default=DEFAULT_RECORDS,
                    help="records.jsonl do corpus v4")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="arquivo pairs.jsonl de saida (ou o diretorio que vai conte-lo)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="consultas por item")
    ap.add_argument("--neg", type=int, default=DEFAULT_NEG, help="negativos por par")
    ap.add_argument("--limit", type=int, default=None, help="usa os N primeiros itens")
    ap.add_argument("--force", action="store_true",
                    help="ignora o pairs.jsonl existente e regera tudo")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="checkpoint do gerador")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chat_fn = build_chat_fn(a.model, device=a.device, dtype=a.dtype,
                            max_new_tokens=a.max_new_tokens)
    linhas = make_pairs(records=a.records, out=a.out, k=a.k, neg=a.neg, limit=a.limit,
                        force=a.force, seed=a.seed, generator_name="llm", chat_fn=chat_fn)
    print(f"{len(linhas)} pares em {_resolve_out(a.out)}")


if __name__ == "__main__":
    main()
