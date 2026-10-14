"""Helpers para falar com checkpoints de *custom code* (`trust_remote_code=True`).

ColEmbed V2 (`colembed.py`) e Qwen3-VL-Embedding (`qwen3vl_embed.py`) trazem o proprio
modeling no repo do Hub, e o nome do metodo de encode muda entre releases do card. Em vez
de fixar um nome (que quebra o job depois de 20 min de download), os wrappers listam os
nomes conhecidos e resolvem na instancia carregada.

# TODO confirmar em runtime: assim que o card for lido no cluster, fixar o nome real no
# topo do wrapper correspondente -- estas tentativas sao um amortecedor, nao a verdade.
"""

from __future__ import annotations

import inspect


def first_method(obj, names: tuple[str, ...]):
    """Primeiro metodo de `names` que existe e e chamavel em `obj` (senao None)."""
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn
    return None


def call_batch(fn, items: list):
    """Chama `fn(items)`, passando `batch_size` so quando a assinatura aceita.

    Algumas APIs (ex.: `forward_queries`/`forward_passages` do ColEmbed) fazem o proprio
    batching interno e pedem `batch_size`. Como os wrappers ja batelam por fora, mandamos
    o lote inteiro de uma vez.
    """
    try:
        has_bs = "batch_size" in inspect.signature(fn).parameters
    except (TypeError, ValueError):   # builtins/C-ext sem assinatura introspectavel
        has_bs = False
    return fn(items, batch_size=max(len(items), 1)) if has_bs else fn(items)
