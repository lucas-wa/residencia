"""API de consulta HTTP (subsistema [4]): serve o baseline A (Qwen3-VL finetunado).

Sem imports pesados aqui de proposito -- `api.retriever` e `api.index_io` tem de ser
importaveis num processo que nunca vai carregar torch/transformers (os testes, e o
proprio `api.app` ate o evento de startup).
"""
