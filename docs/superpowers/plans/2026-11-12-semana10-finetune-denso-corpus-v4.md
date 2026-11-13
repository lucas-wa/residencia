# Semana 10 — Finetune denso do Qwen3-VL + corpus v4 balanceado (código do caminho crítico) — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar o CÓDIGO testável do caminho crítico [1]→[2]: estender o trainer para finetunar o unificado `Qwen3-VL-Embedding-2B` (denso/InfoNCE) em pares de todos os domínios, com gerador de queries por LLM, e o ferramental do corpus v4 balanceado (checagem de piso por eixo + split disjunto). Os runs pagos (captura v4, treino H100, avaliação) são orquestrados DEPOIS, fora deste plano.

**Architecture:** Tudo roda sobre o stack de finetune da S7 (`finetune/train_lora.py`, `make_pairs.py`, runners Slurm/GCP). O caminho denso é NOVO e convive com o caminho late-interaction/MaxSim existente, selecionado por `BaseSpec.kind`. O encoder denso treinável reusa só os helpers de montagem de input de `benchmark/models/qwen3vl_embed.py` (não o `_embed_*`, que é `no_grad`+numpy).

**Tech Stack:** Python 3.11, torch 2.9, transformers, peft, Qwen3-VL. Testes com pytest + dublês (sem GPU/rede).

**Spec:** `docs/superpowers/specs/2026-11-12-semana10-finetune-infra-api-design.md`

## Global Constraints

- **Stack pinado:** torch 2.9 / Qwen3-VL / ColQwen2.5. Sem WeMM / transformers ≥ 5.3.
- **Carregar o Qwen3-VL como `Qwen3VLForConditionalGeneration` e treinar o `.model`** — NUNCA `AutoModel` (reinicializa 100% dos pesos em silêncio, ver `benchmark/models/qwen3vl_embed.py:53-62`).
- **LoRA só no `language_model`** (regex/target restrito) — não adaptar a torre visual (`model.visual.*`).
- **`attn_implementation="sdpa"`** no caminho denso (o `eager` dá OOM na L4 com uma página; ver `qwen3vl_embed.py:216-222`).
- **Encoder denso treina COM gradiente:** sem `no_grad`, sem `.detach()/.cpu().numpy()`; `_pool` do último token não-pad + `F.normalize` em tensores.
- **Pares de todos os domínios:** o positivo pode ser texto (`a_repr`) OU imagem (página); o encoder despacha por modalidade do item.
- **Gerador de queries = LLM local** (injeção de um callable; sem chamada de rede nos testes).
- **Piso N = 8 queries julgadas por eixo** (alvo 10) no corpus v4; split treino/avaliação **disjunto** por query/item.
- **DRY-RUN continua o default** dos runners; nada treina por acidente.
- **Roteamento:** implementação no `implementer`; a thread principal orquestra e revisa.

## Review Focus

- **Qwen3-VL carregado com `AutoModel`** → pesos reinicializados, treino silenciosamente inútil. Task 3 fixa a classe e testa o dispatch.
- **Batch misto texto+imagem** no encoder denso → padding/pooling ou montagem de input errada por modalidade. Task 2/4 testam os dois tipos no mesmo batch.
- **Eixo vazio no v4** (axis sem query julgada) → métrica deve ser `n/a`, não 0; e o balance-check deve ACUSAR o eixo abaixo de N. Task 7 testa axis com 0 e com <N.
- **LLM gerando lista vazia/duplicada** → `make_pairs` não pode escrever par com query vazia nem duplicar. Task 5 testa saída vazia e com duplicatas.
- **Split não-disjunto** (um item/query em treino e avaliação) → vaza avaliação. Task 7 testa a disjunção.

---

### Task 1: `BaseSpec` `qwen3vl` (denso) + dispatch de kind

**Files:**
- Modify: `finetune/train_lora.py` (BASES, `missing_deps`, `BaseSpec.kind` dispatch)
- Test: `tests/test_train_lora.py`

**Interfaces:**
- Produces: `BASES["qwen3vl"]` com `kind="qwen3vl"`, `model_id="Qwen/Qwen3-VL-Embedding-2B"`, dims do config do 2B (hidden/layers/intermediate/heads — marcar "confirmar em runtime"); `resolve_config(base="qwen3vl")` válido.

- [ ] **Step 1: Falha** — `test_base_qwen3vl_resolve`: `resolve_config("qwen3vl", "lora")` devolve cfg com `spec.kind=="qwen3vl"` e `spec.model_id` do Qwen3-VL; `estimate_lora_params`/`estimate_vram_gb` não quebram.
- [ ] **Step 2:** rodar o teste; FALHA (`base desconhecida`).
- [ ] **Step 3:** adicionar `BaseSpec` `qwen3vl` em `BASES`; ajustar `missing_deps` (qwen3vl não precisa `colpali_engine`, igual ao ramo `colembed`).
- [ ] **Step 4:** teste PASSA.
- [ ] **Step 5:** commit.

### Task 2: `DenseEncoder` (forward com gradiente) + InfoNCE denso

**Files:**
- Modify: `finetune/train_lora.py` (nova classe `DenseEncoder`, `dense_infonce_loss`)
- Test: `tests/test_train_lora.py`

**Interfaces:**
- Consumes: helpers de `benchmark/models/qwen3vl_embed.py` (`_build_inputs`, `_pool`, `_cap_pixels`, `_messages`).
- Produces: `DenseEncoder(model, processor).encode_queries(texts)->Tensor[B,D]` e `.encode_items(items)->Tensor[N,D]` (L2-normalizados, `requires_grad` preservado); `dense_infonce_loss(q,d,temperature)->scalar` (cosine `q@d.T`, alvo diagonal).

- [ ] **Step 1: Falha** — `test_dense_encoder_preserva_grad`: com um modelo-dublê cujo forward devolve `last_hidden_state` com `requires_grad=True`, `encode_queries` devolve `[B,D]` normalizado e `.requires_grad is True`; `test_dense_infonce_diagonal`: dados positivos idênticos na diagonal → loss ≈ `log` baixo e `argmax` por linha == `arange(B)`.
- [ ] **Step 2:** rodar; FALHA (classe/fn inexistentes).
- [ ] **Step 3:** implementar `DenseEncoder` (monta input por modalidade, forward SEM `no_grad`, `_pool` + `F.normalize`) e `dense_infonce_loss` (reusa `contrastive_loss` sobre matriz cosine).
- [ ] **Step 4:** testes PASSAM.
- [ ] **Step 5:** commit.

### Task 3: Carregamento correto da base + LoRA só no `language_model`

**Files:**
- Modify: `finetune/train_lora.py` (`load_base_model`, `wrap_lora` dispatch por `kind`)
- Test: `tests/test_train_lora.py`

**Interfaces:**
- Consumes: `BASES["qwen3vl"]` (Task 1), `DenseEncoder` (Task 2).
- Produces: `load_base_model(cfg)` para `kind=="qwen3vl"` usa `Qwen3VLForConditionalGeneration.from_pretrained(..., attn_implementation="sdpa")` e devolve `(model.model, processor)`; `wrap_lora` com `target_modules` restrito a `language_model` quando `kind=="qwen3vl"`.

- [ ] **Step 1: Falha** — `test_qwen3vl_carrega_forcondgen_e_usa_model` (mockando `transformers`): a classe pedida é `Qwen3VLForConditionalGeneration`, NUNCA `AutoModel`; o alvo devolvido é o `.model`. `test_lora_target_so_language_model`: para `kind=="qwen3vl"`, `wrap_lora` não inclui módulos de `visual`.
- [ ] **Step 2:** rodar; FALHA.
- [ ] **Step 3:** implementar o dispatch de classe e o filtro de `target_modules` (usar `attn_implementation="sdpa"`; `task_type` adequado ao `.model` denso).
- [ ] **Step 4:** testes PASSAM.
- [ ] **Step 5:** commit.

### Task 4: `make_pairs` — pares de todos os domínios (texto + imagem)

**Files:**
- Modify: `finetune/make_pairs.py` (`PageContext`→contexto por item com modalidade; positivo texto OU imagem), `finetune/train_lora.py` (`Pair`/`PairBatch` carregam `pos_modality` + `pos_text`)
- Test: `tests/test_make_pairs.py`, `tests/test_train_lora.py`

**Interfaces:**
- Produces: linha do `pairs.jsonl` com `pos_modality` ("texto"|"imagem"), `pos_text` (quando texto) além de `pos_image`; `make_batch` separa itens por modalidade para o `DenseEncoder.encode_items`.

- [ ] **Step 1: Falha** — `test_make_pairs_emite_texto_e_imagem`: fixture de corpus com 1 item texto (`a_repr`) + 1 item imagem (página) → `pairs.jsonl` tem uma linha com `pos_modality=="texto"`/`pos_text` e outra `pos_modality=="imagem"`/`pos_image`. `test_pairbatch_modalidade_mista`: `make_batch` com pares texto+imagem devolve os dois grupos sem misturar caminhos.
- [ ] **Step 2:** rodar; FALHA.
- [ ] **Step 3:** estender a leitura de corpus (itens com texto viram positivo textual), o schema do par e o `make_batch`/`PairBatch`. Manter retrocompat: par só-imagem continua válido (`pos_modality` default "imagem").
- [ ] **Step 4:** testes PASSAM (e os antigos de make_pairs continuam verdes).
- [ ] **Step 5:** commit.

### Task 5: `llm_generate_queries` — backend de LLM local (injetável)

**Files:**
- Modify: `finetune/make_pairs.py` (`llm_generate_queries`, `GENERATORS`)
- Test: `tests/test_make_pairs.py`

**Interfaces:**
- Produces: `llm_generate_queries(ctx, *, chat_fn=None)` — usa um `chat_fn(prompt)->str` injetável (o runner liga ao LLM local no H100); sem `chat_fn` levanta erro claro. Saída: `list[str]` não vazia, sem duplicatas, sem strings vazias.

- [ ] **Step 1: Falha** — `test_llm_generator_usa_chat_fn_injetado`: com um `chat_fn` dublê que devolve 3 linhas (1 duplicada, 1 vazia) → resultado tem 2 queries únicas não vazias. `test_llm_generator_sem_backend_erra`: sem `chat_fn`, `RuntimeError`/`ValueError` claro (não `NotImplementedError` silencioso).
- [ ] **Step 2:** rodar; FALHA.
- [ ] **Step 3:** implementar o parser determinístico da resposta + dedup + filtro de vazios; registrar em `GENERATORS["llm"]`.
- [ ] **Step 4:** testes PASSAM.
- [ ] **Step 5:** commit.

### Task 6: `adapter_path` no retriever + hook de avaliação do ensemble

**Files:**
- Modify: `benchmark/models/qwen3vl_embed.py` (`Qwen3VLEmbedRetriever.__init__` aceita `adapter_path`)
- Modify: `finetune/train_lora.py` (hook de avaliação pós-treino chama o eval do ENSEMBLE/v4, não só ViDoRe)
- Test: `tests/test_qwen3vl_embed_api.py`, `tests/test_train_lora.py`

**Interfaces:**
- Produces: `Qwen3VLEmbedRetriever(model_id, adapter_path=None)` — com `adapter_path`, carrega base + `PeftModel.from_pretrained` + `merge_and_unload` antes do `.eval()`; um hook `run_ensemble_eval(adapter_path, queries, qrels)` que registra nDCG@10 antes/depois no v4.

- [ ] **Step 1: Falha** — `test_retriever_aceita_adapter_path` (mock peft): quando `adapter_path` é dado, `PeftModel.from_pretrained` é chamado e o modelo final é o merge. `test_eval_hook_antes_depois_v4`: o hook produz duas linhas (antes/depois) com a mesma métrica no mesmo conjunto.
- [ ] **Step 2:** rodar; FALHA.
- [ ] **Step 3:** implementar o `adapter_path` e o hook de avaliação do ensemble/v4 (reusa `ensemble/eval`).
- [ ] **Step 4:** testes PASSAM.
- [ ] **Step 5:** commit.

### Task 7: Ferramental do corpus v4 — balance-check + split disjunto

**Files:**
- Create: `ensemble/corpus_v4.py` (`check_balance`, `tag_split`)
- Test: `tests/test_corpus_v4.py`

**Interfaces:**
- Produces: `check_balance(qrels, queries, *, floor=8)->dict[eixo,int]` + `BalanceError` quando algum eixo < floor; `tag_split(queries, *, eval_ids)->dict` que marca treino/avaliação e **falha** se um id aparecer nos dois.

- [ ] **Step 1: Falha** — `test_balance_acusa_eixo_abaixo_do_piso`: qrels com imagem=2 (<8) → `BalanceError` nomeando "imagem". `test_balance_axis_vazio_conta_zero`: eixo sem query → 0 (não KeyError). `test_split_disjunto`: id em treino e avaliação → erro; caso disjunto → dict com os dois conjuntos sem interseção.
- [ ] **Step 2:** rodar; FALHA.
- [ ] **Step 3:** implementar `check_balance` (reusa a contagem "julgadas" de `ensemble/eval/run_eval.py:182-186`) e `tag_split`.
- [ ] **Step 4:** testes PASSAM.
- [ ] **Step 5:** commit.

---

## Fora deste plano (execução orquestrada depois)

- Captura dos itens do **corpus v4** por eixo (GCP/`gcp/slice/`) até o piso N; congelar `records.jsonl` v4.
- Propor queries + qrels v4 (humano no loop: eu proponho, Lucas revisa) e marcar o split.
- Escolher/subir o **LLM pt-BR local no H100** e gerar o `pairs.jsonl` do v4.
- **Treino** no H100/Slurm (`DRY_RUN=0`) com avaliação antes/depois; fallback L4 Spot QLoRA.
- Re-run dos 3 sistemas no v4 e publicação no Notion.
- **Plano B:** infra event-driven [3] + API de consulta [4].
