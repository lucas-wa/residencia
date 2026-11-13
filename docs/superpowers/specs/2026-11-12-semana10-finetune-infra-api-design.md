# Semana 10 — Fechamento: benchmark balanceado, finetune do unificado, infra event-driven e API de consulta

> **Design doc** da décima e última semana da residência. Documento de intenção:
> diz o que cada subsistema deve fazer e por quê; os valores exatos ficam para o plano.
>
> **Data:** 2026-11-12 · **Autoria:** proposto por Claude Opus 4.8; **gate de aprovação: Lucas.**

## Intenção e porquê

É a última semana. O objetivo é fechar a residência com quatro entregas que, juntas,
formam **uma pipeline só** — do dado bruto ao serviço consultável:

1. um **benchmark balanceado por modalidade**, que remove a crítica "a fusão é dominada por documento";
2. um **modelo unificado (Qwen3-VL) adaptado ao domínio político pt-BR** por finetuning;
3. **infra event-driven + Spot** que corta o custo de captura/indexação;
4. uma **API de consulta** que expõe o retriever à ferramenta de chat da GovExe.

Por que agora: os três sistemas (ensemble, A unificado, B especialista) e o gabarito já
existem; o stack de finetuning (LoRA / Slurm-H100 / GCP-L4) **já foi construído na S7**.
Falta (a) tornar o benchmark honesto por modalidade, (b) provar que adaptar o modelo ao
domínio move a agulha, e (c) empacotar tudo como serviço barato de rodar.

## Metas / Não-metas / Anti-metas

**Metas**
- **G1.** Gabarito balanceado: queries julgadas por eixo em número suficiente para a fusão
  não degenerar para documento; recomputar macro-média + conjunto + por-eixo dos 3 sistemas.
- **G2.** Finetune do `Qwen3-VL-Embedding-2B` (denso/InfoNCE) em dados políticos pt-BR
  minerados; avaliação **antes/depois** no gabarito balanceado.
- **G3.** Captura/indexação **event-driven** (dispara com dado novo), com GPU só sob
  demanda (Spot) — custo ocioso zero.
- **G4.** **API HTTP de consulta** (sem UI) servindo o retriever, pronta para um chat plugar.

**Não-metas**
- UI de chat (o item [4] é só a API).
- Treino do zero / full finetune (é **LoRA**).
- WeMM / vídeo multimodal real (segue preterido — fora do escopo).

**Anti-metas (fracasso mesmo "funcionando")**
- Finetune que vira overfit do gabarito (treinar e avaliar no mesmo conjunto) → treino e
  avaliação **disjuntos**.
- "Reduzir custo" que deixe VM/GPU ociosa cobrando.
- Balancear o gabarito inflando queries fáceis → tem de ser **julgamento humano no loop**.

## Restrições globais

- **GPU:** H100 via Slurm (cluster UFG, `finetune/slurm_finetune.sh`) como **primário**;
  **GCP L4 Spot QLoRA** (`finetune/run_gcp_finetune.sh`) como alternativa preparada.
  Disciplina ligar/desligar; custo ocioso zero.
- **Gabarito:** humano no loop — eu proponho, Lucas revisa (regra do projeto).
- **Stack pinado:** torch 2.9 / Qwen3-VL / ColQwen2.5 (sem WeMM / transformers ≥ 5.3).
- **Dados:** fontes públicas gov (Câmara/Senado/DOU/RSS/PLOA); uso livre com crédito.
  `item_id = sha1 da URL`.
- **Roteamento:** thread principal orquestra; implementação vai para o `implementer`.
- **Custo:** GCP pago autorizado (projeto `residencia-511517`), efêmero, teardown garantido.

## Os 4 subsistemas

### [1] Benchmark balanceado por modalidade — a fundação

- **Decisão (2026-11-12):** construir um **corpus v4 novo, maior e balanceado por eixo** (não
  expandir o n=83) — base única de treino e avaliação, com **split disjunto** (queries/itens de
  treino ≠ de avaliação) para o finetune [2] não dar overfit.
- **Abordagem:** capturar mais itens por eixo (áudio/vídeo/imagem/texto) até equilibrar com
  documento, congelar o v4, e propor queries + qrels por eixo (humano no loop: eu proponho,
  Lucas revisa). Recomputar os 3 sistemas no v4.
- **Reusa:** toda a infra de eval (`run_eval.py`, `metrics.py`, `rrf.py`, `qrels.py`) e a captura
  (`gcp/slice/`, `ensemble/capture/`).
- **Novo:** corpus v4 balanceado + gabarito v4; o split treino/avaliação (tag por query/item).
- **Aceite:** distribuição de queries por eixo documentada; nenhum eixo abaixo do piso **N**
  (definido no plano); split disjunto comprovado; re-run dos 3 sistemas no v4 publicado no Notion.
- **Risco/fallback:** faltar conteúdo julgável num eixo → capturar mais daquele eixo antes de julgar.

### [2] Finetune do unificado Qwen3-VL (denso) — domínio político pt-BR

- **Abordagem:** estender `finetune/train_lora.py` com um **caminho denso/InfoNCE** para o
  `Qwen/Qwen3-VL-Embedding-2B` (vetor único por item, cosine), ao lado do caminho
  late-interaction/MaxSim existente. Dados minerados de fontes abertas gov via
  `make_pairs.py`. Treino no H100/Slurm (LoRA bf16), com o caminho L4 Spot (QLoRA) pronto.
  Avaliação antes/depois no gabarito balanceado [1].
- **Reusa:** `train_lora.py` (dry-run, estimativa de VRAM, loss contrastiva, runners),
  `make_pairs.py` (formato de pares, idempotência, negativos), `slurm_finetune.sh` +
  `run_gcp_finetune.sh`. Do encoder do Qwen3-VL (`benchmark/models/qwen3vl_embed.py`) reusa os
  **helpers de montagem de input** (`_build_inputs`, `_pool`, `_cap_pixels`) — **não** o `_embed_*`,
  que roda em `no_grad` + numpy (não-diferenciável; o `embedders.py` só embrulha isso).
- **Novo:**
  - `BaseSpec` `qwen3vl` (denso) + um **`DenseEncoder` novo** em `train_lora.py` (forward **com
    gradiente**, sem `no_grad`/numpy; `_pool` do último token não-pad + `F.normalize`) + loss
    InfoNCE sobre **cosine** (não MaxSim), com dispatch por tipo de base.
  - Carregar a base como `Qwen3VLForConditionalGeneration` e aplicar o LoRA ao `.model` (não ao
    wrapper — senão treina sobre pesos reinicializados em silêncio); `target_modules` restrito a
    `language_model` (não adapta a torre visual sem querer); `attn_implementation=sdpa` no QLoRA/L4
    (eager dá OOM); parâmetro `adapter_path` novo em `Qwen3VLEmbedRetriever` para o hook de
    avaliação carregar o adaptador.
  - **Gerador de queries por LLM local no H100** (`llm_generate_queries`, hoje
    `NotImplementedError`) — um LLM pt-BR roda no próprio H100 e gera queries de eleitor sobre
    cada item político (sem custo de API nem envio de dados).
  - **Pares em todos os domínios:** estender `make_pairs.py` (hoje só query↔imagem) para emitir
    pares de **todas as modalidades** que o unificado embedda — query↔texto (`a_repr`),
    query↔página (imagem), etc. — a partir do corpus v4. Mineração em escala (Câmara/Senado/DOU)
    no layout `corpus.jsonl / queries.jsonl / qrels.tsv`.
  - **Split treino/avaliação disjunto dentro do v4** (anti-overfit): treino e conjunto de
    avaliação do hook não compartilham queries/itens.
- **Aceite:** dry-run verde (VRAM GO no H100); treino conclui e salva o adaptador; nDCG@10 do
  Qwen3-VL **depois > antes** no gabarito balanceado (hook de avaliação); split disjunto comprovado.
- **Risco/fallback:** (a) dados insuficientes → gerador template + hard-negatives; (b) H100
  indisponível → L4 Spot QLoRA; (c) ganho nulo → **reportar honestamente** (resultado negativo
  é resultado) + ablação de r/lr/dados.

### [3] Infra event-driven + Spot — custo

- **Abordagem:** trocar a execução manual de VM por fluxo **acionado por evento**: dado novo
  numa fonte (ou agendamento) dispara captura/indexação; a GPU (embedding) só sobe quando há
  trabalho, em **Spot**, e derruba ao fim. Desenho: evento (Pub/Sub·Eventarc ou Cloud
  Scheduler) → trabalho leve em Cloud Run/Functions → job GPU em Spot VM sob demanda
  (reusa `run_slice.sh`).
- **Reusa:** `gcp/slice/run_slice.sh` (captura/eval em Spot, teardown garantido), a disciplina
  de custo-zero.
- **Novo:** o gatilho (chamada 2026-11-12: **Cloud Scheduler em cron de polling** — as fontes são
  RSS/polling, sem eventos nativos; Pub/Sub·Eventarc fica como expansão), a orquestração
  subir-rodar-derrubar, a idempotência da indexação incremental (upsert por `item_id`), e a
  **persistência do índice em GCS** entre ciclos (o `run_slice.sh` destrói a VM ao fim; o índice
  servível por [4] mora em GCS, não na VM).
- **Aceite:** um evento/agendamento dispara a pipeline ponta-a-ponta sem intervenção; nenhum
  recurso ocioso após o ciclo (checado com `instances list`); custo do ciclo medido.
- **Risco/fallback:** Eventarc/Pub-Sub for demais para a semana → Cloud Scheduler simples (cron)
  disparando a Spot.

### [4] API de consulta — GovExe

- **Decisão (2026-11-12):** a API serve **só o A/Qwen3-VL finetunado** (1 modelo), não o ensemble.
  Índice persiste em **GCS/arquivo**; Cloud Run `min-instances=0` (aceita cold start) para custo
  ocioso zero.
- **Abordagem:** expor o Qwen3-VL finetunado como **serviço HTTP de consulta** (sem UI): recebe
  uma query, devolve itens ranqueados (top-k, com `pointer_uri`/`descriptor`), pronto para a
  ferramenta de chat da GovExe consumir como RAG.
- **Reusa:** `UnifiedRetrieverA.search`, os embedders (cache `get_embedder`), os records
  (`pointer_uri` → fonte), o índice persistido em GCS por [3].
- **Novo:** endpoint HTTP (FastAPI/Cloud Run), contrato request/response (`item_id`, `eixo`,
  `score`, `pointer_uri`, `descriptor`), carregamento do índice do GCS no startup, autenticação
  mínima (API key).
- **Aceite:** `POST /consulta {query}` devolve top-k itens com fonte em latência aceitável; deploy
  acessível; contrato documentado para o time da GovExe.
- **Risco/fallback:** servir o índice completo for pesado → servir só o eixo documento/texto
  primeiro; deploy demorar → rodar local + túnel para a demo.

## Sequência e caminho crítico

```
[1] balanceado ──► (gabarito + split) ──► [2] finetune ──► avaliação depois
                                   └──► [4] API serve o modelo resultante
[3] infra event-driven ──► (paralelo; serve a captura de [1] e o deploy de [4])
```

- **Caminho crítico:** **[1] → [2]**. O gabarito balanceado é pré-requisito do finetune — dá o
  conjunto de avaliação disjunto e a prova de ganho.
- **[3] e [4]** correm em paralelo (implementer), compartilhando o desenho Spot/deploy.
- **Ordem recomendada:** [1] primeiro (rápido, destrava [2] e [4]); [2] começa assim que houver
  split; [3]/[4] em paralelo.

## O que fica para o builder (plano / implementer)

- Valores exatos: piso **N** de queries por eixo [1]; `r`/`alpha`/`lr`/`epochs` do LoRA, tamanho do
  dataset minerado e **qual LLM pt-BR** no H100 [2]; **janela do cron** de polling [3];
  latência-alvo e forma final do contrato do endpoint [4].
- Quais queries/itens do v4 ficam no split de **avaliação** (fora do treino) [2].
