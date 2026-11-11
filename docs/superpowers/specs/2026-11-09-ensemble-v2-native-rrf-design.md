# Ensemble v2 — especialistas nativos fundidos por RRF (design)

> 2ª onda da **Abordagem B** de [`multimodal.md`](../../../fusao/multimodal.md) §3, continuando
> [`ensemble.md`](../../../fusao/ensemble.md) §3. O v1 (pointer-vector, demonstração cross-modal)
> está pronto e em PR. Este documento especifica o **v2: recuperação nativa por modalidade + métricas**.
> Decidido no brainstorm de 2026-11-09 (Lucas), via perguntas selecionáveis.

## 1. Intenção e porquê

O v1 provou a **fusão cross-modal** barata (um índice de texto por *pointer*). O v2 troca cada
ponteiro pelo **especialista nativo** da modalidade — cada um no seu próprio espaço de embedding —
e **funde por RRF**. A pergunta que o v2 responde é a do **futuro (a)** anotado no v1:
**o ensemble de especialistas nativos recupera melhor que o unificado (A) e que o especialista de
documento isolado (B)?** — medido com **nDCG/MRR/recall por eixo e no conjunto**, sobre um
**mini-gabarito (qrels)** de Orçamento. É o passo de "precisão de verdade" depois da demonstração.

## 2. Goals / Non-goals / Anti-goals

**Goals**
- 5 especialistas nativos, **um por modalidade**, cada um com seu índice, **fundidos por RRF**.
- **Métricas comparáveis**: `ensemble × A × B` na **mesma base** (mesmo corpus + mesmas qrels),
  nDCG@{1,5,10} + MRR + recall@{1,5,10}, **por eixo e no conjunto**.
- **Mini-gabarito** (~15-20 queries de Orçamento + qrels de relevância) para imagem/texto/vídeo —
  os eixos que ainda não têm verdade-anotada (documento e áudio já têm).
- Rodar no **Slurm H100** (embed + eval), espelhando o harness `benchmark/` e a STAGE do v1.

**Non-goals (ficam para ondas futuras)**
- **Rerank two-stage** (cross-encoder): nesta onda é **só RRF**. O design §3 de `multimodal.md`
  prevê rerank; fica para depois.
- **RRF ponderado / tuning de pesos**: baseline é **RRF uniforme** (k=60, pesos iguais). O ajuste
  de pesos por eixo (motivado pelo achado de que o CLAP ruidoso diluiu o topo no A+áudio) é a
  próxima iteração, depois de ver onde cada eixo ajuda ou atrapalha.
- **LLM-judge de geração** (objetivo #1 da S9): trilha separada; aqui o juízo é de **recuperação**
  (qrels), não de geração.
- **Grafo de entidades / DOU**: trilha separada (fase futura).

**Anti-goals (seria falha mesmo "funcionando")**
- Gabarito enviesado pelo próprio sistema: as qrels **não** podem ser geradas pelo mesmo modelo que
  está sendo avaliado (por isso o default é **eu proponho, Lucas revisa** — humano no laço).
- Comparar `ensemble × A × B` sobre corpora/queries diferentes (invalidaria o comparativo). **Mesma
  base** é inegociável.
- Introduzir pgvector/DB na máquina do Lucas, ou rodar eval na GCP. (ver constraints.)

## 3. Restrições

- **Roda no Slurm H100** (`slurm_run.sh`, nova STAGE; ex. `ensemble_v2`); **nunca** na máquina do
  Lucas, **nunca** GCP para os runs de modelo. (A fatia GCP fica reservada a validar infra de
  produção — coleta/armazenamento/pgvector/teardown — não estes runs.)
- **Índices em memória** por especialista (cada espaço nativo: cosine ou MaxSim), **sem pgvector,
  sem DB**. B já é assim (MaxSim em memória); os demais seguem o padrão.
- **Tema = Orçamento**; **mesmo corpus do v1** (~100 itens, ~20/eixo), agora com os **binários
  retidos** para embedding nativo. pt-BR nas queries e descritores.
- **Fontes só oficiais e públicas** (as mesmas do v1): documento = **LOA + relatório da CMO**;
  texto = Agência Brasil/Câmara/Senado; imagem = fotos de agências; áudio = EBC/Radioagência;
  vídeo = YouTube Câmara + TV Senado + **Canal Gov**.
- **Commits** com os trailers de atribuição; **push/PR só com OK do Lucas**.

## 4. Arquitetura decidida

### 4.1 Especialistas nativos (um por modalidade)

| Eixo | Especialista nativo | Checkpoint | Estado | Espaço / matching |
|---|---|---|---|---|
| Documento | **ColQwen2.5** | `vidore/colqwen2.5-v0.2` | **pronto (= arch B)** | multi-vetor, **MaxSim** em memória |
| Texto | **BGE-M3** (denso) | `BAAI/bge-m3` | reusa do v1 | denso, **cosine** em memória |
| Imagem | **SigLIP 2** | `google/siglip2-so400m-patch14-384` *(minha escolha)* | a fazer | denso imagem-texto, **cosine** |
| Áudio | **CLAP** | `laion/larger_clap_general` | base pronta (A+áudio) | denso áudio-texto, **cosine** |
| Vídeo | **WeMM-Embedding** | `WeMM-Embedding-2B` (Tencent) *(minha escolha)* | a fazer | denso multimodal, **cosine**; L2-norm, Matryoshka |

Cada especialista **indexa apenas os itens da sua modalidade** e, para uma **query de texto**,
produz uma **lista ranqueada** dos itens daquele eixo (texto→imagem no SigLIP, texto→áudio no CLAP,
texto→vídeo no WeMM, texto→página no ColQwen2.5, texto→texto no BGE-M3).

> **Observação (não muda a escolha):** o WeMM-Embedding é um embedder *universal* (texto/imagem/
> vídeo/docs) — poderia cobrir também imagem/texto, mas no design de "melhor especialista por
> modalidade" ele entra **só no vídeo**; imagem fica no SigLIP 2 e texto no BGE-M3, como o Lucas
> fixou. WeMM **não suporta áudio** (ok, áudio é CLAP).

### 4.2 Fusão — RRF uniforme

As 5 listas ranqueadas são fundidas por **Reciprocal Rank Fusion** (k=60, pesos iguais) numa única
lista cross-modal. Score RRF de um item = Σ_especialista 1/(k + rank_especialista(item)). Itens
aparecem só na lista do seu próprio eixo (cada eixo ranqueia seus itens), então o RRF intercala as
modalidades pela posição relativa. **Sem rerank, sem pesos** nesta onda.

### 4.3 Baselines A e B — re-rodados na mesma base

Para o comparativo `ensemble × A × B` valer, A e B rodam sobre o **mesmo corpus + mesmas qrels**:

- **B (ColQwen2.5, doc)**: já roda em memória (MaxSim); reaproveita o caminho do `arch B`.
- **A (unificado Qwen3-VL-Embedding)**: no v1/provas anteriores A usava **pgvector** (fatia GCP).
  **Minha chamada:** reimplementar a **busca de A em memória no Slurm** (cosine sobre os embeddings
  Qwen3-VL de todos os itens num só espaço) em vez de subir pgvector/GCP — mantém a regra
  **eval→Slurm** e torna o comparativo reprodutível no mesmo job. O embedding Qwen3-VL já é o de A;
  só o *store* muda (pgvector → matriz numpy em memória), o que não altera o ranking (cosine no mesmo
  espaço).

### 4.4 Métricas

Espelhando o `results/comparacao.csv` existente, por **(sistema, eixo)**:
`recall@1, nDCG@1, recall@5, nDCG@5, recall@10, nDCG@10, MRR, latência`. Sistemas =
`{ensemble, A, B}`. Eixos = `{documento, texto, imagem, audio, video}` + linha **conjunto**
(todos os itens). Por-eixo = restringe as qrels àquele eixo; conjunto = sobre a lista fundida.

### 4.5 Corpus + qrels

- **Corpus (pré-requisito compartilhado com o run real do v1):** construir o corpus real de
  Orçamento com a **captura do v1** (as fontes já estão fixadas), **retendo os binários** (PDF/
  imagem/áudio/vídeo) além do descritor+ponteiro, para o embedding nativo. ~20 itens/eixo.
- **Qrels:** **eu proponho** ~15-20 queries de Orçamento (expandindo as ~8 de
  `queries_orcamento.txt`) + um **gabarito de relevância** (query×item, relevância binária, topical)
  sobre o corpus; **Lucas revisa/corrige**.
  **Correção (builder-check):** os qrels existentes (ptbr_planos = planos TSE; audio_eval = mini-set
  EBC) são de **outros corpora**, não do corpus Orçamento do v2 — então o gabarito é **novo para os
  5 eixos**, não só imagem/texto/vídeo. Isso aumenta o esforço de revisão do Lucas (~20 queries ×
  ~100 itens, relevância binária). A proposta é escrita **blind às saídas dos sistemas** (não
  "pooled" dos rankings) e julga **relevância topical do conteúdo** (imagens eu examino; áudio/vídeo
  julgo pela ASR + metadados); o Lucas revisa com os binários à mão.

### 4.6 Semântica de avaliação (minhas chamadas — resolvem o builder-check)

- **Relevância = topical, binária.** Uma query ("orçamento da saúde") pode ser relevante a **vários
  itens em vários eixos** (a matéria, a foto, o áudio, o vídeo e a página sobre o tema) — é o que
  torna o comparativo cross-modal significativo. Não é um-alvo-por-query.
- **Unidade de recuperação (e granularidade das qrels):** documento = **página** (ColQwen2.5 é
  page-level, como no B), vídeo = **janela** (segmento com `#t`, como no v1), imagem/áudio/texto =
  **arquivo**. "~20 itens/eixo" conta **unidades de recuperação** (páginas/janelas/arquivos), não
  fontes. qrels e métricas são **item-level**; se uma unidade aparece duas vezes numa lista, o RRF e
  as métricas **deduplicam à unidade**.
- **Como cada sistema é pontuado:**
  - **ensemble (RRF dos 5):** por eixo = restringe as qrels àquele eixo; **conjunto** = sobre a
    lista fundida (todos os itens).
  - **B (ColQwen2.5):** só indexa documentos → reportado em **documento** e **conjunto** (onde só
    consegue trazer páginas, evidenciando a estreiteza do especialista); **n/a** nos eixos
    imagem/áudio/texto/vídeo (não os indexa — n/a, não zero).
  - **A (unificado Qwen3-VL):** um só espaço. Por-eixo = **ranking global filtrado àquele eixo**
    (não re-ranqueado dentro do eixo). A cobre **áudio via ASR→texto** e **vídeo via keyframes**
    (Qwen3-VL embute os frames), consistente com a baseline "A" das provas anteriores (ASR venceu
    A+áudio no topo). A+CLAP é um variante à parte, fora deste comparativo.

## 5. O que fica para o builder (plano de implementação)

Decisões de implementação deixadas ao plano/implementador (idiomáticas, dado o harness existente):
- Estrutura de pacote (provável `ensemble/native/` por especialista + um módulo de RRF + um driver
  de eval), espelhando `benchmark/` e o padrão de dublês de `test_doc_specialist.py`.
- Como reter os binários na captura (estender o `build_corpus`/`capture` do v1 para salvar os
  arquivos no bind `/work/ufg/residencia`, host `/raid/user_lucasalves/...`).
- Carregamento tardio de cada modelo (como `_build_embedder`/`_build_retriever`), com dublês nos
  testes (sem GPU, sem download).
- Keyframes do vídeo para o WeMM (quantos/como amostrar) — parâmetro com default sensato.
- Formato exato do driver de eval e da escrita do CSV comparativo.
- STAGE nova no `slurm_run.sh` (ex. `ensemble_v2`), espelhando `govexe`/`ensemble`.

## 6. Fluxo de execução e dependências

1. **Captura real do corpus** — **stage Slurm** (o nó tem internet): roda a captura do v1 retendo
   os binários → `records.jsonl` + arquivos no bind `/work/ufg/residencia`. *(Pré-requisito também
   do run real do v1.)*
2. **Qrels**: proponho queries+gabarito → Lucas revisa. *(Humano no laço; bloqueia a medição, não a
   construção dos índices.)*
3. **Índices nativos**: embeda cada eixo com seu especialista (Slurm H100), em memória.
4. **A e B na mesma base**: A em memória (cosine Qwen3-VL), B (MaxSim ColQwen2.5).
5. **RRF** das 5 listas → ranking do ensemble.
6. **Eval**: nDCG/MRR/recall por eixo e conjunto, para `ensemble/A/B` → CSV comparativo + log.

## 7. Decisões

**Travadas (2026-11-09, Lucas):** 5 especialistas nativos de uma vez; entrega = nativo + métricas
(qrels); corpus = reusa o de Orçamento do v1; RRF **uniforme** primeiro; modelos texto=BGE-M3,
imagem=SigLIP 2, vídeo=WeMM; qrels = eu proponho + Lucas revisa; re-rodar A e B na mesma base; só
RRF nesta onda; documento = LOA + relatório da CMO; vídeo inclui Canal Gov.

**Minhas chamadas (builder — revise):**
- Checkpoints: `google/siglip2-so400m-patch14-384` (imagem), `WeMM-Embedding-2B` (vídeo),
  `BAAI/bge-m3` (texto), `vidore/colqwen2.5-v0.2` (doc), `laion/larger_clap_general` (áudio).
- A reimplementado **em memória no Slurm** (sem pgvector/GCP no eval).
- Métricas espelham `comparacao.csv` (nDCG@{1,5,10}+MRR+recall@{1,5,10}+latência) por eixo e conjunto.
- ~20 queries no gabarito; relevância **binária**.
- RRF k=60.
- **Semântica de avaliação** (§4.6): relevância topical; unidade = página(doc)/janela(vídeo)/
  arquivo(resto); qrels item-level com dedup; B n/a fora de documento; A per-eixo = global filtrado,
  com áudio via ASR→texto e vídeo via keyframes.
- **Gabarito novo para os 5 eixos** (os qrels antigos são de outros corpora — §4.5).

**Resolvidos no handoff (Lucas, 2026-11-09):**
- **Captura = stage Slurm.** O nó de compute do cluster UFG tem internet de saída → a captura
  (scrape + `yt-dlp` + download dos PDFs + Whisper) roda como uma **stage do `slurm_run.sh`**,
  gravando os binários direto no bind `/work/ufg/residencia`. Não precisa de etapa separada de
  download-e-copia.
- **Gabarito = ~15-20 queries nos 5 eixos** (~100 itens, relevância binária). Eu proponho, Lucas
  revisa — métricas mais estáveis, revisão maior aceita.

**Ainda em aberto (resolver na execução, não bloqueiam o design):**
- Handle exato do **Canal Gov** no YouTube (verificar via `agenciagov.ebc.com.br/canal-gov` na captura).
- URLs oficiais exatas da **LOA + relatório da CMO** do ano corrente (portal orçamento Câmara/Senado).
- Tamanho do WeMM (2B default; 4B se couber no H100 com os outros modelos) e amostragem de keyframes.
