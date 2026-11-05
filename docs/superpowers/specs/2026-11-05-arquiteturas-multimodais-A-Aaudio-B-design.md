# Comparação empírica de 3 arquiteturas multimodais (A × A+áudio × B) — design

> Spec da Semana 9, objetivos **#3** (ensemble de especialistas) e **#4** (modelo
> distinto só para áudio), estendendo o **#2** (unificado, já provado — ver
> `gcp/slice/` e a página Notion "índice cross-modal com um único modelo"). Companheiro
> de `fusao/multimodal.md` (design A/B/áudio) e `fusao/coleta-gcp.md` (coleta).

## 1. Objetivo e intenção

Subir e **testar empiricamente na GCP** (mesmo rito da run do #2: provisiona → indexa →
avalia → teardown) **três arquiteturas**, medindo **qualidade de recuperação** e
**eficiência**, para decidir o que vai para a camada Gold:

| Arq. | Doc / Imagem / Texto | Áudio | Nº de espaços | Objetivo |
|---|---|---|---|---|
| **A** (unificado) | Qwen3-VL-Embedding-2B | ASR→texto (Qwen3-VL) | 1 | #2 (feito; reindexar) |
| **A+áudio** | Qwen3-VL-Embedding-2B | **CLAP nativo** | 2 | **#4** |
| **B** (ensemble) | texto=BGE-M3 · doc=ColQwen2.5 · imagem=SigLIP 2 | CLAP nativo | N | **#3** |

Decisões já tomadas (brainstorming 2026-11-05): áudio nativo = **CLAP**; escopo =
**funcional + métricas de qualidade**; dataset de áudio = **construir mini-set agora**
(com âncora humana do Lucas). Fusão entre espaços = **RRF** (conforme `multimodal.md §3`).

## 2. Princípio central — RRF funde *rankings*, não vetores

O nó das arquiteturas multi-modelo é que modelos distintos produzem vetores em **espaços
diferentes** (dim e geometria), e cosseno entre eles não tem sentido. A saída é **não**
tentar um espaço comum: cada espaço devolve sua própria **lista ranqueada** top-k, e a
**Reciprocal Rank Fusion** combina por posição:

```
score(d) = Σ_espaços  1 / (c + rank_espaço(d))      # c = 60 (padrão)
```

RRF precisa só de **ordens**, não de vetores comparáveis. Isso tem uma consequência de
projeto importante: backends **heterogêneos** convivem — espaços **densos** (Qwen3-VL,
CLAP, BGE-M3, SigLIP 2) vivem no **pgvector** (HNSW cosseno), e o especialista
**late-interaction** (ColQwen2.5, MaxSim) é ranqueado **fora** do pgvector (harness
`benchmark/`), e as duas listas entram na MESMA fusão. Sem isso, teríamos de forçar tudo a
caber numa coluna HNSW — o que ColQwen2.5 (multi-vetor) não permite.

## 3. Componentes

### 3.1 Índice multi-espaço (pgvector denso)
Hoje há **uma** tabela `govexe_items` com uma coluna `embedding vector(dim)`. Generaliza-se
para **uma tabela por espaço denso**, reusando o DDL já templatizado por `{table}`:
`govexe_qwen3vl`, `govexe_clap`, `govexe_bgem3`, `govexe_siglip2`. Mesmo esquema de colunas
(`item_id` PK, `modality`, `source`, `uri`, `pointer_uri`, `credit`, `content_text`,
`content_sha256`, `model_id`, `embedding`), cada uma com seu `dim` e índice HNSW cosseno
próprios. `item_id` é estável entre tabelas (o mesmo item aparece no espaço do seu modelo).

- **Interface:** manter o `VectorStore`/`PgVectorStore` atuais; parametrizar por `table` e
  `dim` (já quase pronto). `embed_job` ganha `--space <nome>` que escolhe tabela+modelo.
- **ColQwen2.5 NÃO entra no pgvector** (multi-vetor); seu ranking vem do harness `benchmark/`
  (MaxSim), já existente.

### 3.2 Wrappers de modelo
- **Qwen3-VL-Embedding-2B** — existe (`benchmark/models/qwen3vl_embed.py`); reuso.
- **CLAP** (novo) — `embed_audio(paths)` + `embed_texts(queries)` no espaço alinhado do CLAP;
  entra como novo caminho no `embed_job` (modalidade `audio_native`). Modelo e id ⚠ a
  confirmar no model card (licença/pt-BR).
- **BGE-M3** (novo no slice) — embedder de texto denso pt-BR (há wrapper denso análogo em
  `benchmark/models/text_ptbr.py` a reaproveitar/portar para o slice).
- **SigLIP 2** (novo no slice) — imagem densa (`benchmark/models/clip_siglip.py` já cobre a
  família; ligar ao caminho pgvector).
- **ColQwen2.5** — existe em `benchmark/models/colvision.py` (MaxSim); **não** vai ao
  pgvector, só ao harness de ranking.

### 3.3 Camada de consulta + RRF (novo)
`gcp/slice/query.py` ganha um modo **multi-espaço**: dada uma query de texto, embeda-a no
espaço de cada backend ativo da arquitetura, coleta top-k de cada um (pgvector denso e/ou
MaxSim do harness) e funde por **RRF** → ranking unificado. O modo atual (espaço único)
continua para A. Rerank two-stage fica **fora de escopo** por ora.

### 3.4 Mini-set de avaliação de áudio (novo; âncora humana)
`data/govexe/audio_eval/{corpus.jsonl, queries.jsonl, qrels.tsv}` no mesmo formato do set
pt-br (`id`+`audio_path`; `id`+`text`; `qid\tdocid\trel`):
- **Corpus:** ~15 clipes reais (EBC/Radioagência Nacional, Senado Multimídia, TV Câmara via
  `audio_collector`), tópicos **conhecidos e distintos** (saúde, segurança, infraestrutura,
  educação…), crédito preservado, ≤ ~90 s cada.
- **Queries:** ~10-15 consultas de texto pt-BR sobre esses tópicos.
- **Qrels:** relevância **julgada pelo Lucas** (âncora humana) — o passo que destrava a
  métrica. 0/1 binário basta.
- Reuso do set **pt-br doc-visual** existente (92 páginas, 22 queries, 680 qrels) para a
  métrica doc-visual, sem mudança.

### 3.5 Driver de avaliação + comparação (novo, fino)
Reaproveita `benchmark/common/metrics.py` (recall@k, ndcg@k, mrr, aggregate — python puro,
pronto). Um driver recebe listas ranqueadas (do pgvector, do RRF ou do MaxSim) + qrels e
emite um CSV consolidado **uma linha por (arquitetura, tarefa)**: doc-visual e áudio, com
qualidade + eficiência.

### 3.6 Orquestração GCP com teardown garantido (novo)
`gcp/slice/run_slice.sh` idempotente, parametrizado por arquitetura: `provisiona (L4 + Cloud
SQL) → bootstrap → indexa os espaços da arquitetura → avalia → emite métricas → teardown`.
**Teardown via `trap`/`finally`** (roda mesmo com falha na indexação — a fraqueza da run
manual de hoje). Consolida os scripts de scratchpad que funcionaram em arquivos versionados.

## 4. Fluxo de dados (medalhão, já existente)
Bronze (GCS raw por `fonte/dt`, crédito preservado) → Silver (normaliza, resolve entidade,
dedup por hash; **ASR aqui** para o caminho pointer de A) → Gold (embedding pelo modelo do
espaço + upsert por `item_id` na tabela do espaço). CLAP embute o binário de áudio direto
(sem ASR). Cada arquitetura é um conjunto de espaços preenchidos sobre o MESMO Bronze.

## 5. Protocolo de avaliação (recuperação, `multimodal.md §5`)
Controles fixos entre as três: **corpus**, **query-set**, **dim** (1024 onde MRL permite),
**hardware** (L4), **k** (`{1,5,10}`).
- **Tarefa doc-visual** (texto→página): distingue **A × B** (Qwen3-VL denso × ColQwen2.5
  late-interaction). A e A+áudio são idênticos aqui.
- **Tarefa áudio** (texto→clipe): distingue **A (ASR→texto) × A+áudio/B (CLAP nativo)** no
  mini-set novo.
- **Métricas:** nDCG@k, recall@k, MRR; **eficiência:** tempo de indexação, latência de query,
  storage do índice (multi-vetor pesa mais que denso), custo L4-hora.
- **Hipótese (do §5):** unificado vence em cross-modal e custo operacional; ensemble vence em
  precisão por modalidade. **Previsão específica do áudio:** CLAP nativo deve **perder** para
  ASR→texto em fala governamental (CLAP modela evento sonoro/acústica, não semântica de
  transcrição) — resultado que **valida a §4**; é um achado esperado, não uma falha.

## 6. Tratamento de erros / casos de borda
- **Teardown sempre** (trap) — nenhum caminho de falha deixa L4/Cloud SQL de pé.
- **L4 STOCKOUT** — a run do #2 só achou capacidade em `northamerica-northeast2-a`; o script
  tenta uma lista de zonas de fallback.
- **Dim > 2000** — HNSW não indexa; já há guarda acionável em `ensure_schema`.
- **Qrels vazios / doc relevante fora do ranking** — métricas degradam p/ 0.0 (já coberto em
  `metrics.py`).
- **CLAP em pt-BR** — risco de qualidade baixa (ver §5); tratado como achado, com `--asr`
  (pointer) indexado em paralelo para a comparação direta.
- **Licenças** (CLAP, BGE-M3, SigLIP 2, ColQwen2.5, Qwen3-VL) — confirmar no model card antes
  de uso; dados são públicos/governamentais (soberania do dado → open-weight default).

## 7. Testes
- Unitários (local, sem GPU/nuvem): RRF (ordens conhecidas → fusão esperada); driver de
  métricas sobre rankings sintéticos; `embed_job --space` roteia tabela/modelo certos (store
  fake em memória, como `tests/test_gcp_slice.py` já faz); schema por-espaço cria tabela+índice.
- Integração barata: mini-smoke de cada wrapper novo (CLAP/BGE-M3/SigLIP 2) com shapes/normas
  sãs, no padrão do `smoke.py` do #2.
- Métrica de aceite na nuvem: as 3 arquiteturas sobem, indexam, a query cross-modal via RRF
  retorna itens de múltiplas modalidades, o CSV sai preenchido e o teardown deixa
  `gcloud ... list` vazio.

## 8. Decomposição (para o plano)
Grande demais para um plano só; sugiro **sub-projetos** com dependências:
1. **Infra multi-espaço + RRF** — schema por-espaço, `--space`, camada de consulta RRF (base
   de tudo; sem dado novo).
2. **Wrappers CLAP + BGE-M3 + SigLIP 2** no caminho slice; ponte ColQwen2.5↔RRF.
3. **Mini-set de áudio + qrels** — captura + anotação humana (Lucas). **Gate humano**: as
   métricas de áudio dependem deste passo; os demais sub-projetos correm em paralelo.
4. **Driver de avaliação + 3 runs GCP + comparação** — depende de 1-3; produz os números.

## 9. Fora de escopo
Qrels de imagem (imagem fica prova funcional) · rerank two-stage · eixos de geração/LLM-judge
da régua #1 · vídeo · WeMM-9B / hardware A100 · APIs pagas (soberania do dado).

## 10. Questões em aberto
- **Id/licença exatos** de CLAP, BGE-M3, SigLIP 2 a fixar no model card (⚠ `multimodal.md`
  marca os arXiv como "a confirmar").
- **Texto de B:** BGE-M3 (híbrido denso+esparso) × Qwen3-Embedding — default BGE-M3; esparso
  entra no índice? (por ora só o denso, como no `factory.py`).
- **Tamanho/fontes exatas** do mini-set de áudio a combinar com o Lucas na hora da captura.
- **Uma tabela por espaço × uma tabela com coluna `space`** — escolhido "uma por espaço"
  (dims distintos exigem colunas/índices distintos); revisitar se virar muitas tabelas.
