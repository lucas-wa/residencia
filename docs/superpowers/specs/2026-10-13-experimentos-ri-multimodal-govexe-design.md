# Design — Experimentos de RI Multimodal e Fusão de Dados (GovExe), Semana 6

> Spec da disciplina **Residência em IA (BIA/UFG)**, tema *"Fusão de Big Data e
> Recuperação de Informação Multimodal"*. Data: 2026-10-13. Autor: Lucas.
> Fonte de contexto: `README.md` (semanas 1–5) + Notion "Residência" + repositório
> GovExe (`D:\projects\govexe`). Hardware-alvo: **GPU 4090 (24 GB), on-premise**.

## 1. Objetivo e entregáveis da semana

Quatro entregáveis, encaixados num **pipeline único** (Abordagem A, aprovada):

1. **Arquitetura de fusão** de dados não estruturados de setores políticos, como parte
   do GovExe (design doc + diagrama). *Projeto*, não implementação de produção.
2. **Pipeline de benchmark** (código + requirements) de modelos de embeddings
   multimodais, rodável, validado num benchmark estabelecido.
3. **Adaptação pt-BR**: benchmark(s) de embeddings multimodais em **português**,
   construídos a partir de dados do GovExe. Primeiro caso desta semana: RI de
   **documento visual** sobre os PDFs de planos de governo. Outras modalidades
   (imagem↔texto, áudio, vídeo — inclusive YouTube) ficam como extensão futura.
4. **Levantamento de fontes recentes** (2024–2026) embasando cada decisão técnica,
   respeitando as bases já estabelecidas no README.

### Princípio de desenho

O valor do multimodal no GovExe está onde o **FTS do Postgres (que o GovExe já usa)
não alcança**: recuperação sobre **documentos visualmente ricos** (PDFs de planos,
páginas do DOU com layout). Por isso o centro de gravidade é **RI de documento visual**
(linha ColPali/ColQwen2), com um *slice* de imagem↔texto (CLIP vs SigLIP2) para largura.

**Papel do GovExe nesta semana (esclarecimento do aluno):** as trilhas de **RI** e
**Big Data** seguem **separadas** por ora — não há integração num sistema único agora.
O GovExe entra em apenas dois pontos: (a) como estudo de caso para o **plano de
arquitetura de fusão** (visão de *como* as modalidades se fundiriam), e (b) como fonte
dos dados para os **benchmarks multimodais em português**. A arquitetura é um **projeto**
(design), não um sistema implementado nesta semana.

**Escopo multimodal da visão (RI multimodal ≠ só texto/documento):** o plano futuro
abrange **captura e recuperação de várias modalidades** — imagens, áudios, documentos e
**vídeos (inclusive do YouTube: sessões, pronunciamentos, comissões)** — como modalidades
de **primeira classe**. A **transcrição (ASR, fala→texto)** é um **auxílio para casos
específicos**, não a redução padrão de áudio/vídeo a texto. A validação **desta semana**
cobre apenas a **primeira fatia** (documento visual + um slice imagem↔texto); as demais
modalidades entram na arquitetura como projeto (§4) e em semanas futuras.

## 2. Escopo e não-escopo

**No escopo (Semana 6):**
- Documento de arquitetura de fusão + diagrama.
- Pipeline de benchmark reutilizável (loaders, índice, métricas, wrappers de modelo).
- Rodada de validação em **ViDoRe v2** (nDCG@5).
- Construção + rodada do **mini-benchmark pt-BR GovExe** (planos).
- Slice imagem↔texto CLIP vs SigLIP2 (barato, largura MIEB).
- Lista de fontes recentes verificadas (ABNT + data de acesso).

**Fora de escopo (explicitamente):**
- M-BEIR *global* (pool 5,6M / ~170 GB) — inviável como 1º passo no 4090.
- **Embedders MLLM (GME-Qwen2-VL, VLM2Vec-V2)** e modelos > ~8B (MM-Embed, E5-V) —
  **fora desta semana** (decisão do aluno); ficam como ampliação futura. Continuam na
  §7 só como *literatura*.
- **Integrar RI + Big Data num sistema único.** As trilhas seguem separadas nesta
  semana; o GovExe entra só para (a) o plano de arquitetura e (b) o benchmark pt-BR.
- Índice vetorial em **produção** no GovExe (pgvector/RAG). O GovExe **não tem** infra
  de embeddings hoje; aqui é *greenfield*, mas a implementação de produção não é meta
  desta semana — apenas o **projeto** da arquitetura.
- Treino/fine-tuning de modelos. Só inferência/avaliação zero-shot.
- Edição do `README.md` na seção de experimentos (fica para quando os resultados saírem).

## 3. Restrições de hardware (4090, 24 GB)

- Classe de modelos padrão: **≤ 2–3B em bf16/fp16**.
  - ColQwen2-2B ≈ 6 GB · ColPali-3B ≈ 7 GB · CLIP/SigLIP2 < 2 GB ·
    encoders de texto pt-BR < 1 GB. (GME/VLM2Vec — fora desta semana.)
- *Late interaction* (ColPali/ColQwen2) gera **multi-vetor por página** → disco e RAM
  crescem com nº de páginas; dimensionar o índice (ver §6).
- Sem download de M-BEIR global. ViDoRe v2 e planos GovExe são pequenos (poucos GB).
- Projetar tudo para **1 GPU**; `batch_size` configurável; `dtype=bfloat16`.

## 4. Entregável 1 — Arquitetura de fusão (GovExe)

### 4.1 Modalidades não estruturadas no GovExe (fontes políticas)

| Modalidade | Fonte GovExe | Origem oficial |
|---|---|---|
| Texto (ato) | `atos_dou` / config `dou` | DOU 1/2/3, CGU, TCU, Datajud, TJs, BCB, IBGE, Comprasnet, Transparência |
| Texto (lei) | `proposicoes_simplificadas` | Câmara + Senado (Dados Abertos) |
| **Documento visual (PDF)** | `planos_governo` / `pdf_url` | Planos de governo registrados no TSE (2026) |
| Imagem | fotos de candidato (TSE 161×225), fotos de parlamentares | TSE DivulgaCand / Câmara / Senado |
| Áudio + vídeo | reels narrados (`scripts/narrar.ts`, `reel/`) | TTS + Remotion/ffmpeg sobre o conteúdo simplificado |
| **Vídeo externo** (futuro) | captura de vídeos do **YouTube** (sessões, pronunciamentos, comissões) | canais oficiais de órgãos públicos |

> **Visão além do que o GovExe já coleta:** a arquitetura prevê **captura de vídeo externo
> (YouTube)** e **áudio** como modalidades a recuperar. A **transcrição (ASR)** entra como
> sinal auxiliar em **casos específicos**, sem substituir o embedding nativo de áudio/vídeo.

### 4.2 Mapeamento às taxonomias de fusão (base já levantada)

- **Zheng (2015) — cross-domain, *semantic-meaning-based***: ligar entidades entre
  datasets heterogêneos por significado (ex.: proposição ↔ ato do DOU ↔ eixo de plano,
  pela taxonomia de `tema`/`categoria` e por entidade/órgão). É fusão *entre fontes*.
- **Zhao; Zhang; Geng (2024) — deep multimodal fusion**: fusão *de modalidades* dentro
  de uma representação (texto + página-imagem via modelo visão-linguagem; *late/hybrid
  fusion*). Base dos retrievers multimodais do benchmark.
- **Zheng (2025, arXiv:2506.03155)** — ponte entre as duas tradições na era multimodal.
- **Camada de infraestrutura (Semana 2)**: a fusão de fontes ocorre na progressão
  **medalhão Bronze→Silver→Gold** de um lakehouse (DATABRICKS/MICROSOFT, 2026). Bronze =
  bruto por fonte; Silver = padronizado/deduplicado/resolvido; Gold = representação
  unificada + índice de recuperação.

### 4.3 Arquitetura proposta (diagrama + camadas)

Diagrama em Mermaid no doc final (`fusao/arquitetura.md`). Camadas:

1. **Ingestão (Bronze)** — conectores por fonte já existentes no GovExe (`lib/fontes/*`);
   dados brutos por modalidade.
2. **Padronização/Resolução (Silver)** — normalização de texto (pt-BR), extração de
   páginas de PDF como imagem, resolução de entidades (órgão/autor/tema), chave de fusão
   `(tema, janela temporal, entidade)`.
3. **Representação (Gold)** — camada multimodal de embeddings:
   - Texto → encoder pt-BR (BGE-M3 / multilingual-e5 / Serafim).
   - Documento visual (PDF) → ColQwen2/ColPali (multi-vetor).
   - Imagem → SigLIP2/CLIP.
   - Áudio → embedding nativo de áudio (ex.: CLAP / wav2vec2-BERT); ASR (Whisper) só
     para casos específicos, como sinal textual complementar.
   - Vídeo (reels, YouTube) → embedding de vídeo ou frames-chave + faixa de áudio;
     transcrição como sinal auxiliar.
   - *Fusão tardia/híbrida* na recuperação (escore combinado ou re-ranking).
4. **Recuperação/Apresentação** — índice (FAISS p/ dense; late-interaction p/ multi-vetor)
   + API de consulta. É o ponto onde o GovExe ganharia busca semântica multimodal sobre
   o que hoje só tem FTS.

O benchmark (Entregáveis 2–3) **valida a camada 3** (qual encoder recupera melhor), que é
a decisão de maior risco da arquitetura.

## 5. Entregável 2 — Pipeline de benchmark (código + requirements)

### 5.1 Layout do repositório

```
fusao/
  arquitetura.md            # doc da arquitetura de fusão (diagrama Mermaid)
benchmark/
  common/
    datasets.py             # loaders: ViDoRe v2, GovExe pt-BR
    index.py                # FAISS (dense) + late-interaction (multi-vetor)
    metrics.py              # Recall@{1,5}, nDCG@{5,10}, MRR
    runner.py               # laço eval: encode queries/corpus -> search -> métricas
    config.py               # dataclasses de config (modelo, dtype, batch, device)
  models/
    base.py                 # interface Retriever (encode_queries, encode_corpus, score)
    clip_siglip.py          # CLIP, SigLIP2 (open_clip / transformers)
    colvision.py            # ColPali, ColQwen2 (colpali-engine)
    # (mllm_embed.py — GME/VLM2Vec: adiado, fora desta semana)
    text_ptbr.py            # BGE-M3, multilingual-e5, Serafim (sentence-transformers)
  run_vidore.py             # validação estabelecida
  run_govexe_ptbr.py        # mini-benchmark pt-BR GovExe
  configs/*.yaml            # uma config por experimento
data/govexe/
  build_planos_benchmark.py # constrói o conjunto pt-BR a partir dos planos
  (saída: pages/*.png, queries.jsonl, qrels.tsv)
results/                    # CSV + relatório .md por rodada (versionados)
requirements.txt
requirements-gpu.txt        # torch/flash-attn/faiss-gpu (instalação à parte)
README.md                   # como rodar
```

### 5.2 Interface comum (`models/base.py`)

```python
class Retriever(Protocol):
    name: str
    def encode_queries(self, queries: list[Query]) -> Any: ...
    def encode_corpus(self, docs: list[Doc]) -> Any: ...
    def score(self, q_emb, d_emb) -> Float[Tensor, "nq nd"]: ...
```
Dois regimes: **dense single-vector** (CLIP/SigLIP2/texto/GME) usa FAISS;
**multi-vetor late-interaction** (ColPali/ColQwen2) usa MaxSim via `colpali-engine`.

### 5.3 Modelos (confirmar id/revisão exatos na implementação)

- Documento visual: `vidore/colqwen2-v1.0` (Qwen2-VL-2B), `vidore/colpali-v1.3`
  (PaliGemma-3B); opcional `vidore/colqwen2.5-v0.2`.
- Imagem↔texto: `openai/clip-vit-large-patch14`, `google/siglip2-so400m-patch14-384`.
- MLLM embedder (GME/VLM2Vec-V2): **adiado** — fora desta semana.
- Texto pt-BR: `BAAI/bge-m3`, `intfloat/multilingual-e5-large`, Serafim-900m,
  `Alibaba-NLP/gte-multilingual-base`.

## 6. Entregável 3 — Benchmark multimodal pt-BR a partir do GovExe

> É o "benchmark multimodal em português" citado pelo aluno — artefato prático da
> **trilha de RI** (que segue separada da trilha de Big Data nesta semana). O **primeiro
> caso** é RI de **documento visual** sobre os PDFs de planos (abaixo); imagem↔texto,
> áudio e vídeo (fotos TSE, reels, vídeos do YouTube) ficam como **extensões futuras**,
> fora desta semana.

**Corpus**: páginas dos **PDFs de planos de governo** (`planos`, 27 planos, mediana 44
págs → ~1,2k páginas-imagem). Renderizar via `pypdfium2`/`pdf2image` a ~150 dpi.

**Queries (pt-BR)**: derivadas dos alvos já existentes e verificáveis do dataset:
- `destaque` → consulta "ponto mais relevante do plano X".
- cada item de `eixos` (`titulo`/`propostas`) → consulta por eixo temático.
- `resumo` → consulta geral.

**Relevância (qrels)**: a query de um plano é relevante às páginas **daquele** plano
(relevância em nível de documento → página). *Setting local* por plano no início;
pool global (todas as páginas) como passo seguinte.

**Métrica**: nDCG@5 e Recall@{1,5} (comparável ao ViDoRe).

**Limitações herdadas (do `dataset/README.md`)**: só 27 planos, 5 UFs + BR; 3 planos
truncados no pipeline original; alvos gerados por IA (ruído na query). Documentar como
ameaça à validade — é um **mini**-benchmark exploratório, não um benchmark definitivo.

**Fatiamento de recursos**: se o índice multi-vetor de ~1,2k páginas não couber em RAM,
processar por plano e agregar. Medir tempo/VRAM e registrar.

## 7. Entregável 4 — Fontes recentes (verificar antes de citar)

Âncoras sólidas (pré-corte, alta confiança): ColPali (ICLR 2025); ViDoRe v2
(arXiv:2505.17166); MIEB (ICCV 2025, arXiv:2504.10471); VLM2Vec/MMEB (ICLR 2025);
VLM2Vec-V2/MMEB-V2 (arXiv:2507.04590); GME-Qwen2-VL; MM-Embed (ICLR 2025);
Serafim (arXiv:2407.19527).

Recentes pós-corte (**verificar URL/venue antes de citar na banca**): MTEB-PT/MTEB-BR
(2026); Qwen3-VL-Embedding; ColMate; métodos 2026 em MMEB-V2. Entregue como tabela ABNT
com DOI/arXiv + data de acesso, separando "estabelecido" de "preprint a verificar".

## 8. Dependências (requirements)

`torch`, `transformers`, `sentence-transformers`, `open_clip_torch`, `colpali-engine`,
`mteb` (extras de imagem p/ MIEB), `faiss-gpu` (ou `faiss-cpu`), `datasets`, `pypdfium2`
/ `pdf2image`, `pandas`, `pyyaml`, `tqdm`. GPU-específicas (`torch` CUDA, `flash-attn`,
`faiss-gpu`) em `requirements-gpu.txt`, instalação documentada à parte. Python 3.11.
Venv isolado (não reusar `.venv-dataset` do GovExe).

**Ambiente-alvo: WSL2 (Linux) na máquina da 4090** — habilita `faiss-gpu` e `flash-attn`.
`attn_implementation="eager"` e `faiss-cpu` ficam apenas como *fallback*.

## 9. Verificação (como saber que está certo)

- **Smoke test** (CPU ou 4090): encoder de texto pt-BR em uma tarefa pequena do `mteb`;
  pipeline produz nDCG/Recall não-trivial (sanidade da lib/FAISS).
- **Reprodução**: nDCG@5 de ColPali/ColQwen2 no ViDoRe v2 **na mesma ordem de grandeza**
  dos valores publicados (não idêntico — protocolos diferem; registrar desvio).
- **GovExe pt-BR**: cada query recupera páginas do próprio plano acima do acaso;
  inspeção manual de ~10 casos.
- Testes unitários de `metrics.py` (nDCG/Recall/MRR em casos sintéticos conhecidos).
- Tudo com `seed` fixa; config e versões de modelo logadas no CSV de saída.

## 10. Riscos e questões em aberto

- `colpali-engine` + `transformers` têm acoplamento de versão sensível → fixar versões.
- `flash-attn` no Windows pode falhar; ter *fallback* `attn_implementation="eager"`.
- `faiss-gpu` no Windows é problemático; *fallback* `faiss-cpu` (corpora pequenos toleram).
- Queries pt-BR derivadas de alvos de IA carregam viés do `gpt-4o` → validade limitada.
- **Resolvido**: GME/VLM2Vec-V2 **fora** desta semana (escopo enxuto).
- **Resolvido**: ambiente **WSL2 (Linux)** — `faiss-gpu` + `flash-attn` habilitados.

## 11. Critérios de aceite

1. `fusao/arquitetura.md` com diagrama Mermaid e mapeamento às três taxonomias + camada
   medalhão. ✔ quando legível e embasado.
2. `benchmark/` roda `run_vidore.py` e produz CSV de métricas para ≥2 modelos.
3. `data/govexe/build_planos_benchmark.py` gera `pages/`, `queries.jsonl`, `qrels.tsv`;
   `run_govexe_ptbr.py` produz CSV para os mesmos modelos.
4. `requirements.txt` + `requirements-gpu.txt` + `benchmark/README.md` de execução.
5. Tabela de fontes recentes verificadas (§7) em `docs/`.
6. Testes de `metrics.py` passando.
