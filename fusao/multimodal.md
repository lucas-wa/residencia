# Estratégia multimodal — modelo unificado × ensemble de especialistas

> Companheiro de `fusao/arquitetura.md` (taxonomias de fusão) e `fusao/coleta-gcp.md`
> (coleta em GCP). Decisão da Semana 7: **implementar o modelo unificado** na primeira
> fatia (ver `gcp/slice/`) e **documentar aqui o ensemble** para implementação e
> comparação (A/B) futuras. Projeto/visão — levantamento SOTA de 2026.

## 1. O problema

A base GovExe vai além de texto: **documentos** (PDFs com layout), **imagens** (fotos,
infográficos), **áudio** e **vídeo** (YouTube: sessões, pronunciamentos). Queremos
**recuperar** sobre tudo. Duas perguntas de projeto:

1. **Transcrição/descrição × embedding de conteúdo** — representar o não-texto por um
   *descritor textual* (legenda/ASR/OCR) ou por *embedding nativo* da modalidade?
2. **Modelo único × ensemble de especialistas** — um só embedder cobrindo tudo, ou o
   melhor por modalidade e "colapsar" os sinais num só sistema?

O plano é **implementar as duas abordagens** e compará-las; esta semana sobe a primeira.

## 2. Abordagem A — modelo unificado (IMPLEMENTADA nesta semana)

Um único embedder projeta várias modalidades no **mesmo espaço vetorial** → **1 índice**,
consultas cross-modais nativas, operação simples.

- **Escolhido:** **Qwen3-VL-Embedding** — texto + imagem + documento-imagem + **vídeo**
  num só espaço; o unificado self-hostable mais abrangente de 2026. **2B** na fatia GCP
  (cabe folgado em L4 24 GB; VM efêmera); **8B** como opção (L4/A100). Matryoshka (MRL)
  permite truncar a dimensão para baratear o índice. ⚠ confirmar licença no model card.
- **Teto de score (se subir o hardware):** **WeMM-Embedding-9B** (líder MMEB-V2 ~80,6) em
  A100 80 GB; mesma família de uso.
- **Limite central:** **nenhum** unificado cobre **áudio** → áudio sempre em trilha própria
  (ver §4).

## 3. Abordagem B — ensemble de especialistas + *pointer vector* (DESIGN)

O melhor embedder por modalidade; os sinais são combinados por **fusão de ranqueamento**.
O **pointer vector** é o mecanismo que "colapsa" o não-texto no espaço de texto: gera um
**descritor textual** (legenda/ASR/OCR/resumo-VLM) do conteúdo, **embeda o texto** e guarda
um **ponteiro** (URI/ID) ao binário original — 1 índice, busca uniforme, barato e
**auditável** (bom para dados governamentais; coerente com "o resumo é a porta para a fonte").

| Modalidade | Especialista (open-weight) | Opção de API (quando supera) |
|---|---|---|
| Texto pt-BR | **Qwen3-Embedding** (4B/8B) ou **BGE-M3** (híbrido denso+esparso) | Cohere Embed 4; Voyage-3.5; Gemini embedding |
| Documento-imagem | **ColQwen2.5** (late interaction, Apache-2.0) / Nemotron ColEmbed V2 | Cohere Embed 4 (PDF/slides em escala, sem GPU) |
| Imagem | **SigLIP 2** | Voyage multimodal-3.5 (mesmo espaço texto+imagem) |
| Áudio | **Whisper large-v3 → texto** (pointer) · CLAP (embedding direto) | AssemblyAI Universal-2 / ElevenLabs Scribe (ASR) |
| Vídeo | **Qwen3-VL-Embedding / WeMM** (nativo) · frames+ASR (pointer) | TwelveLabs Marengo 3.0 (especialista) |

- **Fusão:** **RRF** (reciprocal rank fusion) dos índices por modalidade → **rerank
  two-stage** (recall rápido → reranker multimodal sobre um candidato menor).
- **APIs:** registradas como opção, **não-default** por licença/custo e por **soberania do
  dado** (base pública governamental). Preço/licença a confirmar nas páginas oficiais.

## 4. Áudio — sempre trilha própria

Nenhum embedder unificado cobre áudio. Dois caminhos: **(a) ASR→texto** (Whisper
large-v3, melhor pt em FLEURS) → embedder de texto; **(b) embedding de áudio direto**
(CLAP e sucessores). Para **fala governamental** (semântica densa), **ASR→texto costuma
ganhar** em recuperação — é o *pointer vector* da §3. ASR **só em casos específicos**
(coerente com `fusao/arquitetura.md`); observar a Resolução TSE 23.732/2024.

## 5. Protocolo de comparação (A/B)

Mesmo **corpus** e **query-set**; medir para cada abordagem:

- **Qualidade:** nDCG@k, MRR, Recall@k (mesma convenção do benchmark da Semana 6).
- **Custo:** GPU-hora de indexação + **storage do índice** (multi-vetor pesa mais que denso).
- **Latência:** tempo de consulta ponta a ponta.
- **Controles:** dimensão do vetor (MRL), tamanho do índice, hardware fixos entre as duas.

**Hipótese:** o **unificado** vence em *cross-modal* e **custo operacional** (1 índice, 1
modelo); o **ensemble** vence em **precisão por modalidade** (especialistas, ex.
late-interaction em documento), ao custo de complexidade e storage. O A/B decide o que vai
para a camada Gold de produção.

## 6. Decisão da Semana 7

- **Implementar A** (unificado **Qwen3-VL-Embedding-2B**) na fatia GCP — ver
  `fusao/coleta-gcp.md` §4.1 e `gcp/slice/` (`README-slice.md`).
- **B** fica desenhado aqui, pronto para implementação incremental e para o A/B.

## Referências

Levantamento SOTA 2026 (ids ⚠ a confirmar no model card do HF antes de uso oficial):
Qwen3-VL-Embedding (arXiv:2601.04720); WeMM-Embedding (arXiv:2608.24053); Nemotron
ColEmbed V2 (arXiv:2602.03992); jina-embeddings-v4 (Qwen Research License); SigLIP 2
(arXiv:2502.14786); VLM2Vec-V2 / MMEB-V2 (arXiv:2507.04590); MTEB-BR (arXiv:2607.04581);
Whisper large-v3; CLAP. APIs: Cohere Embed 4, Voyage-3.5, Gemini embedding, TwelveLabs
Marengo 3.0. Taxonomias de fusão: Zheng (2015); Zhao, Zhang e Geng (2024); Zheng (2025).
