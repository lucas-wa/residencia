# Ensemble multimodal — demonstração cross-modal (v1 *pointer*) → ensemble nativo (v2)

> Implementa a **Abordagem B** de [`multimodal.md`](./multimodal.md) §3 em duas ondas.
> Os eixos documento (objetivo #3, `arch B` = ColQwen2.5) e áudio (objetivo #4, A×A+áudio)
> já foram provados **isolados** na GCP. Este documento especifica o **ensemble dos 5 eixos**.
> Decidido no brainstorm de 2026-11-09 (Lucas).

## 1. Intenção e porquê

Provar, com **fontes públicas reais**, que **uma única consulta de texto** recupera o melhor
conteúdo governamental **seja qual for a modalidade** (documento, imagem, áudio, texto, vídeo),
com **rastreabilidade à fonte oficial**. Duas ondas, deliberadamente nessa ordem:

1. **v1 — barato e auditável primeiro** (*pointer vector*): prova a fusão cross-modal sem
   gastar GPU com vários modelos pesados nem depender de gabarito que ainda não existe.
2. **v2 — nativo e preciso depois** (ensemble de especialistas + RRF): a recuperação de
   verdade por modalidade, onde as métricas do futuro (a) comparam ensemble × A × B.

## 2. v1 — índice único por *pointer vector* (os 5 eixos) · **FOCO DESTA ENTREGA**

Mecanismo (§3 do `multimodal.md`): todo não-texto vira um **descritor textual** + um
**ponteiro** (URI/ID) ao binário original → **embeda o texto** → **1 índice de texto**,
busca uniforme, **cross-modal por construção**, barato e **auditável** ("o resumo é a porta
para a fonte"). É o padrão da prova do objetivo #2, estendido às 5 fontes reais.

**Escopo travado (2026-11-09):** tema único = **Orçamento** (atravessa as 5 modalidades); ~**20
itens por eixo** (~100 no total); roda no **Slurm H100**; fontes da §5 confirmadas. O tema único
é o que torna a fusão cross-modal visível: "orçamento da saúde" deve puxar o documento, a foto da
votação, o áudio da notícia, o texto da matéria e o vídeo da sessão — todos sobre orçamento.

> **Consequência do tema:** o corpus de documento **deixa de ser o TSE/planos** (plano de governo
> ≠ orçamento). O eixo documento passa a usar **documentos orçamentários** oficiais (LOA/LDO/PPA ou
> relatórios da Comissão Mista de Orçamento) — fonte a confirmar na §6.

| Eixo | Fonte pública | Descritor (v1) | Ponteiro |
|---|---|---|---|
| Documento | **orçamento**: LOA/LDO/PPA ou relatórios da CMO (a confirmar) | texto extraído / OCR | `page_id` |
| Imagem | fotos de agências (Câmara/EBC/Senado) | a **legenda** que já acompanha a foto | URL da imagem |
| Áudio | EBC / Radioagência (temos clipes) | Whisper large-v3 → ASR | URL do clipe |
| Texto | **sites oficiais (não-documento)** | o próprio texto (trecho/parágrafo) | URL |
| Vídeo | **YouTube oficial (Câmara/Senado)** | **frames-chave + ASR da fala** | URL + timestamp |

**Decisões de builder (minhas escolhas — revise):**

- **Embedder de texto do índice único:** `BAAI/bge-m3` (denso+esparso, multilíngue, bom
  pt-BR, leve). Alternativa: Qwen3-Embedding-0.6B. *(minha recomendação: BGE-M3)*
- **Store:** **sem pgvector e sem nada na máquina do Lucas** — índice **em memória** (cosine
  sobre os embeddings BGE-M3), igual ao que o `arch B` já faz com MaxSim. v1 é leve.
- **Sem RRF e sem especialistas nativos** no v1: é **um** índice; a fusão cross-modal é
  intrínseca ao espaço de texto. (Nem o ColQwen2.5/B entra aqui — ele é v2.)
- **Demonstração:** N queries de texto → top-k **intercalado**, cada hit com **tag de
  modalidade** + resolução do ponteiro (abre o binário/link). Saída: script CLI que imprime
  o ranking anotado (opção: página simples depois).

**Onde roda:** **Slurm H100** (cluster UFG, nó "H100 2"), via `slurm_run.sh` + harness
`benchmark/` — uma **nova STAGE** (ex. `ensemble`), espelhando as stages `govexe`/`vidore`.
Isto segue a regra do Lucas: **eval/embed → Slurm H100** (`slurm_run.sh`); **teste de infra
de produção → fatia paga GCP** (`run_slice.sh`, Cloud SQL/pgvector/VM efêmera). O v1 é
embed+eval, então é Slurm. A captura (scrape/`yt-dlp`+ASR) é um passo de preparo à parte.

## 3. v2 — ensemble de especialistas nativos (RRF) · *próxima onda*

Troca cada ponteiro pelo **especialista nativo**, cada um no seu índice, **fundidos por RRF**
→ rerank two-stage:

| Eixo | Especialista nativo | Estado |
|---|---|---|
| Documento | **ColQwen2.5** (late interaction, MaxSim) | **pronto** — é o `arch B` |
| Imagem | **SigLIP 2** | a fazer |
| Áudio | **CLAP** (laion/larger_clap_general) | base pronta (espaço do A+áudio) |
| Texto | Qwen3-Embedding / BGE-M3 | a fazer |
| Vídeo | Qwen3-VL / WeMM (nativo) | a fazer |

É aqui que entra o **futuro (a)**: construir mini-gabarito (~15-20 queries + qrels) para
imagem/texto/vídeo e medir **ensemble × A × B** com nDCG/MRR por eixo. v2 **também roda no
Slurm H100** (é embed+eval) — a fatia paga GCP fica reservada a **validar a infra de
produção** (coleta/armazenamento/pgvector/teardown), não estes runs de modelo.

## 4. Non-goals do v1

- **Métricas** em imagem/texto/vídeo — sem qrels; é o futuro (a), no v2.
- **Embeddings nativos** e **RRF** — são o v2.
- **Vídeo nativo** — no v1 o vídeo é *pointer* (frames-chave + ASR), não embedding de vídeo.
- **Ligação de entidades / grafo do DOU** — trilha separada (design "fase futura, pós-S9").

## 5. Captura — fontes públicas oficiais

Imagem e áudio já mapeados em "Semana 9 — fontes de captura" (`coleta-gcp.md §6`). As duas
ingestões novas do v1 (texto e vídeo) têm fonte oficial, pública e **reutilizável com crédito**:

**Texto — portais oficiais de notícia (não-documento):**

- **Agência Brasil** (EBC) — agência de notícias do Executivo federal, licença **Creative
  Commons** (reuso livre com crédito). Principal fonte de texto do Executivo.
- **Agência Câmara Notícias** (`camara.leg.br/noticias`) — reuso permitido com a assinatura
  "Agência Câmara Notícias"; conteúdo dividido em ~21 temas.
- **Agência Senado / Portal Senado Notícias** (`senado.leg.br/noticias`) — produzido pela
  Secretaria de Comunicação do Senado.
- **Acesso programático (para a ingestão):** API de **Dados Abertos da Câmara** v2
  (`dadosabertos.camara.leg.br`, JSON) e de **Dados Abertos do Senado**
  (`legis.senado.leg.br/dadosabertos`, XML/JSON, público sem auth). As notícias em si saem
  do portal/RSS; as APIs cobrem metadados legislativos (bom para ligar texto ↔ entidade depois).

**Vídeo — canais oficiais no YouTube:**

- **Câmara dos Deputados** (canal oficial desde 2016) — transmite ao vivo, **sem cortes e sem
  edição**, audiências, reuniões de comissão e votações de Plenário; tem janela de Libras.
- **TV Senado** (YouTube) — todos os eventos legislativos ao vivo (até 8 simultâneos).
- **Canal Gov** (Executivo federal, ex-TV NBR/EBC) — *a outra fonte oficial que faz sentido*:
  cobre o **Executivo**, complementando o Legislativo das duas Casas. *(confirmar o canal/handle atual.)*

Fluxo Bronze → Silver (descritor: ASR da fala / legenda / texto) → índice, igual ao resto da
pipeline GovExe. A fala de plenário/sessão é **clara e densa** → ótima para ASR (pointer).

## 6. Decisões

**Travadas (2026-11-09):** fontes da §5 confirmadas (texto = Agência Brasil + Câmara + Senado;
vídeo = Câmara + TV Senado + Canal Gov); **20 itens/eixo** (~100 no total); tema = **Orçamento**;
roda no **Slurm H100**; índice em memória (sem pgvector).

**Ainda em aberto:**

- **Fonte dos documentos orçamentários** (eixo documento, por causa do tema): LOA/LDO/PPA ou
  relatórios da **CMO** (Comissão Mista de Orçamento). Proponho LOA + relatório da CMO do ano
  corrente, de fonte oficial (Câmara/Senado/Planalto). Confirmar.
- **Handle atual do Canal Gov** no YouTube (confirmar o canal do Executivo).
- **Formato da demonstração:** CLI anotado agora; página simples depois.
