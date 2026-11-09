# Ensemble v1 (pointer-vector cross-modal) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provar fusão cross-modal com fontes públicas oficiais sobre o tema **Orçamento** — uma query de texto recupera o melhor item de cada modalidade (documento, imagem, áudio, texto, vídeo), representados por descritor textual + ponteiro ao binário, num único índice de texto em memória.

**Architecture:** *Pointer vector* (`fusao/multimodal.md §3`). Todo não-texto vira **descritor textual + ponteiro** (URI/ID). Um índice de texto único (BGE-M3, cosine em memória — sem pgvector, nada na máquina do Lucas). Pipeline: captura por eixo → `PointerRecord` (JSONL) → índice → runner de demonstração. Roda no **Slurm H100** (nova STAGE em `slurm_run.sh`, espelhando `govexe`/`vidore`). Código testável (schema, descritores, índice, demo, parsers de captura) separado dos *fetch* de rede (wrappers finos).

**Tech Stack:** Python 3.10. BGE-M3 (`FlagEmbedding` ou `sentence-transformers`), Whisper large-v3 (`faster-whisper`/`openai-whisper`, já usado), `yt-dlp`, `pypdfium2` (já usado no build do benchmark), `requests`+`beautifulsoup4` (scrape), `numpy` (cosine). Reusa o harness `benchmark/` e `slurm_run.sh`.

**Spec:** `fusao/ensemble.md` (ler junto com este plano; argumenta a partir de `fusao/multimodal.md §3`).

## Global Constraints

- **Roda no Slurm H100** (`slurm_run.sh`, nova STAGE `ensemble`); **nunca** na máquina do Lucas, **nunca** GCP (GCP é só teste de infra de produção).
- **Índice em memória** (cosine sobre BGE-M3 L2-normalizado). **Sem pgvector, sem Cloud SQL, sem DB.**
- **Tema = Orçamento**; **20 itens por eixo** (~100 no total, 5 eixos).
- **Fontes só oficiais e públicas**, reuso com crédito: texto = Agência Brasil (CC-BY) + Agência Câmara + Agência Senado; documento = LOA + relatório da CMO (ano corrente); imagem = fotos de agências (Câmara/EBC/Senado) com legenda; áudio = EBC/Radioagência; vídeo = YouTube Câmara + TV Senado + Canal Gov.
- **pt-BR** em todo descritor e query.
- **Sem RRF e sem especialistas nativos** no v1 (são o v2). ColQwen2.5/B **não** entra aqui.
- **Fetch de rede isolado em wrappers finos**; a lógica de parsing é testada por *fixture* salva (sample de HTML/JSON/PDF), nunca por rede ao vivo no teste.
- **Commits:** commit local por tarefa segue o fluxo TDD; **push/PR só com OK do Lucas**. Toda mensagem de commit termina com os trailers de atribuição (`Co-Authored-By: Claude Opus 4.8 …` + `Claude-Session: …`).

## Review Focus

- **Query/descritor vazio ou só-whitespace** → o índice não pode quebrar nem ranquear lixo (Task 3 testa).
- **Modalidade ausente no corpus** (ex. vídeo falhou na captura) → a demo ainda roda com os eixos presentes e diz quais faltam (Task 4 testa).
- **Ponteiro não-resolúvel** (URL morta / arquivo sumido) → a demo marca o hit como "ponteiro indisponível" em vez de estourar (Task 4 testa).
- **Áudio/vídeo sem fala** (ASR devolve string vazia) → o item é descartado com aviso, não entra como descritor vazio (Task 2 testa).
- **Encoding pt-BR** (acentos em descritor e no output do CLI) → sem mojibake (Task 1 e Task 4 testam).

## File Structure

- `ensemble/__init__.py` — pacote.
- `ensemble/records.py` — `PointerRecord` + I/O JSONL + validação. (Task 1)
- `ensemble/descriptors.py` — adaptadores de descritor por modalidade (backends injetados). (Task 2)
- `ensemble/index.py` — `EnsembleIndex` (embed BGE-M3 + cosine em memória). (Task 3)
- `ensemble/demo.py` — runner de demonstração cross-modal + formatação CLI. (Task 4)
- `ensemble/run_ensemble.py` — entrypoint que a STAGE do Slurm chama (carrega records → índice → demo). (Task 5)
- `slurm_run.sh` — adiciona `STAGE=ensemble`. (Task 5)
- `ensemble/capture/{text,document,image,audio,video}.py` — ingestores (parser testável + fetch fino). (Tasks 6–8)
- `tests/test_ensemble_records.py`, `test_ensemble_descriptors.py`, `test_ensemble_index.py`, `test_ensemble_demo.py`, `test_ensemble_capture.py` — testes (dublês/fixtures, sem rede, sem GPU).
- `tests/fixtures/ensemble/` — samples salvos (HTML/JSON/PDF/áudio curto) para os parsers.

---

### Task 1: `PointerRecord` — schema e I/O

**Files:**
- Create: `ensemble/__init__.py`, `ensemble/records.py`
- Test: `tests/test_ensemble_records.py`

**Interfaces:**
- Produces: `PointerRecord` (frozen dataclass): `id: str`, `modality: Literal["documento","imagem","audio","texto","video"]`, `descriptor: str`, `pointer_uri: str`, `source: str`, `meta: dict`. `load_records(path) -> list[PointerRecord]`, `save_records(records, path)` (JSONL, UTF-8), `validate_record(r) -> None` (levanta `ValueError`).

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_records.py`:
  - `test_roundtrip_jsonl_preserva_acentos`: salva e recarrega um record com `descriptor="Orçamento da saúde"`; o texto volta idêntico.
  - `test_validate_rejeita_modalidade_invalida`: `validate_record` com `modality="pdf"` levanta `ValueError`.
  - `test_validate_rejeita_descriptor_vazio`: `descriptor="  "` levanta `ValueError`.
  - `test_validate_rejeita_pointer_vazio`: `pointer_uri=""` levanta `ValueError`.
- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_ensemble_records.py -v` → FAIL (módulo inexistente).
- [ ] **Step 3: Implementar** `PointerRecord`, `load_records`, `save_records`, `validate_record` em `ensemble/records.py`. JSONL com `ensure_ascii=False`, `encoding="utf-8"`.
- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_ensemble_records.py -v` → PASS.
- [ ] **Step 5: Commit** (`feat(ensemble): PointerRecord + I/O JSONL`).

---

### Task 2: Adaptadores de descritor (ASR / legenda / OCR / texto)

**Files:**
- Create: `ensemble/descriptors.py`
- Test: `tests/test_ensemble_descriptors.py`

**Interfaces:**
- Consumes: nada da Task 1 (opera sobre caminhos/strings crus).
- Produces: `asr_descriptor(media_path, *, transcriber) -> str` (transcriber injetado: `Callable[[str], str]`); `caption_descriptor(caption) -> str` (passthrough normalizado); `ocr_descriptor(doc_path, *, extractor) -> str` (extractor injetado); `text_descriptor(html_or_text, *, html=False) -> str`. Todos devolvem texto limpo (strip, colapsa espaço). **ASR/OCR que devolvem vazio levantam `EmptyDescriptorError`** (subclasse de `ValueError`) — o chamador descarta o item.

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_descriptors.py` (todos com dublês, sem GPU):
  - `test_asr_usa_transcriber_injetado`: `asr_descriptor("x.wav", transcriber=lambda p: "fala sobre orçamento")` devolve "fala sobre orçamento".
  - `test_asr_vazio_levanta_EmptyDescriptorError`: transcriber devolve `"   "` → `EmptyDescriptorError`.
  - `test_text_descriptor_extrai_de_html`: dado `"<p>Orçamento <b>2026</b></p>"` com `html=True`, devolve "Orçamento 2026" (sem tags).
  - `test_caption_normaliza_espacos`: `"  foto da  votação  "` → "foto da votação".
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** os quatro adaptadores + `EmptyDescriptorError`. `text_descriptor(html=True)` usa BeautifulSoup `get_text(" ")`. Normalização comum num helper `_limpa(s)`.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble): adaptadores de descritor por modalidade`).

---

### Task 3: Índice em memória (BGE-M3 + cosine)

**Files:**
- Create: `ensemble/index.py`
- Test: `tests/test_ensemble_index.py`

**Interfaces:**
- Consumes: `PointerRecord` (Task 1).
- Produces: `build_index(records, *, embedder) -> EnsembleIndex` (embedder injetado: `Callable[[list[str]], np.ndarray]` devolvendo `[n, d]`); `EnsembleIndex.search(query, k=10, *, embedder) -> list[tuple[PointerRecord, float]]` (cosine sobre embeddings L2-normalizados, ordem decrescente). Corpus vazio em `build_index` levanta `ValueError`. Query vazia em `search` levanta `ValueError`. **Padrão dublê idêntico ao `tests/test_doc_specialist.py`** (embedder falso, sem baixar modelo).

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_index.py`:
  - `test_search_ordena_por_cosine`: com embedder dublê determinístico (ex. bag-of-words → vetor), a query "orçamento saúde" ranqueia o record cujo descritor é "orçamento da saúde" acima de um record "foto de pônei".
  - `test_corpus_embutido_uma_vez`: espiona o embedder; `build_index` chama-o **uma vez** com todos os descritores (não por query).
  - `test_build_index_corpus_vazio_levanta`: `[]` → `ValueError`.
  - `test_search_query_vazia_levanta`: `search("  ")` → `ValueError`.
  - `test_search_k_maior_que_corpus`: `k=100` sobre 3 records devolve 3 sem estourar.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** `build_index` (embute descritores uma vez, L2-normaliza, guarda matriz + records) e `search` (embute query, dot-product, `argsort` decrescente, corta em k). `numpy` apenas.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble): indice em memoria BGE-M3 + cosine`).

---

### Task 4: Runner de demonstração cross-modal

**Files:**
- Create: `ensemble/demo.py`
- Test: `tests/test_ensemble_demo.py`

**Interfaces:**
- Consumes: `EnsembleIndex` (Task 3), `PointerRecord` (Task 1).
- Produces: `run_demo(index, queries, *, embedder, k=10, resolver=None) -> list[DemoResult]`; `format_cli(result) -> str` (ranking anotado: `[MODALIDADE] descritor-trecho → pointer_uri`, uma linha por hit). `DemoResult` tem `query`, `hits` (record+score), `modalidades_presentes: set[str]`, `modalidades_ausentes: set[str]`. `resolver(uri) -> bool` injetado checa se o ponteiro resolve; hit não-resolúvel recebe marca "ponteiro indisponível" no `format_cli`, sem exceção.

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_demo.py`:
  - `test_demo_interleava_modalidades`: corpus com as 5 modalidades + embedder dublê; para uma query de orçamento, o top-k inclui mais de uma modalidade e `format_cli` imprime a tag `[MODALIDADE]` de cada hit.
  - `test_demo_reporta_modalidade_ausente`: corpus sem `video`; `DemoResult.modalidades_ausentes == {"video"}` e a demo roda.
  - `test_ponteiro_indisponivel_nao_estoura`: `resolver=lambda u: False` → `format_cli` contém "ponteiro indisponível" e não levanta.
  - `test_format_cli_preserva_acentos`: saída contém "Orçamento" sem mojibake.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** `run_demo` (para cada query: `index.search`, computa modalidades presentes/ausentes vs o conjunto canônico de 5) e `format_cli`. `resolver` default = `None` (não checa).
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble): runner de demonstracao cross-modal + CLI`).

---

### Task 5: Entrypoint + STAGE `ensemble` no Slurm

**Files:**
- Create: `ensemble/run_ensemble.py`
- Modify: `slurm_run.sh`
- Test: `tests/test_ensemble_entrypoint.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `main(argv=None, *, embedder=None) -> int` — lê `--records <jsonl>`, `--queries <txt>`, `--out <txt>`, constrói BGE-M3 (import tardio; `embedder` injetável no teste), roda a demo, grava o output anotado. Carrega BGE-M3 via `FlagEmbedding`/`sentence-transformers` numa função `_build_embedder()` com import tardio (como `doc_specialist._build_retriever`).

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_entrypoint.py`:
  - `test_main_grava_output` (embedder dublê injetado): dado um JSONL de records e um arquivo de queries, `main` escreve o arquivo de saída com as tags de modalidade e retorna 0. Não baixa modelo (dublê).
  - `test_main_records_inexistente_retorna_erro`: caminho inexistente → retorna código ≠ 0 com mensagem clara.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** `ensemble/run_ensemble.py` (`main` + `_build_embedder` tardio). Depois adicionar ao `slurm_run.sh` a STAGE `ensemble` espelhando `govexe` (preflight + `python -m ensemble.run_ensemble --records … --queries … --out logs/ensemble_demo.txt`), com `--gres=gpu:h100:1`.
- [ ] **Step 4: Verificar** — `pytest tests/test_ensemble_entrypoint.py -v` → PASS; `bash -n slurm_run.sh` → rc=0; `STAGE=ensemble` aparece no `case` do script.
- [ ] **Step 5: Commit** (`feat(ensemble): entrypoint + STAGE ensemble no slurm_run.sh`).

---

### Task 6: Captura — texto e documento (orçamento)

**Files:**
- Create: `ensemble/capture/__init__.py`, `ensemble/capture/text.py`, `ensemble/capture/document.py`
- Test: `tests/test_ensemble_capture_text_doc.py`, `tests/fixtures/ensemble/` (sample de artigo + sample de PDF orçamentário)

**Interfaces:**
- Produces: `text.parse_article(raw_html, url) -> PointerRecord` (descritor = `text_descriptor(html=True)`, modality `"texto"`); `text.fetch_budget_articles(fontes, n=20) -> list[PointerRecord]` (wrapper fino de rede, sem teste de rede). `document.parse_pages(pdf_path, *, extractor, n=20) -> list[PointerRecord]` (descritor = `ocr_descriptor`, modality `"documento"`, pointer = `page_id`). **Fonte do documento: LOA + relatório da CMO do ano corrente — escolha minha (Task settle); confirmar com o Lucas antes do fetch real.**

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_capture_text_doc.py` (fixtures salvas, sem rede):
  - `test_parse_article_extrai_record`: dado o HTML-fixture de uma matéria de orçamento, `parse_article` devolve `PointerRecord` válido (modality "texto", descritor não-vazio com "orçamento", pointer = url).
  - `test_parse_pages_gera_n_records`: dado o PDF-fixture + extractor dublê, `parse_pages(n=3)` devolve 3 records "documento" com `page_id` único.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** os parsers (testáveis) + os wrappers de fetch finos (`fetch_budget_articles` via RSS/API da §5; `document` baixa o PDF e usa `pypdfium2`). Cada record passa por `validate_record`.
- [ ] **Step 4: Rodar e ver passar** (só os parsers; o fetch real roda na Task 9).
- [ ] **Step 5: Commit** (`feat(ensemble): captura de texto e documento (orcamento)`).

---

### Task 7: Captura — imagem e áudio (orçamento)

**Files:**
- Create: `ensemble/capture/image.py`, `ensemble/capture/audio.py`
- Test: `tests/test_ensemble_capture_img_audio.py`, fixtures (sample de galeria + clip de áudio curto)

**Interfaces:**
- Produces: `image.parse_gallery(raw, base_url) -> list[PointerRecord]` (descritor = `caption_descriptor` da legenda que acompanha a foto, modality "imagem", pointer = url da imagem); `audio.record_from_clip(clip_path, url, *, transcriber) -> PointerRecord` (descritor = `asr_descriptor`, modality "audio", pointer = url); item com ASR vazio é **pulado** (captura o `EmptyDescriptorError` e segue).

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_capture_img_audio.py`:
  - `test_parse_gallery_usa_legenda`: fixture de galeria com 2 fotos+legendas → 2 records "imagem" com descritor = legenda.
  - `test_foto_sem_legenda_e_pulada`: foto sem legenda não vira record (descritor vazio não entra).
  - `test_record_from_clip_usa_asr`: transcriber dublê → record "audio" com descritor da ASR.
  - `test_clip_sem_fala_e_pulado`: transcriber devolve "" → `record_from_clip` sinaliza "pulado" (None), sem estourar.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** parsers + wrappers de fetch (galeria de agência; clipes EBC/Radioagência → Whisper).
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble): captura de imagem e audio (orcamento)`).

---

### Task 8: Captura — vídeo (YouTube oficial, orçamento)

**Files:**
- Create: `ensemble/capture/video.py`
- Test: `tests/test_ensemble_capture_video.py`, fixture (metadados yt-dlp sample)

**Interfaces:**
- Produces: `video.record_from_session(info, audio_path, *, transcriber, start=None, end=None) -> PointerRecord` (descritor = `asr_descriptor` da fala do trecho; modality "video"; pointer = `url#t=<start>`); `video.fetch_sessions(canais, termo="orçamento", n=20)` (wrapper `yt-dlp`, sem teste de rede). Vídeo = *pointer* (frames+ASR), **não** embedding de vídeo (é v2).

- [ ] **Step 1: Testes falhando** — `tests/test_ensemble_capture_video.py`:
  - `test_record_from_session_monta_pointer_com_timestamp`: info-fixture + transcriber dublê → record "video" com `pointer_uri` contendo `#t=`.
  - `test_sessao_sem_fala_e_pulada`: ASR vazio → pulado (None).
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** `record_from_session` + `fetch_sessions` (yt-dlp baixa áudio da sessão sobre orçamento; Whisper ASR do trecho). Keyframes ficam como meta opcional (o descritor v1 é a ASR).
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** (`feat(ensemble): captura de video (youtube oficial, orcamento)`).

---

### Task 9: Corpus de demonstração + run no Slurm H100

**Files:**
- Create: `ensemble/build_corpus.py` (orquestra as capturas → `data/govexe/ensemble_orcamento/records.jsonl`), `ensemble/queries_orcamento.txt`
- Test: `tests/test_ensemble_build_corpus.py`

**Interfaces:**
- Consumes: Tasks 6–8 (os `fetch_*`).
- Produces: `build_corpus(destino, *, por_eixo=20) -> Path` (chama os 5 fetch, concatena, valida, grava JSONL); falha **clara** se um eixo vier com 0 itens (diz qual).

- [ ] **Step 1: Teste falhando** — `test_build_corpus_concatena_e_valida` (com os `fetch_*` monkeypatchados devolvendo records dublê por eixo): grava JSONL com 5 modalidades e todas passam `validate_record`; se um eixo devolve `[]`, `build_corpus` levanta erro nomeando o eixo.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** `build_corpus` + redigir ~8 queries de orçamento em `queries_orcamento.txt` (ex. "orçamento da saúde", "emendas parlamentares", "despesas com educação", "relator do orçamento").
- [ ] **Step 4: Verificar** — teste PASS. **Depois (fora do TDD, passo de dados, requer confirmação do Lucas das fontes):** rodar o fetch real → `records.jsonl` com ~100 itens; submeter `STAGE=ensemble sbatch slurm_run.sh` no cluster; conferir `logs/ensemble_demo.txt` mostrando uma query de orçamento puxando itens de ≥3 modalidades.
- [ ] **Step 5: Commit** (`feat(ensemble): build do corpus de orcamento + queries`).

---

## Self-Review

- **Spec coverage:** v1 pointer (§2) → Tasks 1–5 (schema/descritor/índice/demo/slurm) + 6–9 (as 5 capturas, 20/eixo, tema orçamento). v2 (§3), métricas/qrels (futuro a), RRF e nativos — **fora** deste plano (non-goals), como a spec manda. ✓
- **Tipos consistentes:** `PointerRecord` (Task 1) é consumido com os mesmos campos em 3, 4, 6–9; `embedder: Callable[[list[str]], np.ndarray]` igual em 3 e 5; `transcriber` igual em 2, 7, 8. ✓
- **Review Focus coberto:** query/descritor vazio (T1, T3), modalidade ausente (T4), ponteiro morto (T4), ASR vazio (T2, T7, T8), encoding pt-BR (T1, T4). ✓
- **Proporção:** passos são assinatura + nome/asserção de teste, sem corpos — o builder escreve o código. ✓
- **Dependências externas isoladas:** todo fetch de rede/GPU é wrapper fino não-testado por rede; o testável é parser+lógica com fixtures/dublês (mesmo padrão de `test_doc_specialist.py`). ✓
