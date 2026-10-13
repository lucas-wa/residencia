# Fusão de Big Data e Recuperação de Informação Multimodal

## 1. O que é a disciplina e o tema

- **Disciplina**: Residência em IA — Bacharelado em Inteligência Artificial (BIA),
  Instituto de Informática, Universidade Federal de Goiás (UFG).
- **Objetivo da disciplina**: especializar o aluno em uma área de interesse da IA ao longo
  do semestre, com apresentações periódicas para uma banca.
- **Tema**: **"Fusão de Big Data e Recuperação de Informação Multimodal"**.
- **Gênese do tema**: escolhido a partir dos tópicos de interesse das conferências do
  congresso **CSCE 2026** (em especial ICAI, ICDATA'26 e ACC) — a interseção de
  "Multimodal learning systems" com "Big Data Fusion and Information Retrieval".
- O tema é a interseção de várias subáreas, estudadas separadamente:
  Big Data → Fusão de Big Data → RI (Recuperação de Informação) → modelos de embeddings
  multimodais → RI Multimodal → modelos de linguagem multimodais.

## 2. Processo e etapas (histórico do trabalho até aqui)

### Etapa 1 — História e fundamentos (concluída)
Levantamento da história e dos materiais fundacionais de **Big Data** e **RI**
(papers seminais, cursos, livros, blogs), com metodologia de busca documentada
(busca em camadas / snowballing reverso: vocabulário histórico → artefatos primários
→ surveys agregadores → material didático).

Produtos desta etapa:
- Linha do tempo do Big Data (1998–2015) montada no Canva.
- Lista de citações simples e referências com link + data de acesso.
- Plano de leitura de RI clássica em 1 semana (livro de Manning + papers curtos).
- Trilha paralela de **Engenharia de Dados** (indicada pela coordenação do Grupo de
  Estudos em Engenharia de Dados): lista de leitura de infraestrutura e a conceituação
  **Fusão × Data Warehouse × Data Lake × Lakehouse** (ver §3.5 e §3.6).
- Primeiros contatos com **embeddings multimodais** via dois papers aplicados lidos na
  Semana 1 (ImageBind/manifolds e OpenFlamingo/C2C — ver §3.3).

**Feedback da banca**: tempo demais gasto em história/linha do tempo. Orientação:
avançar para materiais fundacionais **recentes** e para a parte prática.

### Etapa 2 — Fundamentos contemporâneos (concluída)
Levantamento de surveys e papers-marco recentes (2015–2026) de **Fusão de Big Data**
e **RI Multimodal**, com a técnica adaptada: surveys que definem o estado atual →
papers-marco dos últimos ~3 anos → benchmarks vivos.

Leituras selecionadas (com notas no Notion):
1. Methodologies for Cross-Domain Data Fusion: An Overview (Yu Zheng, 2015) — acessado em 03/10/2026.
2. Deep Multimodal Data Fusion (Zhao; Zhang; Geng, 2024 — ACM Computing Surveys) — acessado em 05/10/2026.
3. MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs (Lin et al., 2025 — ICLR) — acessado em 06/10/2026.

Achado analítico importante: existem **duas tradições quase disjuntas** sob o nome
"fusão de dados" — (a) data mining/bancos de dados (Zheng 2015: fusão de conhecimento
entre datasets de domínios diferentes) e (b) deep learning multimodal (Zhao et al. 2024:
fusão de modalidades dentro de um modelo). O survey de 2024 aparentemente **não cita**
Zheng (2015) (confirmação pendente via Ctrl+F no PDF); a ponte foi feita no sentido
inverso: Zheng cita Zhao et al. (2024) em seu paper de 2025
(arXiv:2506.03155, "Fusing Cross-Domain Knowledge from Multimodal Data...").
Survey-ponte entre as duas tradições: arXiv:2402.19348 (urban computing).

### Etapa 3 — Datasets e benchmarks (ATUAL)
Buscar datasets abertos e montar testes de benchmark de **processamento** e de
**modelos de embeddings**. É a etapa em que este repositório entra.

## 3. Referências levantadas

### 3.1 Big Data (etapa histórica)
1. Big Data... and the Next Wave of InfraStress (Mashey, 1999) —
   usenix.org/event/usenix99/invited_talks/mashey.pdf
2. 3D Data Management: Controlling Data Volume, Velocity, and Variety (Laney, 2001) —
   nota técnica META Group (3Vs originais); acesso via Wayback Machine
   (o link original em blogs.gartner.com morreu; META Group foi adquirida pela Gartner em 2005).
3. The Google File System (Ghemawat; Gobioff; Leung, 2003) — SOSP; DOI 10.1145/945445.945450.
4. MapReduce: Simplified Data Processing on Large Clusters (Dean; Ghemawat, 2004) — OSDI;
   versão CACM 2008: DOI 10.1145/1327452.1327492.
5. Bigtable: A Distributed Storage System for Structured Data (Chang et al., 2006) — OSDI;
   versão TOCS 2008: DOI 10.1145/1365815.1365816.
6. Beyond the Hype: Big Data Concepts, Methods, and Analytics (Gandomi; Haider, 2015) —
   consolida os 3 Vs adicionais (Veracidade/IBM, Variabilidade/SAS, Valor/Oracle);
   DOI 10.1016/j.ijinfomgt.2014.10.007.
- Contexto etimológico: Diebold (2012), "On the Origin(s) and Development of the Term
  'Big Data'"; Cox & Ellsworth (1997) = primeira ocorrência acadêmica do termo.

### 3.2 RI pura (plano de 1 semana)
- Introduction to Information Retrieval (Manning; Raghavan; Schütze, 2008) —
  capítulos usados: 1, 2, 6, 8, 11 (+ 9 e 21 na versão "só livro");
  PDFs por capítulo em nlp.stanford.edu/IR-book.
- Papers curtos: Spärck Jones (1972, IDF); Salton; Wong; Yang (1975, modelo vetorial);
  Brin; Page (1998, PageRank); Robertson; Zaragoza (2009, BM25).

### 3.3 RI Multimodal (histórico → recente)
Linhagem: QBIC (1995) → Smeulders et al. (2000, semantic gap) → TRECVID (2001+) →
Rasiwasia et al. (2010, CCA) → DeViSE (2013) → CLIP (Radford et al., 2021) →
**UniIR/M-BEIR (Wei et al., ECCV 2024)** → **MM-Embed (ICLR 2025)** → GME, MegaPairs (ACL 2025).
Surveys: Wang et al. (2016, arXiv:1607.06215); Baltrušaitis; Ahuja; Morency (2018, TPAMI);
Liang; Zadeh; Morency (2022, arXiv:2209.03430); RAG multimodal: arXiv:2502.08826
("Ask in Any Modality", ACL 2025 Findings) e arXiv:2504.08748; VisRAG (ICLR 2025).

Primeiros papers aplicados lidos (Semana 1, porta de entrada em embeddings multimodais):
- From Latent to Engine Manifolds: Analyzing ImageBind's Multimodal Embedding Space
  (Hamara; Rivas, 2024, Baylor Univ.) — arXiv:2409.10528 (CCIS 2024);
  também em rivas.ai/pdfs/hamara2024manifolds.pdf.
- Leveraging OpenFlamingo for Multimodal Embedding Analysis of C2C Car Parts Data
  (Rashid; Rivas, 2025) — arXiv:2503.17408.

### 3.4 Fusão de Big Data (recente)
- Zheng (2015) — taxonomia: stage-based, feature-level-based, semantic meaning-based;
  PDF aberto na Microsoft Research.
- Lahat; Adali; Jutten (2015) — Proceedings of the IEEE; diversidade como conceito-chave.
- Gao et al. (2020) — survey de transição para deep learning (Neural Computation).
- Zhao; Zhang; Geng (2024, ACM CSUR) — nova taxonomia fine-grained em 5 classes
  (encoder-decoder, atenção, GNN, redes generativas, constraint-based); argumenta que
  early/late fusion não basta mais.
- Multimodal Fusion on Low-quality Data (2024, arXiv:2404.18947).
- Zheng (2025, arXiv:2506.03155) — atualização da taxonomia para a era multimodal.

### 3.5 Engenharia de Dados e infraestrutura (trilha paralela, Semana 2)
Lista indicada pela coordenação do Grupo de Estudos em Engenharia de Dados; destaques
(⭐) são os priorizados para o tema de fusão em escala:
- ⭐ Designing Data-Intensive Applications (Kleppmann, 2017, O'Reilly) — fundamenta os
  problemas por trás da fusão em escala.
- ⭐ Fundamentals of Data Engineering (Reis; Housley, 2022, O'Reilly) — ciclo de vida do
  dado: ingestão de fontes heterogêneas, transformação, integração.
- ⭐ The Data Warehouse Toolkit, 3ª ed. (Kimball; Ross, 2013, Wiley) — modelagem dimensional.
- ⭐ Delta Lake: The Definitive Guide (Lee; Wentling; Haines; Babu, 2024, O'Reilly) — base
  técnica do lakehouse.
- ⭐ Building Medallion Architectures (Strengholt, 2025, O'Reilly) — camadas Bronze/Silver/Gold.
- ⭐ Understanding ETL (Palmer, 2023, O'Reilly, report) — etapa de transformação do ETL.
- Demais: Cloud Data Engineering for Dummies (Baum, 2020); Data Engineering Design Patterns
  (Konieczny, 2025); Data Pipelines with Apache Airflow (Harenslak; de Ruiter, 2021);
  Learning Spark 2ª ed. (Damji et al., 2020); Snowflake: The Definitive Guide (Avila, 2022);
  Spark: The Definitive Guide (Chambers; Zaharia, 2018); Python for Data Analysis 3ª ed.
  (McKinney, 2022); Storytelling com Dados (Knaflic, 2015/2017).
- Cursos de Big Data (Semana 1): Big Data Specialization (Coursera) — escolhido pela maior
  procura e avaliações.

### 3.6 Conceitos: Fusão × Data Warehouse × Data Lake × Lakehouse
Distinção levantada para a banca (produto da Semana 2):
- **Fusão de dados**: integra dados de múltiplas fontes em acesso unificado, melhorando a
  qualidade da informação. Organiza-se em camadas (fonte → transformação → fusão →
  apresentação); envolve padronização, deduplicação de entidades e resolução de conflitos
  entre valores divergentes. Em escala de big data lida com volume, variedade e fluxos em
  tempo real (DREMIO, 2024).
- **Data Warehouse**: repositório para análise/decisão; **schema-on-write** (o dado chega
  limpo, transformado e modelado); garante consistência e desempenho sobre dados
  estruturados (AWS, 2026).
- **Data Lake**: armazena dados em formato bruto/nativo em object storage de baixo custo;
  **schema-on-read** (o esquema só é aplicado na leitura); nasceu da limitação do DW com
  dados não estruturados e cargas de ML; risco de virar *data swamp* por menos governança
  (DATABRICKS, 2026).
- **Lakehouse**: combina os dois — estruturas e gestão de DW sobre o storage barato do
  lake; elimina a arquitetura de duas camadas e estabelece fonte única da verdade;
  organizado em camadas (padrão medalhão Bronze/Silver/Gold), e **é nessa progressão que
  ocorre a fusão de fontes** (DATABRICKS, 2026; MICROSOFT, 2026).
- **Engenharia de Dados**: projeta e opera DW/lake/lakehouse e implementa os processos de
  integração e fusão; materializa-se em pipelines (ingestão → transformação →
  disponibilização) com observabilidade (IBM, 2026).

Referências web (ABNT, acesso em 16 set. 2026): AWS, *What is a Data Warehouse?* (2026),
aws.amazon.com/what-is/data-warehouse/; DATABRICKS, *Introduction to Data Lakes* /
*What is a Data Lakehouse?* (2026), databricks.com; DREMIO, *What is Data Fusion?* (2024),
dremio.com/wiki/data-fusion/; MICROSOFT, *What is a data lakehouse?* (2026),
learn.microsoft.com/azure/databricks/lakehouse/; IBM, *What Is Data Engineering?* (2026),
ibm.com/think/topics/data-engineering.

## 4. Datasets e benchmarks escolhidos (etapa atual)

### Benchmark central
- **M-BEIR** (Multimodal BEnchmark for Instructed Retrieval) —
  huggingface.co/datasets/TIGER-Lab/M-BEIR
  - 8 tarefas, 10 datasets, 1,5M queries com instruções, pool de 5,6M candidatos.
  - Imagens: ~170 GB em 4 partes tar.gz (dimensionar disco antes do download).
  - Dois settings: **global** (pool de 5,6M) e **local** (pool por tarefa; usar primeiro).
  - Pipeline de avaliação pronto: github.com/TIGER-AI-Lab/UniIR
    (configurar MBEIR_DATA_DIR nos scripts).
  - Checkpoints abertos dos retrievers UniIR: huggingface.co/TIGER-Lab/UniIR.

### Avaliação de embeddings
- **MTEB** (texto) e **MIEB** (imagem/imagem-texto, ICCV 2025; 130 tarefas, 38 línguas) —
  mesma biblioteca: github.com/embeddings-benchmark/mteb;
  leaderboard: huggingface.co/spaces/mteb/leaderboard.
  - Usar **MIEB-lite** (51 tarefas, baixa correlação entre si) para custo menor.
- **BEIR** (texto puro, baseline) — github.com/beir-cellar/beir.

### Datasets individuais (subconjuntos do M-BEIR, para começar leve)
MS-COCO, Flickr30k, Fashion200k, FashionIQ, CIRR, WebQA, InfoSeek/OVEN.
Primeiro experimento sugerido: MSCOCO + Fashion200k no setting local.

### Documentos visuais (linha VisRAG)
- **ViDoRe** (Visual Document Retrieval / ColPali) — organização `vidore` no HuggingFace
  (confirmar nome exato do leaderboard antes de citar).

### Plano de testes (ordem de custo crescente)
1. BEIR/scifact com 2–3 embeddings de texto → validar o pipeline.
2. MIEB-lite: CLIP vs. SigLIP vs. retriever MLLM (checkpoints UniIR).
3. M-BEIR local (MSCOCO + Fashion200k), métrica **Recall@5** (padrão do M-BEIR).
Stack prevista: Python, FAISS, biblioteca `mteb`, scripts do repositório UniIR; GPU única.

## 5. Metodologia de busca (documentada para a banca)

- **Etapa histórica**: busca em camadas / snowballing reverso —
  (1) vocabulário histórico (quem cunhou o termo), (2) artefatos primários nas
  conferências de origem (OSDI, SOSP, NSDI, VLDB, SIGMOD; ACM MM, SIGIR, ICMR),
  (3) surveys agregadores, (4) material didático de universidades (Stanford, MIT, CMU).
  Atenção à mudança de vocabulário por época (ex.: CBIR → multimedia IR → cross-modal
  → multimodal retrieval): buscar com os termos de cada era.
- **Etapa recente**: survey mining (termo + "survey" + janela 2023–2026),
  benchmark tracing (achar o benchmark dominante e seguir quem publica nele),
  repositórios vivos (awesome lists / survey repos no GitHub),
  "Citado por" no Google Scholar com filtro de ano.
- **Etapa de datasets**: benchmark-first — seção de experimentos dos papers lidos →
  HuggingFace Datasets (fonte canônica) → repositório com pipeline de avaliação →
  datasets individuais.

## 6. Convenções de trabalho

- **Citação simples** (para listas e slides): `Título (Autor, ano)`.
  Multiautor: `(Sobrenome; Sobrenome, ano)`; 4+ autores: `(Sobrenome et al., ano)`.
- **Referência completa**: ABNT, com "Disponível em: URL. Acesso em: DD mês. AAAA."
  quando a fonte é exclusivamente online; preferir DOI quando existir;
  para links mortos, usar Wayback Machine com o timestamp do snapshot.
- **Notas de leitura**: mantidas no Notion (uma página por paper, padrão
  "Título (Autor, ano)" no nome da página).
- **Idioma**: entregas da disciplina em português; papers e código em inglês.
- **Apresentações**: banca periódica; evitar excesso de contexto histórico
  (feedback já recebido); priorizar análise própria e resultados práticos.

## 7. Próximos passos

1. Montar neste repositório o pipeline de avaliação (download M-BEIR local,
   FAISS + mteb, scripts UniIR adaptados).
2. Rodar o plano de testes da seção 4 e registrar resultados (Recall@5, nDCG@10)
   em tabelas/CSV versionados.
3. Confirmar no PDF se Zhao et al. (2024) cita Zheng (2015) e registrar o achado
   das duas tradições de fusão na próxima apresentação.
4. Levantamentos ainda pendentes das subáreas: NLP e Fusão de Big Data "clássica"
   (se a banca pedir), seguindo a mesma metodologia.