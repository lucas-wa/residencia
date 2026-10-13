# Fontes recentes — Semana 6 (embasamento das decisões técnicas)

> Entregável 4. Levantamento para sustentar as escolhas do pipeline de benchmark e do
> doc de arquitetura, respeitando as bases já estabelecidas (README §3). Convenção:
> `✔ verificado` = título/autores/data conferidos em 13 out. 2026 nesta sessão;
> `⚠ conferir` = âncora conhecida/preprint pós-corte ainda a validar antes da banca.
> Acesso em: 13 out. 2026, salvo indicação.

## 1. RI de documento visual (centro de gravidade do benchmark)

- ✔ **ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval.** Macé, Quentin;
  Loison, António; Faysse, Manuel, 2025. arXiv:2505.17166 (v1 22 maio 2025; v2 19 set.
  2025). Sucessor multilíngue e não-saturado do ViDoRe v1; métrica nDCG@5.
  Disponível em: https://arxiv.org/abs/2505.17166.
- ⚠ **ColPali: Efficient Document Retrieval with Vision Language Models.** Faysse et al.,
  ICLR 2025. Modelo de late interaction sobre páginas-imagem; introduziu o ViDoRe v1.
  Disponível em: https://arxiv.org/abs/2407.01449 (conferir).
- ⚠ Família de modelos **ColQwen2 / ColPali / ColSmolVLM** — coleção `vidore` no
  Hugging Face (confirmar ids/revisões exatos em runtime).
  Disponível em: https://huggingface.co/vidore.

## 2. Modelos e benchmarks de embeddings multimodais

- ✔ **MIEB: Massive Image Embedding Benchmark.** Xiao, Chenghao; Chung, Isaac; Kerboua,
  Imene; Stirling, Jamie; Zhang, Xin; Kardos, Márton; Solomatin, Roman; Al Moubayed,
  Noura; Enevoldsen, Kenneth; Muennighoff, Niklas, 2025 (ICCV 2025). arXiv:2504.10471
  (14 abr. 2025). 130+ tarefas, dezenas de línguas; mesma lib do MTEB.
  Disponível em: https://arxiv.org/abs/2504.10471.
- ✔ **VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual
  Documents.** Meng, Rui et al. (13 autores), 2025. arXiv:2507.04590 (7 jul. 2025).
  Modelo único p/ texto/imagem/vídeo/documento + benchmark **MMEB-V2** (5 novas tarefas).
  Disponível em: https://arxiv.org/abs/2507.04590.
- ⚠ **VLM2Vec / MMEB** (versão original). Jiang et al., ICLR 2025. Framework de treino de
  embedding multimodal sobre MMEB. Disponível em: https://github.com/TIGER-AI-Lab/VLM2Vec.
- ⚠ **MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs.** Lin et al., ICLR
  2025 (já no README §2). Disponível em: https://arxiv.org/abs/2411.02571 (conferir).
- ⚠ **GME-Qwen2-VL** (Alibaba-NLP) — embedder multimodal MLLM.
  Disponível em: https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct.
  > Observação: GME/VLM2Vec ficam como **literatura** nesta semana (fora do escopo de
  > execução, por decisão de enxugamento).

## 3. Português (pt-BR) — adaptação

- ✔ **Beyond Multilingual Averages: MTEB-PT, a Benchmark for Portuguese Sentence
  Encoders.** Okamura, Lucas Hideki Takeuchi; Alcoforado, Alexandre; Reali Costa, Anna
  Helena, 2026. arXiv:2607.04071 (v1 5 jul. 2026; v2 18 jul. 2026). Benchmark pt de 14
  datasets (de MMTEB), 17 modelos; rankings multilíngues predizem mal o desempenho em pt.
  Disponível em: https://arxiv.org/abs/2607.04071.
- ⚠ **MTEB-BR / MTEB-PT (Brazilian Portuguese).** Stekel (IFSP), 2026. arXiv:2607.04581 —
  22 tarefas nativas em pt-BR, 93 modelos. (Trabalho distinto do anterior; conferir.)
  Disponível em: https://arxiv.org/abs/2607.04581.
- ✔ **Open Sentence Embeddings for Portuguese with the Serafim PT\* encoders family.**
  Gomes, Luís; Branco, António; Silva, João; Rodrigues, João; Santos, Rodrigo, 2024.
  arXiv:2407.19527 (28 jul. 2024). Encoders pt abertos em três tamanhos (100M/335M/900M).
  Disponível em: https://arxiv.org/abs/2407.19527.
- Modelos candidatos de texto pt-BR p/ o pipeline (abertos): `BAAI/bge-m3`,
  `intfloat/multilingual-e5-large`, `Alibaba-NLP/gte-multilingual-base`, Serafim-900m.

## 4. Fusão de dados (taxonomias — base já estabelecida, README §3.4)

- Zheng, Yu. **Methodologies for Cross-Domain Data Fusion: An Overview.** 2015.
- Zhao; Zhang; Geng. **Deep Multimodal Data Fusion.** 2024 (ACM Computing Surveys).
- ⚠ Zheng. **Fusing Cross-Domain Knowledge from Multimodal Data.** 2025.
  arXiv:2506.03155 (ponte entre as duas tradições; conferir).

## 5. Infraestrutura / lakehouse (trilha de Big Data, README §3.6)

- DREMIO (2024); AWS, DATABRICKS, MICROSOFT, IBM (2026) — ver referências ABNT no
  README §3.6 (Fusão × DW × Data Lake × Lakehouse; padrão medalhão).

## 6. Primeiros contatos com embeddings multimodais (Semana 1)

- ✔ **From Latent to Engine Manifolds: Analyzing ImageBind's Multimodal Embedding
  Space.** Hamara, Andrew; Rivas, Pablo (Baylor Univ.), 2024. arXiv:2409.10528 (30 ago.
  2024; CCIS 2024). Fusão de embeddings ImageBind p/ anúncios de autopeças.
  Disponível em: https://arxiv.org/abs/2409.10528.
- ✔ **Leveraging OpenFlamingo for Multimodal Embedding Analysis of C2C Car Parts Data.**
  Rashid, Maisha Binte; Rivas, Pablo, 2025. arXiv:2503.17408 (20 mar. 2025). Embeddings
  texto+imagem de 1,2M anúncios; clustering k-means. Mesmo grupo (Rivas/Baylor).
  Disponível em: https://arxiv.org/abs/2503.17408.

---

### Notas de verificação
- `✔` = título, autoria e data conferidos via arXiv nesta sessão (13 out. 2026).
- `⚠` = confirmar venue/URL/id antes de citar em entrega formal da banca. Em especial os
  ids arXiv de ColPali (2407.01449), MM-Embed (2411.02571) e Zheng 2025 (2506.03155), e
  os ids/revisões exatos dos modelos no Hugging Face.
- Preprints de 2026 (MTEB-PT/MTEB-BR) podem ter rankings sujeitos a revisão; tratar
  números como referência, não como placar fechado.
