# Imagem do servico de consulta [4] -- Cloud Run, min-instances=0 (cold start aceito).
#
# Build a partir da RAIZ do repo (o contexto precisa de `api/`, `ensemble/`, `benchmark/`):
#     docker build -f api/Dockerfile -t consulta-govexe .
#
# Em Cloud Run, configure:
#     INDEX_PATH=gs://<bucket>/<prefixo>   indice escrito por `python -m api.build_index`
#     ADAPTER_PATH=/opt/adapter            adaptador LoRA (ou um caminho no GCS montado)
#     API_KEY=<segredo>                    chave esperada no header X-API-Key
#     DEVICE=cuda                          (cpu em instancia sem GPU)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# So o que o servico importa: a API, o nucleo do ensemble (records/metrica/rotas) e o
# wrapper do Qwen3-VL. Dados e indice NAO entram na imagem -- vem do GCS no startup.
COPY api/ /app/api/
COPY ensemble/ /app/ensemble/
COPY benchmark/ /app/benchmark/

EXPOSE 8080

# `api.app:app` constroi o app sem baixar modelo; o indice e o Qwen3-VL sobem no evento
# de startup do FastAPI (lifespan), ja dentro do container em execucao.
CMD exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}
