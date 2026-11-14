# Imagem do Cloud Run Job do ciclo [3] -- cron -> Spot L4 -> indice no GCS -> nada de pe.
#
# Build a partir da RAIZ do repo (o contexto precisa de `gcp/`, `ensemble/` e os
# requirements que o bootstrap da VM instala):
#
#     docker build -f gcp/serving/Dockerfile -t gcr.io/<projeto>/ciclo-captura:latest .
#     docker push gcr.io/<projeto>/ciclo-captura:latest
#
# O job roda `bash gcp/serving/trigger_cycle.sh` (e o `--command`/`--args` que
# `gcp/serving/create_scheduler.sh job` configura), e esse script precisa, DENTRO do
# container, de:
#
#   * gcloud -- `compute instances create/delete`, `compute ssh/scp` e `storage cp` sao
#     chamados pelo `gcp/slice/run_slice.sh`; dai a base ser a do SDK e nao python:slim.
#   * openssh-client -- `gcloud compute ssh/scp` delega ao ssh/scp do sistema e gera a
#     chave com `ssh-keygen`; a base `:slim` nao os traz.
#   * python3 + google-cloud-storage -- etapa [2/3], `python -m gcp.serving.upsert_index`
#     com `--index gs://...` (o import do pacote e feito DENTRO da funcao, mas no ciclo
#     real o caminho E gs://, entao ele e exigido).
#   * tar/gzip -- o `run_slice` empacota o repo para a VM e a etapa [3/3] sobe os
#     binarios retidos num tar por ciclo.
#
# O que a imagem NAO precisa: GPU, torch, transformers, modelo. A GPU e a Spot L4 que o
# `run_slice.sh` sobe e derruba; este container so orquestra e faz IO no GCS. Por isso
# tambem nao ha `requirements-gpu.txt` aqui -- quem instala as deps pesadas e o bootstrap
# da VM, com o `requirements-slurm.txt` que viaja no tarball do repo.
FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HOME=/root

RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client tar gzip \
    && rm -rf /var/lib/apt/lists/*

# `--break-system-packages` so existe a partir do pip 23.1 (Debian 12, que marca o python
# do sistema como externally-managed). O fallback cobre as bases mais antigas da imagem do
# SDK, onde a flag nao existe e o install direto funciona.
RUN pip3 install --no-cache-dir google-cloud-storage \
    || pip3 install --no-cache-dir --break-system-packages google-cloud-storage

WORKDIR /app

# O `--args` do Cloud Run Job e um caminho RELATIVO (`gcp/serving/trigger_cycle.sh`),
# entao a arvore tem de ficar na WORKDIR, com os mesmos caminhos do repo.
#
# O que entra: os scripts e o codigo que o ciclo roda aqui dentro (`gcp/`, e `ensemble/`
# porque o `upsert_index` importa `ensemble.records`) mais o que viaja no tarball que o
# `run_slice` manda para a VM (`requirements-slurm.txt` e o codigo que a captura usa la).
# `data/`, `results/` e `logs/` NAO entram: o ciclo os cria do zero a cada execucao.
COPY requirements.txt requirements-slurm.txt /app/
COPY gcp/ /app/gcp/
COPY ensemble/ /app/ensemble/
COPY api/ /app/api/
COPY benchmark/ /app/benchmark/
COPY finetune/ /app/finetune/

# Default coerente com o `--command=bash --args=gcp/serving/trigger_cycle.sh` do
# `create_scheduler.sh`: rodar o container sem override faz exatamente UM ciclo.
ENTRYPOINT ["bash"]
CMD ["gcp/serving/trigger_cycle.sh"]
