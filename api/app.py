"""Servico HTTP de consulta (subsistema [4]) -- serve SO o baseline A finetunado.

Contrato (fechado com a GovExe):

    GET  /health                          -> 200 {"status": "ok"}            (sem auth)
    POST /consulta  X-API-Key: <chave>
         {"query": str, "k": int = 10}    -> 200 {"results": [{item_id, eixo, score,
                                                  pointer_uri, descriptor}], latency_ms}
         k fora de [1, 100] ou query vazia -> 422
         chave ausente/errada              -> 401
         servidor sem `API_KEY`            -> 503 (recusa segura: nao serve sem chave)
         indice ainda nao carregado        -> 503

O modelo pesado e o GCS entram INJETADOS: `create_app(retriever=...)` recebe um
`ConsultaRetriever` pronto (e assim que os testes rodam sem GPU e sem bucket). Sem
retriever, o app carrega o de producao no EVENTO DE STARTUP -- nunca no import, para
`import api.app` (e o build do container) nao baixar o Qwen3-VL.

Producao (Cloud Run, `min-instances=0`):

    INDEX_PATH    diretorio do indice, local ou gs://bucket/prefixo  (obrigatorio)
    ADAPTER_PATH  diretorio do adaptador LoRA da Semana 10           (opcional)
    API_KEY       chave esperada no header X-API-Key                 (obrigatorio)
    DEVICE        device do embedder (default "cuda")
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from api.index_io import load_index
from api.retriever import ConsultaRetriever

log = logging.getLogger(__name__)

DEFAULT_K = 10
MAX_K = 100
DEFAULT_DEVICE = "cuda"


class ConsultaRequest(BaseModel):
    query: str = Field(..., min_length=1, description="consulta em texto livre (pt-BR)")
    k: int = Field(DEFAULT_K, ge=1, le=MAX_K, description="quantos itens devolver")

    @field_validator("query")
    @classmethod
    def _nao_vazia(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query vazia")
        return v


class ResultItem(BaseModel):
    item_id: str
    eixo: str
    score: float
    pointer_uri: str
    descriptor: str


class ConsultaResponse(BaseModel):
    results: list[ResultItem]
    latency_ms: float


def load_production_retriever(*, index_path: str | None = None,
                              adapter_path: str | None = None,
                              device: str | None = None) -> ConsultaRetriever:
    """Monta o retriever real: indice de `INDEX_PATH` + Qwen3-VL (+ adaptador) na GPU.

    NAO roda nos testes -- e o caminho de deploy, chamado so pelo startup. Os imports
    pesados (`benchmark.models.qwen3vl_embed`) ficam dentro da funcao para o modulo
    continuar importavel num processo sem torch.
    """
    index_path = index_path or os.environ.get("INDEX_PATH")
    if not index_path:
        raise RuntimeError("INDEX_PATH nao configurado: sem indice nao ha o que servir")
    adapter_path = adapter_path if adapter_path is not None else os.environ.get("ADAPTER_PATH")
    device = device or os.environ.get("DEVICE", DEFAULT_DEVICE)

    records, matrix = load_index(index_path)

    from benchmark.models.qwen3vl_embed import get_embedder
    from ensemble.native.embedders import UNIFIED_A_MODEL

    log.info("carregando %s%s em %s", UNIFIED_A_MODEL,
             f" + adaptador {adapter_path}" if adapter_path else "", device)
    emb = get_embedder(UNIFIED_A_MODEL, device=device, adapter_path=adapter_path)
    return ConsultaRetriever(records, matrix, embed_query=lambda qs: emb.embed_texts(qs))


def create_app(retriever: ConsultaRetriever | None = None, *,
               api_key_env: str = "API_KEY") -> FastAPI:
    """Factory do app. `retriever` injetado pula a carga de producao (testes/dev)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.retriever is None:
            app.state.retriever = load_production_retriever()
        yield

    app = FastAPI(title="API de consulta GovExe (Qwen3-VL finetunado)", lifespan=lifespan)
    app.state.retriever = retriever

    def exigir_chave(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
        esperada = os.environ.get(api_key_env)
        if not esperada:
            raise HTTPException(
                status_code=503,
                detail=f"servidor sem {api_key_env} configurada; consulta indisponivel")
        if not x_api_key or not secrets.compare_digest(x_api_key, esperada):
            raise HTTPException(status_code=401, detail="X-API-Key ausente ou invalida")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/consulta", response_model=ConsultaResponse,
              dependencies=[Depends(exigir_chave)])
    def consulta(req: ConsultaRequest, request: Request) -> ConsultaResponse:
        r: ConsultaRetriever | None = request.app.state.retriever
        if r is None:
            raise HTTPException(status_code=503, detail="indice ainda nao carregado")
        inicio = time.perf_counter()
        results = r.search(req.query, req.k)
        return ConsultaResponse(
            results=[ResultItem(**item) for item in results],
            latency_ms=(time.perf_counter() - inicio) * 1000.0,
        )

    return app


# Entrypoint do uvicorn (`uvicorn api.app:app`): construir o app NAO carrega modelo --
# o indice e o Qwen3-VL entram no lifespan, ja com o servidor de pe.
app = create_app()
