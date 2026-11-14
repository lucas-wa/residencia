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
    ALLOW_ADAPTER_MISMATCH=1  serve com adaptador != o do indice (ver abaixo)

O startup FALHA quando o adaptador que embeddou os ITENS (carimbado no `index_meta.json`
pelo `api/build_index.py`) nao e o `ADAPTER_PATH` que vai embeddar as QUERIES: espacos
diferentes degradam o ranking em silencio, entao a falha tem de ser fechada. Indice
legado (sem carimbo) so avisa, e `ALLOW_ADAPTER_MISMATCH=1` libera a divergencia
deliberada (servir a base crua de proposito).
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

from api.index_io import META_NAME, load_index
from api.retriever import ConsultaRetriever

log = logging.getLogger(__name__)

DEFAULT_K = 10
MAX_K = 100
DEFAULT_DEVICE = "cuda"
ALLOW_MISMATCH_ENV = "ALLOW_ADAPTER_MISMATCH"


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


def _adapter_normalizado(valor: str | None) -> str | None:
    """Identidade do adaptador: o caminho, sem barra final e sem vazio-como-valor."""
    if valor is None:
        return None
    texto = str(valor).strip()
    return os.path.normpath(texto) if texto else None


def check_adapter_do_indice(meta: dict | None, adapter_path: str | None, *,
                            permitir_divergencia: bool = False) -> None:
    """Exige que os ITENS do indice e as QUERIES do serving usem o MESMO adaptador.

    Funcao pura (por isso testavel sem modelo): recebe o `index_meta.json` ja lido e o
    `ADAPTER_PATH` do processo. Levanta `RuntimeError` na divergencia -- o startup morre
    e o `/consulta` responde 503, que e melhor que servir um ranking degradado em
    silencio. `meta=None` (indice legado, sem carimbo) so avisa, porque nao ha o que
    comparar; `permitir_divergencia` (o env `ALLOW_ADAPTER_MISMATCH=1`) tambem so avisa.
    """
    if meta is None:
        log.warning("indice sem %s: nao da para conferir se o adaptador do indice e o "
                    "do serving (%s). Reconstrua com `python -m api.build_index` para "
                    "carimbar a proveniencia.", META_NAME, adapter_path or "nenhum")
        return

    do_indice = _adapter_normalizado(meta.get("adapter_path"))
    do_serving = _adapter_normalizado(adapter_path)
    if do_indice == do_serving:
        return

    msg = (f"adaptador divergente: os itens do indice foram embeddados com "
           f"{meta.get('adapter_path') or 'nenhum (base crua)'} e as queries seriam "
           f"embeddadas com {adapter_path or 'nenhum (base crua)'} -- espacos diferentes, "
           f"ranking degradado em silencio. Reconstrua o indice com o mesmo --adapter "
           f"(`python -m api.build_index`) ou defina {ALLOW_MISMATCH_ENV}=1 para servir "
           f"assim de proposito.")
    if permitir_divergencia:
        log.warning("%s=1: %s", ALLOW_MISMATCH_ENV, msg)
        return
    raise RuntimeError(msg)


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

    records, matrix, meta = load_index(index_path, with_meta=True)
    check_adapter_do_indice(
        meta, adapter_path,
        permitir_divergencia=os.environ.get(ALLOW_MISMATCH_ENV, "").strip() == "1")

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
