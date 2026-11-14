"""API de consulta [4]: contrato HTTP, nucleo puro e carga do indice -- SEM modelo/GCS.

O retriever entra INJETADO (records fake + matriz pequena + `embed_query` duble), que e
como estes testes exercem o contrato inteiro sem baixar o Qwen3-VL nem tocar no bucket.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.index_io import load_index
from api.retriever import ConsultaRetriever
from ensemble.records import PointerRecord, save_records

CHAVE = "chave-de-teste"

VOCAB = ["orçamento", "saúde", "educação", "praia"]


def _bow(texts: list[str]) -> np.ndarray:
    return np.array(
        [[float(w in t.lower().split()) for w in VOCAB] for t in texts], dtype=np.float32
    )


def _records() -> list[PointerRecord]:
    return [
        PointerRecord("doc1", "documento", "Orçamento da saúde 2026",
                      "https://exemplo.gov.br/doc1.pdf", "camara", {}),
        PointerRecord("aud1", "audio", "Audiência sobre orçamento",
                      "https://exemplo.gov.br/aud1.mp3", "senado", {}),
        PointerRecord("tex1", "texto", "Nota sobre educação",
                      "https://exemplo.gov.br/tex1.html", "dou", {}),
        PointerRecord("img1", "imagem", "Foto da praia",
                      "https://exemplo.gov.br/img1.jpg", "rss", {}),
    ]


def _matrix() -> np.ndarray:
    """Linha `i` = embedding-A do record `i` (mesma ordem), no espaco do `_bow`."""
    return np.array(
        [
            [1.0, 1.0, 0.0, 0.0],  # doc1: orçamento + saúde
            [1.0, 0.0, 0.0, 0.0],  # aud1: orçamento
            [0.0, 0.0, 1.0, 0.0],  # tex1: educação
            [0.0, 0.0, 0.0, 1.0],  # img1: praia
        ],
        dtype=np.float32,
    )


def _retriever() -> ConsultaRetriever:
    return ConsultaRetriever(_records(), _matrix(), embed_query=_bow)


def _client(monkeypatch, *, chave: str | None = CHAVE) -> TestClient:
    if chave is None:
        monkeypatch.delenv("API_KEY", raising=False)
    else:
        monkeypatch.setenv("API_KEY", chave)
    return TestClient(create_app(_retriever()))


# --- 5. nucleo puro: ConsultaRetriever.search -------------------------------------


def test_search_ranqueia_por_cosine_contra_matriz_fixa():
    r = _retriever()
    res = r.search("orçamento saúde", 4)
    assert [x["item_id"] for x in res][:2] == ["doc1", "aud1"]
    scores = [x["score"] for x in res]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.7071, abs=1e-3)


def test_search_devolve_os_campos_do_contrato_com_eixo_igual_a_modality():
    item = _retriever().search("praia", 1)[0]
    assert set(item) == {"item_id", "eixo", "score", "pointer_uri", "descriptor"}
    assert item["item_id"] == "img1"
    assert item["eixo"] == "imagem"
    assert item["pointer_uri"] == "https://exemplo.gov.br/img1.jpg"
    assert item["descriptor"] == "Foto da praia"
    assert isinstance(item["score"], float)


def test_search_respeita_k_e_nao_estoura_o_corpus():
    r = _retriever()
    assert len(r.search("orçamento", 1)) == 1
    assert len(r.search("orçamento", 3)) == 3
    assert len(r.search("orçamento", 99)) == 4  # k > n -> n


def test_search_embeda_a_query_uma_vez_por_chamada():
    chamadas: list[list[str]] = []

    def espiao(qs):
        chamadas.append(list(qs))
        return _bow(qs)

    r = ConsultaRetriever(_records(), _matrix(), embed_query=espiao)
    r.search("orçamento", 2)
    assert chamadas == [["orçamento"]]


def test_retriever_recusa_matriz_com_n_diferente_dos_records():
    with pytest.raises(ValueError, match="4 records"):
        ConsultaRetriever(_records(), _matrix()[:2], embed_query=_bow)


def test_retriever_vazio_devolve_lista_vazia():
    r = ConsultaRetriever([], np.zeros((0, 4), dtype=np.float32), embed_query=_bow)
    assert r.search("orçamento", 5) == []


# --- 1. /health --------------------------------------------------------------------


def test_health_responde_ok_sem_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    c = TestClient(create_app(_retriever()))
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- 2. /consulta feliz ------------------------------------------------------------


def test_consulta_devolve_o_schema_exato_do_contrato(monkeypatch):
    c = _client(monkeypatch)
    resp = c.post("/consulta", json={"query": "orçamento saúde", "k": 2},
                  headers={"X-API-Key": CHAVE})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"results", "latency_ms"}
    assert isinstance(body["latency_ms"], float) and body["latency_ms"] >= 0.0
    assert [r["item_id"] for r in body["results"]] == ["doc1", "aud1"]
    assert [r["eixo"] for r in body["results"]] == ["documento", "audio"]
    for r in body["results"]:
        assert set(r) == {"item_id", "eixo", "score", "pointer_uri", "descriptor"}
    assert body["results"][0]["pointer_uri"] == "https://exemplo.gov.br/doc1.pdf"
    assert body["results"][0]["descriptor"] == "Orçamento da saúde 2026"


def test_consulta_respeita_k_e_o_default_e_10(monkeypatch):
    c = _client(monkeypatch)
    um = c.post("/consulta", json={"query": "orçamento", "k": 1},
                headers={"X-API-Key": CHAVE})
    assert len(um.json()["results"]) == 1

    sem_k = c.post("/consulta", json={"query": "orçamento"}, headers={"X-API-Key": CHAVE})
    assert sem_k.status_code == 200
    assert len(sem_k.json()["results"]) == 4  # default 10, corpus tem 4


# --- 3. auth: 401 / 503 ------------------------------------------------------------


def test_consulta_sem_chave_e_401(monkeypatch):
    c = _client(monkeypatch)
    resp = c.post("/consulta", json={"query": "orçamento"})
    assert resp.status_code == 401


def test_consulta_com_chave_errada_e_401(monkeypatch):
    c = _client(monkeypatch)
    resp = c.post("/consulta", json={"query": "orçamento"},
                  headers={"X-API-Key": "chave-errada"})
    assert resp.status_code == 401


def test_consulta_sem_api_key_no_servidor_e_503(monkeypatch):
    """Recusa segura: servidor sem `API_KEY` configurada nao serve consulta nenhuma."""
    c = _client(monkeypatch, chave=None)
    resp = c.post("/consulta", json={"query": "orçamento"}, headers={"X-API-Key": CHAVE})
    assert resp.status_code == 503


def test_api_key_env_configuravel(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("CHAVE_GOVEXE", "outra")
    c = TestClient(create_app(_retriever(), api_key_env="CHAVE_GOVEXE"))
    assert c.post("/consulta", json={"query": "orçamento"},
                  headers={"X-API-Key": "outra"}).status_code == 200
    assert c.post("/consulta", json={"query": "orçamento"},
                  headers={"X-API-Key": CHAVE}).status_code == 401


# --- 4. validacao: 422 -------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1, 101])
def test_k_fora_de_1_100_e_422(monkeypatch, k):
    c = _client(monkeypatch)
    resp = c.post("/consulta", json={"query": "orçamento", "k": k},
                  headers={"X-API-Key": CHAVE})
    assert resp.status_code == 422


@pytest.mark.parametrize("query", ["", "   "])
def test_query_vazia_e_422(monkeypatch, query):
    c = _client(monkeypatch)
    resp = c.post("/consulta", json={"query": query, "k": 5},
                  headers={"X-API-Key": CHAVE})
    assert resp.status_code == 422


def test_k_no_limite_e_aceito(monkeypatch):
    c = _client(monkeypatch)
    for k in (1, 100):
        resp = c.post("/consulta", json={"query": "orçamento", "k": k},
                      headers={"X-API-Key": CHAVE})
        assert resp.status_code == 200


# --- app de producao: importavel/construivel sem baixar modelo ---------------------


def test_create_app_sem_retriever_nao_carrega_modelo(monkeypatch):
    """A carga real mora no startup event; construir o app nao pode tocar no modelo."""
    monkeypatch.delenv("INDEX_PATH", raising=False)
    app = create_app()
    assert app.state.retriever is None
    assert TestClient(app).get("/health").status_code == 200


def test_consulta_sem_indice_carregado_e_503(monkeypatch):
    monkeypatch.setenv("API_KEY", CHAVE)
    c = TestClient(create_app())
    resp = c.post("/consulta", json={"query": "orçamento"}, headers={"X-API-Key": CHAVE})
    assert resp.status_code == 503


# --- 6. index_io.load_index (local) ------------------------------------------------


def _escrever_indice(dir_, records, matrix):
    dir_.mkdir(parents=True, exist_ok=True)
    save_records(records, dir_ / "records.jsonl")
    np.save(dir_ / "matrix.npy", matrix)
    return dir_


def test_load_index_local_le_records_e_matriz(tmp_path):
    d = _escrever_indice(tmp_path / "idx", _records(), _matrix())
    records, matrix = load_index(str(d))
    assert [r.id for r in records] == ["doc1", "aud1", "tex1", "img1"]
    assert [r.modality for r in records] == ["documento", "audio", "texto", "imagem"]
    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(matrix, _matrix())


def test_load_index_alimenta_o_retriever(tmp_path):
    d = _escrever_indice(tmp_path / "idx", _records(), _matrix())
    r = ConsultaRetriever(*load_index(str(d)), embed_query=_bow)
    assert r.search("praia", 1)[0]["item_id"] == "img1"


def test_load_index_recusa_n_divergente(tmp_path):
    d = _escrever_indice(tmp_path / "idx", _records(), _matrix()[:2])
    with pytest.raises(ValueError, match="4 records"):
        load_index(str(d))


def test_load_index_erro_claro_sem_arquivo(tmp_path):
    d = tmp_path / "vazio"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="records.jsonl"):
        load_index(str(d))
