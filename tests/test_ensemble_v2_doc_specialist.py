"""DocSpecialist: adapter do ranker late-interaction (ColQwen2.5 MaxSim) ao protocolo."""
import numpy as np

from ensemble.native.doc import DocSpecialist
from ensemble.native.specialist import Specialist
from ensemble.records import PointerRecord


class FakeColRetriever:
    """Duble de ColVisionRetriever: 1 token one-hot por pagina; ordem da consulta conhecida."""

    def __init__(self, ordem: dict[str, list[str]]):
        self.ordem = ordem
        self.corpus_paths: list[str] = []
        self._ids: list[str] = []

    def encode_corpus(self, docs):
        self.corpus_paths = [d.image_path for d in docs]
        self._ids = [d.id for d in docs]
        n = len(docs)
        return [np.eye(n, dtype="float32")[i][None, :] for i in range(n)]

    def encode_queries(self, queries):
        n = len(self._ids)
        saida = []
        for q in queries:
            desejada = self.ordem[q.text]
            tokens = np.zeros((len(desejada), n), dtype="float32")
            for pos, item_id in enumerate(desejada):
                tokens[pos, self._ids.index(item_id)] = float(len(desejada) - pos)
            saida.append(tokens)
        return saida


def _rec(id_, modality, path=None):
    meta = {"path": path} if path else {}
    return PointerRecord(id=id_, modality=modality, descriptor=f"desc {id_}",
                         pointer_uri=f"gs://b/{id_}", source="t", meta=meta)


def test_doc_ranqueia_paginas():
    recs = [_rec(f"p0{i}", "documento", f"pages/p0{i}.png") for i in range(3)]
    fake = FakeColRetriever({"saude": ["p02", "p00", "p01"]})
    spec = DocSpecialist(recs, ranker=fake)

    assert isinstance(spec, Specialist) and spec.modality == "documento"
    res = spec.search("saude", 2)
    assert [i for i, _ in res] == ["p02", "p00"]
    assert res[0][1] > res[1][1]
    assert fake.corpus_paths == ["pages/p00.png", "pages/p01.png", "pages/p02.png"]


def test_doc_ignora_nao_documento():
    recs = [_rec("p00", "documento", "pages/p00.png"), _rec("img1", "imagem", "x.png"),
            _rec("t1", "texto"), _rec("p01", "documento")]
    fake = FakeColRetriever({"q": ["p01", "p00"]})
    spec = DocSpecialist(recs, ranker=fake)

    assert [i for i, _ in spec.search("q", 10)] == ["p01", "p00"]
    assert len(fake.corpus_paths) == 2
    assert fake.corpus_paths[1] == "gs://b/p01"   # sem meta["path"]: cai no pointer_uri
