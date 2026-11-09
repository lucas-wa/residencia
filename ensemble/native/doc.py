"""Especialista de documento do ensemble v2: ColQwen2.5 + MaxSim (multi-vetor, em memoria).

Adapta o ranqueador late-interaction do braco B (`gcp/slice/doc_specialist.py`) ao protocolo
`Specialist`. Sem pgvector: cada pagina vira uma matriz [n_tokens, dim]; o ranking usa o
`maxsim_search` de `benchmark.common.index`. `ranker` e injetado (qualquer objeto com
`encode_corpus`/`encode_queries`, como o `ColVisionRetriever`); os testes usam um duble.
"""
from __future__ import annotations

from ensemble.records import PointerRecord

MODALITY = "documento"


def _page_image(record: PointerRecord) -> str:
    """Imagem da pagina: meta["path"] se houver, senao o pointer_uri."""
    return record.meta.get("path") or record.pointer_uri


class DocSpecialist:
    modality = MODALITY

    def __init__(self, records: list[PointerRecord], *, ranker) -> None:
        from benchmark.common.types import Doc

        pages = [r for r in records if r.modality == MODALITY]
        self._ids = [r.id for r in pages]
        self._ranker = ranker
        if pages:
            self._doc_mv = list(ranker.encode_corpus(
                [Doc(id=r.id, image_path=_page_image(r)) for r in pages]))
            if len(self._doc_mv) != len(self._ids):
                raise ValueError(f"{len(self._ids)} paginas para {len(self._doc_mv)} embeddings")
        else:
            self._doc_mv = []

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        from benchmark.common.index import maxsim_search
        from benchmark.common.types import Query

        q_mv = self._ranker.encode_queries([Query(id="q", text=query)])
        order = maxsim_search(q_mv, self._doc_mv, k)[0]
        q = q_mv[0]
        return [(self._ids[i], float((q @ self._doc_mv[i].T).max(axis=1).sum()))
                for i in order]
