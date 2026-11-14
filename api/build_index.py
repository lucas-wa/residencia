"""Constroi OFFLINE o indice servido por `api/app.py` -- precisa de GPU (staged).

Embedda os itens do corpus v4 pelo lado de ITENS do baseline A (`build_unified_a_embedder`,
que ja roteia cada item pela modalidade: `documento`/`imagem`/`video` por `embed_images`,
`texto`/`audio` por `embed_texts`) e escreve o diretorio de indice que a API consome:

    <out>/records.jsonl     copia fiel dos records de entrada (a ORDEM define as linhas)
    <out>/matrix.npy        [n, dim] float32 L2-normalizado
    <out>/index_meta.json   proveniencia: adaptador/modelo que produziram a matriz

O `index_meta.json` existe porque o serving ([4]) embedda a QUERY com o adaptador do
`ADAPTER_PATH` dele: se nao for o mesmo adaptador que embeddou os ITENS aqui, os dois
espacos nao casam e o ranking degrada EM SILENCIO (ambos normalizam e multiplicam sem
erro). Com o carimbo, `api/app.py` recusa a divergencia no startup.

Uso (H100/4090, com o adaptador da Semana 10):

    python -m api.build_index \
        --records data/govexe/ensemble_orcamento_v4/records.jsonl \
        --out data/govexe/indice_api_v4 \
        --adapter finetune/out/qwen3vl-lora

Depois o diretorio sobe para o GCS (infra [3]) e vira o `INDEX_PATH` do Cloud Run.

O embedder real (que so existe com GPU) entra por injecao, entao o que `tests/test_api.py`
exercita aqui e o que NAO depende dele: o carimbo de proveniencia e a sanidade dos
caminhos. O casamento records/matriz e validado na LEITURA, em `api.index_io.load_index`.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from api.index_io import MATRIX_NAME, META_NAME, RECORDS_NAME
from ensemble.corpus_v4 import check_corpus_paths
from ensemble.native.embedders import UNIFIED_A_MODEL
from ensemble.native.specialist import _l2_normalize
from ensemble.native.unified_a import a_repr, a_route
from ensemble.records import load_records

log = logging.getLogger(__name__)

DEFAULT_RECORDS = "data/govexe/ensemble_orcamento_v4/records.jsonl"
DEFAULT_OUT = "data/govexe/indice_api_v4"


def build_index(records_path: str | Path, out_dir: str | Path, *,
                adapter_path: str | None = None, embed_items=None) -> Path:
    """Embedda os itens e escreve `<out_dir>/{records.jsonl,matrix.npy,index_meta.json}`.

    `embed_items` injeta o embedder (duble em dev); sem ele, carrega o Qwen3-VL real via
    `build_unified_a_embedder(adapter_path)` -- import tardio, so quem roda de fato paga.

    Ordem deliberada: records -> `check_corpus_paths` -> embedder -> embeddar. O ciclo [3]
    captura com `build_corpus`, e um item image-route cujo `a_repr` nao esta em disco (um
    keyframe que a retencao nao reteve, por exemplo) so estouraria no `open` do
    `embed_images` DEPOIS de carregar o modelo -- minutos de GPU para um `CorpusPathError`
    que da para dar agora.
    """
    records = load_records(records_path)
    if not records:
        raise ValueError(f"nenhum record em {records_path}: nao ha indice a construir")
    check_corpus_paths(records)

    if embed_items is None:
        from ensemble.native.embedders import build_unified_a_embedder
        embed_items, _ = build_unified_a_embedder(adapter_path=adapter_path)

    pares = [(a_route(r), a_repr(r)) for r in records]
    log.info("embeddando %d itens (%d pela rota de imagem)", len(pares),
             sum(1 for rota, _ in pares if rota == "image"))
    matrix = _l2_normalize(embed_items(pares))
    if matrix.shape[0] != len(records):
        raise ValueError(
            f"{len(records)} records para {matrix.shape[0]} embeddings "
            "(a linha i tem de ser o embedding do record i)")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / MATRIX_NAME, matrix)
    shutil.copyfile(records_path, out / RECORDS_NAME)
    _gravar_meta(out, adapter_path=adapter_path, n=int(matrix.shape[0]),
                 dim=int(matrix.shape[1]))
    log.info("indice: %d itens, dim=%d -> %s", matrix.shape[0], matrix.shape[1], out)
    return out


def _gravar_meta(out: Path, *, adapter_path: str | None, n: int, dim: int) -> dict:
    """Carimba `index_meta.json` com o que produziu esta matriz.

    A identidade do adaptador e a STRING do caminho -- e o mesmo valor que o serving le
    de `ADAPTER_PATH`, entao e com ela que a comparacao do startup acontece. `None` =
    checkpoint publico, sem adaptador.
    """
    meta = {
        "adapter_path": str(adapter_path) if adapter_path else None,
        "model_id": UNIFIED_A_MODEL,
        "dim": dim,
        "n": n,
    }
    (out / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    log.info("proveniencia carimbada em %s: adaptador=%s", META_NAME, meta["adapter_path"])
    return meta


def main():
    ap = argparse.ArgumentParser(
        description="Embedda o corpus v4 pelo baseline A e salva o indice servido pela API.")
    ap.add_argument("--records", default=DEFAULT_RECORDS,
                    help="records.jsonl do corpus v4")
    ap.add_argument("--out", default=DEFAULT_OUT, help="diretorio do indice de saida")
    ap.add_argument("--adapter", default=None,
                    help="diretorio do adaptador LoRA (omitido = checkpoint publico)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out = build_index(a.records, a.out, adapter_path=a.adapter)
    print(f"indice em {out}")


if __name__ == "__main__":
    main()
