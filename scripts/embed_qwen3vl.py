"""CLI: embute paginas (imagens) com o Qwen3-VL-Embedding e grava vetores + manifest.

E o uso do modelo unificado FORA do benchmark -- o mesmo encode que a fatia GCP roda na
VM efemera antes do upsert no pgvector. A saida e deliberadamente simples (dois arquivos)
para o job de ingestao so precisar ler `.npy` + manifest:

    <out>.npy            matriz [N, dim] float32, L2-normalizada, uma linha por imagem
    <out>.manifest.json  {"model_id": ..., "dim": N, "items": [{"row", "id", "path"}]}

O `id` de cada item e o stem do arquivo (ex.: `123456_p07`), que ja e o id de pagina
usado por `data/govexe/build_planos_benchmark.py`.

Uso:
    python scripts/embed_qwen3vl.py --images-dir data/govexe/ptbr_planos/pages \
        --out data/embeddings/planos_qwen3vl --limit 32
    python scripts/embed_qwen3vl.py --images-dir pages --query "metas de saude"

Com `--query`, a consulta e embutida no mesmo espaco e o script imprime as paginas mais
proximas (produto interno = cosseno, porque os dois lados saem L2-normalizados).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# permite tanto `python scripts/embed_qwen3vl.py` quanto `python -m scripts.embed_qwen3vl`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.models.qwen3vl_embed import (  # noqa: E402
    DEFAULT_MODEL_ID,
    embed_images,
    embed_texts,
)

log = logging.getLogger(__name__)

DEFAULT_GLOB = "**/*.png"
DEFAULT_OUT = "data/embeddings/qwen3vl"
TOP_K = 5


def collect_paths(images_dir: str | None, pattern: str = DEFAULT_GLOB,
                  limit: int | None = None) -> list[str]:
    """Caminhos das imagens, ordenados (saida reprodutivel) e cortados por `limit`.

    Sem `images_dir`, `pattern` e resolvido a partir do diretorio corrente -- assim
    `--glob 'data/**/*.jpg'` tambem funciona como "lista de imagens".
    """
    root = Path(images_dir) if images_dir else Path(".")
    paths = sorted(str(p) for p in root.glob(pattern) if p.is_file())
    return paths[:limit] if limit else paths


def build_manifest(paths: list[str], model_id: str, dim: int) -> dict:
    """Manifest id->caminho, na mesma ordem das linhas da matriz."""
    return {
        "model_id": model_id,
        "dim": dim,
        "count": len(paths),
        "items": [{"row": i, "id": Path(p).stem, "path": str(Path(p).resolve())}
                  for i, p in enumerate(paths)],
    }


def write_outputs(out: str, vecs, manifest: dict) -> tuple[Path, Path]:
    """Grava `<out>.npy` e `<out>.manifest.json` (cria o diretorio se preciso)."""
    import numpy as np

    base = Path(out)
    if base.suffix == ".npy":
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    npy, mani = base.with_suffix(".npy"), base.with_suffix(".manifest.json")
    np.save(npy, vecs)
    mani.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return npy, mani


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default=None,
                    help="diretorio das paginas; sem ele, --glob resolve do diretorio atual")
    ap.add_argument("--glob", default=None,
                    help=f"padrao das imagens dentro de --images-dir (default: {DEFAULT_GLOB})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"prefixo da saida .npy/.manifest.json (default: {DEFAULT_OUT})")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help=f"checkpoint do Qwen3-VL-Embedding (default: {DEFAULT_MODEL_ID})")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="embute so as N primeiras imagens")
    ap.add_argument("--query", default=None,
                    help="texto de consulta: embute, imprime e ranqueia as imagens")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--dim", type=int, default=None,
                    help="trunca o vetor para N dimensoes (Matryoshka/MRL) e renormaliza")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    opts = dict(model_id=a.model_id, device=a.device, dtype=a.dtype,
                batch_size=a.batch_size, dim=a.dim)

    # so varre o disco quando o usuario apontou imagens; --query sozinho e modo consulta
    pattern = a.glob or DEFAULT_GLOB
    quer_imagens = bool(a.images_dir or a.glob)
    if not quer_imagens and not a.query:
        ap.error("informe --images-dir/--glob (imagens) e/ou --query (consulta)")
    paths = collect_paths(a.images_dir, pattern, a.limit) if quer_imagens else []
    if quer_imagens and not paths:
        log.error("nenhuma imagem encontrada (images_dir=%s glob=%s)", a.images_dir, pattern)
        return 1

    vecs = None
    if paths:
        log.info("embutindo %d imagens com %s", len(paths), a.model_id)
        vecs = embed_images(paths, **opts)
        npy, mani = write_outputs(a.out, vecs, build_manifest(paths, a.model_id, vecs.shape[1]))
        log.info("vetores: %s %s", npy, tuple(vecs.shape))
        log.info("manifest: %s", mani)

    if a.query:
        import numpy as np

        q = embed_texts([a.query], **opts)
        print(f"query dim={q.shape[1]} norma={float(np.linalg.norm(q[0])):.4f}")
        if vecs is not None and len(vecs):
            scores = (q @ vecs.T)[0]
            for rank, i in enumerate(np.argsort(-scores)[:TOP_K], start=1):
                print(f"{rank:2d}. {scores[i]:.4f}  {Path(paths[i]).stem}  {paths[i]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
