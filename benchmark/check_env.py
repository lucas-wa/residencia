"""Preflight: confere as versoes das libs sensiveis ANTES de gastar GPU.

O par `colpali-engine` x `transformers` e o ponto fragil do pipeline: cada versao do
engine exige uma faixa de `transformers`, e a imagem do container ja traz uma copia
propria (exigida pelo vLLM). Em vez de descobrir isso depois de 20 min de download de
modelo, este modulo olha as versoes instaladas e decide por qual caminho o ColQwen2 pode
ser carregado:

    colpali-engine     -> `from colpali_engine.models import ColQwen2, ColQwen2Processor`
                          (checkpoint `vidore/colqwen2-v1.0`)
    transformers-native-> `transformers.ColQwen2ForRetrieval` (>=4.49), que exige o
                          checkpoint convertido `vidore/colqwen2-v1.0-hf`
    none               -> nenhum caminho viavel; o job deve parar aqui

`analyze()` e pura (recebe o mapa de versoes) para poder ser testada sem importar
nenhuma das libs; `collect()` faz a coleta real e `main()` imprime o relatorio.

Uso:
    python -m benchmark.check_env            # sai 1 so em caso de fail
    python -m benchmark.check_env --strict   # sai 1 tambem em caso de warn
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as _dist_version

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

# Faixas de compatibilidade (ver NOTES do colpali-engine):
#   - engine >0.3.4 e a primeira serie com ColQwen2 estavel;
#   - essas versoes do engine exigem transformers >4.46.1;
#   - transformers >=4.49 ja traz ColQwen2 nativo (classe *ForRetrieval*).
COLPALI_ENGINE_SPEC = SpecifierSet(">0.3.4")
TRANSFORMERS_FOR_COLPALI_SPEC = SpecifierSet(">4.46.1")
TRANSFORMERS_NATIVE_SPEC = SpecifierSet(">=4.49")

# jina-embeddings-v4: o custom code vendoriza `qwen2_5_vl.py` e usa
# ROPE_INIT_FUNCTIONS["default"], chave removida no transformers 5.0.0 -> KeyError.
TRANSFORMERS_FOR_JINAV4_SPEC = SpecifierSet(">=4.52,<5")

HF_CHECKPOINT_NATIVE = "vidore/colqwen2-v1.0-hf"

# distribuicoes consultadas por collect(): nome no PyPI -> chave do relatorio
LIBS = {
    "transformers": "transformers",
    "colpali-engine": "colpali_engine",
    "torch": "torch",
    "huggingface_hub": "huggingface_hub",
    "sentence-transformers": "sentence_transformers",
    "mteb": "mteb",
    "open_clip_torch": "open_clip_torch",
    # o custom code do jina-embeddings-v4 carrega adaptadores LoRA por task -> precisa de peft
    "peft": "peft",
    "pypdfium2": "pypdfium2",
    "numpy": "numpy",
}

# libs sempre reportadas nas mensagens, mesmo ausentes
REPORTED = ("transformers", "colpali_engine", "torch", "huggingface_hub")

_SEVERITY = {"ok": 0, "warn": 1, "fail": 2}


def _raw(versions: dict, key: str) -> str | None:
    """Le a versao aceitando tanto `colpali_engine` quanto `colpali-engine`."""
    value = versions.get(key)
    if value is None:
        value = versions.get(key.replace("_", "-"))
    return value


def _parsed(raw: str | None) -> Version | None:
    if not raw:
        return None
    try:
        return Version(raw)
    except InvalidVersion:
        return None


def analyze(versions: dict[str, str | None]) -> dict:
    """Decide o caminho de carregamento do ColQwen2/ColPali a partir das versoes.

    Retorna `{"status": "ok"|"warn"|"fail", "messages": [...], "colqwen_path": ...}`.
    """
    messages: list[str] = []
    status = "ok"

    def note(level: str, msg: str) -> None:
        nonlocal status
        messages.append(f"[{level}] {msg}")
        if _SEVERITY[level] > _SEVERITY[status]:
            status = level

    # 1) versoes observadas -- sempre no relatorio, mesmo quando ausentes
    for key in REPORTED:
        raw = _raw(versions, key)
        messages.append(f"[info] {key}={raw if raw else '<ausente>'}")

    tf_raw, cp_raw = _raw(versions, "transformers"), _raw(versions, "colpali_engine")
    tf, cp = _parsed(tf_raw), _parsed(cp_raw)
    if tf_raw and tf is None:
        note("fail", f"versao de transformers ilegivel: {tf_raw!r}")
    if cp_raw and cp is None:
        note("warn", f"versao de colpali_engine ilegivel: {cp_raw!r}")

    # 2) caminho de carregamento do ColQwen2
    colpali_ok = cp is not None and cp in COLPALI_ENGINE_SPEC
    tf_for_colpali = tf is not None and tf in TRANSFORMERS_FOR_COLPALI_SPEC
    tf_native = tf is not None and tf in TRANSFORMERS_NATIVE_SPEC

    if colpali_ok and tf_for_colpali:
        path = "colpali-engine"
        note("ok", f"ColQwen2/ColPali via colpali-engine {cp_raw} "
                   f"(transformers {tf_raw} satisfaz {TRANSFORMERS_FOR_COLPALI_SPEC})")
    elif tf_native:
        path = "transformers-native"
        motivo = ("colpali-engine ausente" if cp is None else
                  f"colpali-engine {cp_raw} nao satisfaz {COLPALI_ENGINE_SPEC}")
        note("warn", f"{motivo}; caia para o ColQwen2 nativo do transformers {tf_raw} "
                     f"-- use o checkpoint convertido {HF_CHECKPOINT_NATIVE} "
                     f"(o checkpoint sem sufixo -hf so carrega pelo colpali-engine)")
    else:
        path = "none"
        if tf is None:
            note("fail", "transformers nao esta instalado/legivel: nenhum caminho de "
                         "carregamento do ColQwen2 disponivel")
        else:
            note("fail", f"sem caminho viavel: colpali-engine "
                         f"{cp_raw or '<ausente>'} incompativel e transformers {tf_raw} "
                         f"nao satisfaz {TRANSFORMERS_NATIVE_SPEC} (ColQwen2 nativo)")

    # 2b) jina-embeddings-v4 (v3_jinav4) -- preso a API de RoPE do transformers 4.x
    if tf is not None and tf not in TRANSFORMERS_FOR_JINAV4_SPEC:
        note("warn", f"transformers {tf_raw} fora de {TRANSFORMERS_FOR_JINAV4_SPEC}: o "
                     f"jina-embeddings-v4 (custom code, qwen2_5_vl.py vendorizado) usa "
                     f"ROPE_INIT_FUNCTIONS['default'], removido no 5.0.0 -- o config "
                     f"v3_jinav4 vai estourar KeyError: 'default' no from_pretrained")

    # 3) torch: no cluster vem do container, entao a ausencia aqui e so um aviso
    if _parsed(_raw(versions, "torch")) is None:
        note("warn", "torch ausente (no cluster ele vem do container vLLM; "
                     "localmente a avaliacao em GPU nao roda)")

    return {"status": status, "messages": messages, "colqwen_path": path}


def collect() -> dict[str, str | None]:
    """Versoes instaladas das libs de interesse (`None` quando nao instalada)."""
    out: dict[str, str | None] = {}
    for dist, key in LIBS.items():
        try:
            out[key] = _dist_version(dist)
        except PackageNotFoundError:
            out[key] = None
        except Exception:                    # metadata corrompida: trata como ausente
            out[key] = None
    return out


def format_report(versions: dict[str, str | None], result: dict) -> str:
    lines = ["=== preflight benchmark.check_env ===", "-- versoes instaladas --"]
    lines += [f"  {k:22s} {v if v else '<ausente>'}" for k, v in sorted(versions.items())]
    lines += ["-- diagnostico --"]
    lines += [f"  {m}" for m in result["messages"]]
    lines += [f"caminho ColQwen2: {result['colqwen_path']}", f"status: {result['status']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preflight de versoes do pipeline.")
    ap.add_argument("--strict", action="store_true",
                    help="sai com 1 tambem quando o status for warn")
    a = ap.parse_args(argv)

    versions = collect()
    result = analyze(versions)
    print(format_report(versions, result))
    if result["status"] == "fail" or (a.strict and result["status"] == "warn"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
