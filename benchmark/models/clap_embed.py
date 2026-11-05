"""CLAP -- embedding NATIVO de audio (a forma de onda, nao a transcricao) + texto alinhado.

E o backbone do espaco `clap` do registro (`gcp/slice/spaces.py`): a arquitetura
`A+audio` indexa o audio AQUI, em paralelo ao caminho `audio_asr` (Whisper -> texto ->
Qwen3-VL) da arquitetura A. Dois espacos distintos, fundidos por RRF na consulta
(`gcp/slice/fusion.py`) -- nunca por soma de vetores, que a geometria nao permite.

O CLAP tem DOIS encoders treinados por contraste no mesmo espaco (audio e texto), entao a
consulta continua sendo texto: `embed_texts` embute a pergunta e o produto interno contra
os vetores de `embed_audio` e um cosseno valido.

    from benchmark.models.clap_embed import embed_audio, embed_texts

    vecs = embed_audio(["sessao.m4a", "entrevista.m4a"])   # [N, 512] float32, L2=1
    q = embed_texts(["metas de saude"])                    # [1, 512] float32, L2=1
    scores = q @ vecs.T                                    # produto interno = cosseno

Checkpoint default: `laion/larger_clap_general` (CLAP "general", ~0,2 B -- cabe folgado ao
lado do Qwen3-VL numa L4/4090 24 GB). `CLAP_DIM = 512` e o valor do card, igual ao
`projection_dim` default do `ClapConfig`; e tambem o que a tabela `govexe_clap` declara.
CONFIRMAR no smoke com pesos reais antes de criar a tabela: divergir quebra o upsert.

API CONFIRMADA offline contra transformers 4.52 (`ClapModel`/`ClapProcessor` instanciados
com config default, sem baixar o checkpoint)
-------------------------------------------------------------------------------------
    processor(audios=[wav, ...], sampling_rate=48000, return_tensors="pt")
        -> {"input_features": [n, C, 1001, 64], "is_longer": [n, 1]}
    model.get_audio_features(input_features=..., is_longer=...)   -> [n, 512]
    model.get_text_features(input_ids=..., attention_mask=...)    -> [n, 512]

Dois detalhes que NAO sao cosmeticos:

1. 48 kHz. O `ClapFeatureExtractor` do CLAP roda a 48 kHz (`sampling_rate=48000`), nao a
   16 kHz do Whisper -- mandar a onda em outra taxa nao levanta erro, so desloca o
   espectrograma e degrada o vetor em silencio. Por isso o audio e lido JA reamostrado
   para `_sampling_rate(proc)`, que sai do proprio feature extractor do checkpoint.
2. `truncation` do feature extractor e `enable_fusion` do modelo sao acoplados: com
   `truncation="fusion"` o extractor devolve 4 canais de mel e so um audio tower com
   fusao aceita isso (verificado: a combinacao errada estoura `Given groups=1, weight of
   size [96, 1, 4, 4], expected input[1, 4, 256, 256] to have 1 channels`). Por isso NAO
   passamos `truncation`/`padding` ao lado de audio: valem os defaults do
   `preprocessor_config.json` do checkpoint, que casam com o `config.json` dele.

JANELA DE ~10 s (limite conhecido, nao e bug)
---------------------------------------------
O feature extractor do CLAP tem `max_length_s=10`: um arquivo mais longo NAO e coberto
inteiro -- o extractor recorta/compacta a onda para essa janela e o vetor representa so
esse trecho. Audio longo que cai aqui inteiro sai com um aviso (`log.warning` nomeando o
arquivo e a duracao), para a truncacao nunca ser silenciosa. **Chunking/pooling de audio
longo e responsabilidade do CHAMADOR** (fatiar o item em janelas, embutir cada uma e
agregar/indexar como preferir) -- este wrapper e deliberadamente um vetor por entrada, e
o fatiamento mora em quem monta os itens `audio_native`.

`_audio_vectors`/`_text_vectors` sao metodos de instancia de proposito: o teste de
contrato os substitui para exercitar o wrapper sem baixar pesos. O carregamento real mora
em `_load` (lazy, idempotente), entao importar este modulo -- e construir o embedder --
nao toca em transformers nem na GPU.

`get_embedder` guarda o embedder carregado num cache por `(model_id, device, batch_size)`:
chamadas repetidas no mesmo processo nao recarregam pesos.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "CLAP_DIM",
    "CLAP_MAX_SECONDS",
    "CLAP_SAMPLE_RATE",
    "DEFAULT_CLAP_ID",
    "ClapEmbedder",
    "embed_audio",
    "embed_texts",
    "get_embedder",
]

# Checkpoint do espaco `clap` (espelha `SPACES["clap"].model_id`).
DEFAULT_CLAP_ID = "laion/larger_clap_general"
# Dimensao da projecao do card (= `ClapConfig.projection_dim`); espelha `SPACES["clap"].dim`.
# Usada so como forma da matriz VAZIA -- os vetores reais saem com a dimensao do modelo.
CLAP_DIM = 512
# Taxa do feature extractor do CLAP. E o fallback: a taxa real sai do proprio processor
# do checkpoint (`_sampling_rate`).
CLAP_SAMPLE_RATE = 48_000
# Janela do feature extractor (`ClapFeatureExtractor.max_length_s`). Entrada mais longa e
# truncada POR ELE -- aqui so viramos o aviso. Fallback: a janela real sai do checkpoint
# (`_max_seconds`).
CLAP_MAX_SECONDS = 10.0


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-9)


def _read_audio(path: str, sampling_rate: int,
                max_seconds: float = CLAP_MAX_SECONDS) -> np.ndarray:
    """Arquivo de audio -> onda mono float32 JA reamostrada para `sampling_rate`.

    `librosa` entra tardiamente (mesmo padrao de `gcp/slice/asr.py`): o modulo precisa ser
    importavel -- e os testes offline precisam rodar -- sem o decodificador instalado.
    Ele cobre os formatos que o coletor baixa (m4a/mp3/wav) e ja devolve mono.

    A onda sai INTEIRA (quem trunca e o feature extractor, que so enxerga `max_seconds`):
    acima da janela, emitimos um `log.warning` nomeando arquivo e duracao para a perda ser
    visivel. Fatiar audio longo em janelas e do chamador -- ver docstring do modulo.
    """
    try:
        import librosa
    except ImportError as exc:   # noqa: TRY003 -- mensagem acionavel vale mais que o tipo
        raise ImportError(
            "embedding de audio nativo (CLAP) precisa do librosa para decodificar e "
            "reamostrar o arquivo: pip install librosa soundfile"
        ) from exc

    onda, _sr = librosa.load(path, sr=sampling_rate, mono=True)
    onda = np.asarray(onda, dtype="float32")
    duracao = len(onda) / float(sampling_rate or CLAP_SAMPLE_RATE)
    if max_seconds and duracao > max_seconds:
        log.warning(
            "audio %s tem %.1f s e excede a janela de ~%.0f s do feature extractor do "
            "CLAP: o vetor cobre so o trecho que ele seleciona, o resto e descartado "
            "(fatiar em janelas e do chamador)", path, duracao, max_seconds)
    return onda


class ClapEmbedder:
    """Wrapper do CLAP: audio e texto no MESMO espaco, saida L2-normalizada.

    Construir e barato (nada e carregado); os pesos sobem na primeira chamada de
    `embed_audio`/`embed_texts`, dentro de `_load`.
    """

    def __init__(self, model_id: str = DEFAULT_CLAP_ID, device: str = "cuda",
                 batch_size: int = 8):
        self.name = f"clap:{model_id}"
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._proc = None

    # -- carregamento (nao exercitado pelo teste de contrato) ---------------------

    def _load(self):
        """Carrega `ClapModel` + `ClapProcessor` na primeira chamada; depois e no-op."""
        if self._model is None:
            from transformers import ClapModel, ClapProcessor

            log.info("carregando CLAP %s (%s)", self.model_id, self.device)
            self._proc = ClapProcessor.from_pretrained(self.model_id)
            self._model = ClapModel.from_pretrained(self.model_id).to(self.device).eval()
        return self._model, self._proc

    @staticmethod
    def _sampling_rate(proc) -> int:
        """Taxa que o feature extractor do CHECKPOINT espera (48 kHz no CLAP)."""
        fe = getattr(proc, "feature_extractor", None)
        return int(getattr(fe, "sampling_rate", None) or CLAP_SAMPLE_RATE)

    @staticmethod
    def _max_seconds(proc) -> float:
        """Janela que o feature extractor do CHECKPOINT cobre (~10 s no CLAP)."""
        fe = getattr(proc, "feature_extractor", None)
        return float(getattr(fe, "max_length_s", None) or CLAP_MAX_SECONDS)

    def _to_device(self, inputs) -> dict:
        return {k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in dict(inputs).items()}

    @staticmethod
    def _as_numpy(x) -> np.ndarray:
        """Tensor (torch) ou array -> `[n, dim]` float32, sem exigir torch importado."""
        if hasattr(x, "detach"):
            x = x.detach().to("cpu").float().numpy()
        return np.atleast_2d(np.asarray(x, dtype="float32"))

    # -- embedding ---------------------------------------------------------------

    def _audio_vectors(self, paths: list[str]) -> np.ndarray:
        """Caminhos de audio -> `[N, dim]` CRU (sem normalizar), batelado."""
        if not paths:   # antes de torch/`_load`: lote vazio nao justifica subir os pesos
            return np.zeros((0, CLAP_DIM), dtype="float32")
        import torch

        model, proc = self._load()
        taxa = self._sampling_rate(proc)
        janela = self._max_seconds(proc)      # so para avisar; quem trunca e o extractor
        out: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            ondas = [_read_audio(p, taxa, janela)
                     for p in paths[start:start + self.batch_size]]
            # sem `truncation`/`padding` aqui: valem os defaults do checkpoint (ver docstring)
            inputs = proc(audios=ondas, sampling_rate=taxa, return_tensors="pt")
            with torch.no_grad():
                feats = model.get_audio_features(**self._to_device(inputs))
            out.append(self._as_numpy(feats))
        return np.vstack(out)

    def _text_vectors(self, texts: list[str]) -> np.ndarray:
        """Textos (consultas) -> `[N, dim]` CRU (sem normalizar), batelado."""
        if not texts:
            return np.zeros((0, CLAP_DIM), dtype="float32")
        import torch

        model, proc = self._load()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            # `truncation=True`: consulta longa demais e CORTADA no limite do text tower,
            # em vez de estourar o embedding posicional do RoBERTa.
            inputs = proc(text=batch, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                feats = model.get_text_features(**self._to_device(inputs))
            out.append(self._as_numpy(feats))
        return np.vstack(out)

    def _postprocess(self, emb: np.ndarray) -> np.ndarray:
        """`[N, dim]` float32 L2-normalizado -- o formato que o indice recebe."""
        return _l2_normalize(np.atleast_2d(np.asarray(emb, dtype="float32")))

    # -- API de inferencia -------------------------------------------------------

    def embed_audio(self, paths: list[str]) -> np.ndarray:
        """Caminhos de audio -> `[N, dim]` float32 L2-normalizado (um vetor por item)."""
        return self._postprocess(self._audio_vectors(list(paths)))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Textos -> `[N, dim]` float32 L2-normalizado (mesmo espaco do audio)."""
        return self._postprocess(self._text_vectors(list(texts)))


# -- API de modulo ---------------------------------------------------------------------

_EMBEDDERS: dict[tuple, ClapEmbedder] = {}


def get_embedder(model_id: str = DEFAULT_CLAP_ID, device: str = "cuda",
                 batch_size: int = 8) -> ClapEmbedder:
    """Embedder em cache por `(model_id, device, batch_size)` -- nao recarrega pesos."""
    key = (model_id, device, batch_size)
    emb = _EMBEDDERS.get(key)
    if emb is None:
        emb = _EMBEDDERS[key] = ClapEmbedder(model_id, device=device, batch_size=batch_size)
    return emb


def embed_audio(paths: list[str], *, model_id: str = DEFAULT_CLAP_ID, device: str = "cuda",
                batch_size: int = 8, embedder: ClapEmbedder | None = None) -> np.ndarray:
    """Embute audio NATIVO -> `np.ndarray [N, dim]` float32 L2-normalizado.

    `embedder` injeta um wrapper ja carregado (ou um duble, nos testes); sem ele o modelo
    vem de `get_embedder(...)`.
    """
    emb = embedder or get_embedder(model_id=model_id, device=device, batch_size=batch_size)
    return emb.embed_audio(list(paths))


def embed_texts(texts: list[str], *, model_id: str = DEFAULT_CLAP_ID, device: str = "cuda",
                batch_size: int = 8, embedder: ClapEmbedder | None = None) -> np.ndarray:
    """Embute textos (consultas) no espaco do CLAP -> `[N, dim]` float32 L2-normalizado."""
    emb = embedder or get_embedder(model_id=model_id, device=device, batch_size=batch_size)
    return emb.embed_texts(list(texts))
