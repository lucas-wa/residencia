"""Ferramental do corpus v4: piso por eixo e split treino/avaliacao disjunto."""
import pytest

from ensemble import corpus_v4 as cv4
from ensemble.corpus_v4 import (
    AXES,
    BalanceError,
    CorpusPathError,
    SplitError,
    check_balance,
    check_corpus_paths,
    finalize_split,
    tag_split,
)
from ensemble.records import PointerRecord


def _qrels_por_eixo(contagens: dict[str, int]) -> tuple[dict, dict]:
    """Monta (qrels, queries) com `contagens[eixo]` queries julgadas em cada eixo.

    Cada query recebe UM relevante do seu eixo, com o prefixo de modalidade do
    `item_id` (`documento-`, `imagem-`, ...) que o corpus v2/v4 ja usa.
    """
    qrels: dict[str, set[str]] = {}
    queries: dict[str, str] = {}
    for eixo, n in contagens.items():
        for i in range(n):
            qid = f"{eixo}-q{i}"
            qrels[qid] = {f"{eixo}-item{i}"}
            queries[qid] = f"consulta {eixo} {i}"
    return qrels, queries


def test_balance_acusa_eixo_abaixo_do_piso():
    qrels, queries = _qrels_por_eixo(
        {"documento": 8, "imagem": 2, "audio": 8, "texto": 9, "video": 8})
    with pytest.raises(BalanceError, match="imagem"):
        check_balance(qrels, queries, floor=8)


def test_balance_axis_vazio_conta_zero():
    # video sem nenhuma query julgada: entra no dict como 0 (sem KeyError) e reprova.
    qrels, queries = _qrels_por_eixo(
        {"documento": 8, "imagem": 8, "audio": 8, "texto": 8})
    with pytest.raises(BalanceError) as exc:
        check_balance(qrels, queries, floor=8)
    assert "video" in str(exc.value)

    contagens = check_balance(qrels, queries, floor=0)
    assert set(contagens) == set(AXES)
    assert contagens["video"] == 0
    assert contagens["documento"] == 8


def test_balance_passa_quando_todos_no_piso():
    qrels, queries = _qrels_por_eixo(
        {"documento": 8, "imagem": 10, "audio": 8, "texto": 12, "video": 8})
    assert check_balance(qrels, queries) == {
        "documento": 8, "imagem": 10, "audio": 8, "texto": 12, "video": 8,
    }


def test_balance_conta_query_uma_vez_por_eixo_e_ignora_qid_fora_de_queries():
    # query cross-modal conta nos DOIS eixos (criterio do run_eval), mas uma vez em cada;
    # qid presente so nas qrels nao e julgado (nao esta no conjunto de queries avaliadas).
    qrels = {
        "q1": {"imagem-a", "imagem-b", "texto-a"},
        "q2": {"imagem-c"},
        "orfa": {"imagem-z"},
    }
    queries = {"q1": "x", "q2": "y"}
    assert check_balance(qrels, queries, floor=0) == {
        "documento": 0, "imagem": 2, "audio": 0, "texto": 1, "video": 0,
    }


def test_split_disjunto():
    queries = {f"q{i}": f"texto {i}" for i in range(5)}
    split = tag_split(queries, eval_ids=["q1", "q3"])
    assert split["eval"] == ["q1", "q3"]
    assert split["train"] == ["q0", "q2", "q4"]
    assert not set(split["train"]) & set(split["eval"])
    assert set(split["train"]) | set(split["eval"]) == set(queries)


def test_split_eval_id_inexistente_levanta():
    queries = {"q0": "a", "q1": "b"}
    with pytest.raises(SplitError, match="q9"):
        tag_split(queries, eval_ids=["q1", "q9"])


def test_split_id_nos_dois_lados_levanta():
    queries = {"q0": "a", "q1": "b", "q2": "c"}
    with pytest.raises(SplitError, match="q1"):
        tag_split(queries, eval_ids=["q1"], train_ids=["q0", "q1"])


def test_split_train_id_inexistente_levanta():
    queries = {"q0": "a", "q1": "b"}
    with pytest.raises(SplitError, match="q7"):
        tag_split(queries, eval_ids=["q1"], train_ids=["q0", "q7"])


def test_split_erros_sao_valueerror():
    assert issubclass(SplitError, ValueError)
    assert issubclass(BalanceError, ValueError)
    assert issubclass(CorpusPathError, ValueError)


# --- check_corpus_paths: itens image-route precisam de arquivo em disco ----------------

def _rec(modality: str, i: int, *, path: str | None = None,
         a_repr: str | None = None) -> PointerRecord:
    meta = {}
    if path is not None:
        meta["path"] = path
    if a_repr is not None:
        meta["a_repr"] = a_repr
    return PointerRecord(
        id=f"{modality}-{i}",
        modality=modality,
        descriptor=f"descritor {modality} {i}",
        pointer_uri=f"https://exemplo.gov.br/{modality}/{i}",
        source="duble",
        meta=meta,
    )


def _arquivo(tmp_path, nome: str) -> str:
    p = tmp_path / nome
    p.write_bytes(b"\x89PNG conteudo de duble")
    return str(p)


def test_paths_ok_quando_image_route_aponta_arquivo_real(tmp_path):
    records = [
        _rec("documento", 0, path=_arquivo(tmp_path, "doc0.png")),
        _rec("imagem", 0, path=_arquivo(tmp_path, "img0.jpg")),
        # video: o a_repr (keyframe da janela) vence o meta["path"] (o video inteiro)
        _rec("video", 0, path=_arquivo(tmp_path, "vid0.mp4"),
             a_repr=_arquivo(tmp_path, "vid0-kf3.png")),
    ]
    assert check_corpus_paths(records) is None


def test_paths_ignora_itens_text_route(tmp_path):
    # texto/audio tem a_repr TEXTUAL (artigo, transcricao ASR): nao e caminho, nao e checado
    records = [
        _rec("texto", 0),
        _rec("audio", 0, a_repr="transcricao ASR do boletim do Senado"),
        _rec("imagem", 0, path=_arquivo(tmp_path, "img0.jpg")),
    ]
    assert check_corpus_paths(records) is None


def test_paths_acusa_image_route_sem_arquivo(tmp_path):
    records = [
        _rec("imagem", 0, path=_arquivo(tmp_path, "img0.jpg")),
        _rec("documento", 7, path=str(tmp_path / "sumido.png")),
    ]
    with pytest.raises(CorpusPathError) as exc:
        check_corpus_paths(records)
    msg = str(exc.value)
    assert "documento-7" in msg
    assert "imagem-0" not in msg


def test_paths_acusa_image_route_com_a_repr_textual():
    # sem meta["path"]/meta["a_repr"], o a_repr cai no descriptor: o embed_images quebraria
    with pytest.raises(CorpusPathError, match="video-3"):
        check_corpus_paths([_rec("video", 3)])


def test_paths_limita_a_lista_de_ofensores():
    records = [_rec("imagem", i, path=f"/nao/existe/{i}.png") for i in range(12)]
    with pytest.raises(CorpusPathError) as exc:
        check_corpus_paths(records)
    msg = str(exc.value)
    assert "12" in msg                                  # o TOTAL aparece
    assert msg.count("imagem-") <= cv4.MAX_OFENSORES    # a lista e truncada


# --- finalize_split: piso + reserva ~eval_frac por eixo, disjunta -----------------------

def _eixos_de(qid: str, qrels: dict[str, set[str]]) -> set[str]:
    return {cv4.axis_of(item) for item in qrels[qid]}


def test_finalize_split_particiona_disjunto_com_fracao_por_eixo():
    qrels, queries = _qrels_por_eixo({e: 10 for e in AXES})
    train, eval_ = finalize_split(queries, qrels)

    assert not train & eval_
    assert train | eval_ == set(queries)
    for eixo in AXES:
        do_eixo = {q for q in queries if eixo in _eixos_de(q, qrels)}
        assert len(do_eixo & eval_) == 3          # ~30% de 10
        assert len(do_eixo & train) == 7


def test_finalize_split_e_deterministico_por_seed():
    qrels, queries = _qrels_por_eixo({e: 10 for e in AXES})
    assert finalize_split(queries, qrels) == finalize_split(queries, qrels)
    assert finalize_split(queries, qrels, seed=1) == finalize_split(queries, qrels, seed=1)
    # seeds diferentes sorteiam conjuntos diferentes (mesmos tamanhos)
    _, eval_a = finalize_split(queries, qrels, seed=1)
    _, eval_b = finalize_split(queries, qrels, seed=2)
    assert eval_a != eval_b
    assert len(eval_a) == len(eval_b)


def test_finalize_split_reserva_ao_menos_uma_eval_por_eixo_com_duas_julgadas():
    qrels, queries = _qrels_por_eixo({e: 2 for e in AXES})
    train, eval_ = finalize_split(queries, qrels, floor=2)
    for eixo in AXES:
        do_eixo = {q for q in queries if eixo in _eixos_de(q, qrels)}
        assert len(do_eixo & eval_) >= 1          # eixo com >=2 julgadas sempre em eval
        assert do_eixo & train                    # e nunca fica sem treino


def test_finalize_split_query_cross_modal_fica_de_um_lado_so():
    qrels, queries = _qrels_por_eixo({e: 9 for e in AXES})
    qrels["texto-q0"] = {"texto-item0", "imagem-item0", "video-item0"}
    train, eval_ = finalize_split(queries, qrels)
    assert ("texto-q0" in train) != ("texto-q0" in eval_)
    assert not train & eval_


def test_finalize_split_propaga_balance_error():
    qrels, queries = _qrels_por_eixo(
        {"documento": 8, "imagem": 2, "audio": 8, "texto": 8, "video": 8})
    with pytest.raises(BalanceError, match="imagem"):
        finalize_split(queries, qrels)


def test_finalize_split_propaga_split_error(monkeypatch):
    # guarda de sanidade: selecao que devolvesse qid fora de `queries` nao pode virar split
    qrels, queries = _qrels_por_eixo({e: 10 for e in AXES})
    monkeypatch.setattr(cv4, "_selecionar_eval", lambda *a, **k: {"fantasma-q0"})
    with pytest.raises(SplitError, match="fantasma-q0"):
        finalize_split(queries, qrels)


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_finalize_split_recusa_fracao_fora_do_intervalo(frac):
    qrels, queries = _qrels_por_eixo({e: 10 for e in AXES})
    with pytest.raises(ValueError, match="eval_frac"):
        finalize_split(queries, qrels, eval_frac=frac)
