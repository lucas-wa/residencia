"""Nucleo do ciclo event-driven [3]: upsert idempotente dos records do indice.

So o nucleo Python e testado aqui. A orquestracao (`gcp/serving/trigger_cycle.sh`,
`gcp/serving/create_scheduler.sh`) e shell/gcloud e fica fora do teste de unidade; o ramo
`gs://` do IO tambem, porque o import do `google-cloud-storage` e guardado dentro da
funcao -- o que se testa aqui e o ramo LOCAL, que e o mesmo codigo de serializacao.
"""
from pathlib import Path

from ensemble.records import PointerRecord
from gcp.serving.upsert_index import (
    RECORDS_NAME,
    load_index_gcs,
    save_index_gcs,
    upsert_records,
)


def _rec(id_, descriptor="descritor", **kw):
    base = dict(
        id=id_,
        modality="texto",
        descriptor=descriptor,
        pointer_uri=f"gs://bucket/{id_}.txt",
        source="portal",
        meta={},
    )
    base.update(kw)
    return PointerRecord(**base)


# -- upsert_records (puro) -------------------------------------------------------------

def test_upsert_dedup_por_id_o_novo_vence_e_a_ordem_e_estavel():
    existentes = [_rec("texto-a", "velho a"), _rec("texto-b", "velho b"),
                  _rec("texto-c", "velho c")]
    novos = [_rec("texto-b", "novo b"), _rec("texto-d", "novo d")]

    saida = upsert_records(existentes, novos)

    # Os existentes mantem a ORDEM original (b atualizado no lugar); o id inedito vai ao fim.
    assert [r.id for r in saida] == ["texto-a", "texto-b", "texto-c", "texto-d"]
    assert [r.descriptor for r in saida] == ["velho a", "novo b", "velho c", "novo d"]


def test_upsert_sem_sobreposicao_concatena_preservando_as_duas_ordens():
    existentes = [_rec("texto-a"), _rec("texto-b")]
    novos = [_rec("imagem-x"), _rec("imagem-y")]

    saida = upsert_records(existentes, novos)

    assert [r.id for r in saida] == ["texto-a", "texto-b", "imagem-x", "imagem-y"]


def test_upsert_de_dois_lados_vazios_e_vazio():
    assert upsert_records([], []) == []


def test_upsert_com_existente_vazio_e_a_lista_de_novos():
    novos = [_rec("texto-a"), _rec("texto-b")]
    assert upsert_records([], novos) == novos


def test_upsert_e_idempotente_aplicar_duas_vezes_nao_muda_nada():
    existentes = [_rec("texto-a", "velho a"), _rec("texto-b", "velho b")]
    novos = [_rec("texto-b", "novo b"), _rec("texto-c", "novo c")]

    uma_vez = upsert_records(existentes, novos)
    duas_vezes = upsert_records(uma_vez, novos)

    assert duas_vezes == uma_vez
    # E o ciclo que nao capturou nada de novo tambem nao mexe no indice.
    assert upsert_records(uma_vez, []) == uma_vez


def test_upsert_deduplica_ids_repetidos_dentro_dos_novos_o_ultimo_vence():
    novos = [_rec("texto-a", "primeiro"), _rec("texto-a", "ultimo")]

    saida = upsert_records([], novos)

    assert [r.id for r in saida] == ["texto-a"]
    assert saida[0].descriptor == "ultimo"


# -- IO local (o ramo gs:// tem import guardado e nao e testado) ------------------------

def test_roundtrip_local_preserva_os_records(tmp_path):
    indice = tmp_path / "indice"
    records = [_rec("texto-a", "Orçamento da saúde"), _rec("imagem-b", "gráfico")]

    save_index_gcs(str(indice), records)

    assert (indice / RECORDS_NAME).is_file()
    assert load_index_gcs(str(indice)) == records


def test_load_de_indice_que_ainda_nao_existe_e_lista_vazia(tmp_path):
    # Primeiro ciclo: nao ha indice no bucket e o upsert tem de comecar do zero,
    # nao explodir.
    assert load_index_gcs(str(tmp_path / "ainda-nao-existe")) == []


def test_save_cria_o_diretorio_do_indice(tmp_path):
    indice = tmp_path / "novo" / "indice"

    save_index_gcs(str(indice), [_rec("texto-a")])

    assert Path(indice, RECORDS_NAME).is_file()


def test_ciclo_completo_local_upsert_persiste_e_e_idempotente(tmp_path):
    indice = tmp_path / "indice"
    save_index_gcs(str(indice), [_rec("texto-a", "velho a"), _rec("texto-b", "velho b")])
    novos = [_rec("texto-b", "novo b"), _rec("texto-c", "novo c")]

    save_index_gcs(str(indice), upsert_records(load_index_gcs(str(indice)), novos))
    primeiro = load_index_gcs(str(indice))
    save_index_gcs(str(indice), upsert_records(load_index_gcs(str(indice)), novos))

    assert [r.id for r in primeiro] == ["texto-a", "texto-b", "texto-c"]
    assert load_index_gcs(str(indice)) == primeiro
