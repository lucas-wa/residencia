"""Provisionamento do finetune v4 (`gcp/serving/finetune_v4_spot.sh`) -- tudo em `--dry-run`.

Nenhum teste toca a nuvem, e isso e verificado e nao suposto: o PATH de cada rodada recebe
um `gcloud` POSTICO que, se for EXECUTADO, anota a chamada num arquivo-sentinela e falha.
Como o `--dry-run` do script so IMPRIME os comandos, a sentinela tem de continuar vazia --
e qualquer caminho que escape do `run`/`existe` aparece aqui como teste vermelho.

O que estes testes guardam e a disciplina de custo do script: GCS_OUT obrigatorio (sem ele
a GPU treinaria para jogar o adaptador fora), `--project` explicito em toda chamada gcloud,
e o TEARDOWN rodando em qualquer saida -- inclusive quando uma etapa falha no meio
(`FINETUNE_FAIL_AT`).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from finetune import train_lora as T

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "gcp" / "serving" / "finetune_v4_spot.sh"

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(
    BASH is None, reason="bash ausente (o script de provisionamento e POSIX)")

# Destino de mentira: o dry-run nunca o resolve, mas o preflight exige que exista e seja
# `gs://`. Fora do prefixo do corpus de entrada, de proposito.
GCS_OUT = "gs://bucket-de-teste/finetune/qwen3vl-qlora"

# Fases do script remoto, na ordem em que a orquestracao local as dispara.
FASES_REMOTAS = ("bootstrap", "stage", "pairs", "smoke", "finetune", "upload")


def _gcloud_postico(tmp_path: Path) -> tuple[Path, Path]:
    """Cria um `gcloud` que denuncia a propria execucao. Devolve (dir_no_path, sentinela)."""
    binario = tmp_path / "bin"
    binario.mkdir(exist_ok=True)
    sentinela = tmp_path / "gcloud_foi_chamado.txt"
    falso = binario / "gcloud"
    falso.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{sentinela.as_posix()}"\n'
        "exit 97\n",
        encoding="utf-8",
    )
    falso.chmod(0o755)
    return binario, sentinela


def _shims_de_captura(binario: Path, destino: Path) -> None:
    """Fixa o `TMP_DIR` do script em `destino` e neutraliza o `rm` do teardown.

    `gerar_scripts_remotos` e chamada no PREFLIGHT, antes do `trap ... EXIT`: os scripts
    remotos sao escritos mesmo em `--dry-run`. Quem os apaga depois e o `rm -rf "$TMP_DIR"`
    do teardown -- o unico `rm` do fluxo local. Com estes dois shims no PATH, o
    `finetune_env.sh`/`finetune_remote.sh` GERADOS (e nao o texto do heredoc lido do fonte)
    sobrevivem a rodada e podem ser inspecionados aqui.
    """
    falso_mktemp = binario / "mktemp"
    falso_mktemp.write_text(
        "#!/usr/bin/env bash\n"
        f'mkdir -p "{destino.as_posix()}"\n'
        f'printf "%s\\n" "{destino.as_posix()}"\n',
        encoding="utf-8",
    )
    falso_mktemp.chmod(0o755)
    falso_rm = binario / "rm"
    falso_rm.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    falso_rm.chmod(0o755)


def _rodar(tmp_path: Path, *args: str, fail_at: str | None = None,
           capturar_tmp: Path | None = None, env_extra: dict[str, str] | None = None):
    """Roda o script em `--dry-run` e devolve (CompletedProcess, saida, sentinela)."""
    binario, sentinela = _gcloud_postico(tmp_path)
    if capturar_tmp is not None:
        _shims_de_captura(binario, capturar_tmp)
    env = dict(os.environ)
    env["PATH"] = f"{binario}{os.pathsep}{env.get('PATH', '')}"
    # Variaveis do operador nao podem vazar para dentro do teste.
    for chave in ("FINETUNE_FAIL_AT", "GCS_OUT", "GCS_CORPUS", "BRANCH", "VM", "ZONES",
                  "PROJECT", "BASE", "METHOD", "EPOCHS", "BATCH_SIZE", "MAX_NEG", "LORA_R",
                  "NO_GRAD_CKPT"):
        env.pop(chave, None)
    if fail_at is not None:
        env["FINETUNE_FAIL_AT"] = fail_at
    env.update(env_extra or {})
    proc = subprocess.run(
        [BASH, str(SCRIPT), *args, "--dry-run"],
        cwd=str(RAIZ), env=env, capture_output=True, text=True, timeout=180)
    return proc, proc.stdout + proc.stderr, sentinela


def _rodada_completa(tmp_path: Path, *extra: str, fail_at: str | None = None,
                     capturar_tmp: Path | None = None):
    return _rodar(tmp_path, "--gcs-out", GCS_OUT, *extra, fail_at=fail_at,
                  capturar_tmp=capturar_tmp)


def _scripts_remotos_gerados(tmp_path: Path, *extra: str) -> tuple[str, str]:
    """Roda o dry-run e devolve o (finetune_env.sh, finetune_remote.sh) que ele ESCREVEU."""
    destino = tmp_path / "tmp_do_script"
    proc, saida, _ = _rodada_completa(tmp_path, *extra, capturar_tmp=destino)
    assert proc.returncode == 0, saida
    env = destino / "finetune_env.sh"
    remoto = destino / "finetune_remote.sh"
    assert env.is_file() and remoto.is_file(), f"scripts remotos nao foram gerados: {saida}"
    return env.read_text(encoding="utf-8"), remoto.read_text(encoding="utf-8")


# -- preflight: o que custa caro e barrado ANTES de criar a VM -------------------------


def test_sem_gcs_out_a_rodada_nem_comeca(tmp_path):
    """Sem destino, o treino pagaria GPU para jogar o adaptador fora com a VM."""
    proc, saida, sentinela = _rodar(tmp_path)
    assert proc.returncode != 0
    assert "GCS_OUT e obrigatorio" in saida
    # Nem o create chegou a ser IMPRESSO: o preflight corta antes.
    assert "instances create" not in saida
    assert not sentinela.exists()


def test_gcs_out_tem_de_ser_uri_gs(tmp_path):
    proc, saida, _ = _rodar(tmp_path, "--gcs-out", "/tmp/adaptador")
    assert proc.returncode != 0
    assert "tem de ser um URI gs://" in saida


def test_gcs_out_dentro_do_corpus_de_entrada_e_recusado(tmp_path):
    """O `rsync` do adaptador escreveria por cima de `records.jsonl`/`bin/`."""
    corpus = "gs://bucket-de-teste/ensemble/v4"
    proc, saida, _ = _rodar(
        tmp_path, "--gcs-corpus", corpus, "--gcs-out", f"{corpus}/adapter")
    assert proc.returncode != 0
    assert "esta dentro do corpus de entrada" in saida


def test_corpus_dir_absoluto_e_recusado(tmp_path):
    """Caminhos de dados viajam no tarball: so relativos a raiz do repo resolvem na VM."""
    proc, saida, _ = _rodada_completa(tmp_path, "--corpus-dir", "/mnt/dados/v4")
    assert proc.returncode != 0
    assert "tem de ser relativo a raiz do repo" in saida


def test_branch_divergente_so_avisa_em_dry_run(tmp_path):
    """O ensaio nao pode depender do checkout de quem revisa (numa rodada real e fatal)."""
    proc, saida, _ = _rodada_completa(tmp_path, "--branch", "branch-que-nao-existe")
    assert proc.returncode == 0
    assert "aviso (dry-run segue assim mesmo)" in saida
    assert "branch-que-nao-existe" in saida


# -- a rodada inteira, impressa e sem nuvem --------------------------------------------


def test_dry_run_imprime_a_rodada_inteira(tmp_path):
    proc, saida, _ = _rodada_completa(tmp_path)
    assert proc.returncode == 0, saida
    for passo in (f"[{i}/9]" for i in (1, 2, 4, 5, 6, 7, 8, 9)):
        assert passo in saida, f"etapa {passo} ausente"
    for fase in FASES_REMOTAS:
        assert f"finetune_remote.sh {fase}" in saida, f"fase remota '{fase}' ausente"


def test_dry_run_nao_executa_nenhum_gcloud(tmp_path):
    """A sentinela so aparece se um gcloud for EXECUTADO (e nao apenas impresso)."""
    proc, saida, sentinela = _rodada_completa(tmp_path)
    assert proc.returncode == 0
    assert "[dry-run] gcloud compute instances create" in saida
    assert not sentinela.exists(), f"gcloud executado: {sentinela.read_text()}"


def test_toda_chamada_gcloud_leva_project_explicito(tmp_path):
    """Nada de depender do `gcloud config set project` do operador."""
    _, saida, _ = _rodada_completa(tmp_path)
    linhas = [ln for ln in saida.splitlines() if "] gcloud " in ln]
    assert linhas, "nenhuma chamada gcloud impressa"
    sem_project = [ln for ln in linhas if "--project=" not in ln]
    assert not sem_project, f"gcloud sem --project: {sem_project}"


def test_vm_e_spot_que_se_apaga_na_preempcao(tmp_path):
    _, saida, _ = _rodada_completa(tmp_path)
    assert "--provisioning-model=SPOT" in saida
    assert "--instance-termination-action=DELETE" in saida


def test_o_corpus_v4_nao_viaja_no_tarball(tmp_path):
    """Ele vem do bucket na fase `stage`; mandar uma copia local so gastaria link."""
    _, saida, _ = _rodada_completa(tmp_path)
    tar = next(ln for ln in saida.splitlines() if "tar czf" in ln)
    assert "--exclude=data/govexe/ensemble_orcamento_v4" in tar
    assert "--exclude=.git" in tar


# -- teardown: roda em QUALQUER saida ---------------------------------------------------


def test_sucesso_termina_com_o_teardown_conferido(tmp_path):
    _, saida, _ = _rodada_completa(tmp_path)
    assert "TEARDOWN (saida com status 0)" in saida
    assert "conferencia ok: nenhuma VM sobrou" in saida
    assert "teardown concluido (status final: 0)" in saida


@pytest.mark.parametrize("etapa", ["provision", "bootstrap", "stage", "finetune", "upload"])
def test_falha_injetada_em_qualquer_etapa_ainda_roda_o_teardown(tmp_path, etapa):
    proc, saida, sentinela = _rodada_completa(tmp_path, fail_at=etapa)
    assert proc.returncode != 0
    assert f"falha INJETADA na etapa '{etapa}'" in saida
    assert "=== teardown concluido" in saida
    assert not sentinela.exists()


def test_falha_no_provision_nao_tenta_apagar_vm_nenhuma(tmp_path):
    """Nenhum `create` chegou a acontecer: o teardown e um no-op honesto."""
    _, saida, _ = _rodada_completa(tmp_path, fail_at="provision")
    assert "nenhuma VM foi criada nesta rodada" in saida
    assert "instances delete" not in saida


def test_falha_no_treino_tenta_resgatar_o_adaptador_antes_dos_deletes(tmp_path):
    """Apagar a VM destroi o adaptador: o resgate vem antes, e com teto de tempo."""
    _, saida, _ = _rodada_completa(tmp_path, fail_at="finetune")
    resgate = saida.index("tentativa final de subir o adaptador")
    delete = saida.index("instances delete")
    assert resgate < delete
    assert "timeout 1800 gcloud compute ssh" in saida


# -- o contrato com o treinador ---------------------------------------------------------


def test_a_fase_upload_sobe_o_diretorio_QUE_O_TREINADOR_ESCREVE(tmp_path):
    """`train_lora.py` grava em `finetune/runs/<base>-<metodo>`; a fase `upload` sobe ESSE.

    Se o default do treinador mudar, este teste quebra aqui -- e nao numa rodada paga que
    termina subindo um diretorio vazio.
    """
    esperado = T.resolve_config("qwen3vl", "qlora").out_path.as_posix()
    assert esperado == "finetune/runs/qwen3vl-qlora"
    fonte = SCRIPT.read_text(encoding="utf-8")
    assert 'OUT_REL="finetune/runs/${BASE}-${METHOD}"' in fonte

    _, saida, _ = _rodada_completa(tmp_path)
    assert f"adaptador na VM: {esperado}" in saida


def test_a_fase_de_treino_delega_ao_run_gcp_finetune(tmp_path):
    """Venv, torch, bitsandbytes e o train_lora sao dele: aqui nao ha segunda copia."""
    fonte = SCRIPT.read_text(encoding="utf-8")
    assert fonte.count("bash finetune/run_gcp_finetune.sh") == 2   # smoke + treino real
    assert (RAIZ / "finetune" / "run_gcp_finetune.sh").exists()
    # O hook do antes/depois so liga com EVAL_CORPUS apontando para o records do v4.
    assert 'EVAL_CORPUS="$RECORDS_REL"' in fonte
    assert "DRY_RUN=0" in fonte


def test_a_vm_de_treino_nao_instala_ffmpeg_nem_whisper(tmp_path):
    """O embed do baseline A nao reprocessa midia (`a_route`/`a_repr` leem o que ja existe).

    Video entra pela rota de imagem com o KEYFRAME ja extraido em `bin/`, audio entra pela
    rota de texto com a transcricao ja gravada no record. Instalar codec aqui seria minuto
    de L4 pago por nada.
    """
    # So o CODIGO conta: o cabecalho cita ffmpeg/faster-whisper justamente para explicar
    # por que eles nao estao aqui.
    codigo = "\n".join(ln for ln in SCRIPT.read_text(encoding="utf-8").splitlines()
                       if not ln.lstrip().startswith("#"))
    assert "apt-get" not in codigo
    assert "faster-whisper" not in codigo
    assert "ffmpeg" not in codigo


# -- NO_GRAD_CKPT: o que custou uma rodada paga ----------------------------------------
#
# Com QLoRA (base 4-bit congelada) + gradient checkpointing REENTRANT no caminho de imagem
# do Qwen3-VL, o passo morre com "element 0 of tensors does not require grad". O pico de
# VRAM medido foi 3.7GB de 23GB: o checkpointing so custava tempo. Por isso o default DESTE
# script e NO_GRAD_CKPT=1 -- e por isso a env var tem de CHEGAR as duas chamadas do
# run_gcp_finetune.sh (smoke e treino), que e o que faltava.


def test_preflight_mostra_o_no_grad_ckpt_em_uso(tmp_path):
    _, saida, _ = _rodada_completa(tmp_path)
    assert "no_grad_ckpt=1" in saida


def test_o_env_gerado_leva_no_grad_ckpt_para_a_vm(tmp_path):
    """A configuracao viaja no `finetune_env.sh`; sem a linha, a VM nem ve a flag."""
    env, _ = _scripts_remotos_gerados(tmp_path)
    assert "NO_GRAD_CKPT='1'" in env


def test_as_duas_chamadas_do_run_gcp_recebem_no_grad_ckpt(tmp_path):
    """Smoke E treino: passar so numa delas repetiria a falha num dos dois momentos."""
    _, remoto = _scripts_remotos_gerados(tmp_path)
    chamadas = remoto.split("bash finetune/run_gcp_finetune.sh")
    assert len(chamadas) == 3, "esperado exatamente 2 invocacoes do run_gcp_finetune.sh"
    for i, bloco in enumerate(chamadas[:-1]):
        # A lista de env vars precede a invocacao: olhar so o ultimo comando do bloco.
        env_da_chamada = bloco[bloco.rindex("DRY_RUN=0"):]
        assert 'NO_GRAD_CKPT="$NO_GRAD_CKPT"' in env_da_chamada, \
            f"a {i + 1}a chamada do run_gcp_finetune.sh nao repassa NO_GRAD_CKPT"


def test_grad_ckpt_religa_o_checkpointing(tmp_path):
    """O toggle existe para quem quiser medir o caminho antigo -- mas nao e o default."""
    env, _ = _scripts_remotos_gerados(tmp_path, "--grad-ckpt")
    assert "NO_GRAD_CKPT='0'" in env
    _, saida, _ = _rodada_completa(tmp_path, "--no-grad-ckpt")
    assert "no_grad_ckpt=1" in saida


def test_no_grad_ckpt_invalido_e_fatal_no_preflight(tmp_path):
    """`run_gcp_finetune.sh` so compara com "1": um "2" religaria o checkpointing calado."""
    proc, saida, sentinela = _rodar(
        tmp_path, "--gcs-out", GCS_OUT, env_extra={"NO_GRAD_CKPT": "2"})
    assert proc.returncode != 0
    assert "NO_GRAD_CKPT tem de ser 0 ou 1" in saida
    assert "instances create" not in saida
    assert not sentinela.exists()
