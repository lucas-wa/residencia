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

import json
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


# -- PAIRS_MODE: pares por LLM + split por documento + eval duplo -----------------------
#
# A 1a rodada paga mostrou duas fraquezas: a loss do InfoNCE colapsando em poucos passos
# (88 pares humanos num corpus de 150 = memorizacao) e 7 queries de eval (CI largo). O
# modo `llm` ataca as duas -- consultas escritas pelo Qwen2.5-7B sobre os docs FORA do
# eval, e um 2o conjunto de avaliacao sintetico sobre os 22 docs DE eval. O default
# continua `template`: o fluxo provado nao pode mudar de comportamento por engano.


def test_pairs_mode_default_e_template(tmp_path):
    _, saida, _ = _rodada_completa(tmp_path)
    assert "pairs_mode=template" in saida
    assert "gerador template" in saida
    # nenhuma fase nova aparece no fluxo provado
    assert "finetune_remote.sh eval_syn" not in saida
    assert "split por documento" not in saida


def test_pairs_mode_llm_imprime_a_geracao_e_a_2a_medicao(tmp_path):
    _, saida, _ = _rodada_completa(tmp_path, "--pairs-mode", "llm")
    assert "pairs_mode=llm" in saida
    assert "split por documento" in saida
    assert "[6/9] pares de treino por LLM (Qwen/Qwen2.5-7B-Instruct, k=3)" in saida
    assert "finetune_remote.sh eval_syn" in saida
    # o treino continua sendo UM so: a 2a medicao nao e uma 2a rodada
    assert saida.count("finetune_remote.sh finetune") == 1


def test_pairs_mode_llm_nao_mexe_nas_fases_provadas(tmp_path):
    """Aditivo: as 6 fases de sempre continuam todas la, na mesma ordem."""
    _, saida, _ = _rodada_completa(tmp_path, "--pairs-mode", "llm")
    posicoes = [saida.index(f"finetune_remote.sh {fase}") for fase in FASES_REMOTAS]
    assert posicoes == sorted(posicoes)
    assert saida.index("finetune_remote.sh eval_syn") < saida.index(
        "finetune_remote.sh upload")


def test_pairs_mode_invalido_e_fatal_no_preflight(tmp_path):
    """Um valor errado cairia no `case` remoto em SILENCIO, treinando a rodada errada."""
    proc, saida, sentinela = _rodar(tmp_path, "--gcs-out", GCS_OUT, "--pairs-mode", "xpto")
    assert proc.returncode != 0
    assert "PAIRS_MODE invalido" in saida
    assert "instances create" not in saida
    assert not sentinela.exists()


@pytest.mark.parametrize("valor", ["0", "tres", ""])
def test_gen_k_invalido_e_fatal_no_preflight(tmp_path, valor):
    """k=0 produziria zero pares -- e so a VM paga descobriria isso."""
    proc, saida, sentinela = _rodar(tmp_path, "--gcs-out", GCS_OUT, "--gen-k", valor)
    assert proc.returncode != 0
    assert "GEN_K tem de ser" in saida
    assert not sentinela.exists()


def test_o_env_gerado_leva_pairs_mode_gen_k_e_gen_model(tmp_path):
    """A fase de pares roda NA VM: sem estas linhas ela cairia no fluxo template."""
    env, _ = _scripts_remotos_gerados(tmp_path, "--pairs-mode", "llm",
                                      "--gen-k", "4", "--gen-model", "Qwen/Outro-7B")
    assert "PAIRS_MODE='llm'" in env
    assert "GEN_K='4'" in env
    assert "GEN_MODEL='Qwen/Outro-7B'" in env


def test_a_fase_pairs_deriva_os_docs_de_eval_do_qrels_staged(tmp_path):
    """EVAL_DOCS sai da 3a coluna do qrels que o `stage` baixou: nenhum dado novo."""
    _, remoto = _scripts_remotos_gerados(tmp_path)
    assert '"$QRELS_REL" | sort -u' in remoto
    assert 'EVAL_DOCS_REL="finetune/data/eval_docs.txt"' in remoto


def test_a_fase_pairs_llm_separa_treino_e_eval_por_documento(tmp_path):
    """Treino: tudo MENOS os docs de eval. Eval sintetico: SO os docs de eval."""
    _, remoto = _scripts_remotos_gerados(tmp_path)
    bloco = remoto[remoto.index("PAIRS_MODE:-template"):remoto.index("smoke)")]
    treino = bloco[bloco.index('--out "$PAIRS_REL" --k'):]
    assert '--exclude-docs "$EVAL_DOCS_REL"' in treino
    sintetico = bloco[bloco.index('--out "$SYN_PAIRS_REL"'):]
    assert '--include-docs "$EVAL_DOCS_REL"' in sintetico
    # os dois sao SUBPROCESSOS que saem: a VRAM do 7B volta inteira antes do treino
    assert bloco.count("python3 -m finetune.gen_queries_llm") == 2


def test_a_fase_pairs_llm_repara_o_torchaudio_antes_de_carregar_o_gerador(tmp_path):
    """O torchaudio da DLVM e ABI-incompativel; sem reparo, o transformers morre com
    `undefined symbol: torch_library_impl` ao resolver a classe do modelo. A geracao roda
    antes do venv do treino, entao o reparo (espelho do ensure_torchaudio) tem de estar
    aqui -- e ANTES da 1a chamada do gerador."""
    _, remoto = _scripts_remotos_gerados(tmp_path)
    bloco = remoto[remoto.index("PAIRS_MODE:-template"):remoto.index("smoke)")]
    reparo = bloco.index('python3 -c "import torchaudio"')
    assert reparo < bloco.index("python3 -m finetune.gen_queries_llm")
    assert '"torchaudio==$TA_VER"' in bloco
    assert "--index-url" in bloco and "--force-reinstall" in bloco
    assert "--user" in bloco[reparo:]                  # sem venv: user-site vence o dist-packages


def test_a_2a_medicao_reusa_o_hook_sem_retreinar(tmp_path):
    """1 treino, 2 medicoes: a fase `eval_syn` mede o adaptador JA salvo em disco."""
    fonte = SCRIPT.read_text(encoding="utf-8")
    bloco = fonte[fonte.index("\neval_syn)"):fonte.index("\nupload)")]
    assert "python3 -m finetune.eval_ensemble" in bloco
    assert "run_gcp_finetune.sh" not in bloco          # nao ha 2o treino
    assert '--adapter "$OUT_REL"' in bloco
    assert '--qrels "$SYN_QRELS_REL"' in bloco
    assert "sintetico" in bloco                        # rotulado, nao confundido com o ouro
    assert 'tee -a "$TRAIN_LOG"' in bloco              # os 2 deltas no log que sobe ao GCS


def test_a_2a_medicao_nao_pode_custar_o_upload_do_adaptador(tmp_path):
    """Perder a medicao extra e barato; perder o adaptador de uma rodada paga, nao."""
    fonte = SCRIPT.read_text(encoding="utf-8")
    # o `|| echo` tem de estar ANTES da etapa seguinte (a do upload), senao uma falha da
    # medicao extra derrubaria a rodada com `set -e` e o adaptador morreria com a VM
    chamada = fonte[fonte.index('remoto "bash ~/finetune_remote.sh eval_syn"'):]
    ate_a_proxima_etapa = chamada[:chamada.index("etapa ")]
    assert "|| echo" in ate_a_proxima_etapa
    assert "a 2a medicao falhou" in ate_a_proxima_etapa


# -- a fase `pairs` do modo llm, EXECUTADA de verdade (sem LLM, sem nuvem) --------------
#
# O awk que deriva os docs de eval, o guard anti-vazamento e a conversao
# pairs.jsonl -> (queries_syn.txt, qrels_syn.tsv) sao codigo que so roda NA VM. Aqui eles
# rodam na maquina de quem revisa: o unico duble e o `python3 -m finetune.gen_queries_llm`
# (que carregaria 15GB de Qwen2.5-7B); todo o resto -- awk, sort, o python inline -- e o
# codigo de verdade.

_STUB_PYTHON = """#!/usr/bin/env bash
# Duble do gerador por LLM; qualquer outro comando vai para o python REAL.
case "$*" in
    # torchaudio entra aqui para que o reparo ABI (espelho do ensure_torchaudio) seja
    # PULADO no teste: reparar de verdade exigiria rede e o indice do pytorch.
    *"import transformers"*|*"import accelerate"*|*"import torchaudio"*) exit 0 ;;
    *finetune.gen_queries_llm*) ;;
    *) exec "$REAL_PYTHON" "$@" ;;
esac
out=""; tipo=""; filtro=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out) out="$2"; shift 2 ;;
        --include-docs) tipo="include"; filtro="$2"; shift 2 ;;
        --exclude-docs) tipo="exclude"; filtro="$2"; shift 2 ;;
        *) shift ;;
    esac
done
mkdir -p "$(dirname "$out")"
: > "$out"
for id in documento-001 video-003 texto-004; do
    if [ "${STUB_VAZA:-0}" != "1" ]; then
        if [ "$tipo" = "include" ] && ! grep -qxF "$id" "$filtro"; then continue; fi
        if [ "$tipo" = "exclude" ] && grep -qxF "$id" "$filtro"; then continue; fi
    fi
    printf '{"id": "%s-q1", "query": "o que diz %s?", "pos_id": "%s"}\n' \
        "$id" "$id" "$id" >> "$out"
done
"""

# qrels de avaliacao no formato que o `stage` baixa: TREC 4 colunas, com repeticao de doc
# (2 queries sobre o mesmo documento) e uma linha de comentario.
QRELS_EVAL = ("# gabarito humano\n"
              "q01\t0\tdocumento-001\t1\n"
              "q01\t0\tvideo-003\t1\n"
              "q02\t0\tdocumento-001\t1\n")


def _escrever_sh(destino: Path, texto: str) -> Path:
    """Escreve um .sh com LF (no Windows o write_text poria CRLF e o bash engasgaria)."""
    destino.write_bytes(texto.encode("utf-8"))
    destino.chmod(0o755)
    return destino


def _vm_de_mentira(tmp_path: Path, **env_extra) -> tuple[Path, dict]:
    """Monta um `$HOME` com o finetune_remote.sh GERADO e um repo minimo. Devolve (home, env)."""
    import sys

    gerados = tmp_path / "gerados"
    gerados.mkdir()
    _, remoto_txt = _scripts_remotos_gerados(gerados, "--pairs-mode", "llm")

    home = tmp_path / "home"
    repo = home / "residencia"
    (repo / "ensemble" / "eval").mkdir(parents=True)
    (repo / "data" / "govexe" / "ensemble_orcamento_v4").mkdir(parents=True)
    (repo / "ensemble" / "eval" / "qrels_v4.tsv").write_text(QRELS_EVAL, encoding="utf-8")
    (repo / "data" / "govexe" / "ensemble_orcamento_v4" / "records.jsonl").write_text(
        "", encoding="utf-8")
    _escrever_sh(home / "finetune_remote.sh", remoto_txt)
    _escrever_sh(home / "finetune_env.sh", "\n".join([
        "PROJECT='p'", "GCS_CORPUS='gs://c'", "GCS_OUT='gs://o'",
        "CORPUS_DIR='data/govexe/ensemble_orcamento_v4'",
        "BASE='qwen3vl'", "METHOD='qlora'", "EPOCHS='1'", "BATCH_SIZE='1'",
        "MAX_NEG='1'", "LORA_R='32'", "NO_GRAD_CKPT='1'",
        "PAIRS_MODE='llm'", "GEN_K='1'", "GEN_MODEL='Qwen/Duble'", "BRANCH='b'", ""]))

    binario = tmp_path / "bin_stub"
    binario.mkdir()
    _escrever_sh(binario / "python3", _STUB_PYTHON)

    env = dict(os.environ)
    env["PATH"] = f"{binario.as_posix()}{os.pathsep}{env.get('PATH', '')}"
    env["HOME"] = home.as_posix()
    env["REAL_PYTHON"] = sys.executable
    env.update(env_extra)
    return home, env


def _rodar_fase_pairs(home: Path, env: dict):
    return subprocess.run([BASH, str(home / "finetune_remote.sh"), "pairs"],
                          cwd=str(home), env=env, capture_output=True, text=True,
                          timeout=180)


def test_fase_pairs_llm_produz_treino_sem_os_docs_de_eval_e_o_eval_sintetico(tmp_path):
    home, env = _vm_de_mentira(tmp_path)

    proc = _rodar_fase_pairs(home, env)

    saida = proc.stdout + proc.stderr
    assert proc.returncode == 0, saida
    dados = home / "residencia" / "finetune" / "data"
    # EVAL_DOCS: 3a coluna do qrels, unica e ordenada (o comentario nao entra)
    assert (dados / "eval_docs.txt").read_text(encoding="utf-8").split() == [
        "documento-001", "video-003"]
    # treino: so o doc que NAO esta no eval
    treino = [json.loads(ln) for ln in
              (dados / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert {r["pos_id"] for r in treino} == {"texto-004"}
    assert "0 doc de eval como positivo de treino" in saida
    # eval sintetico: SO os docs do gabarito, em queries.txt + qrels TREC de 4 colunas
    queries = (dados / "queries_syn.txt").read_text(encoding="utf-8").splitlines()
    qrels = [ln.split("\t") for ln in
             (dados / "qrels_syn.tsv").read_text(encoding="utf-8").splitlines()]
    assert [ln.split("\t")[0] for ln in queries] == ["syn-documento-001-q1",
                                                     "syn-video-003-q1"]
    assert [ln.split("\t")[1] for ln in queries] == ["o que diz documento-001?",
                                                     "o que diz video-003?"]
    assert qrels == [["syn-documento-001-q1", "0", "documento-001", "1"],
                     ["syn-video-003-q1", "0", "video-003", "1"]]


def test_fase_pairs_llm_aborta_se_um_doc_de_eval_virar_positivo_de_treino(tmp_path):
    """O guard e sobre o ARQUIVO, nao sobre a intencao: se o recorte falhar, a fase para."""
    home, env = _vm_de_mentira(tmp_path, STUB_VAZA="1")

    proc = _rodar_fase_pairs(home, env)

    saida = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "doc(s) de eval viraram positivo de treino" in saida
    # parou ANTES de gerar o conjunto sintetico (e antes de qualquer minuto de treino)
    assert not (home / "residencia" / "finetune" / "data" / "queries_syn.txt").exists()


# -- auth do gcloud: o token que venceu NO MEIO de uma rodada paga ----------------------
#
# Numa rodada longa o access token da conta ativa (politica de reauth) venceu no meio: o
# upload do adaptador falhou E o `instances delete` do teardown falhou -- a L4 ficou de pe
# cobrando ate alguem reparar. Os testes abaixo cobrem as duas pontas: falhar ANTES de
# pagar qualquer coisa, e deixar instrucoes impossiveis de perder quando quebra no fim.
#
# Aqui o `gcloud` dublado nao e mais o delator que sempre falha (aquele serve ao `--dry-run`):
# e um dublê PROGRAMAVEL, para que o fluxo de verdade -- SEM `--dry-run` -- possa rodar sem
# tocar a nuvem. Ele registra cada chamada num log, e o que nao foi programado nunca vira
# rede, so um rc.

_GCLOUD_PROGRAMAVEL = """#!/usr/bin/env bash
# gcloud dublado e OFFLINE: cada chamada vai para o log e o rc vem do ambiente.
printf '%s\\n' "$*" >> "$GCLOUD_LOG"
case "$*" in
    "auth print-access-token"*)  exit "${AUTH_RC:-0}" ;;
    "config get-value account"*) printf '%s\\n' "${CONTA_ATIVA:-dev@example.com}"; exit 0 ;;
    *"instances create"*)        exit "${CREATE_RC:-0}" ;;
    *"instances delete"*)
        if [ "${DELETE_REAUTH:-0}" = "1" ]; then
            echo "ERROR: (gcloud.compute.instances.delete) There was a problem refreshing" \\
                 "your current auth tokens: Reauthentication failed. cannot prompt during" \\
                 "non-interactive execution." >&2
            exit 1
        fi
        exit 0 ;;
    *"instances describe"*)      exit "${DESCRIBE_RC:-1}" ;;
    *) exit 0 ;;
esac
"""


def _rodar_de_verdade(tmp_path: Path, *args: str, **env_extra: str):
    """Roda o script SEM `--dry-run`, com um gcloud programavel que nao toca a nuvem.

    Devolve (CompletedProcess, saida, chamadas_ao_gcloud). `--branch ''` desliga a checagem
    de checkout (que e fatal fora do dry-run) e o `sleep` dublado tira os minutos de espera
    do teardown -- nenhum dos dois e o objeto destes testes.
    """
    binario = tmp_path / "bin_real"
    binario.mkdir()
    log = tmp_path / "chamadas_gcloud.txt"
    _escrever_sh(binario / "gcloud", _GCLOUD_PROGRAMAVEL)
    _escrever_sh(binario / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    env = dict(os.environ)
    env["PATH"] = f"{binario.as_posix()}{os.pathsep}{env.get('PATH', '')}"
    for chave in ("FINETUNE_FAIL_AT", "GCS_OUT", "GCS_CORPUS", "BRANCH", "VM", "ZONES",
                  "PROJECT", "BASE", "METHOD", "EPOCHS", "BATCH_SIZE", "MAX_NEG", "LORA_R",
                  "NO_GRAD_CKPT"):
        env.pop(chave, None)
    env["GCLOUD_LOG"] = log.as_posix()
    env.update(env_extra)
    proc = subprocess.run(
        [BASH, str(SCRIPT), "--gcs-out", GCS_OUT, "--branch", "", *args],
        cwd=str(RAIZ), env=env, capture_output=True, text=True, timeout=180)
    chamadas = log.read_text(encoding="utf-8") if log.exists() else ""
    return proc, proc.stdout + proc.stderr, chamadas


def test_token_invalido_aborta_o_preflight_antes_de_qualquer_vm(tmp_path):
    """A ponta (a): token ja vencido na largada nao pode virar VM paga.

    Sem isto a rodada sobe a L4, treina, e so descobre o problema na hora de subir o
    adaptador e apagar a VM -- que foi exatamente o incidente.
    """
    proc, saida, chamadas = _rodar_de_verdade(tmp_path, AUTH_RC="1")

    assert proc.returncode != 0
    assert "gcloud sem token valido" in saida
    assert "gcloud auth login" in saida
    # nada pago chegou nem a ser tentado
    assert "instances create" not in saida
    assert "instances create" not in chamadas


def test_token_valido_deixa_o_preflight_seguir(tmp_path):
    """A ponta (b): com token bom o preflight nao muda nada do fluxo de hoje."""
    proc, saida, chamadas = _rodar_de_verdade(
        tmp_path, AUTH_RC="0", FINETUNE_FAIL_AT="provision")

    assert "gcloud sem token valido" not in saida
    assert "auth print-access-token" in chamadas       # a checagem rodou de verdade
    # seguiu ate a etapa seguinte do preflight e so parou na falha INJETADA
    assert "falha INJETADA na etapa 'provision'" in saida
    assert "nenhuma VM foi criada nesta rodada" in saida
    assert "instances create" not in chamadas
    assert proc.returncode != 0


def test_dry_run_nao_executa_o_check_de_auth(tmp_path):
    """A ponta (c): `print-access-token` e chamada de REDE -- o ensaio so a anuncia."""
    proc, saida, sentinela = _rodada_completa(tmp_path)

    assert proc.returncode == 0, saida
    assert "[dry-run] gcloud auth print-access-token" in saida
    # o gcloud delator do dry-run continua intocado: nenhum gcloud foi EXECUTADO
    assert not sentinela.exists()


def test_conta_sujeita_a_reauth_so_avisa(tmp_path):
    """Aviso, nao barreira: a conta academica roda, mas o operador fica sabendo do risco."""
    proc, saida, _ = _rodar_de_verdade(
        tmp_path, AUTH_RC="0", FINETUNE_FAIL_AT="provision",
        CONTA_ATIVA="fulano@discente.ufg.br")

    assert "AVISO: conta fulano@discente.ufg.br pode exigir reauth" in saida
    assert "service account" in saida
    # avisou e SEGUIU: quem parou a rodada foi a falha injetada, nao o aviso
    assert "falha INJETADA na etapa 'provision'" in saida


def test_delete_que_falha_por_auth_imprime_o_banner_de_recuperacao(tmp_path):
    """A ponta (d): se o token morreu no meio, o teardown nao consegue apagar a VM.

    O retry nao resolve (so um `gcloud auth login` humano resolve), entao o que resta e
    nao deixar o operador adivinhar: VM, zona, projeto e o resgate do adaptador, prontos.
    """
    proc, saida, _ = _rodar_de_verdade(
        tmp_path, "--zones", "us-central1-a",
        AUTH_RC="0", CREATE_RC="1", DELETE_REAUTH="1", DESCRIBE_RC="0",
        TEARDOWN_RETRIES="1")

    assert proc.returncode != 0
    assert "RECUPERACAO MANUAL NECESSARIA (auth do gcloud expirou)" in saida
    assert "1. gcloud auth login" in saida
    assert ("3. gcloud compute instances delete govexe-finetune-v4 "
            "--zone=us-central1-a --project=residencia-511517 --quiet") in saida
    assert "finetune/runs/qwen3vl-qlora" in saida      # o resgate do adaptador
    assert f"{GCS_OUT}/adapter" in saida
    # o anti-leak continua valendo: a VM sobreviveu e o script grita
    assert "LEAK" in saida


def test_delete_que_falha_por_outro_motivo_nao_vira_banner_de_auth(tmp_path):
    """O banner e um diagnostico, nao um rodape: falha comum segue no caminho de sempre."""
    proc, saida, _ = _rodar_de_verdade(
        tmp_path, "--zones", "us-central1-a",
        AUTH_RC="0", CREATE_RC="1", DELETE_REAUTH="0", DESCRIBE_RC="1",
        TEARDOWN_RETRIES="1")

    assert "RECUPERACAO MANUAL NECESSARIA" not in saida
    assert "conferencia ok: nenhuma VM sobrou" in saida
