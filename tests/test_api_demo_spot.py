"""Demo efemera da API [4] (`gcp/serving/api_demo_spot.sh`) -- quase tudo em `--dry-run`.

Mesma tecnica de `tests/test_finetune_v4_spot.py`: o PATH de cada rodada recebe um
`gcloud` POSTICO que, se for EXECUTADO, anota a chamada num arquivo-sentinela e falha.
Como o `--dry-run` do script so IMPRIME os comandos, a sentinela tem de continuar vazia.
Aqui ha uma sentinela a mais -- a do `curl` --, porque nesta demo a prova "de fora" tambem
e rede: o ensaio nao pode abrir port-forward nem bater na API.

O que estes testes guardam:

  * disciplina de custo: INDEX_URI/ADAPTER obrigatorios e fora dos prefixos de entrada,
    `--project` explicito em toda chamada, SPOT que se apaga na preempcao, e o TEARDOWN
    rodando em qualquer saida (`API_DEMO_FAIL_AT`);
  * disciplina de vazamento LOCAL: o port-forward e um processo DESTA maquina, e o
    teardown tem de mata-lo (o unico teste que roda o fluxo inteiro existe para isso);
  * o contrato com o codigo do [4]: o MESMO caminho de adaptador embedda os itens
    (`api.build_index --adapter`) e as queries (`ADAPTER_PATH` do uvicorn) -- se divergirem,
    `api/app.py` recusa o startup e a VM paga nao serve nada.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "gcp" / "serving" / "api_demo_spot.sh"

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(
    BASH is None, reason="bash ausente (o script de provisionamento e POSIX)")

# Destinos de mentira: o dry-run nunca os resolve, mas o preflight exige `gs://` e que o
# indice nao caia dentro do corpus nem do adaptador.
ADAPTER = "gs://bucket-de-teste/finetune/qwen3vl-qlora/adapter"
INDEX_URI = "gs://bucket-de-teste/api/indice_v4"

# Fases do script remoto, na ordem em que a orquestracao local as dispara.
FASES_REMOTAS = ("bootstrap", "stage", "build_index", "upload_index", "serve")

# Variaveis do operador que nao podem vazar para dentro do teste.
VARS_DO_OPERADOR = ("API_DEMO_FAIL_AT", "ADAPTER_URI", "INDEX_URI", "GCS_CORPUS", "BRANCH",
                    "VM", "ZONES", "PROJECT", "CORPUS_DIR", "INDICE_DIR", "API_KEY",
                    "SAIDA", "K", "PORTA", "PORTA_LOCAL", "DEVICE")


def _escrever_sh(destino: Path, texto: str) -> Path:
    """Escreve um .sh com LF (no Windows o write_text poria CRLF e o bash engasgaria)."""
    destino.write_bytes(texto.encode("utf-8"))
    destino.chmod(0o755)
    return destino


def _delatores(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Cria `gcloud` e `curl` que denunciam a propria execucao.

    Devolve (dir_no_path, sentinela_gcloud, sentinela_curl). No `--dry-run` as duas
    sentinelas tem de continuar inexistentes: nenhuma chamada de REDE pode acontecer.
    """
    binario = tmp_path / "bin"
    binario.mkdir(exist_ok=True)
    sentinela_gcloud = tmp_path / "gcloud_foi_chamado.txt"
    sentinela_curl = tmp_path / "curl_foi_chamado.txt"
    for nome, sentinela in (("gcloud", sentinela_gcloud), ("curl", sentinela_curl)):
        _escrever_sh(binario / nome,
                     "#!/usr/bin/env bash\n"
                     f'echo "$@" >> "{sentinela.as_posix()}"\n'
                     "exit 97\n")
    return binario, sentinela_gcloud, sentinela_curl


def _shims_de_captura(binario: Path, destino: Path) -> None:
    """Fixa o `TMP_DIR` do script em `destino` e neutraliza o `rm` do teardown.

    `gerar_scripts_remotos` roda no PREFLIGHT, antes do `trap ... EXIT`: os scripts remotos
    sao escritos mesmo em `--dry-run`. Quem os apaga depois e o `rm -rf "$TMP_DIR"` do
    teardown -- o unico `rm` do fluxo local. Com estes dois shims no PATH, o
    `api_demo_env.sh`/`api_demo_remote.sh` GERADOS (e nao o texto do heredoc lido do fonte)
    sobrevivem a rodada e podem ser inspecionados aqui.
    """
    _escrever_sh(binario / "mktemp",
                 "#!/usr/bin/env bash\n"
                 f'mkdir -p "{destino.as_posix()}"\n'
                 f'printf "%s\\n" "{destino.as_posix()}"\n')
    _escrever_sh(binario / "rm", "#!/usr/bin/env bash\nexit 0\n")


def _rodar(tmp_path: Path, *args: str, fail_at: str | None = None,
           capturar_tmp: Path | None = None, env_extra: dict[str, str] | None = None):
    """Roda o script em `--dry-run` e devolve (CompletedProcess, saida, sentinelas)."""
    binario, sentinela_gcloud, sentinela_curl = _delatores(tmp_path)
    if capturar_tmp is not None:
        _shims_de_captura(binario, capturar_tmp)
    env = dict(os.environ)
    env["PATH"] = f"{binario}{os.pathsep}{env.get('PATH', '')}"
    for chave in VARS_DO_OPERADOR:
        env.pop(chave, None)
    if fail_at is not None:
        env["API_DEMO_FAIL_AT"] = fail_at
    env.update(env_extra or {})
    proc = subprocess.run(
        [BASH, str(SCRIPT), *args, "--dry-run"],
        cwd=str(RAIZ), env=env, capture_output=True, text=True, timeout=180)
    return proc, proc.stdout + proc.stderr, (sentinela_gcloud, sentinela_curl)


def _rodada_completa(tmp_path: Path, *extra: str, fail_at: str | None = None,
                     capturar_tmp: Path | None = None):
    return _rodar(tmp_path, "--adapter", ADAPTER, "--index-uri", INDEX_URI, *extra,
                  fail_at=fail_at, capturar_tmp=capturar_tmp)


def _scripts_remotos_gerados(tmp_path: Path, *extra: str) -> tuple[str, str]:
    """Roda o dry-run e devolve o (api_demo_env.sh, api_demo_remote.sh) que ele ESCREVEU."""
    destino = tmp_path / "tmp_do_script"
    proc, saida, _ = _rodada_completa(tmp_path, *extra, capturar_tmp=destino)
    assert proc.returncode == 0, saida
    env = destino / "api_demo_env.sh"
    remoto = destino / "api_demo_remote.sh"
    assert env.is_file() and remoto.is_file(), f"scripts remotos nao foram gerados: {saida}"
    return env.read_text(encoding="utf-8"), remoto.read_text(encoding="utf-8")


def _bloco_da_fase(remoto: str, fase: str) -> str:
    """Recorta o bloco `<fase>)` ... `;;` do `case` do script remoto."""
    inicio = remoto.index(f"\n{fase})")
    return remoto[inicio:remoto.index("\n    ;;", inicio)]


# -- preflight: o que custa caro e barrado ANTES de criar a VM -------------------------


def test_sem_adaptador_a_rodada_nem_comeca(tmp_path):
    """Sem adaptador o indice sairia da base crua: a demo provaria a coisa errada."""
    proc, saida, (gcloud, _) = _rodar(tmp_path, "--adapter", "", "--index-uri", INDEX_URI)
    assert proc.returncode != 0
    assert "ADAPTER_URI e obrigatorio" in saida
    assert "instances create" not in saida
    assert not gcloud.exists()


def test_sem_index_uri_a_rodada_nem_comeca(tmp_path):
    """Sem destino, a GPU embeddaria 150 itens para jogar o indice fora com a VM."""
    proc, saida, _ = _rodar(tmp_path, "--adapter", ADAPTER, "--index-uri", "")
    assert proc.returncode != 0
    assert "INDEX_URI e obrigatorio" in saida
    assert "instances create" not in saida


@pytest.mark.parametrize("flag,valor", [("--index-uri", "/tmp/indice"),
                                        ("--adapter", "/mnt/adapter")])
def test_entradas_e_saida_tem_de_ser_uri_gs(tmp_path, flag, valor):
    proc, saida, _ = _rodada_completa(tmp_path, flag, valor)
    assert proc.returncode != 0
    assert "tem de ser um URI gs://" in saida


def test_index_uri_dentro_do_corpus_de_entrada_e_recusado(tmp_path):
    """O `rsync` do indice escreveria por cima de `records.jsonl`/`bin/`."""
    corpus = "gs://bucket-de-teste/ensemble/v4"
    proc, saida, _ = _rodar(
        tmp_path, "--adapter", ADAPTER, "--gcs-corpus", corpus,
        "--index-uri", f"{corpus}/indice")
    assert proc.returncode != 0
    assert "esta dentro do corpus de entrada" in saida
    assert "instances create" not in saida


def test_index_uri_dentro_do_adaptador_e_recusado(tmp_path):
    """Mesmo espirito: o indice sobrescreveria os pesos PEFT da Semana 10."""
    proc, saida, _ = _rodar(
        tmp_path, "--adapter", ADAPTER, "--index-uri", f"{ADAPTER}/indice")
    assert proc.returncode != 0
    assert "esta dentro do adaptador" in saida


@pytest.mark.parametrize("k", ["0", "101", "cinco", ""])
def test_k_fora_da_faixa_da_api_e_fatal_no_preflight(tmp_path, k):
    """`ConsultaRequest` responde 422 fora de [1,100]: a VM nao pode ser paga para isso."""
    proc, saida, (gcloud, _) = _rodada_completa(tmp_path, "--k", k)
    assert proc.returncode != 0
    assert "K tem de " in saida
    assert not gcloud.exists()


@pytest.mark.parametrize("flag", ["--corpus-dir", "--indice-dir"])
def test_caminho_absoluto_na_vm_e_recusado(tmp_path, flag):
    """Caminhos de dados viajam no tarball: so relativos a raiz do repo resolvem na VM."""
    proc, saida, _ = _rodada_completa(tmp_path, flag, "/mnt/dados/v4")
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
    """As 8 etapas locais, as 5 fases remotas e o teardown -- nesta ordem."""
    proc, saida, _ = _rodada_completa(tmp_path)
    assert proc.returncode == 0, saida
    for passo in (f"[{i}/8]" for i in (1, 2, 4, 5, 6, 7, 8)):
        assert passo in saida, f"etapa {passo} ausente"
    posicoes = []
    for fase in FASES_REMOTAS:
        assert f"api_demo_remote.sh {fase}" in saida, f"fase remota '{fase}' ausente"
        posicoes.append(saida.index(f"api_demo_remote.sh {fase}"))
    assert posicoes == sorted(posicoes), "as fases remotas sairam fora de ordem"
    # a prova "de fora" vem DEPOIS do serve, e o teardown depois de tudo
    assert saida.index("api_demo_remote.sh serve") < saida.index("-N -L 8080:127.0.0.1:8080")
    assert saida.index("-N -L 8080:127.0.0.1:8080") < saida.index("=== TEARDOWN")


def test_dry_run_nao_executa_nenhum_gcloud_nem_curl(tmp_path):
    """As sentinelas so aparecem se um comando de REDE for EXECUTADO (e nao impresso)."""
    proc, saida, (gcloud, curl) = _rodada_completa(tmp_path)
    assert proc.returncode == 0
    assert "[dry-run] gcloud compute instances create" in saida
    assert not gcloud.exists(), f"gcloud executado: {gcloud.read_text()}"
    assert not curl.exists(), f"curl executado: {curl.read_text()}"


def test_dry_run_nao_abre_port_forward(tmp_path):
    """O tunel e `gcloud compute ssh` em background: no ensaio ele so e ANUNCIADO."""
    proc, saida, (gcloud, curl) = _rodada_completa(tmp_path)
    assert proc.returncode == 0
    linha = next(ln for ln in saida.splitlines() if "-N -L" in ln)
    assert linha.lstrip().startswith("[dry-run]"), f"port-forward fora do guard: {linha}"
    assert "-L 8080:127.0.0.1:8080" in linha
    assert not gcloud.exists()
    # e nenhuma consulta foi disparada: elas tambem sao rede
    assert "[dry-run] curl -sS -X POST" in saida
    assert not curl.exists()
    # o tunel nunca abriu, entao nao ha o que fechar
    assert "fechando o port-forward local" not in saida


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


def test_a_chave_efemera_e_gerada_e_viaja_para_a_vm(tmp_path):
    """Gerada LOCALMENTE (quem a usa e o curl desta maquina) e impressa: ela morre com a VM."""
    destino = tmp_path / "tmp_do_script"
    proc, saida, _ = _rodada_completa(tmp_path, capturar_tmp=destino)
    assert proc.returncode == 0, saida
    linha = next(ln for ln in saida.splitlines() if "API_KEY efemera" in ln)
    chave = linha.split(":")[-1].strip()
    assert len(chave) >= 16 and chave.isalnum()
    env = (destino / "api_demo_env.sh").read_text(encoding="utf-8")
    assert f"API_KEY='{chave}'" in env


def test_a_chave_passada_a_mao_e_respeitada(tmp_path):
    env, _ = _scripts_remotos_gerados(tmp_path, "--api-key", "chave-de-teste")
    assert "API_KEY='chave-de-teste'" in env


# -- teardown: roda em QUALQUER saida ---------------------------------------------------


def test_sucesso_termina_com_o_teardown_conferido(tmp_path):
    _, saida, _ = _rodada_completa(tmp_path)
    assert "TEARDOWN (saida com status 0)" in saida
    assert "conferencia ok: nenhuma VM sobrou" in saida
    assert "teardown concluido (status final: 0)" in saida
    # o indice e o unico sobrevivente -- e de proposito
    assert f"o indice esta em {INDEX_URI}" in saida


@pytest.mark.parametrize("etapa", ["provision", "bootstrap", "stage", "build_index",
                                   "upload_index", "serve", "probe"])
def test_falha_injetada_em_qualquer_etapa_ainda_roda_o_teardown(tmp_path, etapa):
    proc, saida, (gcloud, curl) = _rodada_completa(tmp_path, fail_at=etapa)
    assert proc.returncode != 0
    assert f"falha INJETADA na etapa '{etapa}'" in saida
    assert "=== teardown concluido" in saida
    assert not gcloud.exists()
    assert not curl.exists()


def test_falha_no_provision_nao_tenta_apagar_vm_nenhuma(tmp_path):
    """Nenhum `create` chegou a acontecer: o teardown e um no-op honesto."""
    _, saida, _ = _rodada_completa(tmp_path, fail_at="provision")
    assert "nenhuma VM foi criada nesta rodada" in saida
    assert "instances delete" not in saida


def test_falha_no_build_tenta_resgatar_o_indice_antes_dos_deletes(tmp_path):
    """Apagar a VM destroi o embed: o resgate vem antes, e com teto de tempo."""
    _, saida, _ = _rodada_completa(tmp_path, fail_at="upload_index")
    resgate = saida.index("tentativa final de subir o indice")
    delete = saida.index("instances delete")
    assert resgate < delete
    assert "timeout 1800 gcloud compute ssh" in saida


def test_o_upload_do_indice_vem_antes_do_serve(tmp_path):
    """Se a Spot for preemptada durante a demo, o que custou GPU ja esta no bucket."""
    _, saida, _ = _rodada_completa(tmp_path)
    assert saida.index("api_demo_remote.sh upload_index") < saida.index(
        "api_demo_remote.sh serve")


# -- o contrato com o codigo do [4] -----------------------------------------------------


def test_o_mesmo_adaptador_embedda_os_itens_e_as_queries(tmp_path):
    """`api/app.py` recusa o startup se o adaptador do indice != o do serving.

    O script tem de usar UMA variavel para os dois lados; se alguem trocar um deles por
    outro caminho, a VM paga sobe e a API se recusa a servir -- e o erro aparece aqui.
    """
    _, remoto = _scripts_remotos_gerados(tmp_path)
    assert 'ADAPTER_LOCAL="$HOME/adapter"' in remoto
    build = _bloco_da_fase(remoto, "build_index")
    serve = _bloco_da_fase(remoto, "serve")
    assert '--adapter "$ADAPTER_LOCAL"' in build
    assert 'ADAPTER_PATH="$ADAPTER_LOCAL"' in serve

    # e a regra do outro lado, exercitada no codigo de verdade:
    pytest.importorskip("fastapi")
    from api.app import check_adapter_do_indice
    meta = {"adapter_path": "/home/u/adapter"}
    check_adapter_do_indice(meta, "/home/u/adapter")          # nao levanta
    with pytest.raises(RuntimeError):
        check_adapter_do_indice(meta, "/home/u/outro")


def test_o_indice_que_sobe_e_o_que_o_build_index_escreve(tmp_path):
    """Se o default do `api/build_index.py` mudar, este teste quebra aqui -- e nao numa
    rodada paga que termina subindo um diretorio vazio."""
    from api.build_index import DEFAULT_OUT
    fonte = SCRIPT.read_text(encoding="utf-8")
    assert f'INDICE_DIR="${{INDICE_DIR:-{DEFAULT_OUT}}}"' in fonte

    _, remoto = _scripts_remotos_gerados(tmp_path)
    build = _bloco_da_fase(remoto, "build_index")
    upload = _bloco_da_fase(remoto, "upload_index")
    assert '--out "$INDICE_LOCAL"' in build
    assert 'gcloud storage rsync -r "$INDICE_LOCAL" "$INDEX_URI"' in upload
    # os tres arquivos do indice sao conferidos ANTES do upload
    for arquivo in ("records.jsonl", "matrix.npy", "index_meta.json"):
        assert arquivo in build


def test_o_stage_baixa_corpus_E_adaptador_e_falha_alto_sem_eles(tmp_path):
    _, remoto = _scripts_remotos_gerados(tmp_path)
    stage = _bloco_da_fase(remoto, "stage")
    assert 'rsync -r "$GCS_CORPUS/bin" "$BIN_REL"' in stage
    assert 'cp "$GCS_CORPUS/records.jsonl" "$RECORDS_REL"' in stage
    assert 'rsync -r "$ADAPTER_URI" "$ADAPTER_LOCAL"' in stage
    # sentinela do PEFT: sem ela a demo embeddaria com a base crua, depois de pagar a GPU
    assert 'adapter_config.json' in stage


def test_o_uvicorn_escuta_so_no_loopback_da_vm(tmp_path):
    """Nenhuma porta aberta na nuvem: quem chega na API e o tunel ssh, e so ele."""
    _, remoto = _scripts_remotos_gerados(tmp_path)
    serve = _bloco_da_fase(remoto, "serve")
    assert "--host 127.0.0.1" in serve
    assert "0.0.0.0" not in remoto
    fonte = SCRIPT.read_text(encoding="utf-8")
    assert "firewall-rules" not in fonte           # nem de longe
    assert "--address" not in fonte                # nenhum IP publico pedido a mao


def test_o_serve_deixa_o_servidor_de_pe_e_so_volta_com_o_health(tmp_path):
    """O `nohup` e NA VM: a sessao ssh termina e o uvicorn tem de sobreviver a ela."""
    _, remoto = _scripts_remotos_gerados(tmp_path)
    serve = _bloco_da_fase(remoto, "serve")
    assert "nohup python3 -m uvicorn api.app:app" in serve
    assert "< /dev/null" in serve                       # sem stdin preso na sessao ssh
    assert "echo $! > \"$UVICORN_PID\"" in serve        # e `$!` e o pid do uvicorn mesmo
    # `setsid` faria `$!` ser de um intermediario que sai na hora, e a espera abaixo
    # acusaria uma morte falsa do startup (o comentario do script explica; o CODIGO e que
    # nao pode te-lo)
    codigo_do_serve = "\n".join(ln for ln in serve.splitlines()
                                if not ln.lstrip().startswith("#"))
    assert "setsid" not in codigo_do_serve
    # espera ATIVA pelo /health, com teto e com o log do uvicorn no erro
    assert 'while [ "$ESPEROU" -lt "$SERVE_TIMEOUT" ]' in serve
    assert "health_ok" in serve
    assert 'tail -n 40 "$UVICORN_LOG"' in serve
    assert "SERVINDO" in serve
    # as env vars que `api/app.py` le no startup
    for var in ('INDEX_PATH="$INDICE_LOCAL"', 'API_KEY="$API_KEY"', 'DEVICE="$DEVICE"'):
        assert var in serve


def test_a_fase_serve_instala_o_servidor_e_o_bootstrap_nao_rebaixa_o_transformers(tmp_path):
    """fastapi/uvicorn/pydantic entram na `serve`; o resto vem do requirements-slurm.

    `api/requirements.txt` NAO pode ser usado na VM: ele pina `transformers<4.51` (alvo do
    Cloud Run) e o Qwen3-VL-Embedding exige >=4.57 -- instala-lo aqui rebaixaria o
    transformers e quebraria o embedder DEPOIS da GPU paga.
    """
    _, remoto = _scripts_remotos_gerados(tmp_path)
    serve = _bloco_da_fase(remoto, "serve")
    assert 'pip install --quiet fastapi "uvicorn[standard]" pydantic' in serve
    bootstrap = _bloco_da_fase(remoto, "bootstrap")
    assert "requirements-slurm.txt" in bootstrap
    # so o CODIGO conta: o comentario cita api/requirements.txt justamente para explicar
    # por que ele NAO e instalado aqui
    codigo = "\n".join(ln for ln in remoto.splitlines() if not ln.lstrip().startswith("#"))
    assert "api/requirements.txt" not in codigo


def test_o_bootstrap_preserva_o_torch_da_imagem(tmp_path):
    """Espelho do `prefer_host_torch`: um torch vindo do pip sombreia o da imagem (sem CUDA)."""
    _, remoto = _scripts_remotos_gerados(tmp_path)
    bootstrap = _bloco_da_fase(remoto, "bootstrap")
    assert "--system-site-packages" in bootstrap
    assert 'rm -rf "$_sp"/torch' in bootstrap


def test_a_demo_nao_instala_ffmpeg_nem_whisper(tmp_path):
    """O embed do baseline A nao reprocessa midia: os binarios ja vem prontos do bucket."""
    codigo = "\n".join(ln for ln in SCRIPT.read_text(encoding="utf-8").splitlines()
                       if not ln.lstrip().startswith("#"))
    assert "apt-get" not in codigo
    assert "faster-whisper" not in codigo
    assert "ffmpeg" not in codigo


# -- auth do gcloud: o token que venceu NO MEIO de uma rodada paga ----------------------
#
# Mesmas quatro pontas do finetune (o incidente que derrubou uma rodada la): falhar ANTES
# de pagar qualquer coisa, e deixar instrucoes impossiveis de perder quando quebra no fim.
# Aqui o `gcloud` dublado nao e o delator: e um duble PROGRAMAVEL, para que o fluxo de
# verdade -- SEM `--dry-run` -- possa rodar sem tocar a nuvem.

_GCLOUD_PROGRAMAVEL = """#!/usr/bin/env bash
# gcloud dublado e OFFLINE: cada chamada vai para o log e o rc vem do ambiente.
printf '%s\\n' "$*" >> "$GCLOUD_LOG"
case "$*" in
    *"-N -L"*)
        # port-forward: fica de pe ate alguem matar. O trap so dispara no TERM, entao o
        # arquivo so existe se o script REALMENTE matou o tunel. A pausa usa o `sleep` do
        # sistema por CAMINHO ABSOLUTO: o do PATH esta dublado (devolve na hora) e o duble
        # faria este processo morrer sozinho, mascarando um tunel orfao.
        trap 'printf "term\\n" >> "$PF_FIM"; exit 0' TERM INT
        PAUSA=""
        for s in /usr/bin/sleep /bin/sleep; do
            if [ -x "$s" ]; then PAUSA="$s"; break; fi
        done
        i=0
        while [ "$i" -lt 60 ]; do
            if [ -n "$PAUSA" ]; then "$PAUSA" 1; else i=$((i + 59)); fi
            i=$((i + 1))
        done
        exit 0 ;;
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
        if [ "${DELETE_NOTFOUND:-0}" = "1" ]; then
            echo "ERROR: (gcloud.compute.instances.delete) Could not fetch resource:" >&2
            echo " - The resource 'projects/p/zones/z/instances/govexe-api-demo' was" \\
                 "not found" >&2
            exit 1
        fi
        if [ "${DELETE_ERRO:-0}" = "1" ]; then
            echo "ERROR: (gcloud.compute.instances.delete) The resource is not ready" >&2
            exit 1
        fi
        exit 0 ;;
    *"instances describe"*)      exit "${DESCRIBE_RC:-1}" ;;
    *) exit 0 ;;
esac
"""

# curl dublado: responde /health e /consulta sem rede, e registra o que recebeu (e assim
# que o teste confere que a chave efemera viajou no header).
_CURL_PROGRAMAVEL = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CURL_LOG"
# FALHA_CURL=1: a API nao responde pelo tunel (o cenario em que a rodada morre com o
# port-forward ABERTO, e quem tem de fecha-lo e o teardown).
if [ "${FALHA_CURL:-0}" = "1" ]; then exit 7; fi
case "$*" in
    *"/health"*)   printf '{"status": "ok"}'; exit 0 ;;
    *"/consulta"*)
        printf '{"results": [{"item_id": "documento-001", "eixo": "documento", '
        printf '"score": 0.81, "pointer_uri": "gs://b/documento-001.png", '
        printf '"descriptor": "PLOA 2026, pagina 12"}], "latency_ms": 42.0}'
        exit 0 ;;
esac
exit 0
"""


def _rodar_de_verdade(tmp_path: Path, *args: str, **env_extra: str):
    """Roda o script SEM `--dry-run`, com gcloud/curl programaveis que nao tocam a nuvem.

    Devolve (CompletedProcess, saida, chamadas_ao_gcloud). `--branch ''` desliga a checagem
    de checkout (fatal fora do dry-run), o `tar` dublado evita empacotar o repo inteiro e o
    `sleep` dublado tira os minutos de espera do teardown -- nenhum dos tres e o objeto
    destes testes.
    """
    binario = tmp_path / "bin_real"
    binario.mkdir()
    log = tmp_path / "chamadas_gcloud.txt"
    curl_log = tmp_path / "chamadas_curl.txt"
    _escrever_sh(binario / "gcloud", _GCLOUD_PROGRAMAVEL)
    _escrever_sh(binario / "curl", _CURL_PROGRAMAVEL)
    _escrever_sh(binario / "tar", "#!/usr/bin/env bash\nexit 0\n")
    _escrever_sh(binario / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    env = dict(os.environ)
    env["PATH"] = f"{binario.as_posix()}{os.pathsep}{env.get('PATH', '')}"
    for chave in VARS_DO_OPERADOR:
        env.pop(chave, None)
    env["GCLOUD_LOG"] = log.as_posix()
    env["CURL_LOG"] = curl_log.as_posix()
    env["PF_FIM"] = (tmp_path / "port_forward_morreu.txt").as_posix()
    env.update(env_extra)
    proc = subprocess.run(
        [BASH, str(SCRIPT), "--adapter", ADAPTER, "--index-uri", INDEX_URI,
         "--branch", "", *args],
        cwd=str(RAIZ), env=env, capture_output=True, text=True, timeout=300)
    return proc, proc.stdout + proc.stderr, (log.read_text(encoding="utf-8")
                                             if log.exists() else "")


def test_token_invalido_aborta_o_preflight_antes_de_qualquer_vm(tmp_path):
    """Token ja vencido na largada nao pode virar VM paga.

    Sem isto a rodada sobe a L4, embedda o corpus, e so descobre o problema na hora de
    subir o indice e apagar a VM -- que foi exatamente o incidente do finetune.
    """
    proc, saida, chamadas = _rodar_de_verdade(tmp_path, AUTH_RC="1")

    assert proc.returncode != 0
    assert "gcloud sem token valido" in saida
    assert "gcloud auth login" in saida
    assert "instances create" not in saida
    assert "instances create" not in chamadas


def test_token_valido_deixa_o_preflight_seguir(tmp_path):
    proc, saida, chamadas = _rodar_de_verdade(
        tmp_path, AUTH_RC="0", API_DEMO_FAIL_AT="provision")

    assert "gcloud sem token valido" not in saida
    assert "auth print-access-token" in chamadas       # a checagem rodou de verdade
    assert "falha INJETADA na etapa 'provision'" in saida
    assert "nenhuma VM foi criada nesta rodada" in saida
    assert "instances create" not in chamadas
    assert proc.returncode != 0


def test_dry_run_nao_executa_o_check_de_auth(tmp_path):
    """`print-access-token` e chamada de REDE -- o ensaio so a anuncia."""
    proc, saida, (gcloud, _) = _rodada_completa(tmp_path)

    assert proc.returncode == 0, saida
    assert "[dry-run] gcloud auth print-access-token" in saida
    assert not gcloud.exists()


def test_conta_sujeita_a_reauth_so_avisa(tmp_path):
    proc, saida, _ = _rodar_de_verdade(
        tmp_path, AUTH_RC="0", API_DEMO_FAIL_AT="provision",
        CONTA_ATIVA="fulano@discente.ufg.br")

    assert "AVISO: conta fulano@discente.ufg.br pode exigir reauth" in saida
    assert "service account" in saida
    assert "falha INJETADA na etapa 'provision'" in saida


def test_delete_que_falha_por_auth_imprime_o_banner_de_recuperacao(tmp_path):
    """Se o token morreu no meio, o teardown nao consegue apagar a VM.

    O retry nao resolve (so um `gcloud auth login` humano resolve), entao o que resta e
    nao deixar o operador adivinhar: VM, zona, projeto e o resgate do indice, prontos.
    """
    proc, saida, _ = _rodar_de_verdade(
        tmp_path, "--zones", "us-central1-a",
        AUTH_RC="0", CREATE_RC="1", DELETE_REAUTH="1", DESCRIBE_RC="0",
        TEARDOWN_RETRIES="1")

    assert proc.returncode != 0
    assert "RECUPERACAO MANUAL NECESSARIA (auth do gcloud expirou)" in saida
    assert "1. gcloud auth login" in saida
    assert ("3. gcloud compute instances delete govexe-api-demo "
            "--zone=us-central1-a --project=residencia-511517 --quiet") in saida
    assert "data/govexe/indice_api_v4" in saida        # o resgate do indice
    assert INDEX_URI in saida
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


# -- `not found` no delete: zona vazia nao e falha a retentar ---------------------------
#
# Com fallback de zona, o teardown varre TODAS as zonas tentadas e as que nunca tiveram VM
# respondem "was not found". Retentar isso queima TEARDOWN_RETRIES x TEARDOWN_ESPERA por
# zona vazia -- com a VM de verdade, na zona seguinte, ainda cobrando.


def test_delete_em_zona_sem_vm_e_sucesso_sem_retry(tmp_path):
    proc, saida, chamadas = _rodar_de_verdade(
        tmp_path, "--zones", "us-central1-a",
        AUTH_RC="0", CREATE_RC="1", DELETE_NOTFOUND="1", DESCRIBE_RC="1",
        TEARDOWN_RETRIES="3")

    assert "nada a apagar nessa zona (VM nao existe aqui)" in saida
    # curto-circuito: UMA tentativa, sem a linha de retry e sem o veredito de falha
    assert len([ln for ln in chamadas.splitlines() if "instances delete" in ln]) == 1
    assert "tentativa 1 falhou" not in saida
    assert "FALHOU em 3 tentativas" not in saida
    # e nao se confunde com auth: o banner continua sendo so do token vencido
    assert "RECUPERACAO MANUAL NECESSARIA" not in saida
    assert "conferencia ok: nenhuma VM sobrou" in saida


def test_delete_que_falha_de_verdade_continua_retentando(tmp_path):
    """So o `not found` curto-circuita: erro generico segue no caminho de retry de sempre."""
    proc, saida, chamadas = _rodar_de_verdade(
        tmp_path, "--zones", "us-central1-a",
        AUTH_RC="0", CREATE_RC="1", DELETE_ERRO="1", DESCRIBE_RC="1",
        TEARDOWN_RETRIES="2")

    assert "nada a apagar nessa zona" not in saida
    assert len([ln for ln in chamadas.splitlines() if "instances delete" in ln]) == 2
    assert "tentativa 1 falhou" in saida
    assert "FALHOU em 2 tentativas" in saida


# -- o vazamento LOCAL: o tunel e processo DESTA maquina --------------------------------
#
# O unico teste que roda o FLUXO INTEIRO (sem --dry-run, com gcloud/curl/tar dublados). Ele
# existe por uma razao so: provar que o port-forward morre. Um `ssh -N` orfao segura a
# porta local e mantem sessao contra uma VM que o teardown acabou de apagar -- vazamento
# que nenhuma conferencia de `instances list` pegaria.


def test_o_fluxo_inteiro_prova_a_api_de_fora_e_mata_o_tunel(tmp_path):
    saida_json = tmp_path / "consultas.jsonl"
    proc, saida, chamadas = _rodar_de_verdade(
        tmp_path, "--zones", "us-central1-a", "--saida", saida_json.as_posix(),
        AUTH_RC="0", TEARDOWN_RETRIES="1")

    assert proc.returncode == 0, saida
    # a demo passou pelas 5 fases remotas e pelo tunel
    for fase in FASES_REMOTAS:
        assert f"api_demo_remote.sh {fase}" in chamadas
    assert "-N -L 8080:127.0.0.1:8080" in chamadas
    assert "GET /health (da maquina local, pelo tunel)" in saida
    assert "4/4 consultas responderam" in saida

    # as consultas viraram JSON em disco, uma por linha, com results + latency_ms
    linhas = [json.loads(ln) for ln in
              saida_json.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(linhas) == 4
    assert all(r["resposta"]["results"] for r in linhas)
    assert all("latency_ms" in r["resposta"] for r in linhas)
    assert "orcamento" in linhas[0]["query"]

    # a chave efemera viajou no header (e e a mesma que o log imprimiu)
    chave = next(ln for ln in saida.splitlines() if "API_KEY efemera" in ln).split(":")[-1].strip()
    curl_log = (tmp_path / "chamadas_curl.txt").read_text(encoding="utf-8")
    assert f"X-API-Key: {chave}" in curl_log

    # E O PONTO DESTE TESTE: o tunel foi morto (o duble so grava no SIGTERM)
    assert "fechando o port-forward local" in saida
    assert (tmp_path / "port_forward_morreu.txt").is_file(), \
        "o port-forward local sobreviveu ao teardown (tunel orfao)"
    assert "teardown concluido (status final: 0)" in saida


def test_tunel_aberto_quando_a_rodada_morre_e_fechado_pelo_teardown(tmp_path):
    """O caso que importa: a API nao responde, o script morre com o tunel JA ABERTO.

    Quem fecha, entao, e o teardown -- e antes dos deletes, porque matar processo local
    nao custa rede e nao pode depender de o gcloud ainda funcionar.
    """
    saida_json = tmp_path / "consultas.jsonl"
    proc, saida, _ = _rodar_de_verdade(
        tmp_path, "--zones", "us-central1-a", "--saida", saida_json.as_posix(),
        AUTH_RC="0", TEARDOWN_RETRIES="1", FALHA_CURL="1", PF_TIMEOUT="3")

    assert proc.returncode != 0
    assert "o /health nao respondeu pelo tunel" in saida
    teardown = saida[saida.index("=== TEARDOWN"):]
    assert "fechando o port-forward local" in teardown
    # fora do dry-run o delete nao e ecoado como comando: o marcador e o anuncio dele
    assert teardown.index("fechando o port-forward local") < teardown.index(
        "apagando a VM govexe-api-demo em us-central1-a")
    assert (tmp_path / "port_forward_morreu.txt").is_file(), \
        "o port-forward local sobreviveu ao teardown (tunel orfao)"
