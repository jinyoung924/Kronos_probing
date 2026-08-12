# ==========================================================================
#  Kronos_probing — Colab control cell
#
#  이 파일 전체를 Colab 셀 하나에 붙여넣고 실행한다.
#
#  런타임 설정: 런타임 > 런타임 유형 변경 > T4 GPU (표준 RAM)
#    - Kronos-small(24.7M)에는 T4 로 충분하다. Stage 5 의 AR 추론이 느리면
#      그때 L4 로 올린다.
#
#  아래 STAGE 값만 바꿔가며 재실행하면 된다.
# ==========================================================================

CONTROL_VERSION = "v3-multistage"   # 실행 시 출력된다. 이 값이 안 보이면 옛 셀이다.

# 실행할 stage. 아래 형태를 모두 받는다.
#   3            단일
#   [1, 2, 3]    순서대로
#   "1,2,3"      문자열도 허용
STAGE = 0

# stage 인자. 문자열이면 모든 stage 에, dict 면 stage 별로 적용된다.
#   ""                    인자 없음
#   "--smoke"             전 stage 에 --smoke
#   {1: "--force"}        stage 1 에만 --force
EXTRA_ARGS = ""

BRANCH = "master"
REINSTALL_DEPS = False   # 의존성을 다시 설치하려면 True

# --------------------------------------------------------------------------

import ast
import glob
import os
import subprocess


def parse_stages(value):
    """STAGE 를 정수 리스트로 정규화한다."""
    if isinstance(value, str):
        value = ast.literal_eval(value.strip()) if value.strip() else []
    if isinstance(value, (int, float)):
        value = [value]
    stages = [int(s) for s in value]
    if not stages:
        raise ValueError("STAGE 가 비어 있다.")
    return stages


def args_for(stage, extra):
    if isinstance(extra, dict):
        return str(extra.get(stage, extra.get(str(stage), "")))
    return str(extra or "")

REPO_URL = "https://github.com/jinyoung924/Kronos_probing.git"
REPO_DIR = "/content/Kronos_probing"
DRIVE_DIR = "/content/drive/MyDrive/Kronos_probing"
SCRATCH_DIR = "/content/kexp_scratch"


def sh(cmd, cwd=None, check=True):
    """출력을 실시간으로 흘려보내며 명령을 실행한다."""
    print(f"\n$ {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if check and proc.returncode != 0:
        raise RuntimeError(f"실패 (exit {proc.returncode}): {cmd}")
    return proc.returncode


# 0) 실행 계획 확인 ----------------------------------------------------------
# 인자 오류로 긴 clone/설치를 낭비하지 않도록 가장 먼저 검사한다.
STAGES = parse_stages(STAGE)
print(f"[colab_control {CONTROL_VERSION}]")
print(f"  실행할 stage: {STAGES}")
for _s in STAGES:
    print(f"    stage {_s}: python stage{_s}_*.py {args_for(_s, EXTRA_ARGS)}")

# 1) Google Drive 마운트 -----------------------------------------------------
from google.colab import drive  # noqa: E402

drive.mount("/content/drive")
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

# 2) 저장소 clone / 최신화 ---------------------------------------------------
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    sh("git fetch origin", cwd=REPO_DIR)
    sh(f"git checkout {BRANCH}", cwd=REPO_DIR)
    sh(f"git reset --hard origin/{BRANCH}", cwd=REPO_DIR)
else:
    sh(f"git clone --depth 50 --branch {BRANCH} {REPO_URL} {REPO_DIR}")
sh("git log --oneline -1", cwd=REPO_DIR)

# 3) 의존성 (최초 1회) -------------------------------------------------------
MARKER = "/content/.kexp_deps_installed"
if REINSTALL_DEPS or not os.path.exists(MARKER):
    sh("pip install -q einops huggingface_hub safetensors seaborn")
    open(MARKER, "w").close()
else:
    print("\n[deps] 이미 설치됨 (다시 설치하려면 REINSTALL_DEPS = True)")

# 4) stage 실행 --------------------------------------------------------------
env_prefix = f"KEXP_DRIVE={DRIVE_DIR} KEXP_SCRATCH={SCRATCH_DIR} PYTHONUNBUFFERED=1"
results = {}

for stage in STAGES:
    pattern = f"{REPO_DIR}/experiment/code/stages/stage{stage}_*.py"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"stage 스크립트를 찾을 수 없다: {pattern}\n"
            f"저장소에 해당 stage 가 아직 푸시되지 않았을 수 있다.")
    if len(matches) > 1:
        raise RuntimeError(f"stage {stage} 에 해당하는 스크립트가 여러 개다: {matches}")
    script = matches[0]
    extra = args_for(stage, EXTRA_ARGS)

    print(f"\n{'=' * 70}\n실행: {os.path.basename(script)} {extra}\n{'=' * 70}")
    rc = sh(f"{env_prefix} python {script} {extra}", cwd=REPO_DIR, check=False)
    results[stage] = rc
    print(f"\n{'-' * 70}\nstage {stage} 종료 코드: {rc}\n{'-' * 70}")
    if rc != 0:
        print(f"stage {stage} 가 실패했다. 이후 stage 는 실행하지 않는다.")
        break

print(f"\n{'=' * 70}\n전체 결과: {results}\n{'=' * 70}")
