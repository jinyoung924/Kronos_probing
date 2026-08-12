"""경로 해석 — Colab / 로컬 어디서 돌아도 같은 코드가 동작하도록 한다.

저장소 배치
  <repo>/experiment/code/kexp/paths.py   <- 이 파일
  <repo>/Kronos/model/kronos.py          <- Kronos 구현체

산출물 배치 원칙
  DRIVE  : 작고 오래 보존해야 하는 것 (합성 데이터셋, LDR/steering 결과, 그림, 리포트)
  SCRATCH: 크고 재계산이 싼 것 (레이어별 활성화 텐서). Colab 로컬 디스크.
           세션이 끊기면 사라지지만 Stage 2 재실행이 몇 분이면 되므로 Drive 용량을
           잡아먹는 것보다 낫다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# <repo>/experiment/code/kexp/paths.py -> parents[3] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[3]
KRONOS_DIR = REPO_ROOT / "Kronos"
CODE_DIR = REPO_ROOT / "experiment" / "code"
CONFIRM_DIR = REPO_ROOT / "experiment" / "confirm_data"


def in_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").is_dir()


def drive_root() -> Path:
    """오래 보존할 산출물의 루트."""
    if in_colab():
        return Path(os.environ.get("KEXP_DRIVE", "/content/drive/MyDrive/Kronos_probing"))
    return REPO_ROOT / "experiment" / "_out"


def scratch_root() -> Path:
    """대용량 중간 산출물의 루트 (재계산 가능, 백업하지 않음)."""
    if in_colab():
        return Path(os.environ.get("KEXP_SCRATCH", "/content/kexp_scratch"))
    return REPO_ROOT / "experiment" / "_scratch"


# --- 하위 디렉토리 ---------------------------------------------------------

def data_dir(tag: str, variant: str = "") -> Path:
    """variant 는 노이즈 종류('ou' / 'rw') 처럼 같은 tag 안의 갈래를 가리킨다."""
    return drive_root() / "data" / tag / variant if variant else drive_root() / "data" / tag


def activations_dir(tag: str, variant: str = "") -> Path:
    base = scratch_root() / "activations" / tag
    return base / variant if variant else base


def results_dir(tag: str) -> Path:
    return drive_root() / "results" / tag


def figs_dir(tag: str) -> Path:
    return drive_root() / "figs" / tag


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def bootstrap_sys_path() -> None:
    """Kronos 구현체와 kexp 패키지를 임포트 가능하게 만든다.

    Kronos 내부 코드가 `from model.module import *` 처럼 `model` 을 최상위로
    임포트하므로, Kronos 디렉토리 자체가 sys.path 에 있어야 한다.
    """
    for p in (str(CODE_DIR), str(KRONOS_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
