"""kexp — Kronos concept-probing / steering 실험 패키지.

패키지 이름을 `kexp`로 둔 이유:
  - `experiment/code` 를 sys.path 에 올리므로, 하위 패키지가 `model` 이면
    Kronos 저장소의 `model` 패키지(`from model.kronos import Kronos`)와 충돌한다.
  - `code` 자체도 파이썬 표준 라이브러리 모듈명이라 sys.path 최상위로 노출하면 안 된다.
"""

__all__ = ["config", "paths", "kronos_loader", "hooks"]
