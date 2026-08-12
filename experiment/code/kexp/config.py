"""실험 하이퍼파라미터 단일 소스.

모든 stage 는 여기서 값을 읽고, 산출물의 meta.json 에 config 해시를 남긴다.
Stage 0 에서 모델 스펙(d_model, n_layers)을 실측한 뒤 확정할 값은
주석에 TODO(stage0) 로 표시해 두었다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


# --- 모델 --------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCfg:
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    model_id: str = "NeoQuasar/Kronos-small"
    max_context: int = 512   # Kronos-small / base 의 상한
    clip: float = 5.0        # KronosPredictor 의 outlier clipping 범위 (명세 3절)


# --- 합성 데이터셋 (명세 1, 2절) ----------------------------------------------

@dataclass(frozen=True)
class DataCfg:
    T: int = 512             # K-line 스텝 수 = 트랜스포머 시퀀스 위치 수
    n_micro: int = 15        # 봉 하나를 만드는 미세 격자 수 N
    sigma: float = 0.01      # 고빈도 변동성 (미세 스텝당 표준편차)
    m: float = 0.002         # trend 클래스의 미세 스텝당 drift. TODO(stage1): 클리핑률 보고 캘리브레이션
    base_level: float = 100.0  # 가격을 양수 구간에 두기 위한 상수 b (z-score 로 상쇄됨)
    n_samples: int = 1024    # 클래스당 샘플 수. LDA 조건수를 위해 d_model 보다 커야 한다
    seed: int = 20250812
    # 명세 2절: Volume / Amount 는 교란 변수라 0 으로 고정
    zero_volume: bool = True
    # 두 클래스가 동일한 타임스탬프를 쓰도록 고정 (TemporalEmbedding 이 교란되지 않게)
    freq_minutes: int = 5
    start_time: str = "2024-01-01 09:00:00"


# --- 프로빙 / LDR (명세 4, 5, 6절) --------------------------------------------

@dataclass(frozen=True)
class ProbeCfg:
    extract_batch_size: int = 16
    store_dtype: str = "float16"
    shrinkage: float = 1e-3      # (Sigma_w + eps*I) 의 eps. 상대 스케일로 적용
    test_ratio: float = 0.3      # held-out LDR 도 함께 보고


# --- Steering (명세 7절) -------------------------------------------------------

@dataclass(frozen=True)
class SteerCfg:
    methods: tuple = ("median", "mean", "lda")
    lambdas: tuple = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
    pred_len: int = 64
    sample_count: int = 5
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9


@dataclass(frozen=True)
class Config:
    tag: str = "v1"
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    probe: ProbeCfg = field(default_factory=ProbeCfg)
    steer: SteerCfg = field(default_factory=SteerCfg)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


CFG = Config()
