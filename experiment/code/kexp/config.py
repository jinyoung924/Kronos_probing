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
    """Stage 0 실측 결과 (--compare):
        mini   ctx=2048  L=4   d=256   4.1M
        small  ctx=512   L=8   d=512  24.7M
        base   ctx=512   L=12  d=832 102.3M   <- 명세의 L=12, T=512 와 일치
    """
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    model_id: str = "NeoQuasar/Kronos-base"
    max_context: int = 512   # Kronos-small / base 의 상한
    clip: float = 5.0        # KronosPredictor 의 outlier clipping 범위 (명세 3절)


# 이름 -> (tokenizer_id, model_id, 학습 시 컨텍스트). --model 플래그로 전환한다.
#
# max_context 는 가중치의 속성이 아니라 런타임 인자다. Kronos 는 절대 위치
# 임베딩 없이 RoPE 만 쓰므로 컨텍스트 길이를 자유롭게 줄일 수 있다. mini 를
# 로컬 프록시로 쓸 때는 반드시 512 를 강제해야 base 와 같은 AR 롤링 분기를 탄다.
MODEL_ZOO = {
    "mini": ("NeoQuasar/Kronos-Tokenizer-2k", "NeoQuasar/Kronos-mini", 2048),
    "small": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small", 512),
    "base": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-base", 512),
}


def resolve_model(name: str, max_context: int = 512) -> ModelCfg:
    """--model 인자 -> ModelCfg. max_context 는 항상 실험 설정값으로 강제한다."""
    if name not in MODEL_ZOO:
        raise ValueError(f"알 수 없는 모델: {name}. 가능: {list(MODEL_ZOO)}")
    tok_id, model_id, _ = MODEL_ZOO[name]
    return ModelCfg(tokenizer_id=tok_id, model_id=model_id, max_context=max_context)


# --- 합성 데이터셋 (명세 1, 2절) ----------------------------------------------

@dataclass(frozen=True)
class DataCfg:
    """명세 1, 2절의 고빈도 잠재 경로 -> OHLC 집계.

    drift 를 raw `m` 대신 SNR 로 파라미터화한 이유:
        m 값 자체는 해석이 불가능하다. 실제로 중요한 것은 시퀀스 전체에 걸친
        총 drift 가 종단 노이즈 대비 얼마나 큰가이며, 그 비율이
            snr = (m * n_micro * T) / (sigma * sqrt(n_micro * T))
        이다. 따라서 snr 을 지정하고 m 을 역산한다.
            m = snr * sigma / sqrt(n_micro * T)
        또한 z-score 정규화가 스케일을 없애므로, 절대적인 m/sigma 가 아니라
        오직 이 비율만이 모델이 보는 입력을 결정한다.
    """
    T: int = 512             # K-line 스텝 수 = 트랜스포머 시퀀스 위치 수
    n_micro: int = 15        # 봉 하나를 만드는 미세 격자 수 N
    sigma: float = 0.01      # 고빈도 변동성 (미세 스텝당 표준편차)

    # 노이즈 프로세스. 명세 1절은 랜덤워크와 OU 를 모두 허용한다.
    #   "ou" : 평균회귀. base 가 진짜로 추세를 갖지 않아 개념이 명확히 정의된다.
    #   "rw" : 랜덤워크. 금융 현실에 가깝지만 base 표본의 상당수가 우연한 추세를
    #          가지므로 레이블 노이즈가 생기고, 입력단 LDR 이 ~1.9 에서 포화한다
    #          (z-score 가 스케일을 없애므로 snr 을 키워도 개선되지 않는다).
    # 1차 실험은 ou 가 주, rw 는 robustness check.
    noise_type: str = "ou"
    ou_theta: float = 0.01   # 미세 스텝당 평균회귀율. 1/theta=100 스텝 ~= 6.7 봉

    # trend 클래스의 drift 세기. 전체설계 Step 1 을 따라 샘플마다 무작위로 뽑는다.
    # snr_min == snr_max 로 두면 구현명세 1절의 고정 m 케이스가 된다.
    snr_min: float = 3.0
    snr_max: float = 6.0

    base_level: float = 100.0  # 가격을 양수 구간에 두기 위한 상수 b (z-score 로 상쇄됨)
    n_samples: int = 2048    # 클래스당 샘플 수. LDA 조건수를 위해 d_model(=832) 보다 충분히 커야 한다
    seed: int = 20250812
    # 명세 2절: Volume / Amount 는 교란 변수라 0 으로 고정
    zero_volume: bool = True
    # 두 클래스가 동일한 타임스탬프를 쓰도록 고정 (TemporalEmbedding 이 교란되지 않게)
    freq_minutes: int = 5
    start_time: str = "2024-01-01 09:00:00"

    @property
    def n_micro_total(self) -> int:
        return self.n_micro * self.T

    def noise_scale(self, noise_type: str | None = None) -> float:
        """시퀀스 끝에서의 노이즈 크기 — snr 의 분모.

        rw : 종단 표준편차 sigma * sqrt(n_total) (시간에 따라 커진다)
        ou : 정상상태 표준편차 sigma / sqrt(1 - (1-theta)^2) (시간과 무관)
        """
        nt = noise_type or self.noise_type
        if nt == "rw":
            return self.sigma * (self.n_micro_total ** 0.5)
        if nt == "ou":
            a = 1.0 - self.ou_theta
            return self.sigma / ((1.0 - a * a) ** 0.5)
        raise ValueError(f"알 수 없는 noise_type: {nt}")

    def drift_per_micro(self, snr: float, noise_type: str | None = None) -> float:
        """snr(= 총 drift / 노이즈 크기) -> 미세 스텝당 drift m."""
        return snr * self.noise_scale(noise_type) / self.n_micro_total


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
