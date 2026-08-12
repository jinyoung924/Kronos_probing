"""명세 1, 2절 — 고빈도 잠재 경로 기반 OHLC candlestick 합성.

핵심 아이디어
    미세 시간 격자 tau in {1..N*T} 위에서 로그가격 경로 p(tau) 를 만들고,
    비중복 구간 [(t-1)N+1, tN] 마다 OHLC 를 집계한다.

        O_t = p((t-1)N+1)
        H_t = max p(tau),  L_t = min p(tau)
        C_t = p(tN)

    이렇게 하면 coherence 제약 L <= min(O,C) <= max(O,C) <= H 가 정의상
    자동 충족된다 (사후 보정이 필요 없다).

두 클래스
    base  : p_c(tau) = b + W(tau)
    trend : p_s(tau) = b + m * tau + W(tau)

    같은 샘플 인덱스 k 에 대해 **동일한 노이즈 실현치** W_k 를 공유한다.
    따라서 두 클래스의 차이는 정확히 m*tau 항 하나뿐이다.

노이즈 프로세스 (명세 1절은 랜덤워크와 OU 를 모두 허용)
    rw : W(tau) = sigma * cumsum(eps)             — 누적, 비정상
    ou : W(tau) = (1-theta) W(tau-1) + sigma*eps  — 평균회귀, 정상

    rw 를 쓰면 base 클래스 표본의 상당수가 우연히 강한 추세를 갖는다. 이는
    "추세 없음" 이라는 레이블을 부분적으로 거짓으로 만들고, z-score 가 스케일을
    없애기 때문에 snr 을 아무리 키워도 입력단 LDR 이 약 1.9 에서 포화한다.
    ou 는 장기 drift 가 구조적으로 제거되므로 개념이 명확히 정의된다.

가격 레벨 처리
    명세는 p(tau) 를 로그가격으로 정의하지만, 여기서는 그 값을 그대로 가격
    레벨로 사용한다(옵션 A). exp() 를 취하면 선형 drift 가 지수 추세가 되어
    "선형 증가 모멘텀" 이라는 개념 자체가 흐려지고, 명세 3절의
    Affinity Preservation 논증(z-score 는 일차 변환이므로 선형성 보존)도
    성립하지 않게 된다. base_level 상수 b 는 가격을 양수 구간에 두기 위한
    것이며 채널별 z-score 가 이를 상쇄하므로 모델 입력에는 영향이 없다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# x[:, :, k] 의 채널 순서. KronosPredictor 가 기대하는 순서와 같다.
CH_OPEN, CH_HIGH, CH_LOW, CH_CLOSE, CH_VOL, CH_AMT = range(6)


def aggregate_paths(paths: np.ndarray, n_micro: int) -> np.ndarray:
    """미세 경로 [n, N*T] -> OHLCVA [n, T, 6].

    Volume/Amount 는 명세 2절에 따라 0 으로 둔다.
    """
    n = paths.shape[0]
    grid = paths.reshape(n, -1, n_micro)          # [n, T, N]
    T = grid.shape[1]

    x = np.zeros((n, T, 6), dtype=np.float32)
    x[:, :, CH_OPEN] = grid[:, :, 0]
    x[:, :, CH_HIGH] = grid.max(axis=2)
    x[:, :, CH_LOW] = grid.min(axis=2)
    x[:, :, CH_CLOSE] = grid[:, :, -1]
    return x


def make_noise(rng, n: int, n_total: int, cfg, noise_type: str) -> np.ndarray:
    """[n, n_total] 노이즈 경로 W(tau)."""
    if noise_type == "rw":
        # 랜덤워크는 정상 분포가 없다. 0 에서 출발하는 것이 정의 그 자체다.
        return np.cumsum(rng.standard_normal((n, n_total)) * cfg.sigma, axis=1)

    if noise_type == "ou":
        from scipy.signal import lfilter
        # burn-in 이 필수다. 영 초기조건으로 시작하면 모든 샘플이 같은 지점에서
        # 출발해 초기 구간의 분산이 정상상태보다 작아진다. 그 인위적인 공통
        # 출발점은 시퀀스 전체 z-score 와 결합해 첫 토큰의 클래스 분리도를
        # 실제보다 크게 부풀린다 (정상상태 도달까지 약 1/theta 스텝).
        burn = min(int(10.0 / cfg.ou_theta), 20000)
        eps = rng.standard_normal((n, n_total + burn)) * cfg.sigma
        w = lfilter([1.0], [1.0, -(1.0 - cfg.ou_theta)], eps, axis=1)
        return w[:, burn:]

    raise ValueError(f"알 수 없는 noise_type: {noise_type}")


def generate_pair(cfg, n_samples: int, seed: int, noise_type: str | None = None,
                  chunk: int = 256):
    """base / trend 두 클래스를 생성한다.

    Returns:
        x_base  [n, T, 6] float32
        x_trend [n, T, 6] float32
        snrs    [n] float64  — 샘플별로 뽑힌 drift 세기
    """
    noise_type = noise_type or cfg.noise_type
    rng = np.random.default_rng(seed)
    n_total = cfg.n_micro_total
    tau = np.arange(1, n_total + 1, dtype=np.float64)

    snrs = rng.uniform(cfg.snr_min, cfg.snr_max, size=n_samples)

    xb_parts, xs_parts = [], []
    for start in range(0, n_samples, chunk):
        end = min(start + chunk, n_samples)
        noise = make_noise(rng, end - start, n_total, cfg, noise_type)

        p_base = cfg.base_level + noise
        # 샘플별 drift 기울기 (전체설계 Step 1 의 m ~ U(m_min, m_max))
        m = np.array([cfg.drift_per_micro(s, noise_type) for s in snrs[start:end]])[:, None]
        p_trend = p_base + m * tau[None, :]

        xb_parts.append(aggregate_paths(p_base, cfg.n_micro))
        xs_parts.append(aggregate_paths(p_trend, cfg.n_micro))

    return np.concatenate(xb_parts), np.concatenate(xs_parts), snrs


def make_timestamps(cfg) -> pd.DatetimeIndex:
    """모든 샘플·두 클래스가 공유하는 단일 타임스탬프.

    Kronos 는 TemporalEmbedding 을 더하므로, 클래스마다 타임스탬프가 다르면
    그것이 곧 교란 변수가 된다.
    """
    return pd.date_range(cfg.start_time, periods=cfg.T, freq=f"{cfg.freq_minutes}min")


# --- 검증 -------------------------------------------------------------------

def fingerprint(*arrays: np.ndarray, n_head: int = 8) -> str:
    """데이터셋 내용 지문.

    config_hash 만으로는 부족하다. 생성 로직(synth.py)이 바뀌어도 설정값은
    그대로일 수 있고, 그러면 하위 stage 가 낡은 산출물을 조용히 재사용한다.
    실제 값에서 지문을 떠서 하위 stage 가 대조하게 한다.
    """
    import hashlib
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a[:n_head], dtype=np.float32).tobytes())
        h.update(str(a.shape).encode())
    return h.hexdigest()[:16]


def check_coherence(x: np.ndarray, tol: float = 1e-5) -> dict:
    """L <= min(O,C) <= max(O,C) <= H 위반 여부."""
    o, h, l, c = x[:, :, CH_OPEN], x[:, :, CH_HIGH], x[:, :, CH_LOW], x[:, :, CH_CLOSE]
    low_ok = l <= np.minimum(o, c) + tol
    high_ok = np.maximum(o, c) <= h + tol
    hl_ok = l <= h + tol
    return {
        "violations_low": int((~low_ok).sum()),
        "violations_high": int((~high_ok).sum()),
        "violations_hl": int((~hl_ok).sum()),
        "ok": bool(low_ok.all() and high_ok.all() and hl_ok.all()),
    }


def check_noise_sharing(x_base: np.ndarray, x_trend: np.ndarray, cfg, snrs: np.ndarray,
                        noise_type: str | None = None) -> dict:
    """두 클래스가 동일 노이즈를 공유하는지 확인한다.

    close 채널의 차이는 정확히 m * (t * N) 이어야 한다 (t = 1..T).
    High/Low 는 max/min 연산이 drift 와 교환되지 않으므로 근사만 성립한다
    (drift 가 단조 증가라 실제로는 같은 위치에서 극값을 갖는 경우가 많다).
    """
    t_idx = np.arange(1, cfg.T + 1, dtype=np.float64) * cfg.n_micro
    m = np.array([cfg.drift_per_micro(s, noise_type) for s in snrs])[:, None]
    expected = m * t_idx[None, :]
    actual = (x_trend[:, :, CH_CLOSE] - x_base[:, :, CH_CLOSE]).astype(np.float64)
    err = np.abs(actual - expected)
    return {
        "max_abs_err_close": float(err.max()),
        "rel_err_close": float(err.max() / (np.abs(expected).max() + 1e-12)),
    }


def normalized_stats(x: np.ndarray, clip: float) -> dict:
    """Kronos 전처리(채널별 z-score) 후의 분포 통계 — 명세 3절 캘리브레이션."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    z = (x - mean) / (std + 1e-5)
    price = z[:, :, :4]
    return {
        "clip_fraction": float((np.abs(price) > clip).mean()),
        "min": float(price.min()),
        "max": float(price.max()),
        "p01": float(np.percentile(price, 1)),
        "p99": float(np.percentile(price, 99)),
        # 정규화된 close 의 시간에 대한 회귀 기울기 — 개념의 세기를 나타내는 지표
        "close_slope_mean": float(_slope(z[:, :, CH_CLOSE]).mean()),
        "close_slope_std": float(_slope(z[:, :, CH_CLOSE]).std()),
    }


def _slope(series: np.ndarray) -> np.ndarray:
    """[n, T] 각 행에 대한 최소제곱 기울기."""
    T = series.shape[1]
    t = np.arange(T, dtype=np.float64)
    t_c = t - t.mean()
    return (series * t_c).sum(axis=1) / (t_c**2).sum()
