"""Stage 4 — 개념 조정 벡터 산출 (설계 Step 4, 명세 7절).

    S_i = M_{i,s} - M_{i,c}

명세 7절은 이 하나의 벡터를 컨텍스트 전 위치와 AR 생성 매 스텝에 동일하게
더한다. 따라서 위치별 차이 벡터 S_i^(t) [T, D] 를 레이어당 벡터 하나 [D] 로
집약해야 하고, 어느 구간에서 집약하느냐가 선택지가 된다.

    token_mean : 전 위치 평균. "모든 위치에 더한다" 는 개입 방식과 형식적으로 일관
    late       : 후반 구간 평균. AR 생성이 일어나는 인과 경계 근처
    peak       : Stage 3 의 평활 LDR 정점 부근. 개념이 가장 선명한 구간

기본값이 late 인 이유
    ||S_i^(t)|| 는 t=0 에서 거대한 스파이크를 갖는다. 시퀀스 전체 z-score 때문에
    trend 표본의 첫 봉이 램프 바닥(정규화값 약 -1.4)에 위치하는 반면 base 는 0
    부근이라, 첫 토큰의 클래스 평균차가 구조적으로 크다. 그런데 같은 위치에서
    LDR 은 가장 낮다 (클래스내 분산도 그만큼 크기 때문). 즉 t=0 의 큰 평균차는
    모멘텀 개념이 아니라 "가격 레벨 오프셋" 방향이다. token_mean 은 이 성분에
    지배되어 late 와 코사인이 0.2 수준까지 떨어진다. 개념 방향을 원한다면
    이 구간을 피해야 한다.

세 방식이 실질적으로 같은 방향인지(코사인 유사도)를 함께 보고해, 선택이
결과를 좌우하는지 판단할 수 있게 한다.

조정 강도 lambda 에 대하여
    명세 7절은 lambda in [0.05, 0.5] 를 권장하지만, 이는 ||S_i|| 와 ||h_i|| 의
    비에 의존하는 값이라 절대값으로는 의미가 없다. Kronos-base 는 활성화 노름이
    레이어별로 6.6 -> 47.0 까지 7 배 변한다. 같은 lambda 를 전 레이어에 쓰면
    레이어마다 개입 세기가 7 배 달라진다. 따라서 상대 강도

        lambda_rel = lambda_i * ||S_i|| / ||h_i||

    를 고정하고 레이어별 lambda_i 를 역산한다.
"""

from __future__ import annotations

import numpy as np

METHODS = ("median", "mean", "lda")
REGIONS = ("token_mean", "late", "peak")


def region_slice(T: int, region: str, peak_token: int, frac: float = 0.0625) -> slice:
    k = max(1, int(T * frac))
    if region == "token_mean":
        return slice(0, T)
    if region == "late":
        return slice(T - k, T)
    if region == "peak":
        lo = max(0, min(T - k, peak_token - k // 2))
        return slice(lo, lo + k)
    raise ValueError(f"알 수 없는 region: {region}")


def build_vectors(stats: dict, peak_token: int) -> dict:
    """한 레이어의 클래스 통계 -> 방식 x 구간별 steering vector.

    Args:
        stats: class_stats_layer{i}.npz 의 내용.
               median_base/median_trend/mean_base/mean_trend 각 [T, D]
    Returns:
        {(method, region): [D]} 및 위치별 차이 벡터 diff_median [T, D]
    """
    T = stats["median_base"].shape[0]
    diffs = {
        "median": stats["median_trend"] - stats["median_base"],   # [T, D]
        "mean": stats["mean_trend"] - stats["mean_base"],
    }
    out = {}
    for method, d in diffs.items():
        for region in REGIONS:
            out[(method, region)] = d[region_slice(T, region, peak_token)].mean(axis=0)
    return out, diffs


def scale_like(vec: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """방향은 vec, 크기는 reference 에 맞춘다.

    LDA 방향은 스케일이 임의라, median 차이 벡터와 같은 노름으로 맞춰야
    동일한 lambda 로 비교할 수 있다.
    """
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n * np.linalg.norm(reference)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def lambda_for_relative(rel: float, s_norm: np.ndarray, h_norm: np.ndarray) -> np.ndarray:
    """상대 강도 rel 을 레이어별 절대 lambda 로 환산한다.

        lambda_i = rel * ||h_i|| / ||S_i||
    """
    return rel * h_norm / np.maximum(s_norm, 1e-12)


def activation_l2(layer_arr, n_samples: int = 64) -> float:
    """||h_i^(t)||_2 의 대표값 (표본·위치 평균). lambda 캘리브레이션의 분모."""
    a = np.asarray(layer_arr[:n_samples], dtype=np.float32)
    return float(np.linalg.norm(a, axis=-1).mean())


def ood_profile(h_base: np.ndarray, h_trend: np.ndarray, s_vec: np.ndarray,
                lambdas: np.ndarray) -> dict:
    """개입이 표현을 데이터 다양체 밖으로 밀어내는 정도를 lambda 별로 측정한다.

    Stage 3 의 PCA 그림에서 드러났듯, Kronos 의 마지막 토큰 표현은 두 개의
    가우시안 덩어리가 아니라 **휘어진 1 차원 다양체**다. S = M_s - M_c 는 그
    호를 가로지르는 직선(현)이므로, h + lambda*S 는 호를 따라가지 않고
    다양체 바깥으로 벗어난다. 모델이 학습 중 본 적 없는 영역에 들어가면
    예측이 붕괴한다 (명세 7절이 Chronos 사례로 경고한 현상).

    측정 방법
        실제 활성화들 사이의 최근접 이웃 거리(중앙값)를 기준자로 삼고,
        조정된 점에서 가장 가까운 실제 활성화까지의 거리를 그 기준자로 나눈다.
        비율이 1 근처면 다양체 위에 있고, 크게 넘어가면 OOD 다.

    Args:
        h_base, h_trend: [n, D] 특정 레이어·토큰의 실제 활성화
        s_vec: [D] steering vector
        lambdas: 시험할 절대 lambda 값들
    Returns:
        {"ratio": [len(lambdas)], "ref_nn": float, "reached": [len(lambdas)]}
        reached 는 조정된 점이 trend 평균에 얼마나 다가갔는지 (0=base, 1=trend)
    """
    import torch

    a = torch.from_numpy(np.asarray(h_base, dtype=np.float32))
    b = torch.from_numpy(np.asarray(h_trend, dtype=np.float32))
    s = torch.from_numpy(np.asarray(s_vec, dtype=np.float32))
    real = torch.cat([a, b], dim=0)

    # 기준자: 실제 활성화끼리의 최근접 거리 (자기 자신 제외)
    d_real = torch.cdist(real, real)
    d_real.fill_diagonal_(float("inf"))
    ref_nn = float(d_real.min(dim=1).values.median())

    mu_a, mu_b = a.mean(0), b.mean(0)
    gap = torch.linalg.norm(mu_b - mu_a) + 1e-12

    ratio, reached = [], []
    for lam in lambdas:
        steered = a + float(lam) * s
        d = torch.cdist(steered, real).min(dim=1).values
        ratio.append(float(d.median()) / ref_nn)
        reached.append(float(torch.linalg.norm(steered.mean(0) - mu_a) / gap))
    return {"ratio": np.array(ratio), "ref_nn": ref_nn, "reached": np.array(reached)}
