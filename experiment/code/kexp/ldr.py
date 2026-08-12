"""Stage 3 — Fisher criterion 선형 프로브와 Linear Discriminant Ratio.

명세 4, 5절의 목적함수

    L_Fisher(c, s) = - (mu_s - mu_c)^2 / (sigma_s^2 + sigma_c^2)

는 LDA 의 닫힌 해로 즉시 최적화된다. 경사하강 루프가 필요 없다.

    w  = (Sigma_w + eps*I)^{-1} (mu_s - mu_c)
    LDR = (mean(w.x_s) - mean(w.x_c))^2 / (var(w.x_s) + var(w.x_c))

steertool 의 compute_linear_separability 는 (layer, patch) 조합마다 sklearn LDA 를
새로 fit 한다. 우리는 512 위치 x 12 레이어 = 6144 개라 그 방식이면 매우 느리다.
여기서는 한 레이어의 모든 토큰 위치를 GPU 배치 연산 한 번으로 처리한다.

정칙화가 필수인 이유
    d_model=832 인데 클래스당 샘플이 2048 이다. 클래스내 산포행렬은 형식적으로
    가역이지만 조건수가 나쁘다. trace 기반 상대 shrinkage 를 적용해 스케일에
    무관하게 만든다.

과적합 통제
    학습에 쓴 표본으로 LDR 을 재면 낙관 편향이 생긴다. train 으로 w 를 구하고
    held-out test 로 LDR 을 재는 값을 함께 보고한다. 레이블을 섞은 null 대조군도
    같이 계산해, 관측된 LDR 이 차원수에 의한 우연인지 판별한다.
"""

from __future__ import annotations

import numpy as np
import torch


def _shrink(sw: torch.Tensor, rel_eps: float) -> torch.Tensor:
    """Sigma_w + eps*I. eps 를 trace/D 에 비례시켜 스케일 불변으로 만든다."""
    d = sw.shape[-1]
    diag = torch.diagonal(sw, dim1=-2, dim2=-1).mean(dim=-1)      # [P]
    eye = torch.eye(d, device=sw.device, dtype=sw.dtype)
    return sw + (rel_eps * diag).view(-1, 1, 1) * eye


def fisher_directions(a: torch.Tensor, b: torch.Tensor, rel_eps: float) -> torch.Tensor:
    """클래스별 활성화 -> 위치별 LDA 방향 w.

    Args:
        a: [n_a, P, D] 클래스 c (base)
        b: [n_b, P, D] 클래스 s (trend)
    Returns:
        w: [P, D]
    """
    mu_a, mu_b = a.mean(0), b.mean(0)                              # [P, D]
    ac, bc = a - mu_a, b - mu_b
    # 클래스내 산포행렬 (pooled)
    sw = torch.einsum("npd,npe->pde", ac, ac) + torch.einsum("npd,npe->pde", bc, bc)
    sw = sw / (a.shape[0] + b.shape[0] - 2)
    delta = (mu_b - mu_a).unsqueeze(-1)                            # [P, D, 1]
    w = torch.linalg.solve(_shrink(sw, rel_eps), delta).squeeze(-1)
    return w


def project_ldr(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """주어진 방향 w 로 투영한 뒤 Fisher LDR 을 구한다. -> [P]"""
    pa = torch.einsum("npd,pd->np", a, w)
    pb = torch.einsum("npd,pd->np", b, w)
    num = (pb.mean(0) - pa.mean(0)) ** 2
    # ddof=0. steertool 의 compute_linear_separability 가 numpy .var() 기본값을
    # 쓰므로 논문 구현체와 수치를 정확히 일치시키기 위해 맞춘다.
    den = pa.var(0, unbiased=False) + pb.var(0, unbiased=False)
    return num / (den + 1e-12)


def ldr_for_layer(a_np: np.ndarray, b_np: np.ndarray, idx_tr: np.ndarray,
                  idx_te: np.ndarray, rel_eps: float, device: torch.device,
                  chunk: int = 32, shuffle_labels: bool = False,
                  seed: int = 0) -> dict:
    """한 레이어의 모든 토큰 위치에 대해 LDR 을 구한다.

    Args:
        a_np, b_np: [N, T, D] fp16 (base, trend)
        idx_tr, idx_te: 표본 인덱스 분할
    Returns:
        {"train": [T], "test": [T], "w_last": [D], "delta_norm": [T]}
    """
    T = a_np.shape[1]
    out_tr = np.zeros(T, dtype=np.float64)
    out_te = np.zeros(T, dtype=np.float64)
    dnorm = np.zeros(T, dtype=np.float64)
    w_last = None

    for p0 in range(0, T, chunk):
        p1 = min(p0 + chunk, T)
        a = torch.from_numpy(np.ascontiguousarray(a_np[:, p0:p1])).to(device).float()
        b = torch.from_numpy(np.ascontiguousarray(b_np[:, p0:p1])).to(device).float()

        if shuffle_labels:
            # 두 클래스를 합쳐 무작위로 다시 나눈다 -> 개념이 없는 null 대조군
            g = torch.Generator(device="cpu").manual_seed(seed + p0)
            cat = torch.cat([a, b], dim=0)
            perm = torch.randperm(cat.shape[0], generator=g).to(device)
            a, b = cat[perm[: a.shape[0]]], cat[perm[a.shape[0]:]]

        a_tr, b_tr = a[idx_tr], b[idx_tr]
        a_te, b_te = a[idx_te], b[idx_te]

        w = fisher_directions(a_tr, b_tr, rel_eps)                 # [P, D]
        out_tr[p0:p1] = project_ldr(a_tr, b_tr, w).cpu().numpy()
        out_te[p0:p1] = project_ldr(a_te, b_te, w).cpu().numpy()
        dnorm[p0:p1] = (b.mean(0) - a.mean(0)).norm(dim=-1).cpu().numpy()
        if p1 == T:
            w_last = w[-1].cpu().numpy()
        del a, b, a_tr, b_tr, a_te, b_te, w

    return {"train": out_tr, "test": out_te, "w_last": w_last, "delta_norm": dnorm}


def class_statistics(a_np: np.ndarray, b_np: np.ndarray, device: torch.device,
                     chunk: int = 32) -> dict:
    """Stage 4 용 위치별 클래스 통계 (median / mean).

    Stage 3 에서 함께 구해 둔다. 활성화 39 GB 를 다시 읽지 않기 위해서다.
    """
    T, D = a_np.shape[1], a_np.shape[2]
    out = {k: np.zeros((T, D), dtype=np.float32)
           for k in ("median_base", "median_trend", "mean_base", "mean_trend")}
    for p0 in range(0, T, chunk):
        p1 = min(p0 + chunk, T)
        a = torch.from_numpy(np.ascontiguousarray(a_np[:, p0:p1])).to(device).float()
        b = torch.from_numpy(np.ascontiguousarray(b_np[:, p0:p1])).to(device).float()
        out["median_base"][p0:p1] = a.median(dim=0).values.cpu().numpy()
        out["median_trend"][p0:p1] = b.median(dim=0).values.cpu().numpy()
        out["mean_base"][p0:p1] = a.mean(dim=0).cpu().numpy()
        out["mean_trend"][p0:p1] = b.mean(dim=0).cpu().numpy()
        del a, b
    return out


def smooth(x: np.ndarray, w: int = 25) -> np.ndarray:
    """토큰 축 이동평균.

    위치별 LDR 은 표본 노이즈가 크다. 512 개 위치에서 raw argmax 를 뽑으면
    구조가 아니라 노이즈의 최댓값을 고르게 된다. 구조를 볼 때는 평활한 곡선을,
    판정에는 구간 평균을 쓴다.
    """
    if w <= 1:
        return x
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge") if x.ndim == 2 else np.pad(x, (pad, pad), mode="edge")
    if x.ndim == 2:
        return np.stack([np.convolve(r, k, mode="valid")[: x.shape[1]] for r in xp])
    return np.convolve(xp, k, mode="valid")[: x.shape[0]]


def token_bands(ldr: np.ndarray, frac: float = 0.0625) -> dict:
    """토큰 위치를 구간으로 나눈 평균 LDR. argmax 보다 표본 노이즈에 강하다.

    Args:
        ldr: [L, T]
    Returns:
        early / mid / late / last 각 [L]
    """
    T = ldr.shape[1]
    k = max(1, int(T * frac))
    mid0 = T // 2 - k // 2
    return {
        "early": ldr[:, :k].mean(axis=1),
        "mid": ldr[:, mid0:mid0 + k].mean(axis=1),
        "late": ldr[:, -k:].mean(axis=1),
        "last": ldr[:, -1],
    }


def minmax(x: np.ndarray) -> np.ndarray:
    """명세 5절의 Min-Max Scaling."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def split_indices(n: int, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_te = int(round(n * test_ratio))
    return perm[n_te:], perm[:n_te]
