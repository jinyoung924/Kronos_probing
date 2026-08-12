"""Stage 5 — 개념 조정 개입 추론 (명세 7절, 설계 Step 5).

요인 설계 (전 격자 252 조합은 과하므로 기준 조건에서 한 축씩 바꾼다)

    기준 A : median x matrix x all-tokens x 전체 레이어   <- 논문의 주 방법
    B : mean            (논문 절제 실험 1)
    C : single-token    (논문 절제 실험 2)
    D : vector 형태     (명세 7절의 축약형, Kronos AR 대응)
    E : LDA 방향        (우리 추가 — steertool 코드에는 있으나 논문 방법은 아님)
    F : 레이어 부분집합 (우리 추가 — Stage 3 LDR + Stage 4 OOD 근거)

입력은 base 클래스(추세 없음)의 held-out 표본이다. 개입이 예측을 모멘텀
방향으로 미는지를 본다.

실행:
    python experiment/code/stages/stage5_intervene.py
    python experiment/code/stages/stage5_intervene.py --smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from kexp import intervene as IV  # noqa: E402
from kexp import kronos_loader as kl  # noqa: E402
from kexp import ldr as LD  # noqa: E402
from kexp import paths  # noqa: E402
from kexp import steering_vec as SV  # noqa: E402
from kexp.config import CFG, resolve_model  # noqa: E402

CH_O, CH_H, CH_L, CH_C = 0, 1, 2, 3


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def ols_slope(series: np.ndarray) -> np.ndarray:
    """[n, H] 각 행의 최소제곱 기울기."""
    h = series.shape[1]
    t = np.arange(h, dtype=np.float64)
    tc = t - t.mean()
    return (series * tc).sum(axis=1) / (tc**2).sum()


def evaluate(preds: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    """예측 [n, H, 6] (정규화 공간) -> 지표.

    기울기는 정규화 공간에서 잰다. generate() 의 반환값이 입력 z-score 단위라
    샘플별 스케일 차이가 제거되고 Stage 1 의 close_slope 와 직접 비교된다.

    coherence 는 반드시 **역정규화 후** 봐야 한다. Kronos 는 O/H/L/C 를 채널별로
    독립 z-score 하므로 채널마다 mean/std 가 다르고, 정규화 공간에서는
    L <= min(O,C) <= max(O,C) <= H 순서가 보존되지 않는다.
    """
    close = preds[:, :, CH_C]
    slope = ols_slope(close)

    raw = preds * (std[:, None, :] + 1e-5) + mean[:, None, :]
    o, h, low, c = raw[:, :, CH_O], raw[:, :, CH_H], raw[:, :, CH_L], raw[:, :, CH_C]
    tol = 1e-4 * np.maximum(np.abs(raw[:, :, :4]).max(), 1.0)
    viol = ((low > np.minimum(o, c) + tol) | (np.maximum(o, c) > h + tol)).mean()
    return {
        "slope_mean": float(slope.mean()),
        "slope_std": float(slope.std()),
        "slope_median": float(np.median(slope)),
        "frac_positive": float((slope > 0).mean()),
        "close_drift": float((close[:, -1] - close[:, 0]).mean()),
        "pred_std": float(close.std(axis=1).mean()),
        "coherence_violation": float(viol),
        "nan_frac": float(np.isnan(preds).mean()),
        "slopes": slope.tolist(),
    }


def run_config(predictor, model, x_norm, x_stamp, y_stamp, cfg, payload, form, scope,
               layers, seed: int) -> np.ndarray:
    """한 조합에 대해 AR 예측을 수행한다. 반환은 정규화 공간의 [n, H, 6]."""
    torch.manual_seed(seed)
    s = cfg.steer
    if payload is None:
        return predictor.generate(x_norm, x_stamp, y_stamp, s.pred_len, s.temperature,
                                  s.top_k, s.top_p, s.sample_count, False)
    with IV.Steerer(model, layers, payload, x_norm.shape[1], form=form, scope=scope):
        return predictor.generate(x_norm, x_stamp, y_stamp, s.pred_len, s.temperature,
                                  s.top_k, s.top_p, s.sample_count, False)


def main(args) -> int:
    t0 = time.time()
    cfg = dataclasses.replace(CFG, model=resolve_model(args.model, CFG.model.max_context))
    data_dir = paths.data_dir(cfg.tag, args.noise)
    res_dir = paths.results_dir(cfg.tag) / f"{args.noise}_{args.model}"
    out_dir = paths.ensure(res_dir / "steer")

    for f in ("steering.npz", "stage3_summary.json"):
        if not (res_dir / f).exists():
            print(f"Stage 4 산출물이 없다: {res_dir / f}")
            return 1
    st = np.load(res_dir / "steering.npz")
    summ = json.loads((res_dir / "stage3_summary.json").read_text())
    L, T, D = summ["L"], summ["T"], summ["D"]

    section("1. 설정")
    n_eval = args.smoke_n if args.smoke else args.n_eval
    pred_len = 8 if args.smoke else cfg.steer.pred_len
    sample_count = 1 if args.smoke else cfg.steer.sample_count
    cfg = dataclasses.replace(cfg, steer=dataclasses.replace(
        cfg.steer, pred_len=pred_len, sample_count=sample_count))
    rels = [r for r in cfg.steer.lambdas_rel] if not args.smoke else [0.0, 0.25]
    print(f"  모델 {args.model}, 데이터 {args.noise}, [L={L}, T={T}, D={D}]")
    print(f"  평가 표본 {n_eval}, pred_len {pred_len}, sample_count {sample_count}")
    print(f"  lambda_rel 격자: {rels}")

    # 입력: base 클래스(추세 없음)의 held-out 표본
    idx_tr, idx_te = LD.split_indices(cfg.data.n_samples, cfg.probe.test_ratio, cfg.data.seed)
    x_all = np.load(data_dir / "base.npz")["x"]
    x_raw = x_all[idx_te[:n_eval]]
    ts = pd.DatetimeIndex(np.load(data_dir / "timestamps.npy"))
    x_norm, x_mean, x_std = kl.normalize_batch(x_raw, cfg.model.clip)
    stamp = kl.make_stamps(ts)
    y_ts = pd.date_range(ts[-1] + pd.Timedelta(minutes=cfg.data.freq_minutes),
                         periods=pred_len, freq=f"{cfg.data.freq_minutes}min")
    x_stamp = np.repeat(stamp[None], len(x_raw), axis=0)
    y_stamp = np.repeat(kl.make_stamps(y_ts)[None], len(x_raw), axis=0)
    print(f"  입력 {x_norm.shape} (base 클래스 held-out)")

    section("2. 모델 로드")
    tokenizer, model, device = kl.load_kronos(cfg.model)
    from model.kronos import KronosPredictor  # noqa: E402
    predictor = KronosPredictor(model, tokenizer, device=str(device),
                                max_context=cfg.model.max_context, clip=cfg.model.clip)
    print(f"  device={device}, max_context={cfg.model.max_context}")

    section("3. 요인 설계")
    all_layers = list(range(L))
    deep = list(range(max(0, L - 3), L))
    best = [int(summ["late_best_layer"])]
    shallow = [0, 1]
    runs = [
        ("A_baseline_median_matrix_all", dict(method="median", form="matrix", scope="all", layers=all_layers)),
        ("B_mean_matrix_all",            dict(method="mean",   form="matrix", scope="all", layers=all_layers)),
        ("C_median_matrix_single",       dict(method="median", form="matrix", scope="single", layers=all_layers)),
        ("D_median_vector_all",          dict(method="median", form="vector", scope="all", layers=all_layers)),
        ("E_lda_vector_all",             dict(method="lda",    form="vector", scope="all", layers=all_layers)),
        ("F_median_matrix_deep",         dict(method="median", form="matrix", scope="all", layers=deep)),
        ("F_median_matrix_shallow",      dict(method="median", form="matrix", scope="all", layers=shallow)),
        ("F_median_matrix_best",         dict(method="median", form="matrix", scope="all", layers=best)),
    ]
    if args.smoke:
        runs = runs[:3]
    for name, c in runs:
        print(f"  {name:32s} {c['method']:>6} {c['form']:>7} {c['scope']:>6} "
              f"layers={c['layers'] if len(c['layers']) < 5 else 'all'}")

    # lambda 캘리브레이션의 분모
    h_l2 = st["h_norm_l2"]
    h_tok = st["h_token_l2"] if "h_token_l2" in st.files else None

    section("4. 실행")
    results = {}
    baseline_metrics = None
    print(f"  {'run':32s} {'lam_rel':>8} {'slope_mean':>11} {'slope_std':>10} "
          f"{'>0 비율':>8} {'coh 위반':>9} {'sec':>6}")
    for name, c in runs:
        for rel in rels:
            key = f"{name}|{rel}"
            t1 = time.time()
            if rel == 0.0:
                if baseline_metrics is None:
                    preds = run_config(predictor, model, x_norm, x_stamp, y_stamp, cfg,
                                       None, None, None, None, args.seed)
                    baseline_metrics = evaluate(preds, x_mean, x_std)
                m = dict(baseline_metrics)
            else:
                if c["method"] == "lda":
                    S_vec, S_mat = st["S_lda"], None
                    s_norm = np.linalg.norm(st["S_lda"], axis=-1)
                else:
                    S_vec = st[f"S_{c['method']}_late"]
                    S_mat = st[f"S_matrix_{c['method']}"]
                    s_norm = np.linalg.norm(S_vec, axis=-1)
                lam = SV.lambda_for_relative(rel, s_norm, h_l2)
                payload = IV.build_payload(S_vec, S_mat, c["layers"], lam, c["form"])
                preds = run_config(predictor, model, x_norm, x_stamp, y_stamp, cfg,
                                   payload, c["form"], c["scope"], c["layers"], args.seed)
                m = evaluate(preds, x_mean, x_std)
            m["lambda_rel"] = rel
            m["config"] = {k: (v if k != "layers" else list(v)) for k, v in c.items()}
            m["seconds"] = round(time.time() - t1, 1)
            results[key] = m
            print(f"  {name:32s} {rel:>8.2f} {m['slope_mean']:>11.5f} "
                  f"{m['slope_std']:>10.5f} {m['frac_positive']:>8.1%} "
                  f"{m['coherence_violation']:>9.2%} {m['seconds']:>6.1f}")

    section("5. 저장")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  {out_dir / 'results.json'}")
    print(f"  총 소요: {time.time() - t0:.1f}s")
    print("\n  다음: stage6 에서 slope vs lambda 곡선, 붕괴 지표, PCA 이동을 시각화한다.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 5 — steering 개입 추론")
    p.add_argument("--model", default="base", choices=["mini", "small", "base"])
    p.add_argument("--noise", default=CFG.data.noise_type, choices=["ou", "rw"])
    p.add_argument("--n-eval", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=4)
    raise SystemExit(main(p.parse_args()))
