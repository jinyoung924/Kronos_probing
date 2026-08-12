"""Stage 5 — 개념 조정 개입 추론 (명세 7절, 설계 Step 5).

요인 설계 — 원논문 방법과 데이터에서 발견한 개선안을 모두 실행한다.

원논문(MOMENT Steering) 방법
    A : median x matrix x all-tokens x 전체 레이어   <- 논문의 주 방법
    B : mean          (논문 절제 실험 1)
    C : single-token  (논문 절제 실험 2)

데이터에서 발견한 개선안 (근거는 base 실측)
    D : layer 0 제외   — OOD 진단에서 layer 0 만 lambda_rel~0.06 에서 다양체 이탈
                        (0.25 에서 비율 8.4, 1.0 에서 28.6). BSQ 이산 임베딩
                        격자라 연속 섭동이 갈 곳이 없다.
    E : 깊은 층만      — Stage 3 에서 layer 4->11 로 갈수록 마지막 토큰 LDR 이
                        17.83->23.52 로 단조 증가 (인과 경계에 개념 보존)
    F : 얕은 정점층만  — 전역 LDR 최대는 layer 1 (29.27, 중간 토큰)
    G : LDA 방향 + 읽기량 정합 — cos(LDA, median)=0.04 이므로 median 팔은 변위의
                        96% 를 판별과 무관한 방향에 쓴다. 같은 개념 읽기 이동량을
                        LDA 방향으로 내면 변위가 약 1/25 이고 OOD 이탈이 줄어든다.

대조군 (위 개선안을 해석하기 위해 필요)
    H : LDA 방향 + 동일 노름 — G 의 효과가 방향 때문인지 크기 때문인지 분리
    I : median x vector      — G/H 가 vector 형태라, 형태 효과를 분리

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
    pred_len = 8 if args.smoke else (args.pred_len or cfg.steer.pred_len)
    sample_count = 1 if args.smoke else (args.sample_count or cfg.steer.sample_count)
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
    no_l0 = list(range(1, L))            # layer 0 은 OOD 진단상 즉시 이탈한다
    deep = list(range(max(0, L - 3), L))
    shallow_peak = [1, 2]                # Stage 3 의 전역 LDR 최대 구역
    runs = [
        # --- 원논문 방법 (재현) ---------------------------------------------
        ("A_paper_median_matrix_all",   dict(method="median", form="matrix", scope="all",    layers=all_layers)),
        ("B_paper_mean_matrix_all",     dict(method="mean",   form="matrix", scope="all",    layers=all_layers)),
        ("C_paper_single_token",        dict(method="median", form="matrix", scope="single", layers=all_layers)),
        # --- 데이터에서 발견한 개선안 ---------------------------------------
        ("D_ours_skip_layer0",          dict(method="median", form="matrix", scope="all", layers=no_l0)),
        ("E_ours_deep_only",            dict(method="median", form="matrix", scope="all", layers=deep)),
        ("F_ours_shallow_peak",         dict(method="median", form="matrix", scope="all", layers=shallow_peak)),
        ("G_ours_lda_matched",          dict(method="lda", form="vector", scope="all", layers=no_l0, lda_scale="matched")),
        # --- 위 두 개선안을 해석하기 위한 대조군 -----------------------------
        ("H_ctrl_lda_equalnorm",        dict(method="lda", form="vector", scope="all", layers=no_l0, lda_scale="equal")),
        ("I_ctrl_median_vector",        dict(method="median", form="vector", scope="all", layers=no_l0)),
    ]
    if args.arms == "paper":
        runs = [r for r in runs if r[0][0] in "ABC"]
    elif args.arms == "ours":
        runs = [r for r in runs if r[0][0] in "DEFGHI"]
    if args.smoke:
        runs = [r for r in runs if r[0][0] in "ADG"] or runs[:3]
    for name, c in runs:
        ls = c["layers"]
        print(f"  {name:28s} {c['method']:>6} {c['form']:>7} {c['scope']:>6} "
              f"{c.get('lda_scale', ''):>8} layers={ls if len(ls) < 5 else f'{ls[0]}..{ls[-1]}'}")

    # lambda 캘리브레이션의 분모
    h_l2 = st["h_norm_l2"]
    # LDA 방향과 median 차이 방향의 코사인 — "개념 읽기 이동량 정합" 에 쓴다.
    # base 실측 0.04 이므로, 같은 읽기 이동량을 내는 데 필요한 변위가 약 1/25 이다.
    ldr_late = np.array(summ["bands"]["late"])
    S_med_late = st["S_median_late"]
    cos_lda = np.array([SV.cosine(st["S_lda"][i], S_med_late[i]) for i in range(L)])
    print(f"\n  cos(LDA, median/late) 레이어별 중앙값: {np.median(cos_lda):.4f} "
          f"-> matched 스케일은 변위가 약 {1/np.median(cos_lda):.0f}배 작다")

    section("4. 실행")
    # 1~3 시간짜리 실행이라 중간에 세션이 끊길 수 있다. 매 조합마다 저장하고,
    # 다시 실행하면 이미 끝난 조합은 건너뛴다.
    out_file = out_dir / ("results_smoke.json" if args.smoke else "results.json")
    results = json.loads(out_file.read_text()) if out_file.exists() and not args.force else {}
    # 실행 조건(표본 수/예측 길이/샘플 수)이 다른 항목은 비교가 불가능하므로 버린다.
    # 예전 스모크 결과가 같은 파일에 섞여 들어가 본 실행이 건너뛰어지는 사고를 막는다.
    want = {"n_eval": n_eval, "pred_len": pred_len, "sample_count": sample_count}

    def matches(v: dict) -> bool:
        rs = v.get("run_settings")
        if rs is not None:
            return rs == want
        # run_settings 기록 이전에 만들어진 항목은 표본 수로 판별한다.
        # (스모크는 n_eval 이 작아 slopes 길이가 다르다)
        return len(v.get("slopes", [])) == n_eval

    stale = [k for k, v in results.items() if not matches(v)]
    for k in stale:
        del results[k]
    if stale:
        print(f"  실행 조건이 다른 기존 항목 {len(stale)}개를 버린다: {sorted(stale)[:6]}...")
    if results:
        print(f"  기존 결과 {len(results)}개를 이어받는다 ({out_file})")
    baseline_metrics = None
    n_todo = sum(1 for n, _ in runs for r in rels if f"{n}|{r}" not in results)
    print(f"  실행할 조합 {n_todo}개")
    done = 0
    print(f"  {'run':28s} {'lam_rel':>8} {'개념축이동':>10} {'slope_mean':>11} "
          f"{'slope_std':>10} {'>0 비율':>8} {'coh 위반':>9} {'sec':>6}")
    for name, c in runs:
        for rel in rels:
            key = f"{name}|{rel}"
            if key in results:
                continue
            t1 = time.time()
            if rel == 0.0:
                if baseline_metrics is None:
                    preds = run_config(predictor, model, x_norm, x_stamp, y_stamp, cfg,
                                       None, None, None, None, args.seed)
                    baseline_metrics = evaluate(preds, x_mean, x_std)
                m = dict(baseline_metrics); shift = 0.0
            else:
                if c["method"] == "lda":
                    S_vec, S_mat = st["S_lda"], None
                    s_norm = np.linalg.norm(S_vec, axis=-1)
                    lam = SV.lambda_for_relative(rel, s_norm, h_l2)
                    if c.get("lda_scale") == "matched":
                        # median 팔의 개념 읽기 이동량은 lambda_rel*||h||*cos 이다.
                        # LDA 팔은 그 방향 자체로 움직이므로 cos 를 곱하면 같은
                        # 이동량이 되고, 변위는 그만큼(약 1/25) 작아진다.
                        lam = lam * cos_lda
                else:
                    S_vec = st[f"S_{c['method']}_late"]
                    S_mat = st[f"S_matrix_{c['method']}"]
                    s_norm = np.linalg.norm(S_vec, axis=-1)
                    lam = SV.lambda_for_relative(rel, s_norm, h_l2)

                payload = IV.build_payload(S_vec, S_mat, c["layers"], lam, c["form"])
                # 개입이 "의도한 만큼 개념 축을 밀었는지" 의 해석적 확인.
                # payload 를 LDA 축 w 에 투영한 이동량을 클래스내 산포 단위로 잰다.
                # 클래스 간격은 <S_median, w> 이고 그 산포 단위 크기가 sqrt(LDR) 이므로
                #   shift = lambda_i * <S_payload, w> / <S_median, w> * sqrt(LDR_i)
                # median 팔은 S_payload = S_median 이라 계수가 1 이지만,
                # LDA 팔은 S_lda 가 이미 w 방향이라 <S_lda, w> = ||S_lda|| 이고
                # 계수가 1/cos (약 22) 이 된다. 이 보정을 빠뜨리면 matched 팔이
                # 실제로는 정합인데도 22 배 작게 보고된다.
                proj = (1.0 / np.maximum(np.abs(cos_lda), 1e-6)
                        if c["method"] == "lda" else np.ones(L))
                shift = float(np.mean([lam[i] * proj[i] * np.sqrt(max(ldr_late[i], 0.0))
                                       for i in c["layers"]]))
                preds = run_config(predictor, model, x_norm, x_stamp, y_stamp, cfg,
                                   payload, c["form"], c["scope"], c["layers"], args.seed)
                m = evaluate(preds, x_mean, x_std)
            m["lambda_rel"] = rel
            m["readout_shift_sigma"] = shift
            m["run_settings"] = {"n_eval": n_eval, "pred_len": pred_len,
                                 "sample_count": sample_count}
            m["config"] = {k: (v if k != "layers" else list(v)) for k, v in c.items()}
            m["seconds"] = round(time.time() - t1, 1)
            results[key] = m
            out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            done += 1
            eta = (time.time() - t0) / done * (n_todo - done)
            print(f"  {name:28s} {rel:>8.2f} {shift:>10.3f} {m['slope_mean']:>11.5f} "
                  f"{m['slope_std']:>10.5f} {m['frac_positive']:>8.1%} "
                  f"{m['coherence_violation']:>9.2%} {m['seconds']:>6.1f}"
                  f"  ETA {eta/60:.0f}m")

    section("5. 저장")
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  {out_file}  ({len(results)}개 조합)")
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
    p.add_argument("--arms", default="all", choices=["all", "paper", "ours"],
                   help="paper=A~C(논문 재현), ours=D~I(개선안+대조군)")
    p.add_argument("--force", action="store_true", help="기존 결과를 무시하고 처음부터")
    p.add_argument("--pred-len", type=int, default=None,
                   help="예측 길이 override. 시간이 부족하면 32 로 줄인다")
    p.add_argument("--sample-count", type=int, default=None,
                   help="AR 샘플 수 override. 배치 크기에 비례해 시간이 든다")
    raise SystemExit(main(p.parse_args()))
