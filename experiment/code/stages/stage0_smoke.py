"""Stage 0 — 환경 부트스트랩 & 모델 스펙 실측.

여기서 확인하는 것 (이후 모든 stage 의 파라미터가 이 값에 의존한다)
  1. GPU / RAM / 디스크 가용량
  2. Kronos-small 의 실측 스펙: n_layers, d_model, s1/s2 bits
  3. TransformerBlock forward hook 출력 shape 가 [B, T, D] 인지
  4. V=0, A=0 입력에서 NaN 없이 예측이 나오는지 (명세 2절 검증)
  5. eval 모드에서 활성화가 결정적(deterministic)인지 -> LDR 신뢰성의 전제
  6. Stage 2 활성화 저장 용량 추정

실행:
    python experiment/code/stages/stage0_smoke.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

# --- bootstrap: 이 파일을 경로로 직접 실행해도 kexp 를 임포트할 수 있게 ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from kexp import paths  # noqa: E402
from kexp.config import CFG  # noqa: E402
from kexp import kronos_loader as kl  # noqa: E402
from kexp.hooks import ActivationRecorder  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def report_env() -> dict:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "in_colab": paths.in_colab(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_vram_GB"] = round(props.total_memory / 1024**3, 1)
        info["cuda_version"] = torch.version.cuda

    try:  # 시스템 RAM (Linux)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["system_ram_GB"] = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except OSError:
        pass

    for label, p in (("scratch", paths.scratch_root()), ("drive", paths.drive_root())):
        try:
            paths.ensure(p)
            usage = shutil.disk_usage(p)
            info[f"{label}_free_GB"] = round(usage.free / 1024**3, 1)
            info[f"{label}_path"] = str(p)
        except OSError as e:
            info[f"{label}_error"] = str(e)

    for k, v in info.items():
        print(f"  {k:22s}: {v}")
    return info


def dummy_ohlcva(T: int, seed: int = 0) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Stage 1 의 micro-path 집계를 축소 재현한 더미 입력.

    coherence 제약(L <= min(O,C) <= max(O,C) <= H)이 정의상 만족되는지도
    여기서 미리 확인해 둔다.
    """
    rng = np.random.default_rng(seed)
    n_micro = CFG.data.n_micro
    steps = rng.normal(0.0, CFG.data.sigma, size=T * n_micro)
    path = CFG.data.base_level + np.cumsum(steps)
    grid = path.reshape(T, n_micro)

    x = np.zeros((T, 6), dtype=np.float32)
    x[:, 0] = grid[:, 0]           # open
    x[:, 1] = grid.max(axis=1)     # high
    x[:, 2] = grid.min(axis=1)     # low
    x[:, 3] = grid[:, -1]          # close
    # volume / amount 는 명세 2절에 따라 0 고정
    ts = pd.date_range(CFG.data.start_time, periods=T, freq=f"{CFG.data.freq_minutes}min")
    return x, ts


def main() -> int:
    t0 = time.time()

    section("1. 실행 환경")
    env = report_env()

    section("2. 모델 로드")
    tokenizer, model, device = kl.load_kronos(CFG.model)
    print(f"  device               : {device}")
    spec = kl.describe(tokenizer, model)
    for k, v in spec.items():
        print(f"  {k:22s}: {v}")

    ok = True
    if spec["n_layers"] != 12:
        print(f"  [경고] 명세는 L=12 를 가정하는데 실측은 {spec['n_layers']} 이다.")
        ok = False

    section("3. 더미 입력 생성 및 전처리 (명세 2, 3절)")
    T = CFG.data.T
    x_raw, ts = dummy_ohlcva(T)
    coherent = bool(
        np.all(x_raw[:, 2] <= np.minimum(x_raw[:, 0], x_raw[:, 3]) + 1e-6)
        and np.all(np.maximum(x_raw[:, 0], x_raw[:, 3]) <= x_raw[:, 1] + 1e-6)
    )
    print(f"  OHLC coherence       : {coherent}")

    xb = x_raw[None, ...]
    mean, std = xb.mean(axis=1), xb.std(axis=1)
    x_unclipped = (xb - mean[:, None, :]) / (std[:, None, :] + 1e-5)
    x_norm, _, _ = kl.normalize_batch(xb, CFG.model.clip)
    print(f"  V/A 채널 std         : {std[0, 4]:.6g}, {std[0, 5]:.6g}  (0 이어야 정상)")
    print(f"  정규화 후 NaN        : {bool(np.isnan(x_norm).any())}  (False 여야 정상)")
    print(f"  V/A 정규화 결과      : {np.unique(x_norm[:, :, 4:])}  ([0.] 이어야 정상)")
    print(f"  클리핑률             : {kl.clip_fraction(x_unclipped, CFG.model.clip):.4%}")
    print(f"  정규화 값 범위       : [{x_norm.min():.3f}, {x_norm.max():.3f}]")

    section("4. Hook 출력 shape 확인")
    s1, s2 = kl.to_tokens(tokenizer, x_norm, device)
    print(f"  token shape          : s1={tuple(s1.shape)}, s2={tuple(s2.shape)}")
    stamp = torch.from_numpy(kl.make_stamps(ts)[None, ...]).to(device)

    with ActivationRecorder(model) as rec:
        with torch.no_grad():
            _ = model.decode_s1(s1, s2, stamp)
        acts = rec.stacked()
    print(f"  activations shape    : {tuple(acts.shape)}  (=[L, B, T, D])")
    print(f"  dtype                : {acts.dtype}")
    print(f"  NaN 포함             : {bool(torch.isnan(acts.float()).any())}")

    L, B, Tt, D = acts.shape
    shape_ok = (L == spec["n_layers"]) and (Tt == T) and (D == spec["d_model"])
    print(f"  shape 정합           : {shape_ok}")
    ok = ok and shape_ok

    section("5. 결정성(determinism) 확인 — eval 모드에서 dropout 이 꺼졌는가")
    with ActivationRecorder(model) as rec2:
        with torch.no_grad():
            _ = model.decode_s1(s1, s2, stamp)
        acts2 = rec2.stacked()
    deterministic = bool(torch.equal(acts, acts2))
    print(f"  두 번의 추출이 동일   : {deterministic}")
    if not deterministic:
        print("  [경고] model.eval() 이 적용되지 않았거나 비결정적 커널이 쓰이고 있다.")
    ok = ok and deterministic

    section("6. V=0, A=0 상태에서의 예측 (명세 2절)")
    from model.kronos import KronosPredictor  # noqa: E402

    predictor = KronosPredictor(model, tokenizer, device=str(device),
                                max_context=CFG.model.max_context, clip=CFG.model.clip)
    df = pd.DataFrame(x_raw, columns=kl.FEAT_COLS)
    pred_len = 32
    y_ts = pd.date_range(ts[-1] + pd.Timedelta(minutes=CFG.data.freq_minutes),
                         periods=pred_len, freq=f"{CFG.data.freq_minutes}min")
    t_pred = time.time()
    pred = predictor.predict(df, pd.Series(ts), pd.Series(y_ts), pred_len=pred_len,
                             T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=False)
    dt_pred = time.time() - t_pred
    pred_nan = bool(pred.isnull().values.any())
    print(f"  예측 shape           : {pred.shape}")
    print(f"  NaN 포함             : {pred_nan}")
    print(f"  close 범위           : [{pred['close'].min():.3f}, {pred['close'].max():.3f}]")
    print(f"  소요 시간            : {dt_pred:.2f}s (pred_len={pred_len}, sample_count=1)")
    ok = ok and not pred_nan

    section("7. Stage 2 저장 용량 추정")
    itemsize = np.dtype(CFG.probe.store_dtype).itemsize
    per_layer_class = CFG.data.n_samples * T * D * itemsize
    total = per_layer_class * spec["n_layers"] * 2
    print(f"  설정: n_samples={CFG.data.n_samples}/클래스, T={T}, D={D}, dtype={CFG.probe.store_dtype}")
    print(f"  레이어 1개 / 클래스 1개 : {per_layer_class / 1024**2:.0f} MB")
    print(f"  전체 (12레이어 x 2클래스): {total / 1024**3:.2f} GB  -> scratch 에 저장")
    if CFG.data.n_samples <= D:
        print(f"  [경고] n_samples({CFG.data.n_samples}) <= d_model({D}). "
              "LDA 클래스내 공분산이 특이행렬이 된다. Stage 3 에서 shrinkage 필수.")

    section("결과")
    print(f"  전체 판정: {'PASS' if ok else 'FAIL — 위 경고 확인 필요'}")
    print(f"  총 소요  : {time.time() - t0:.1f}s")

    out = paths.ensure(paths.results_dir(CFG.tag)) / "stage0_report.json"
    out.write_text(json.dumps(
        {"env": env, "model_spec": spec, "config_hash": CFG.hash(),
         "activation_shape": [L, B, Tt, D], "deterministic": deterministic,
         "coherent": coherent, "pred_ok": not pred_nan, "pass": ok},
        indent=2, ensure_ascii=False))
    print(f"  리포트   : {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
