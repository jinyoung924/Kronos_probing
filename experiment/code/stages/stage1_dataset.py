"""Stage 1 — 합성 데이터셋 생성 (명세 1, 2, 3절).

산출물 (Drive)
    data/<tag>/base.npz    x [n, T, 6] float32, snrs [n]
    data/<tag>/trend.npz
    data/<tag>/meta.json
    figs/<tag>/stage1_*.png

검증 항목
    1. OHLC coherence 위반 0 건 (명세 1절이 정의상 보장하는지 실측)
    2. 두 클래스의 노이즈 실현치 공유 (close 차이 == m*tau 인지)
    3. z-score 후 [-5, 5] 클리핑률 (명세 3절)
    4. 정규화된 close 의 기울기 분포가 두 클래스에서 분리되는지
       -> 분리가 안 되면 Kronos 에 넣기도 전에 개념이 존재하지 않는다는 뜻이므로
          여기서 snr 을 올려야 한다

실행:
    python experiment/code/stages/stage1_dataset.py
    python experiment/code/stages/stage1_dataset.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kexp import paths, synth  # noqa: E402
from kexp.config import CFG  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# 그림의 모든 문자는 영어로 쓴다. Colab / 로컬 matplotlib 모두 CJK 폰트가
# 없어서 한글은 두부(□)로 렌더링된다.
BLUE, RED = "tab:blue", "tab:red"


def plot_samples(x_base, x_trend, out: Path, noise_type: str, n_show: int = 3) -> None:
    """개별 샘플의 원시 경로와 z-score 후 경로."""
    fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 7), sharex=True)
    for j in range(n_show):
        for name, x, color in (("base", x_base, BLUE), ("trend", x_trend, RED)):
            axes[0, j].plot(x[j, :, synth.CH_CLOSE], color=color, lw=0.9, label=f"{name} close")
            axes[0, j].fill_between(np.arange(x.shape[1]), x[j, :, synth.CH_LOW],
                                    x[j, :, synth.CH_HIGH], color=color, alpha=0.2,
                                    lw=0, label=f"{name} high-low")
            z = (x[j] - x[j].mean(axis=0)) / (x[j].std(axis=0) + 1e-5)
            axes[1, j].plot(z[:, synth.CH_CLOSE], color=color, lw=0.9)
        axes[0, j].set_title(f"sample {j}: raw price")
        axes[1, j].set_title("after per-channel z-score")
        for v in (-CFG.model.clip, CFG.model.clip):
            axes[1, j].axhline(v, color="k", ls=":", lw=0.8)
        axes[1, j].set_xlabel("K-line step t")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("price")
    axes[1, 0].set_ylabel("z-scored close")
    fig.suptitle(f"Stage 1 [{noise_type}] base vs trend — identical noise realization, "
                 "drift is the only difference")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_diagnostics(x_base, x_trend, snrs, out: Path, noise_type: str) -> None:
    zb = _z(x_base)[:, :, synth.CH_CLOSE]
    zs = _z(x_trend)[:, :, synth.CH_CLOSE]
    slope_b, slope_s = synth._slope(zb), synth._slope(zs)
    ldr = (slope_s.mean() - slope_b.mean()) ** 2 / (slope_s.var() + slope_b.var())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) 개념을 가장 직접적으로 보여주는 그림: 샘플 평균 궤적
    t = np.arange(zb.shape[1])
    for name, z, color in (("base", zb, BLUE), ("trend", zs, RED)):
        mu, sd = z.mean(axis=0), z.std(axis=0)
        axes[0, 0].plot(t, mu, color=color, lw=1.6, label=f"{name} mean")
        axes[0, 0].fill_between(t, mu - sd, mu + sd, color=color, alpha=0.2, lw=0)
    axes[0, 0].set_title("Mean z-scored close across samples (+-1 std)")
    axes[0, 0].set_xlabel("K-line step t"); axes[0, 0].set_ylabel("z-scored close")
    axes[0, 0].legend()

    # (0,1) 1차원 특징에서의 분리도
    bins = np.histogram_bin_edges(np.concatenate([slope_b, slope_s]), bins=60)
    axes[0, 1].hist(slope_b, bins=bins, alpha=0.6, label="base", color=BLUE)
    axes[0, 1].hist(slope_s, bins=bins, alpha=0.6, label="trend", color=RED)
    axes[0, 1].axvline(np.percentile(slope_b, 95), color="k", ls="--", lw=1,
                       label="base 95th pct")
    axes[0, 1].set_title(f"OLS slope of z-scored close   (input LDR = {ldr:.2f})")
    axes[0, 1].set_xlabel("slope"); axes[0, 1].legend(fontsize=9)

    # (1,0) 클리핑 경계까지의 여유 (명세 3절)
    axes[1, 0].hist(_z(x_base)[:, :, :4].ravel(), bins=120, alpha=0.6, label="base", color=BLUE)
    axes[1, 0].hist(_z(x_trend)[:, :, :4].ravel(), bins=120, alpha=0.6, label="trend", color=RED)
    for v in (-CFG.model.clip, CFG.model.clip):
        axes[1, 0].axvline(v, color="k", ls=":", lw=1.2)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title(f"z-scored OHLC values (dotted = +-{CFG.model.clip} clipping bound)")
    axes[1, 0].set_xlabel("z-score"); axes[1, 0].legend()

    # (1,1) drift 세기와 관측 기울기의 관계
    axes[1, 1].scatter(snrs, slope_s, s=5, alpha=0.35, color=RED, label="trend")
    axes[1, 1].scatter(snrs, slope_b, s=5, alpha=0.35, color=BLUE, label="base (no drift)")
    axes[1, 1].set_xlabel("snr (per-sample drift strength)")
    axes[1, 1].set_ylabel("observed slope")
    axes[1, 1].set_title("drift strength vs observed slope"); axes[1, 1].legend()

    fig.suptitle(f"Stage 1 diagnostics [{noise_type}]")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-5)


def build_variant(cfg, noise_type: str, force: bool) -> dict:
    """노이즈 종류 하나에 대해 데이터셋을 생성·검증·저장한다."""
    t0 = time.time()
    d = cfg.data
    out_dir = paths.ensure(paths.data_dir(cfg.tag, noise_type))
    fig_dir = paths.ensure(paths.figs_dir(cfg.tag))

    if (out_dir / "meta.json").exists() and not force:
        print(f"  이미 존재한다: {out_dir}. 다시 만들려면 --force")
        return json.loads((out_dir / "meta.json").read_text())

    print(f"  노이즈        : {noise_type}" + (f" (theta={d.ou_theta})" if noise_type == "ou" else ""))
    print(f"  노이즈 크기   : {d.noise_scale(noise_type):.5f}  (snr 의 분모)")
    print(f"  m 범위        : [{d.drift_per_micro(d.snr_min, noise_type):.3e}, "
          f"{d.drift_per_micro(d.snr_max, noise_type):.3e}] (미세 스텝당)")

    x_base, x_trend, snrs = synth.generate_pair(d, d.n_samples, d.seed, noise_type)
    ts = synth.make_timestamps(d)
    print(f"  생성 완료     : {x_base.shape} x2, {time.time() - t0:.1f}s")

    coh_b = synth.check_coherence(x_base)
    coh_s = synth.check_coherence(x_trend)
    share = synth.check_noise_sharing(x_base, x_trend, d, snrs, noise_type)
    print(f"  coherence     : base={coh_b['ok']}, trend={coh_s['ok']}")
    print(f"  노이즈 공유   : close 차이의 상대오차 {share['rel_err_close']:.2e}")

    st_b = synth.normalized_stats(x_base, cfg.model.clip)
    st_s = synth.normalized_stats(x_trend, cfg.model.clip)
    print(f"\n  {'지표':22s} {'base':>14s} {'trend':>14s}")
    for k in st_b:
        print(f"  {k:22s} {st_b[k]:14.6g} {st_s[k]:14.6g}")

    # Kronos 에 넣기 전 sanity check: 개념이 입력 단계에서 분리되는가.
    # 이 값은 latent LDR 의 상한이 아니다(모델은 832 차원을 쓴다). 다만 여기서
    # 분리가 안 되면 표현 공간에서 분리되기를 기대할 근거가 약하다.
    sb = synth._slope(_z(x_base)[:, :, synth.CH_CLOSE])
    ss = synth._slope(_z(x_trend)[:, :, synth.CH_CLOSE])
    input_ldr = float((ss.mean() - sb.mean()) ** 2 / (ss.var() + sb.var()))
    thr = np.percentile(sb, 95)
    print(f"\n  입력단 LDR (close 기울기 1차원): {input_ldr:.3f}")
    print(f"  base 95 퍼센타일 이상인 trend 비율: {(ss > thr).mean():.1%}")

    ok = coh_b["ok"] and coh_s["ok"] and share["rel_err_close"] < 1e-4
    if st_s["clip_fraction"] > 0.01:
        print(f"  [주의] trend 클리핑률 {st_s['clip_fraction']:.2%} > 1%. snr 을 낮출 것.")
    if input_ldr < 1.0:
        print(f"  [주의] 입력단 LDR 이 낮다({input_ldr:.2f}). snr 을 올리거나 노이즈를 바꿀 것.")

    f1 = fig_dir / f"stage1_{noise_type}_samples.png"
    f2 = fig_dir / f"stage1_{noise_type}_diagnostics.png"
    plot_samples(x_base, x_trend, f1, noise_type)
    plot_diagnostics(x_base, x_trend, snrs, f2, noise_type)

    np.savez_compressed(out_dir / "base.npz", x=x_base, snrs=np.zeros_like(snrs))
    np.savez_compressed(out_dir / "trend.npz", x=x_trend, snrs=snrs)
    np.save(out_dir / "timestamps.npy", ts.values)
    meta = {
        "config_hash": cfg.hash(),
        "noise_type": noise_type,
        "data_cfg": cfg.to_dict()["data"],
        "coherence": {"base": coh_b, "trend": coh_s},
        "noise_sharing": share,
        "normalized_stats": {"base": st_b, "trend": st_s},
        "input_ldr_close_slope": input_ldr,
        "pass": bool(ok),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    mb = sum((out_dir / n).stat().st_size for n in ("base.npz", "trend.npz")) / 1024**2
    print(f"\n  저장: {out_dir}  ({mb:.1f} MB)")
    print(f"  그림: {f1.name}, {f2.name}")
    print(f"  판정: {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s)")
    return meta


def main(args) -> int:
    cfg = CFG
    d = cfg.data
    variants = ["ou", "rw"] if args.noise == "both" else [args.noise]

    section("설정")
    print(f"  T={d.T}, n_micro={d.n_micro} -> 미세 스텝 {d.n_micro_total}")
    print(f"  sigma={d.sigma}, snr ~ U({d.snr_min}, {d.snr_max})")
    print(f"  n_samples={d.n_samples}/클래스, seed={d.seed}")
    print(f"  주 모델 {cfg.model.model_id} 의 d_model=832 -> LDA 조건 n_samples > 832: "
          f"{'OK' if d.n_samples > 832 else '미달'}")
    print(f"  생성할 갈래: {variants}  (주 데이터셋 = {d.noise_type})")

    metas = {}
    for nt in variants:
        section(f"노이즈 = {nt}")
        metas[nt] = build_variant(cfg, nt, args.force)

    section("요약")
    print(f"  {'noise':>6} {'input LDR':>11} {'clip%(trend)':>14} {'coherence':>11} {'판정':>6}")
    for nt, m in metas.items():
        print(f"  {nt:>6} {m['input_ldr_close_slope']:>11.2f} "
              f"{m['normalized_stats']['trend']['clip_fraction']:>13.4%} "
              f"{str(m['coherence']['trend']['ok']):>11} {'PASS' if m['pass'] else 'FAIL':>6}")
    return 0 if all(m["pass"] for m in metas.values()) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1 — 합성 데이터셋 생성")
    p.add_argument("--force", action="store_true", help="기존 산출물을 덮어쓴다")
    p.add_argument("--noise", choices=["ou", "rw", "both"], default="both",
                   help="생성할 노이즈 갈래 (기본: 둘 다)")
    raise SystemExit(main(p.parse_args()))
