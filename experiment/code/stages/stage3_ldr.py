"""Stage 3 — Fisher probe 학습과 LDR 기반 개념 국소화 (명세 4, 5, 6절).

산출물 (Drive)
    results/<tag>/<noise>_<model>/ldr.npz
        ldr_train, ldr_test        [L, T]
        ldr_null_train/test        [L, T]   레이블을 섞은 null 대조군
        ldr_scaled                 [L, T]   min-max 정규화 (명세 5절)
        ldr_tokenmean_train/test   [L]      토큰 평균 활성화의 분리도
        delta_norm, w_last, h_norm
    results/<tag>/<noise>_<model>/class_stats_layer{i}.npz   Stage 4 용 median/mean
    figs/<tag>/stage3_*.png

핵심 판정
    1. 마지막 토큰 t=T 가 실제로 최대 분리 지점인가 (명세 6절의 causal frontier)
    2. held-out LDR 이 null 대조군보다 유의하게 큰가
       -> 크지 않으면 관측된 분리는 d_model=832 차원에 의한 과적합일 뿐이다

실행:
    python experiment/code/stages/stage3_ldr.py
    python experiment/code/stages/stage3_ldr.py --smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from kexp import activations as A  # noqa: E402
from kexp import ldr as LD  # noqa: E402
from kexp import paths  # noqa: E402
from kexp.config import CFG, resolve_model  # noqa: E402

BLUE, RED, GREY = "tab:blue", "tab:red", "0.55"


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def plot_heatmap(scaled: np.ndarray, layer_mean: np.ndarray, out: Path, title: str) -> None:
    """Wilinski et al. Figure 7 구조: 토큰 x 레이어 히트맵 + 레이어 평균 오버레이."""
    L, T = scaled.shape
    fig, ax1 = plt.subplots(figsize=(11, 8))
    im = ax1.imshow(scaled.T, aspect="auto", origin="lower", cmap="viridis",
                    extent=[-0.5, L - 0.5, -0.5, T - 0.5])
    ax1.set_xlabel("Model depth (layer)")
    ax1.set_ylabel("Token position t")
    ax1.set_xticks(range(L))
    fig.colorbar(im, ax=ax1, pad=0.10, label="Scaled LDR")

    ax2 = ax1.twinx()
    ax2.plot(np.arange(L), layer_mean, color="red", lw=2.5, marker="o", ms=5,
             label="token-mean activation")
    ax2.set_ylabel("Scaled LDR of token-mean activation", color="red")
    ax2.tick_params(axis="y", colors="red")
    ax2.set_xlim(-0.5, L - 0.5)
    ax2.legend(loc="lower right", fontsize=9)

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_profiles(res: dict, out: Path, title: str) -> None:
    """마지막 토큰의 레이어별 1차원 프로파일 + 토큰 위치별 프로파일."""
    L, T = res["ldr_test"].shape
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    layers = np.arange(L)
    axes[0].plot(layers, res["ldr_train"][:, -1], "o-", color=BLUE, label="train (in-sample)")
    axes[0].plot(layers, res["ldr_test"][:, -1], "s-", color=RED, label="held-out test")
    axes[0].plot(layers, res["ldr_null_test"][:, -1], "^--", color=GREY,
                 label="null (shuffled labels, test)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("LDR (log scale)")
    axes[0].set_xticks(layers)
    axes[0].set_title("Causal frontier: LDR at the last token h_i^(T)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    sm = LD.smooth(res["ldr_test"], 25)
    for li in np.linspace(0, L - 1, min(L, 6)).astype(int):
        axes[1].plot(np.arange(T), res["ldr_test"][li], lw=0.5, alpha=0.25, color=f"C{li % 10}")
        axes[1].plot(np.arange(T), sm[li], lw=1.8, color=f"C{li % 10}", label=f"layer {li}")
    axes[1].axvline(T - 1, color="k", ls=":", lw=1)
    axes[1].set_xlabel("Token position t"); axes[1].set_ylabel("held-out LDR")
    axes[1].set_title("LDR across token positions (thin = raw, thick = 25-pt moving average)")
    axes[1].legend(fontsize=8, ncol=2); axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_pca(act_dir_b: Path, act_dir_s: Path, coords: list, out: Path, title: str) -> None:
    """개입 전 표현 공간의 기하 구조. Stage 6 의 개입 후 비교와 짝을 이룬다."""
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(1, len(coords), figsize=(6 * len(coords), 5.5))
    axes = np.atleast_1d(axes)
    for ax, (layer, tok, label) in zip(axes, coords):
        a = np.asarray(A.load_layer(act_dir_b, layer)[:, tok, :], dtype=np.float32)
        b = np.asarray(A.load_layer(act_dir_s, layer)[:, tok, :], dtype=np.float32)
        pca = PCA(n_components=2).fit(np.vstack([a, b]))
        ra, rb = pca.transform(a), pca.transform(b)
        ax.scatter(ra[:, 0], ra[:, 1], s=6, alpha=0.4, color=BLUE, label="base")
        ax.scatter(rb[:, 0], rb[:, 1], s=6, alpha=0.4, color=RED, label="trend")
        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})"); ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_title(f"{label}\nlayer {layer}, token {tok}")
        ax.legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_lda_view(act_b: Path, act_s: Path, coords: list, idx_tr, idx_te,
                  rel_eps: float, device, out: Path, title: str) -> dict:
    """LDR 이 재는 분리를 직접 본다 — LDA 방향 1차원 투영.

    PCA 는 레이블을 보지 않고 전체 분산이 큰 방향을 찾는 반면, LDA 는
    클래스 내 분산으로 백색화한 뒤 평균차 방향을 찾는다. 832 차원에서
    공분산이 비등방적이면 두 방향은 거의 무관해질 수 있다. 따라서
    "LDR 은 높은데 PCA 산점도에서 안 갈라진다" 는 모순이 아니다.

    이 그림은 그 점을 직접 확인한다.
      상단: 학습 분할로 구한 LDA 방향에 held-out 표본을 투영한 히스토그램
      하단: 누적 설명분산, LDA 방향이 상위 k 개 PC 부분공간에 담긴 비율,
            그리고 PC1 축과 LDA 축의 산점도 비교
    """
    from sklearn.decomposition import PCA

    n_c = len(coords)
    fig, axes = plt.subplots(2, n_c, figsize=(6 * n_c, 10))
    info = {}

    for j, (layer, tok, label) in enumerate(coords):
        a = np.asarray(A.load_layer(act_b, layer)[:, tok, :], dtype=np.float32)
        b = np.asarray(A.load_layer(act_s, layer)[:, tok, :], dtype=np.float32)

        # 학습 분할로만 방향을 구하고 held-out 에 투영한다
        ta = torch.from_numpy(a[idx_tr]).to(device).unsqueeze(1)
        tb = torch.from_numpy(b[idx_tr]).to(device).unsqueeze(1)
        w = LD.fisher_directions(ta, tb, rel_eps)[0].cpu().numpy()
        w = w / (np.linalg.norm(w) + 1e-12)

        pa, pb = a[idx_te] @ w, b[idx_te] @ w
        ldr = (pb.mean() - pa.mean()) ** 2 / (pa.var() + pb.var() + 1e-12)

        ax = axes[0, j]
        bins = np.histogram_bin_edges(np.concatenate([pa, pb]), bins=60)
        ax.hist(pa, bins=bins, alpha=0.6, color=BLUE, label="base")
        ax.hist(pb, bins=bins, alpha=0.6, color=RED, label="trend")
        ax.set_title(f"{label}\nlayer {layer}, token {tok}  |  held-out LDR = {ldr:.1f}")
        ax.set_xlabel("projection onto LDA direction"); ax.legend(fontsize=9)

        # PCA 와의 관계
        X = np.vstack([a, b])
        k_max = min(64, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=k_max).fit(X)
        cum = np.cumsum(pca.explained_variance_ratio_)
        energy = np.cumsum((pca.components_ @ w) ** 2)   # w 는 단위벡터

        ax2 = axes[1, j]
        ax2.plot(np.arange(1, k_max + 1), cum, "-", color="tab:green",
                 label="cumulative explained variance")
        ax2.plot(np.arange(1, k_max + 1), energy, "-", color="tab:purple",
                 label="fraction of LDA direction in top-k PCs")
        ax2.axvline(2, color="k", ls=":", lw=1)
        ax2.set_xscale("log"); ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("number of principal components k")
        ax2.set_title(f"top-2 PCs hold {cum[1]:.1%} of variance\n"
                      f"but only {energy[1]:.2%} of the LDA direction")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        info[label] = {
            "layer": int(layer), "token": int(tok), "heldout_ldr": float(ldr),
            "var_top2": float(cum[1]), "lda_energy_top2": float(energy[1]),
            "lda_energy_top16": float(energy[min(15, k_max - 1)]),
            "cos_lda_pc1": float(abs(pca.components_[0] @ w)),
        }

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return info


def main(args) -> int:
    t0 = time.time()
    cfg = dataclasses.replace(CFG, model=resolve_model(args.model, CFG.model.max_context))
    variant = f"{args.noise}/{args.model}" + ("_smoke" if args.smoke else "")
    act_base = paths.activations_dir(cfg.tag, variant) / "base"
    act_trend = paths.activations_dir(cfg.tag, variant) / "trend"
    res_dir = paths.ensure(paths.results_dir(cfg.tag) / f"{args.noise}_{args.model}")
    fig_dir = paths.ensure(paths.figs_dir(cfg.tag))

    if not (act_base / "meta.json").exists():
        print(f"Stage 2 산출물이 없다: {act_base}")
        return 1

    meta = A.read_meta(act_base)
    L, N, T, D = meta["n_layers"], meta["n"], meta["T"], meta["D"]

    section("1. 설정")
    print(f"  활성화   : {act_base.parent}  [L={L}, N={N}, T={T}, D={D}]")
    print(f"  shrinkage: {cfg.probe.shrinkage} (trace 기반 상대값)")
    idx_tr, idx_te = LD.split_indices(N, cfg.probe.test_ratio, cfg.data.seed)
    print(f"  분할     : train {len(idx_tr)}, test {len(idx_te)}")
    if len(idx_tr) <= D:
        print(f"  [주의] train 표본({len(idx_tr)}) <= d_model({D}). shrinkage 의존도가 높다.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  device   : {device}")

    section("2. 레이어별 LDR")
    res = {k: np.zeros((L, T)) for k in
           ("ldr_train", "ldr_test", "ldr_null_train", "ldr_null_test", "delta_norm")}
    tokenmean = {"train": np.zeros(L), "test": np.zeros(L)}
    w_last = np.zeros((L, D), dtype=np.float32)
    h_norm = np.zeros(L)

    print(f"  {'layer':>5} {'LDR_train[T]':>13} {'LDR_test[T]':>12} {'null_test[T]':>13} "
          f"{'test 최대':>10} {'argmax t':>9} {'sec':>6}")
    for i in range(L):
        t1 = time.time()
        # mmap 으로 연다. 레이어 하나가 N=2048 기준 클래스당 1.7 GB 라 통째로
        # 올리면 Colab 의 12.7 GB RAM 에서 여유가 없다. ldr_for_layer 는
        # 토큰 위치 chunk 단위로만 읽으므로 실제 상주량은 수백 MB 에 그친다.
        a = np.load(A.layer_path(act_base, i), mmap_mode="r")
        b = np.load(A.layer_path(act_trend, i), mmap_mode="r")
        h_norm[i] = float(np.abs(a[: min(64, N)]).mean())

        r = LD.ldr_for_layer(a, b, idx_tr, idx_te, cfg.probe.shrinkage, device, args.chunk)
        rn = LD.ldr_for_layer(a, b, idx_tr, idx_te, cfg.probe.shrinkage, device, args.chunk,
                              shuffle_labels=True, seed=cfg.data.seed + i)
        res["ldr_train"][i], res["ldr_test"][i] = r["train"], r["test"]
        res["ldr_null_train"][i], res["ldr_null_test"][i] = rn["train"], rn["test"]
        res["delta_norm"][i] = r["delta_norm"]
        w_last[i] = r["w_last"]

        # 토큰 평균 활성화의 분리도 (steertool 의 빨간 선에 해당)
        am = torch.from_numpy(a.mean(axis=1, dtype=np.float32)).to(device).unsqueeze(1)
        bm = torch.from_numpy(b.mean(axis=1, dtype=np.float32)).to(device).unsqueeze(1)
        wm = LD.fisher_directions(am[idx_tr], bm[idx_tr], cfg.probe.shrinkage)
        tokenmean["train"][i] = LD.project_ldr(am[idx_tr], bm[idx_tr], wm).item()
        tokenmean["test"][i] = LD.project_ldr(am[idx_te], bm[idx_te], wm).item()
        del am, bm

        if args.save_class_stats:
            st = LD.class_statistics(a, b, device, args.chunk)
            np.savez_compressed(res_dir / f"class_stats_layer{i:02d}.npz", **st)

        del a, b
        print(f"  {i:>5} {r['train'][-1]:>13.2f} {r['test'][-1]:>12.2f} "
              f"{rn['test'][-1]:>13.4f} {r['test'].max():>10.2f} "
              f"{int(r['test'].argmax()):>9} {time.time() - t1:>6.1f}")

    section("3. 판정")
    ldr_te = res["ldr_test"]
    sm = LD.smooth(ldr_te, 25)          # 평활 곡선으로 구조를 본다
    bands = LD.token_bands(ldr_te)

    print("  토큰 구간별 held-out LDR 평균 (각 구간은 전체의 1/16)")
    print(f"  {'layer':>5} {'early':>8} {'mid':>8} {'late':>8} {'last':>8} "
          f"{'peak t':>7} {'peak':>8} {'train/test':>11}")
    for i in range(L):
        ratio = res["ldr_train"][i, -1] / (ldr_te[i, -1] + 1e-12)
        print(f"  {i:>5} {bands['early'][i]:>8.2f} {bands['mid'][i]:>8.2f} "
              f"{bands['late'][i]:>8.2f} {bands['last'][i]:>8.2f} "
              f"{int(sm[i].argmax()):>7} {sm[i].max():>8.2f} {ratio:>11.2f}")

    # 명세 6절의 causal frontier 가설: 마지막 토큰이 최대 분리 지점인가?
    # argmax 는 표본 노이즈의 최댓값을 고르므로, 평활 곡선의 정점 위치와
    # late 구간 대 peak 구간의 비로 판정한다.
    peak_pos = sm.argmax(axis=1)
    late_vs_peak = bands["late"] / (sm.max(axis=1) + 1e-12)
    frontier_holds = bool(np.median(peak_pos) > T * 0.9)
    print(f"\n  평활 곡선 정점 위치 (중앙값): t={int(np.median(peak_pos))} / {T-1}")
    print(f"  late 구간 / 정점 비율 (중앙값): {np.median(late_vs_peak):.2f}")
    print(f"  causal frontier 가설 (정점이 마지막 10% 안): {frontier_holds}")
    if not frontier_holds:
        print("  -> 마지막 토큰은 최대 분리 지점이 아니다. Stage 4 의 steering vector 는"
              " 마지막 토큰이 아니라 정점 구간에서 유도해야 한다.")

    best_layer, best_tok = np.unravel_index(np.argmax(sm), sm.shape)
    last_tok_best_layer = int(np.argmax(bands["late"]))
    print(f"  전역 최대(평활): layer {best_layer}, token {best_tok}, LDR {sm.max():.2f}")
    print(f"  late 구간 최고 레이어: {last_tok_best_layer} (LDR {bands['late'].max():.2f})")

    null_max = res["ldr_null_test"].max()
    signal = float(bands["late"].max())
    print(f"  null(test) 최대 {null_max:.4f} vs late 구간 신호 {signal:.2f} "
          f"-> 비율 {signal / (null_max + 1e-12):.1f}배")
    ok = signal > 10 * null_max
    if not ok:
        print("  [실패] held-out LDR 이 null 대조군 대비 충분히 크지 않다. "
              "관측된 분리는 차원수에 의한 과적합일 수 있다.")
    max_overfit = float((res["ldr_train"][:, -1] / (ldr_te[:, -1] + 1e-12)).max())
    print(f"  최대 train/test 비 {max_overfit:.2f} "
          f"(train {len(idx_tr)} / d_model {D} = {len(idx_tr)/D:.2f})")

    section("4. 저장 및 그림")
    scaled = LD.minmax(ldr_te)
    np.savez_compressed(
        res_dir / "ldr.npz", ldr_scaled=scaled, w_last=w_last, h_norm=h_norm,
        tokenmean_train=tokenmean["train"], tokenmean_test=tokenmean["test"],
        idx_train=idx_tr, idx_test=idx_te, **res)

    tag = f"{args.noise}_{args.model}" + ("_smoke" if args.smoke else "")
    coords = [(int(last_tok_best_layer), T - 1, "best layer @ last token"),
              (0, T - 1, "layer 0 @ last token"),
              (int(best_layer), int(best_tok), "global max")]

    f1 = fig_dir / f"stage3_{tag}_heatmap.png"
    f2 = fig_dir / f"stage3_{tag}_profiles.png"
    f3 = fig_dir / f"stage3_{tag}_pca.png"
    f4 = fig_dir / f"stage3_{tag}_lda.png"
    plot_heatmap(scaled, LD.minmax(tokenmean["test"]), f1,
                 f"Scaled LDR: linear increasing momentum [{args.noise}, {args.model}]")
    plot_profiles(res, f2, f"LDR profiles [{args.noise}, {args.model}]")
    plot_pca(act_base, act_trend, coords, f3,
             f"Latent geometry before intervention [{args.noise}, {args.model}]")
    lda_info = plot_lda_view(act_base, act_trend, coords, idx_tr, idx_te,
                             cfg.probe.shrinkage, device, f4,
                             f"LDA view: what the LDR actually measures "
                             f"[{args.noise}, {args.model}]")
    for f in (f1, f2, f3, f4):
        print(f"  {f}")

    print("\n  PCA 가 판별 방향을 보여주지 못하는 정도")
    print(f"  {'좌표':28s} {'held-out LDR':>13} {'상위2 PC 분산':>14} "
          f"{'그 안의 LDA 방향':>17} {'|cos(LDA,PC1)|':>15}")
    for label, d in lda_info.items():
        print(f"  {label:28s} {d['heldout_ldr']:>13.2f} {d['var_top2']:>13.1%} "
              f"{d['lda_energy_top2']:>16.2%} {d['cos_lda_pc1']:>15.4f}")

    summary = {
        "model": args.model, "noise": args.noise, "L": L, "N": N, "T": T, "D": D,
        "best_layer": int(best_layer), "best_token": int(best_tok),
        "best_ldr_smoothed": float(sm.max()),
        "late_best_layer": last_tok_best_layer,
        "bands": {k: v.tolist() for k, v in bands.items()},
        "smoothed_peak_token_by_layer": peak_pos.tolist(),
        "late_vs_peak_ratio": late_vs_peak.tolist(),
        "causal_frontier_holds": frontier_holds,
        "null_test_max": float(null_max),
        "max_train_test_ratio": max_overfit,
        "train_over_dmodel": len(idx_tr) / D,
        "h_norm_by_layer": h_norm.tolist(),
        "lda_vs_pca": lda_info,
        "pass": bool(ok),
    }
    (res_dir / "stage3_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    section("결과")
    print(f"  판정: {'PASS' if ok else 'FAIL'}")
    print(f"  총 소요: {time.time() - t0:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 3 — Fisher probe / LDR")
    p.add_argument("--model", default="base", choices=["mini", "small", "base"])
    p.add_argument("--noise", default=CFG.data.noise_type, choices=["ou", "rw"])
    p.add_argument("--chunk", type=int, default=32, help="한 번에 처리할 토큰 위치 수")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-class-stats", dest="save_class_stats", action="store_false",
                   help="Stage 4 용 median/mean 통계를 저장하지 않는다")
    raise SystemExit(main(p.parse_args()))
