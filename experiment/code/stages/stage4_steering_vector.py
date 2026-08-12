"""Stage 4 — 개념 조정 벡터 S_i 산출 (설계 Step 4).

Stage 3 이 저장한 위치별 클래스 통계에서 S_i = M_{i,s} - M_{i,c} 를 만든다.
활성화 39 GB 를 다시 읽지 않는다 (||h_i|| 캘리브레이션용으로 소수 표본만 읽는다).

프로브 위치와 레이어 선택은 여기서 정하지 않는다. Stage 3 이 남긴
stage3_summary.json 의 실측값(평활 LDR 정점, 구간별 LDR)을 읽어 쓴다.

산출물
    results/<tag>/<noise>_<model>/steering.npz
        S[method][region]  [L, D]
        s_norm             [L, len(REGIONS)]   ||S_i||
        h_norm_l2          [L]                 ||h_i^(t)||_2 대표값
        lambda_table       상대 강도 -> 레이어별 절대 lambda
    figs/<tag>/stage4_*.png

실행:
    python experiment/code/stages/stage4_steering_vector.py
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

from kexp import activations as A  # noqa: E402
from kexp import paths  # noqa: E402
from kexp import steering_vec as SV  # noqa: E402
from kexp.config import CFG, resolve_model  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def plot_diagnostics(s_norm, h_l2, cos_region, cos_layer, diff_norm_by_token,
                     ldr_late, ood_ratio, ood_reached, rels, threshold,
                     out: Path, title: str) -> None:
    L = len(h_l2)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # (0,0) lambda 캘리브레이션의 근거
    ax = axes[0, 0]
    ax.plot(range(L), h_l2, "o-", color="tab:blue", label="||h_i|| (activation)")
    ax.plot(range(L), s_norm, "s-", color="tab:red", label="||S_i|| (steering vector)")
    ax.set_xlabel("Layer"); ax.set_ylabel("L2 norm"); ax.set_xticks(range(L))
    ax.set_title("Norms grow with depth -> a fixed absolute lambda is not comparable")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ratio = s_norm / h_l2
    ax.plot(range(L), ratio, "o-", color="tab:purple")
    ax.set_xlabel("Layer"); ax.set_ylabel("||S_i|| / ||h_i||"); ax.set_xticks(range(L))
    ax.set_title("Relative size of the steering vector\n(lambda_rel = lambda_i * this)")
    ax.grid(alpha=0.3)

    # (1,0) 구간 선택이 방향을 바꾸는가
    ax = axes[1, 0]
    for label, vals in cos_region.items():
        ax.plot(range(L), vals, "o-", label=label)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Layer"); ax.set_ylabel("cosine similarity"); ax.set_xticks(range(L))
    ax.set_title("Do region / method choices give the same direction?")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (1,1) 위치별 차이 벡터의 크기 vs LDR
    ax = axes[1, 1]
    T = diff_norm_by_token.shape[1]
    for li in np.linspace(0, L - 1, min(L, 6)).astype(int):
        ax.plot(np.arange(T), diff_norm_by_token[li], lw=1.2, label=f"layer {li}")
    ax.set_xlabel("Token position t"); ax.set_ylabel("||S_i^(t)||")
    ax.set_title("Class-mean difference across token positions")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # (2,0) 어느 lambda 까지 데이터 다양체 위에 머무는가
    ax = axes[2, 0]
    for li in np.linspace(0, L - 1, min(L, 6)).astype(int):
        ax.plot(rels, ood_ratio[li], "o-", lw=1.4, label=f"layer {li}")
    ax.axhline(threshold, color="k", ls="--", lw=1.2, label=f"threshold {threshold}")
    ax.axhline(1.0, color="0.6", ls=":", lw=1)
    ax.set_xscale("symlog", linthresh=0.05); ax.set_xlim(0, rels.max() * 1.1)
    ax.set_xlabel("lambda_rel"); ax.set_ylabel("NN distance / real NN distance")
    ax.set_title("Off-manifold drift: steering is a straight chord\nacross a curved 1-D manifold")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # (2,1) 같은 lambda 에서 실제로 얼마나 이동했는가
    ax = axes[2, 1]
    for li in np.linspace(0, L - 1, min(L, 6)).astype(int):
        ax.plot(rels, ood_reached[li], "o-", lw=1.4, label=f"layer {li}")
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="reaches trend mean")
    ax.set_xscale("symlog", linthresh=0.05); ax.set_xlim(0, rels.max() * 1.1)
    ax.set_xlabel("lambda_rel"); ax.set_ylabel("fraction of base->trend gap covered")
    ax.set_title("How far the intervention actually moves the representation")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(args) -> int:
    t0 = time.time()
    cfg = dataclasses.replace(CFG, model=resolve_model(args.model, CFG.model.max_context))
    variant = f"{args.noise}/{args.model}" + ("_smoke" if args.smoke else "")
    act_dir = paths.activations_dir(cfg.tag, variant)
    res_dir = paths.results_dir(cfg.tag) / f"{args.noise}_{args.model}"
    fig_dir = paths.ensure(paths.figs_dir(cfg.tag))

    summ_path = res_dir / "stage3_summary.json"
    if not summ_path.exists():
        print(f"Stage 3 산출물이 없다: {summ_path}")
        return 1
    summ = json.loads(summ_path.read_text())
    L, T, D = summ["L"], summ["T"], summ["D"]
    peaks = summ["smoothed_peak_token_by_layer"]
    ldr_late = np.array(summ["bands"]["late"])

    section("1. Stage 3 실측값 (여기서 위치·레이어를 읽어온다)")
    print(f"  [L={L}, T={T}, D={D}]  train/d_model = {summ['train_over_dmodel']:.2f}")
    print(f"  평활 LDR 정점 위치 (레이어별): {peaks}")
    print(f"  late 구간 LDR 최고 레이어    : {summ['late_best_layer']} ({ldr_late.max():.2f})")
    print(f"  causal frontier 가설         : {summ['causal_frontier_holds']}")
    print(f"  null 대비 신호               : {ldr_late.max() / (summ['null_test_max'] + 1e-12):.0f}배")

    section("2. S_i 산출")
    ldr = np.load(res_dir / "ldr.npz")
    w_last = ldr["w_last"]                    # [L, D] 마지막 토큰의 LDA 방향

    S = {(m, r): np.zeros((L, D), dtype=np.float32)
         for m in ("median", "mean") for r in SV.REGIONS}
    # 논문(5쪽)의 S_i 는 R^{N x D} 행렬이다. 토큰 위치마다 다른 벡터를 더한다.
    # 위의 [D] 벡터들은 명세 7절이 AR 생성의 신규 토큰에 쓰려고 축약한 형태이며,
    # Stage 5 에서 두 형태를 모두 시험한다.
    S_mat = {m: np.zeros((L, T, D), dtype=np.float16) for m in ("median", "mean")}
    S_lda = np.zeros((L, D), dtype=np.float32)
    diff_norm_by_token = np.zeros((L, T))
    h_token_l2 = np.zeros((L, T))
    h_l2 = np.zeros(L)

    for i in range(L):
        f = res_dir / f"class_stats_layer{i:02d}.npz"
        if not f.exists():
            print(f"  [실패] {f} 가 없다. Stage 3 을 --no-class-stats 없이 다시 실행할 것.")
            return 1
        st = dict(np.load(f))
        vecs, diffs = SV.build_vectors(st, peaks[i])
        for key, v in vecs.items():
            S[key][i] = v
        for m in ("median", "mean"):
            S_mat[m][i] = diffs[m].astype(np.float16)
        diff_norm_by_token[i] = np.linalg.norm(diffs["median"], axis=-1)
        # 행렬 형태의 lambda 캘리브레이션에는 위치별 ||h|| 가 필요하다
        sample = np.asarray(A.load_layer(act_dir / "base", i)[:64], dtype=np.float32)
        h_token_l2[i] = np.linalg.norm(sample, axis=-1).mean(axis=0)
        # LDA 방향은 스케일이 임의라 median 벡터와 같은 노름으로 맞춘다
        S_lda[i] = SV.scale_like(w_last[i], S[("median", "late")][i])
        h_l2[i] = SV.activation_l2(A.load_layer(act_dir / "base", i))

    s_norm = {k: np.linalg.norm(v, axis=-1) for k, v in S.items()}
    s_norm[("lda", "last")] = np.linalg.norm(S_lda, axis=-1)

    print(f"  {'layer':>5} {'||h_i||':>9} {'||S median,late||':>18} "
          f"{'비율':>8} {'late LDR':>9}")
    for i in range(L):
        r = s_norm[("median", "late")][i] / h_l2[i]
        print(f"  {i:>5} {h_l2[i]:>9.2f} {s_norm[('median','late')][i]:>18.2f} "
              f"{r:>8.3f} {ldr_late[i]:>9.2f}")

    section("3. 선택이 결과를 좌우하는가 (코사인 유사도)")
    cos_region = {}
    ref = ("median", "late")
    for key in [("median", "token_mean"), ("median", "peak"), ("mean", "late")]:
        cos_region[f"{key[0]}/{key[1]} vs median/late"] = [
            SV.cosine(S[key][i], S[ref][i]) for i in range(L)]
    cos_region["lda(last) vs median/late"] = [
        SV.cosine(S_lda[i], S[ref][i]) for i in range(L)]

    for label, vals in cos_region.items():
        print(f"  {label:32s} 중앙값 {np.median(vals):.4f}  범위 [{min(vals):.4f}, {max(vals):.4f}]")

    # 레이어 간 방향 일치도 — 전 레이어에 같은 개념 방향이 있는가
    cos_layer = np.array([[SV.cosine(S[ref][i], S[ref][j]) for j in range(L)] for i in range(L)])
    print(f"  인접 레이어 간 코사인 (중앙값): "
          f"{np.median([cos_layer[i, i+1] for i in range(L-1)]):.4f}")

    section("4. lambda 캘리브레이션표")
    print(f"  상대 강도 lambda_rel = lambda_i * ||S_i|| / ||h_i||")
    print(f"  {'lambda_rel':>11} " + " ".join(f"{'L'+str(i):>7}" for i in range(L)))
    lam_table = {}
    for rel in cfg.steer.lambdas_rel:
        lam = SV.lambda_for_relative(rel, s_norm[ref], h_l2)
        lam_table[str(rel)] = lam.tolist()
        print(f"  {rel:>11.2f} " + " ".join(f"{v:>7.2f}" for v in lam))

    section("5. OOD 진단 — 개입이 표현을 다양체 밖으로 미는 lambda 한계")
    print("  Stage 3 의 PCA 에서 마지막 토큰 표현이 휘어진 1 차원 다양체로 나타났다.")
    print("  S 는 그 호를 가로지르는 직선이므로 큰 lambda 는 다양체를 벗어난다.")
    print("  비율 = (조정된 점의 최근접 실제 활성화 거리) / (실제 활성화끼리의 최근접 거리)\n")

    rels = np.array(cfg.steer.lambdas_rel)
    ood_ratio = np.zeros((L, len(rels)))
    ood_reached = np.zeros((L, len(rels)))
    for i in range(L):
        hb = np.asarray(A.load_layer(act_dir / "base", i)[: args.ood_n, -1, :], dtype=np.float32)
        ht = np.asarray(A.load_layer(act_dir / "trend", i)[: args.ood_n, -1, :], dtype=np.float32)
        lam_abs = SV.lambda_for_relative(rels, s_norm[ref][i], h_l2[i])
        prof = SV.ood_profile(hb, ht, S[ref][i], lam_abs)
        ood_ratio[i], ood_reached[i] = prof["ratio"], prof["reached"]

    print(f"  {'layer':>5} " + " ".join(f"{'rel=' + str(r):>9}" for r in rels))
    for i in range(L):
        print(f"  {i:>5} " + " ".join(f"{v:>9.2f}" for v in ood_ratio[i]))
    print("\n  같은 lambda 에서 trend 평균까지 도달한 비율 (1.0 = trend 평균에 도착)")
    print(f"  {'layer':>5} " + " ".join(f"{'rel=' + str(r):>9}" for r in rels))
    for i in (0, L // 2, L - 1):
        print(f"  {i:>5} " + " ".join(f"{v:>9.2f}" for v in ood_reached[i]))

    safe = [float(rels[max(0, int(np.searchsorted(ood_ratio[i], args.ood_threshold)) - 1)])
            for i in range(L)]
    print(f"\n  비율 {args.ood_threshold} 이하를 유지하는 최대 lambda_rel (레이어별): {safe}")
    print(f"  -> Stage 5 의 lambda 격자는 이 범위를 넘는 지점에서 붕괴가 예상된다.")

    section("6. Stage 6 용 활성화 부분집합 저장")
    # 활성화 39 GB 는 Colab 로컬 scratch 에 있어 세션이 끝나면 사라진다.
    # Stage 6 의 개입 전/후 PCA 비교(설계 Step 5)에는 원시 활성화가 필요하므로,
    # 관심 좌표만 잘라 Drive 에 남긴다. 이렇게 두면 나중에 Stage 2 를 9 분
    # 다시 돌릴 필요가 없다.
    peak_tok = int(np.median(peaks))
    toks = [T - 1, peak_tok]
    n_sub = min(args.pca_n, summ["N"] if "N" in summ else args.pca_n)
    sub = np.zeros((2, L, len(toks), n_sub, D), dtype=np.float16)
    for ci, cls in enumerate(("base", "trend")):
        for i in range(L):
            arr = A.load_layer(act_dir / cls, i)
            for ti, tk in enumerate(toks):
                sub[ci, i, ti] = arr[:n_sub, tk, :]
    np.savez_compressed(res_dir / "pca_subset.npz", act=sub, tokens=np.array(toks),
                        classes=np.array(["base", "trend"]))
    mb = (res_dir / "pca_subset.npz").stat().st_size / 1024**2
    print(f"  좌표: 토큰 {toks} (마지막, 평활 정점 중앙값), 표본 {n_sub}/클래스")
    print(f"  {res_dir / 'pca_subset.npz'}  ({mb:.1f} MB)")

    section("7. 저장 및 그림")
    np.savez_compressed(
        res_dir / "steering.npz",
        S_lda=S_lda, h_norm_l2=h_l2, ldr_late=ldr_late,
        peaks=np.array(peaks), cos_layer=cos_layer,
        diff_norm_by_token=diff_norm_by_token, h_token_l2=h_token_l2,
        ood_ratio=ood_ratio, ood_reached=ood_reached, lambdas_rel=rels,
        S_matrix_median=S_mat["median"], S_matrix_mean=S_mat["mean"],
        **{f"S_{m}_{r}": v for (m, r), v in S.items()},
        **{f"snorm_{m}_{r}": v for (m, r), v in s_norm.items()})

    tag = f"{args.noise}_{args.model}" + ("_smoke" if args.smoke else "")
    f1 = fig_dir / f"stage4_{tag}_vectors.png"
    plot_diagnostics(s_norm[ref], h_l2, cos_region, cos_layer, diff_norm_by_token,
                     ldr_late, ood_ratio, ood_reached, rels, args.ood_threshold,
                     f1, f"Steering vectors S_i [{args.noise}, {args.model}]")
    print(f"  {res_dir / 'steering.npz'}")
    print(f"  {f1}")

    (res_dir / "stage4_summary.json").write_text(json.dumps({
        "h_norm_l2": h_l2.tolist(),
        "s_norm_median_late": s_norm[ref].tolist(),
        "s_over_h": (s_norm[ref] / h_l2).tolist(),
        "cosine": {k: list(map(float, v)) for k, v in cos_region.items()},
        "lambda_table": lam_table,
        "late_best_layer": summ["late_best_layer"],
        "ldr_late": ldr_late.tolist(),
        "lambdas_rel": rels.tolist(),
        "ood_ratio": ood_ratio.tolist(),
        "ood_reached": ood_reached.tolist(),
        "ood_threshold": args.ood_threshold,
        "safe_lambda_rel_by_layer": safe,
    }, indent=2, ensure_ascii=False))

    section("결과")
    print(f"  판정: PASS   총 소요: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 4 — steering vector 산출")
    p.add_argument("--model", default="base", choices=["mini", "small", "base"])
    p.add_argument("--noise", default=CFG.data.noise_type, choices=["ou", "rw"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--ood-n", type=int, default=512,
                   help="OOD 진단에 쓸 표본 수 (레이어·클래스당)")
    p.add_argument("--ood-threshold", type=float, default=2.0,
                   help="최근접 거리 비율이 이 값을 넘으면 다양체 밖으로 본다")
    p.add_argument("--pca-n", type=int, default=512,
                   help="Stage 6 PCA 용으로 Drive 에 남길 표본 수 (클래스당)")
    raise SystemExit(main(p.parse_args()))
