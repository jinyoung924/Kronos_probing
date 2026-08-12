"""Stage 6 — 개입 결과 평가와 시각화 (설계 Step 5).

Stage 5 의 결과(JSON)와 Stage 3/4 의 산출물만 읽는다. 모델도 활성화 39 GB 도
필요 없다 — Stage 4 가 Drive 에 남긴 pca_subset.npz 로 기하 검증까지 마친다.

측정
    주 지표   예측 close 의 회귀 기울기 (정규화 공간) vs lambda
    통계      기준선(lambda=0)과 같은 입력 계열을 쓰므로 **대응표본** 검정
    붕괴      OHLC coherence 위반율, 예측 변동성, 기울기 분산
    기하      개입 전/후 표현이 trend 클래스 방향으로 이동했는가

기하 검증에 대한 주의
    설계 Step 5 는 PCA 를 지정하지만, Stage 3 에서 상위 2 개 PC 가 분산의
    48~80% 를 담으면서도 판별 방향은 그 안에 0.01% 뿐임을 확인했다. 따라서
    PCA 산점도는 개념 이동을 보여주지 못한다. LDA 축 투영을 함께 그린다.

    또한 여기서 그리는 "개입 후" 는 개입 지점에서의 **직접 가산 효과**
    (h + lambda*S) 이지, 이후 레이어를 통과하며 전파된 표현이 아니다.

실행:
    python experiment/code/stages/stage6_report.py
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

from kexp import paths  # noqa: E402
from kexp.config import CFG, resolve_model  # noqa: E402

# 팔의 분류. 색과 그룹 패널을 결정한다.
GROUPS = {
    "paper": ("원논문 방법", "tab:blue"),
    "ours": ("데이터 기반", "tab:red"),
    "ctrl": ("대조군", "0.45"),
}


def group_of(arm: str) -> str:
    if "_paper_" in arm:
        return "paper"
    if "_ours_" in arm:
        return "ours"
    return "ctrl"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def outcome_stats(slopes: list, base: np.ndarray) -> dict:
    """개입 결과의 통계.

    주 검정을 **결과 부호의 이항검정**으로 잡는다. 기준선과의 대응표본 평균
    비교가 아니다. 이유는 개입의 작동 방식에 있다.

        기준선   평균 -0.001, std 0.023  — 표본마다 제각각, 절반이 양수
        개입 후  평균  0.004, std 0.001  — 전부 작은 양수로 수렴

    즉 개입은 drift 를 더하는 것이 아니라 모델의 자체 동학을 통제된 추세로
    **덮어쓴다**. 원래 크게 상승하던 표본은 오히려 끌어내려진다. 그래서
    쌍별 차이의 부호는 절반씩 갈리고(17/32) 대응표본 검정은 무력하지만,
    결과가 전부 양수라는 사실(32/32, p=5e-10)은 압도적이다.

    변동성 붕괴 자체도 보고한다. 개입 강도의 직접적 지표다.
    """
    from scipy import stats as st
    s = np.asarray(slopes, dtype=np.float64)
    n = len(s)
    n_pos = int((s > 0).sum())
    b_sd = base.std(ddof=1)
    s_sd = s.std(ddof=1)
    diff = s - base
    return {
        "frac_positive": n_pos / n,
        "p_sign": float(st.binomtest(n_pos, n, 0.5).pvalue),
        "var_ratio": float(s_sd / b_sd) if b_sd > 0 else 1.0,
        "slope_mean": float(s.mean()),
        "slope_ci95": float(1.96 * s_sd / np.sqrt(n)),
        # 참고용. 위 설명대로 이 지표는 이 개입의 효과를 제대로 못 잡는다.
        "paired_delta_mean": float(diff.mean()),
        "p_paired": float(st.wilcoxon(diff).pvalue) if not np.allclose(diff, 0) else 1.0,
    }


def plot_dose_response(res, arms, lams, base, out: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # 팔마다 고유 색, 선 모양으로 그룹을 구분한다.
    # (그룹별 단일 색으로 묶으면 같은 색 4개가 겹쳐 판별이 안 된다)
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    color = {a: palette[i % 10] for i, a in enumerate(arms)}
    style = {"paper": "-", "ours": "--", "ctrl": ":"}

    def series(a, key):
        return [res[f"{a}|{l}"][key] for l in lams]

    def draw(ax, a, y, **kw):
        g = group_of(a)
        ax.plot(lams, y, marker="o", ms=4, lw=1.8, color=color[a],
                ls=style[g], label=f"{a[0]} {a[2:]}", **kw)

    # (0,0) 주 지표
    ax = axes[0, 0]
    for a in arms:
        g = group_of(a)
        draw(ax, a, series(a, "slope_mean"))
    ax.axhline(base.mean(), color="k", ls=":", lw=1, label="baseline (lambda=0)")
    ax.set_xscale("symlog", linthresh=0.05); ax.set_xlim(0, max(lams) * 1.1)
    ax.set_xlabel("lambda_rel"); ax.set_ylabel("slope of predicted close")
    ax.set_title("Dose-response: does the intervention push the forecast upward?")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # (0,1) 방향 일치율 — 비율이라 이상치에 강하다
    ax = axes[0, 1]
    for a in arms:
        g = group_of(a)
        draw(ax, a, [v * 100 for v in series(a, "frac_positive")])
    ax.axhline(50, color="k", ls=":", lw=1)
    ax.set_xscale("symlog", linthresh=0.05); ax.set_xlim(0, max(lams) * 1.1)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("lambda_rel"); ax.set_ylabel("% of samples with positive slope")
    ax.set_title("Fraction steered in the intended direction\n"
                 "(solid = paper method, dashed = ours, dotted = control)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # (1,0) 기준선 대비 효과 + 95% CI
    ax = axes[1, 0]
    for a in arms:
        g = group_of(a)
        d = [outcome_stats(res[f"{a}|{l}"]["slopes"], base) for l in lams]
        m = np.array([x["slope_mean"] for x in d])
        e = np.array([x["slope_ci95"] for x in d])
        draw(ax, a, m)
        ax.fill_between(lams, m - e, m + e, color=color[a], alpha=0.08, lw=0)
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xscale("symlog", linthresh=0.05); ax.set_xlim(0, max(lams) * 1.1)
    ax.set_xlabel("lambda_rel"); ax.set_ylabel("slope with 95% CI")
    ax.set_title("Outcome slope with 95% CI (not a paired difference —\n"
                 "the intervention overwrites rather than adds)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # (1,1) 분산 붕괴 — 예상 밖 현상
    ax = axes[1, 1]
    for a in arms:
        g = group_of(a)
        draw(ax, a, series(a, "slope_std"))
    ax.set_xscale("symlog", linthresh=0.05); ax.set_yscale("log")
    ax.set_xlim(0, max(lams) * 1.1)
    ax.set_xlabel("lambda_rel"); ax.set_ylabel("std of slope across samples (log)")
    ax.set_title("Forecast variance collapses as the intervention strengthens")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_geometry(sub, tokens, w_last, s_med, s_norm, h_l2, lams, ood, out: Path,
                  layer: int, title: str) -> None:
    """개입 전/후 표현이 trend 방향으로 이동했는가.

    sub: [2(class), L, n_tokens, n, D] 실제 활성화 (Stage 4 가 Drive 에 남긴 것)
    """
    from sklearn.decomposition import PCA
    ti = 0                                    # tokens[0] == 마지막 토큰
    a = sub[0, layer, ti].astype(np.float32)  # base
    b = sub[1, layer, ti].astype(np.float32)  # trend
    S = s_med[layer].astype(np.float32)
    w = w_last[layer].astype(np.float32)
    w = w / (np.linalg.norm(w) + 1e-12)

    show = [l for l in lams if l in (0.05, 0.15, 0.25, 0.5)]
    lam_abs = {l: l * h_l2[layer] / max(s_norm[layer], 1e-12) for l in show}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (0,0) LDA 축 투영 — 개념을 실제로 담고 있는 축
    ax = axes[0, 0]
    pa, pb = a @ w, b @ w
    bins = np.histogram_bin_edges(np.concatenate([pa, pb]), bins=50)
    ax.hist(pa, bins=bins, alpha=0.55, color="tab:blue", label="base (real)")
    ax.hist(pb, bins=bins, alpha=0.55, color="tab:red", label="trend (real)")
    for l in show:
        ax.axvline(((a + lam_abs[l] * S) @ w).mean(), lw=2,
                   color=plt.cm.viridis(l / max(show)), label=f"steered mean, lambda={l}")
    ax.set_xlabel("projection onto LDA direction")
    ax.set_title(f"Layer {layer}, last token: does steering land where trend lives?")
    ax.legend(fontsize=8)

    # (0,1) 클래스 간격 대비 이동 비율
    ax = axes[0, 1]
    gap = pb.mean() - pa.mean()
    frac = [((a + (l * h_l2[layer] / max(s_norm[layer], 1e-12)) * S) @ w).mean()
            for l in lams]
    frac = [(f - pa.mean()) / gap for f in frac]
    ax.plot(lams, frac, "o-", color="tab:purple", lw=1.8)
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="reaches trend mean")
    ax.axhline(0.0, color="0.6", ls=":", lw=1)
    ax.set_xscale("symlog", linthresh=0.05); ax.set_xlim(0, max(lams) * 1.1)
    ax.set_xlabel("lambda_rel"); ax.set_ylabel("fraction of base->trend gap")
    ax.set_title("Displacement along the concept axis")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (1,0) 설계 Step 5 가 지정한 PCA. 개념 방향이 안 보인다는 점을 함께 보인다.
    ax = axes[1, 0]
    pca = PCA(n_components=2).fit(np.vstack([a, b]))
    ra, rb = pca.transform(a), pca.transform(b)
    ax.scatter(ra[:, 0], ra[:, 1], s=6, alpha=0.35, color="tab:blue", label="base")
    ax.scatter(rb[:, 0], rb[:, 1], s=6, alpha=0.35, color="tab:red", label="trend")
    for l in show:
        rs = pca.transform(a + lam_abs[l] * S)
        ax.scatter(rs[:, 0], rs[:, 1], s=8, alpha=0.5,
                   color=plt.cm.viridis(l / max(show)), label=f"steered {l}")
    ev = pca.explained_variance_ratio_
    energy = float(np.sum((pca.components_ @ w) ** 2))
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})"); ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.set_title(f"PCA view (holds only {energy:.2%} of the concept direction)")
    ax.legend(fontsize=8)

    # (1,1) Stage 4 OOD 진단과 실제 결과를 잇는다
    ax = axes[1, 1]
    ax.plot(lams, ood[layer], "o-", color="tab:orange", lw=1.8, label="off-manifold ratio")
    ax.axhline(2.0, color="k", ls="--", lw=1.2, label="threshold")
    ax.set_xscale("symlog", linthresh=0.05); ax.set_xlim(0, max(lams) * 1.1)
    ax.set_xlabel("lambda_rel"); ax.set_ylabel("NN distance ratio")
    ax.set_title("Stage 4 off-manifold diagnostic at this layer")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(args) -> int:
    t0 = time.time()
    cfg = dataclasses.replace(CFG, model=resolve_model(args.model, CFG.model.max_context))
    res_dir = paths.results_dir(cfg.tag) / f"{args.noise}_{args.model}"
    fig_dir = paths.ensure(paths.figs_dir(cfg.tag))
    steer_file = res_dir / "steer" / "results.json"

    if not steer_file.exists():
        print(f"Stage 5 결과가 없다: {steer_file}")
        return 1
    res = json.loads(steer_file.read_text())

    arms, lam_set = [], set()
    for k in res:
        a, l = k.rsplit("|", 1)
        if a not in arms:
            arms.append(a)
        lam_set.add(float(l))
    lams = sorted(lam_set)
    arms.sort()

    section("1. 입력")
    print(f"  {steer_file}")
    print(f"  팔 {len(arms)}개 x lambda {len(lams)}개 = {len(res)}개 조합")
    settings = {json.dumps(v.get("run_settings"), sort_keys=True) for v in res.values()}
    print(f"  실행 조건: {settings}")
    n_sl = {len(v["slopes"]) for v in res.values()}
    if len(n_sl) != 1:
        print(f"  [실패] 표본 수가 조합마다 다르다: {n_sl}. 비교 불가.")
        return 1
    print(f"  표본 수: {n_sl.pop()}")

    base = np.asarray(res[f"{arms[0]}|0.0"]["slopes"], dtype=np.float64)

    section("2. 팔별 요약")
    nz = [l for l in lams if l > 0]
    rows = []
    print("  상향/하향은 '결과 기울기가 양수인 표본 비율'의 이항검정으로 판정한다.")
    print(f"\n  {'arm':<28}{'상향최대λ':>10}{'일치율':>8}{'p':>9}"
          f"{'하향최소λ':>10}{'일치율':>8}{'p':>9}{'분산비':>8}{'coh위반':>9}")
    for a in arms:
        st_by_l = {l: outcome_stats(res[f"{a}|{l}"]["slopes"], base) for l in nz}
        l_up = max(nz, key=lambda l: st_by_l[l]["frac_positive"])
        l_dn = min(nz, key=lambda l: st_by_l[l]["frac_positive"])
        up, dn = st_by_l[l_up], st_by_l[l_dn]
        # 방향 판정: 유의하게 0.5 를 넘거나 밑도는가
        direction = "none"
        if up["frac_positive"] > 0.5 and up["p_sign"] < 0.01:
            direction = "up"
        if dn["frac_positive"] < 0.5 and dn["p_sign"] < 0.01:
            direction = "down" if direction == "none" else "both"
        min_var = min(st_by_l[l]["var_ratio"] for l in nz)
        viol = max(res[f"{a}|{l}"]["coherence_violation"] for l in lams)
        rows.append({"arm": a, "group": group_of(a), "direction": direction,
                     "lambda_up": l_up, "frac_positive_up": up["frac_positive"],
                     "p_sign_up": up["p_sign"], "slope_at_up": up["slope_mean"],
                     "lambda_down": l_dn, "frac_positive_down": dn["frac_positive"],
                     "p_sign_down": dn["p_sign"], "slope_at_down": dn["slope_mean"],
                     "min_var_ratio": min_var, "coherence_violation": viol,
                     "readout_shift_at_up": res[f"{a}|{l_up}"].get("readout_shift_sigma")})
        print(f"  {a:<28}{l_up:>10}{up['frac_positive']:>8.1%}{up['p_sign']:>9.1e}"
              f"{l_dn:>10}{dn['frac_positive']:>8.1%}{dn['p_sign']:>9.1e}"
              f"{min_var:>8.3f}{viol:>9.2%}")

    section("3. 판정")
    paper = [r for r in rows if r["group"] == "paper"]
    steers_up = [r for r in rows if r["direction"] in ("up", "both")]
    inverted = [r for r in rows if r["direction"] in ("down", "both")]
    inert = [r for r in rows if r["direction"] == "none"]
    best = max(steers_up, key=lambda r: (r["frac_positive_up"], -r["lambda_up"])) \
        if steers_up else None

    print(f"  논문 방법(A/B/C) 작동 : "
          f"{'예' if all(r['direction'] in ('up', 'both') for r in paper) else '아니오'}"
          f"  (최고 일치율 {max(r['frac_positive_up'] for r in paper):.1%})")
    if best:
        print(f"  가장 적은 lambda 로 100%: {best['arm']} (lambda={best['lambda_up']})")
    print(f"  상향 조정 성공        : {[r['arm'] for r in steers_up] or '없음'}")
    print(f"  역방향 조정 관측      : {[r['arm'] for r in inverted] or '없음'}")
    print(f"  효과 없음             : {[r['arm'] for r in inert] or '없음'}")
    max_viol = max(r["coherence_violation"] for r in rows)
    print(f"  최대 coherence 위반   : {max_viol:.3%} "
          f"-> {'붕괴 없음' if max_viol < 0.01 else '붕괴 관측'}")
    print(f"  최소 분산비 (붕괴 정도): {min(r['min_var_ratio'] for r in rows):.3f} "
          f"(1.0 = 기준선과 동일)")

    section("4. 그림")
    tag = f"{args.noise}_{args.model}"
    f1 = fig_dir / f"stage6_{tag}_dose_response.png"
    plot_dose_response(res, arms, lams, base, f1,
                       f"Concept steering dose-response [{args.noise}, {args.model}]")
    print(f"  {f1}")

    f2 = None
    need = [res_dir / n for n in ("pca_subset.npz", "steering.npz", "ldr.npz")]
    if all(p.exists() for p in need):
        sub = np.load(need[0])
        stv = np.load(need[1])
        ldr = np.load(need[2])
        summ = json.loads((res_dir / "stage3_summary.json").read_text())
        layer = int(summ["late_best_layer"])
        f2 = fig_dir / f"stage6_{tag}_geometry.png"
        plot_geometry(sub["act"], sub["tokens"], ldr["w_last"],
                      stv["S_median_late"], np.linalg.norm(stv["S_median_late"], axis=-1),
                      stv["h_norm_l2"], lams, stv["ood_ratio"], f2, layer,
                      f"Latent geometry of the intervention [{args.noise}, {args.model}]")
        print(f"  {f2}")
    else:
        missing = [p.name for p in need if not p.exists()]
        print(f"  [건너뜀] 기하 검증에 필요한 파일이 없다: {missing}")

    section("5. 저장")
    out = res_dir / "stage6_summary.json"
    out.write_text(json.dumps({"rows": rows, "lambdas": lams,
                               "max_coherence_violation": max_viol,
                               "best_arm": best["arm"] if best else None,
                               "steers_up": [r["arm"] for r in steers_up],
                               "inverted_arms": [r["arm"] for r in inverted],
                               "inert_arms": [r["arm"] for r in inert]},
                              indent=2, ensure_ascii=False))
    csv = res_dir / "stage6_summary.csv"
    cols = ["arm", "group", "direction", "lambda_up", "frac_positive_up", "p_sign_up",
            "slope_at_up", "lambda_down", "frac_positive_down", "p_sign_down",
            "slope_at_down", "min_var_ratio", "coherence_violation"]
    csv.write_text(",".join(cols) + "\n" + "\n".join(
        ",".join(str(r[c]) for c in cols) for r in rows))
    print(f"  {out}\n  {csv}")

    section("결과")
    print(f"  총 소요: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 6 — 개입 결과 평가")
    p.add_argument("--model", default="base", choices=["mini", "small", "base"])
    p.add_argument("--noise", default=CFG.data.noise_type, choices=["ou", "rw"])
    raise SystemExit(main(p.parse_args()))
