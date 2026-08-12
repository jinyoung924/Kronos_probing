"""Stage 2 — Kronos residual stream 활성화 추출.

Stage 1 의 합성 데이터셋을 Kronos 에 통과시키고, 각 TransformerBlock 출력
h_i^(t) 를 레이어별 파일로 저장한다. AR 생성이 아니라 컨텍스트 1회 forward
(decode_s1) 만 돌린다 — 프로빙에 필요한 것은 입력 문맥의 표현이다.

실행:
    python experiment/code/stages/stage2_activations.py                 # base 모델, OU
    python experiment/code/stages/stage2_activations.py --smoke         # 8샘플 점검
    python experiment/code/stages/stage2_activations.py --model small
    python experiment/code/stages/stage2_activations.py --noise rw
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

from kexp import activations as A  # noqa: E402
from kexp import kronos_loader as kl  # noqa: E402
from kexp import paths  # noqa: E402
from kexp.config import CFG, resolve_model  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main(args) -> int:
    t0 = time.time()
    cfg = dataclasses.replace(CFG, model=resolve_model(args.model, CFG.model.max_context))
    data_dir = paths.data_dir(cfg.tag, args.noise)
    out_base = paths.activations_dir(cfg.tag, f"{args.noise}/{args.model}")

    if not (data_dir / "meta.json").exists():
        print(f"Stage 1 산출물이 없다: {data_dir}. 먼저 stage1 을 실행할 것.")
        return 1

    section("1. 설정")
    print(f"  모델      : {args.model} ({cfg.model.model_id})")
    print(f"  데이터셋  : {args.noise}  <- {data_dir}")
    print(f"  출력      : {out_base}")
    print(f"  배치 크기 : {args.batch_size}")

    ts = pd.DatetimeIndex(np.load(data_dir / "timestamps.npy"))
    stamp_np = kl.make_stamps(ts)
    print(f"  타임스탬프: {ts[0]} .. {ts[-1]}  -> stamp {stamp_np.shape}")

    section("2. 모델 로드")
    tokenizer, model, device = kl.load_kronos(cfg.model)
    spec = kl.describe(tokenizer, model)
    print(f"  device={device}  L={spec['n_layers']}  d_model={spec['d_model']}")

    section("3. 추출")
    stats = {}
    for cls in ("base", "trend"):
        x = np.load(data_dir / f"{cls}.npz")["x"]
        if args.smoke:
            x = x[: args.smoke_n]
        out_root = out_base / cls
        done = (out_root / "meta.json").exists()
        if done and not args.force:
            print(f"  [{cls}] 이미 존재한다. 다시 만들려면 --force")
            stats[cls] = A.read_meta(out_root)
            continue

        print(f"  [{cls}] {x.shape} 추출 시작")
        t1 = time.time()
        st = A.extract_class(tokenizer, model, x, stamp_np, out_root, cfg, device,
                             args.batch_size)
        st["seconds"] = round(time.time() - t1, 1)
        st["model"] = args.model
        st["noise"] = args.noise
        st["config_hash"] = cfg.hash()
        (out_root / "meta.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
        stats[cls] = st
        gb = st["n_layers"] * st["n"] * st["T"] * st["D"] * 2 / 1024**3
        print(f"  [{cls}] 완료 {st['seconds']}s, {gb:.2f} GB, NaN {st['n_nan']}개")

    section("4. 검증")
    ok = True
    for cls, st in stats.items():
        if st["n_nan"]:
            print(f"  [실패] {cls} 에 NaN {st['n_nan']}개")
            ok = False
    shapes = {c: (s["n"], s["T"], s["D"], s["n_layers"]) for c, s in stats.items()}
    print(f"  shape: {shapes}")
    if len(set(shapes.values())) != 1:
        print("  [실패] 두 클래스의 shape 이 다르다.")
        ok = False

    # 저장된 값이 실제로 다시 읽히는지, 그리고 두 클래스가 실제로 다른지 확인.
    # (전처리 버그로 두 클래스가 같은 입력이 되어버리는 실수를 여기서 잡는다.)
    n_l = stats["base"]["n_layers"]
    for layer in (0, n_l // 2, n_l - 1):
        a = A.load_layer(out_base / "base", layer)
        b = A.load_layer(out_base / "trend", layer)
        k = min(64, a.shape[0])
        diff = np.abs(a[:k].astype(np.float32) - b[:k].astype(np.float32))
        last = np.abs(a[:k, -1].astype(np.float32) - b[:k, -1].astype(np.float32))
        print(f"  layer {layer:2d}: |base-trend| 평균 {diff.mean():.5f}, "
              f"마지막 토큰 평균 {last.mean():.5f}, |h| 평균 {np.abs(a[:k]).mean():.3f}")
        if diff.mean() == 0:
            print(f"  [실패] layer {layer} 에서 두 클래스의 활성화가 완전히 같다.")
            ok = False

    # 결정성: 같은 배치를 다시 추출해 bitwise 동일한지
    x0 = np.load(data_dir / "base.npz")["x"][:4]
    reps = []
    for _ in range(2):
        xn, _, _ = kl.normalize_batch(x0, cfg.model.clip)
        s1, s2 = kl.to_tokens(tokenizer, xn, device)
        from kexp.hooks import ActivationRecorder
        with ActivationRecorder(model) as rec:
            with torch.no_grad():
                model.decode_s1(s1, s2, torch.from_numpy(stamp_np).to(device).expand(4, -1, -1))
            reps.append(rec.stacked())
    det = bool(torch.equal(reps[0], reps[1]))
    print(f"  결정성: {det}")
    ok = ok and det

    section("결과")
    print(f"  판정: {'PASS' if ok else 'FAIL'}")
    print(f"  총 소요: {time.time() - t0:.1f}s")
    print(f"  활성화 위치: {out_base}")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 2 — 활성화 추출")
    p.add_argument("--model", default="base", choices=["mini", "small", "base"])
    p.add_argument("--noise", default=CFG.data.noise_type, choices=["ou", "rw"])
    p.add_argument("--batch-size", type=int, default=CFG.probe.extract_batch_size)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true", help="소수 샘플로 경로만 점검")
    p.add_argument("--smoke-n", type=int, default=8)
    raise SystemExit(main(p.parse_args()))
