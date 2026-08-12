"""Stage 2 — Kronos residual stream 활성화 추출·저장.

저장 형식
    <scratch>/activations/<tag>/<noise>/<model>/<class>/layer{i:02d}.npy
    각 파일 shape [N, T, D], dtype float16

레이어별로 파일을 쪼개는 이유
    Kronos-base 기준 전체 텐서는 [12, 2048, 512, 832] fp16 = 39 GB 다.
    Colab 시스템 RAM 이 12.7 GB 이므로 통째로 못 올린다. 레이어 하나는
    1.6 GB/클래스라 Stage 3 에서 한 레이어씩 스트리밍하면 여유 있게 처리된다.

    쓰기는 np.lib.format.open_memmap 으로 배치마다 조금씩 흘려보낸다.
    (배치 결과를 리스트에 모았다가 마지막에 concat 하면 RAM 이 터진다.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from kexp import kronos_loader as kl
from kexp.hooks import ActivationRecorder


def layer_path(root: Path, layer: int) -> Path:
    return root / f"layer{layer:02d}.npy"


def open_writers(root: Path, n_layers: int, n: int, T: int, D: int, dtype: str):
    root.mkdir(parents=True, exist_ok=True)
    return [
        np.lib.format.open_memmap(layer_path(root, i), mode="w+",
                                  dtype=np.dtype(dtype), shape=(n, T, D))
        for i in range(n_layers)
    ]


def extract_class(tokenizer, model, x_raw: np.ndarray, stamp_np: np.ndarray,
                  out_root: Path, cfg, device: torch.device, batch_size: int,
                  progress_every: int = 20) -> dict:
    """한 클래스의 활성화를 추출해 레이어별 memmap 에 쓴다.

    Args:
        x_raw: [N, T, 6] 원시 OHLCVA
        stamp_np: [T, 5] 시간 특징. 모든 샘플이 공유한다.
    """
    n, T, _ = x_raw.shape
    n_layers = len(model.transformer)
    D = model.d_model

    writers = open_writers(out_root, n_layers, n, T, D, cfg.probe.store_dtype)
    stamp = torch.from_numpy(stamp_np).to(device).unsqueeze(0)

    n_nan = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = x_raw[start:end]
        x_norm, _, _ = kl.normalize_batch(xb, cfg.model.clip)
        s1, s2 = kl.to_tokens(tokenizer, x_norm, device)
        stamp_b = stamp.expand(end - start, -1, -1)

        with ActivationRecorder(model, dtype=getattr(torch, cfg.probe.store_dtype)) as rec:
            with torch.no_grad():
                model.decode_s1(s1, s2, stamp_b)
            acts = rec.stacked()          # [L, B, T, D] fp16 on CPU

        arr = acts.numpy()
        n_nan += int(np.isnan(arr).sum())
        for i in range(n_layers):
            writers[i][start:end] = arr[i]

        if progress_every and (start // batch_size) % progress_every == 0:
            print(f"    {end}/{n}", flush=True)

    for w in writers:
        w.flush()
    del writers

    return {"n": n, "T": T, "D": D, "n_layers": n_layers, "n_nan": n_nan}


def load_layer(root: Path, layer: int, mmap: bool = True) -> np.ndarray:
    """Stage 3/4 에서 한 레이어만 읽는다."""
    return np.load(layer_path(root, layer), mmap_mode="r" if mmap else None)


def read_meta(root: Path) -> dict:
    return json.loads((root / "meta.json").read_text())
