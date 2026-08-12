"""Kronos residual stream 후킹 — 추출(Stage 2)과 개입(Stage 5) 공용.

후킹 지점
    model.transformer[i]  (TransformerBlock, i = 0..L-1)
TransformerBlock.forward 는 attention + FFN 을 residual 로 더한 값을 돌려주므로,
그 출력이 명세가 말하는 "FFN 이후 residual stream" h_i^(t) 그대로다.
모델 내부를 고칠 필요가 없다.

개입 시 중요한 성질
    auto_regressive_inference 는 매 디코딩 스텝마다 model.decode_s1 을 다시
    호출한다 (Kronos/model/kronos.py 의 생성 루프). 따라서 forward hook 을 걸어
    두면 아래 두 가지가 자동으로 동시에 충족된다.
      (a) 입력 컨텍스트 전체 토큰 위치에 대한 개입
      (b) 자기회귀 생성 매 스텝의 신규 토큰에 대한 개입
    decode_s2 가 쓰는 context 도 이미 개입된 값이라 일관성이 유지된다.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


class ActivationRecorder:
    """레이어별 residual stream 을 모은다.

    한 번의 forward 마다 layer -> [B, T, D] 텐서 하나가 쌓인다.
    AR 생성처럼 forward 가 여러 번 일어나는 경우에는 마지막 호출만 남기고
    싶을 수 있으므로 `keep` 로 정책을 고른다.
    """

    def __init__(self, model, layers=None, keep: str = "all", to_cpu: bool = True,
                 dtype: torch.dtype | None = torch.float16):
        self.model = model
        self.n_layers = len(model.transformer)
        self.layers = list(range(self.n_layers)) if layers is None else list(layers)
        self.keep = keep
        self.to_cpu = to_cpu
        self.dtype = dtype
        self.acts: dict[int, list[torch.Tensor]] = {i: [] for i in self.layers}
        self._handles: list = []

    def _hook(self, layer_idx: int):
        def fn(module, inputs, output):
            t = output.detach()
            if self.dtype is not None:
                t = t.to(self.dtype)
            if self.to_cpu:
                t = t.cpu()
            if self.keep == "last" and self.acts[layer_idx]:
                self.acts[layer_idx][-1] = t
            else:
                self.acts[layer_idx].append(t)
        return fn

    def __enter__(self):
        for i in self.layers:
            self._handles.append(self.model.transformer[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def stacked(self) -> torch.Tensor:
        """[n_layers, B, T, D] 로 합친다 (forward 를 한 번만 돌린 경우)."""
        out = []
        for i in self.layers:
            chunks = self.acts[i]
            if len(chunks) != 1:
                raise RuntimeError(
                    f"layer {i}: forward 가 {len(chunks)}회 실행되었다. "
                    "stacked() 는 단일 forward 전용이다."
                )
            out.append(chunks[0])
        return torch.stack(out, dim=0)

    def clear(self) -> None:
        for i in self.layers:
            self.acts[i].clear()


@contextmanager
def steering(model, steer_vectors: dict[int, torch.Tensor], scale: float = 1.0):
    """h_i^(t) <- h_i^(t) + scale * S_i 를 모든 시퀀스 위치에 더한다 (명세 7절).

    Args:
        steer_vectors: {layer_idx: [D] 텐서}. 개입할 레이어만 담는다.
        scale: lambda.

    현재 윈도우의 전 위치에 브로드캐스팅으로 더하므로, 컨텍스트 토큰과
    AR 로 새로 생성되는 토큰이 구분 없이 동일하게 조정된다.
    """
    handles = []

    def make(vec: torch.Tensor):
        def fn(module, inputs, output):
            return output + scale * vec.to(device=output.device, dtype=output.dtype)
        return fn

    try:
        for layer_idx, vec in steer_vectors.items():
            if vec.ndim != 1:
                raise ValueError(f"steering vector 는 [D] 여야 한다: layer {layer_idx}, shape {tuple(vec.shape)}")
            handles.append(model.transformer[layer_idx].register_forward_hook(make(vec)))
        yield
    finally:
        for h in handles:
            h.remove()
