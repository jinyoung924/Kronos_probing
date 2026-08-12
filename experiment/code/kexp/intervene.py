"""Stage 5 — 개념 조정 개입 (명세 7절, 논문 3.3절).

논문의 개입 규칙
    h_i <- h_i + lambda * S_i,  S_i in R^{N x D}
    즉 토큰 위치마다 다른 벡터를 더하는 **행렬** 개입이며, 여러 토큰과 여러
    레이어에 동시에 적용하는 편이 단일 토큰 개입보다 효과적이었다고 보고한다.

Kronos 고유의 문제
    논문의 대상(MOMENT 인코더, Chronos 인코더)은 토큰 수가 고정이라 [N, D]
    행렬을 그대로 얹으면 된다. Kronos 는 디코더-only 라 AR 생성 중 컨텍스트가
    롤링하며 위치가 밀린다. max_context == 초기 컨텍스트 길이인 경우
    (base/small 의 T=512) 첫 생성 스텝부터 윈도우가 한 칸씩 미끄러진다.

    윈도우 위치 j 에 대응하는 원래 컨텍스트 인덱스는

        orig = initial_seq_len + step - seq_len + j

    이며, orig >= T 인 위치(새로 생성된 토큰)에는 마지막 행 S_i[T-1] 을 쓴다.
    이것이 명세 7절의 "신규 토큰에도 동일한 스케일의 lambda*S_i 를 가산" 을
    행렬 형태에서 구현한 것이다.

형태 두 가지
    matrix : S_i^(t) 를 위치에 맞춰 더한다 (논문 방법)
    vector : 레이어당 [D] 벡터 하나를 전 위치에 더한다 (명세 7절의 축약형이자
             AR 신규 토큰 처리에 필요한 형태)
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


class Steerer:
    """AR 디코딩 스텝을 추적하며 위치 정합을 맞추는 개입 훅.

    transformer 블록 훅은 decode_s1 에서만 발화한다 (decode_s2 는 dep_layer 와
    head 만 쓴다). 따라서 발화 횟수가 곧 디코딩 스텝 수다.
    """

    def __init__(self, model, layers, payload, initial_seq_len: int,
                 form: str = "matrix", scope: str = "all"):
        """
        Args:
            payload: form="matrix" 이면 {layer: [T, D]}, "vector" 이면 {layer: [D]}
                     이미 lambda 가 곱해진 상태로 받는다.
            initial_seq_len: 개입 시작 시점의 컨텍스트 길이 T
            scope: "all" 모든 위치 / "single" 마지막 위치만 (논문 절제 실험 2)
        """
        self.model = model
        self.layers = sorted(layers)
        self.payload = payload
        self.initial_seq_len = initial_seq_len
        self.form = form
        self.scope = scope
        self.step = 0
        self._first = self.layers[0]
        self._handles = []

    def _hook(self, layer_idx: int):
        def fn(module, inputs, output):
            # 가장 얕은 개입 레이어가 발화할 때마다 새 디코딩 스텝으로 센다
            if layer_idx == self._first:
                self.step += 1
            p = self.payload[layer_idx].to(device=output.device, dtype=output.dtype)
            s = output.shape[1]

            if self.form == "vector":
                delta = p.view(1, 1, -1).expand(1, s, -1)
            else:
                T = p.shape[0]
                start = self.initial_seq_len + (self.step - 1) - s
                idx = torch.arange(s, device=output.device) + start
                idx = idx.clamp(0, T - 1)      # 신규 토큰은 마지막 행을 쓴다
                delta = p.index_select(0, idx).unsqueeze(0)

            if self.scope == "single":
                mask = torch.zeros(1, s, 1, device=output.device, dtype=output.dtype)
                mask[:, -1] = 1.0
                delta = delta * mask
            return output + delta
        return fn

    def __enter__(self):
        self.step = 0
        for i in self.layers:
            self._handles.append(self.model.transformer[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


@contextmanager
def no_steering():
    """lambda=0 대조군. 훅을 아예 걸지 않아 기준 경로를 그대로 탄다."""
    yield None


def build_payload(S_vec, S_mat, layers, lam_by_layer, form: str) -> dict:
    """레이어별 lambda 를 곱한 개입 payload 를 만든다.

    Args:
        S_vec: [L, D] numpy, S_mat: [L, T, D] numpy (form 에 따라 하나만 쓰임)
        lam_by_layer: [L] 레이어별 절대 lambda
    """
    out = {}
    for i in layers:
        src = S_vec[i] if form == "vector" else S_mat[i]
        out[i] = torch.from_numpy(src.astype("float32")) * float(lam_by_layer[i])
    return out
