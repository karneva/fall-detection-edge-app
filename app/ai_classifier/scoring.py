"""
torch 모델 -> p_fall 점수 함수 (운영 v2 분류기와 오프라인 실험이 공유, torch 필요)

- softmax_p1: 창 하나 (C, L) -> softmax(model(x[None]))[0, 1]
- nn_batch_scorer: 묶음 X (B, C, L) -> p_fall (B,) (event_window.SlidingWindowVerifier의 score_fn)
"""
from typing import Callable, Optional

import numpy as np
import torch

__all__ = ["resolve_device", "softmax_p1", "nn_batch_scorer"]


def resolve_device(model, device=None):
    """device가 없으면 모델 파라미터가 있는 장치 (파라미터가 없으면 CPU)."""
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def softmax_p1(model, x: np.ndarray, device) -> float:
    """x (C, L) -> softmax(model(x[None]))[0, 1]"""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)
    return float(probs[0, 1].item())


def nn_batch_scorer(model, device=None, preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                    batched: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """
    v2 모델 -> 묶음 점수 함수 X (B, 92, L) -> p_fall (B,). 모델은 eval 모드로 바꾸고 장치로 옮깁니다.
    preprocess(예: 채널 표준화)는 창 하나 (92, L)씩 적용합니다.
    batched=False(기본, 운영 경로): 창마다 배치 1로 계산 (EventWindowVerifier의 첫 창 p와 비트 단위로 같음).
      CUDA에서는 cuDNN(TF32) 알고리즘이 배치 크기마다 달라 B개를 한 번에 돌리면 p가 1e-4 정도 달라짐.
    batched=True: B개를 한 번에 (빠름, 위 차이 허용 시).
    """
    dev = resolve_device(model, device)
    model = model.to(dev).eval()

    def score(X: np.ndarray) -> np.ndarray:
        xs = [np.asarray(preprocess(x), dtype=np.float32) if preprocess is not None else x for x in X]
        if not batched:
            return np.asarray([softmax_p1(model, x, dev) for x in xs], dtype=np.float64)
        t = torch.from_numpy(np.ascontiguousarray(np.stack(xs), dtype=np.float32)).to(dev)
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1)[:, 1]
        return probs.cpu().numpy().astype(np.float64)

    return score
