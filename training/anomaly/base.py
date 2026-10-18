"""
이상 탐지기 공통 인터페이스와 v2 창 표준화 (numpy만 사용)

Detector
  kind      : "md" | "ae" | "nf"
  n_params  : 학습된 수 (가중치 또는 통계량 개수)
  info      : 학습 기록 (손실 곡선, best epoch, 시간, 창/세그먼트 수 등, JSON 직렬화 가능)
  score(X)  : (B, 92, L) 또는 (92, L) -> (B,) float64 이상 점수 (클수록 이상), 쓸 수 있는 입력이 없으면 NaN
  state()   : torch.save/pickle로 저장할 dict (numpy 배열, CPU state_dict). load_detector(kind, state)로 복원
"""
from typing import Any, Dict

import numpy as np

__all__ = ["Detector", "Standardizer", "as_windows", "no_input"]

_MASK0 = 75   # features.V2_CHANNELS.index("mask_nose")


def as_windows(X) -> np.ndarray:
    """(B, 92, L) 또는 (92, L) -> (B, 92, L) float32"""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[None]
    if X.ndim != 3:
        raise ValueError(f"X shape {X.shape}: (B, 92, L)이어야 합니다")
    return X


def no_input(X: np.ndarray) -> np.ndarray:
    """(B,) bool: 보이는 관절이 하나도 없거나 값이 유한하지 않은 창 (점수 NaN)"""
    if X.shape[-1] == 0:
        return np.ones(len(X), dtype=bool)
    vis = X[:, _MASK0:_MASK0 + 17, :].max(axis=(1, 2)) > 0.5
    finite = np.isfinite(X).all(axis=(1, 2))
    return ~(vis & finite)


class Standardizer:
    """채널별 표준화 (학습 창의 N, T 축 평균/표준편차, sd <= 1e-6이면 1). run_progression._Standardizer와 같은 규칙."""

    def __init__(self, mu, sd):
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sd = np.asarray(sd, dtype=np.float32)

    @classmethod
    def fit(cls, X) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(axis=(0, 2))
        sd = X.std(axis=(0, 2))
        return cls(mu, np.where(sd > 1e-6, sd, 1.0))

    def __call__(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mu[None, :, None]) / self.sd[None, :, None]


class Detector:
    kind = ""

    def __init__(self):
        self.n_params = 0
        self.info: Dict[str, Any] = {}

    def score(self, X) -> np.ndarray:  # pragma: no cover - 하위 클래스
        raise NotImplementedError

    def state(self) -> Dict[str, Any]:  # pragma: no cover - 하위 클래스
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kind={self.kind!r}, n_params={self.n_params})"
