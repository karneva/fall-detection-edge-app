"""
이상 점수 -> 의사 p_fall 보정 (numpy만 사용)

검증 정상 창 점수의 경험 분포로 점수를 [0, 1]로 바꿔 기존 verifier 자리(억제 임계값 0.4)에 넣습니다.
- 매듭(knot): x_q = 검증 점수의 q 분위수 (np.quantile, 선형) -> p_at_q (기본 0.8 -> 0.4)
  x_q 아래 검증 점수 x -> p_at_q * #(v <= x) / (#(v < x_q) + 1)            (0 < p < p_at_q)
  x_q 위 검증 점수 x   -> p_at_q + (1 - p_at_q) * #(x_q < v <= x) / (#(v > x_q) + 1)   (p_at_q < p < 1)
  양 끝: 최솟값 아래로 (x_q - 최솟값)만큼 가서 0, 최댓값 위로 (최댓값 - x_q)만큼 가서 1, 그 밖은 0/1로 자름
- 매듭 사이는 선형 보간 (np.interp): 단조 증가, 매듭에서 x_q 양쪽의 경험 CDF를 따름
- 판정 보장: 점수 <= x_q 이면 p <= p_at_q (억제), 점수 > x_q 이면 p > p_at_q (부동소수점 경계도 강제)
  즉 창은 점수가 검증 정상 창의 q 분위수 이하일 때만 억제됩니다.
- NaN 점수 -> NaN p (검증기에서 모든 창이 NaN이면 None = fail-open으로 바꿔야 함)

q = 0.8은 사전 등록 값입니다 (E가 ADL 알람의 약 80%를 억제했기 때문). test 결과로 고치지 않습니다.
"""
from typing import Any, Dict, Optional, Sequence

import numpy as np

__all__ = ["Calibrator", "calibrated_scorer", "DEFAULT_Q", "DEFAULT_P_AT_Q"]

DEFAULT_Q = 0.8
DEFAULT_P_AT_Q = 0.4


class Calibrator:
    """점수 -> 의사 p_fall. fit으로 만들고, params()/from_params로 JSON 저장/복원."""

    def __init__(self, xk: Sequence[float], pk: Sequence[float], threshold: float, q: float, p_at_q: float,
                 n_val: int = 0):
        self.xk = np.asarray(xk, dtype=np.float64)
        self.pk = np.asarray(pk, dtype=np.float64)
        self.threshold = float(threshold)
        self.q = float(q)
        self.p_at_q = float(p_at_q)
        self.n_val = int(n_val)
        if len(self.xk) < 2 or len(self.xk) != len(self.pk) or np.any(np.diff(self.xk) <= 0):
            raise ValueError("매듭 x는 2개 이상, 엄격히 증가해야 합니다")

    @classmethod
    def fit(cls, val_scores, q: float = DEFAULT_Q, p_at_q: float = DEFAULT_P_AT_Q) -> "Calibrator":
        """검증 정상 창 점수(NaN 제외)로 보정기를 만듭니다. 유효 점수가 없으면 ValueError."""
        if not (0.0 < q < 1.0 and 0.0 < p_at_q < 1.0):
            raise ValueError(f"q {q}, p_at_q {p_at_q}는 (0, 1) 안이어야 합니다")
        v = np.asarray(val_scores, dtype=np.float64).reshape(-1)
        v = np.sort(v[np.isfinite(v)])
        if len(v) == 0:
            raise ValueError("유효한 검증 점수가 없습니다")
        xq = float(np.quantile(v, q))
        lo, hi = v[v < xq], v[v > xq]
        ux_lo = np.unique(lo)
        ux_hi = np.unique(hi)
        # 아래쪽: #(v <= x) / (#(v < xq) + 1)
        p_lo = p_at_q * np.searchsorted(lo, ux_lo, side="right") / (len(lo) + 1)
        # 위쪽: #(xq < v <= x) / (#(v > xq) + 1)
        p_hi = p_at_q + (1.0 - p_at_q) * np.searchsorted(hi, ux_hi, side="right") / (len(hi) + 1)
        scale = max(abs(xq), 1.0) * 1e-6
        span_lo = max(xq - float(v[0]), scale)
        span_hi = max(float(v[-1]) - xq, scale)
        x0 = (float(ux_lo[0]) if len(ux_lo) else xq) - span_lo
        x1 = (float(ux_hi[-1]) if len(ux_hi) else xq) + span_hi
        xk = np.concatenate([[x0], ux_lo, [xq], ux_hi, [x1]])
        pk = np.concatenate([[0.0], p_lo, [p_at_q], p_hi, [1.0]])
        return cls(xk, pk, xq, q, p_at_q, n_val=len(v))

    def __call__(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        p = np.interp(s, self.xk, self.pk)
        # 판정 경계 강제: 점수 <= x_q -> p <= p_at_q, 점수 > x_q -> p > p_at_q
        below = s <= self.threshold
        p = np.where(below, np.minimum(p, self.p_at_q), np.maximum(p, np.nextafter(self.p_at_q, 1.0)))
        p = np.clip(p, 0.0, 1.0)
        return np.where(np.isnan(s), np.nan, p)

    transform = __call__

    def params(self) -> Dict[str, Any]:
        return {"q": self.q, "p_at_q": self.p_at_q, "threshold": self.threshold, "n_val": self.n_val,
                "xk": [float(x) for x in self.xk], "pk": [float(x) for x in self.pk]}

    @classmethod
    def from_params(cls, d: Dict[str, Any]) -> "Calibrator":
        return cls(d["xk"], d["pk"], d["threshold"], d["q"], d["p_at_q"], n_val=d.get("n_val", 0))

    def __repr__(self) -> str:
        return (f"Calibrator(q={self.q}, p_at_q={self.p_at_q}, threshold={self.threshold:.6g}, "
                f"n_val={self.n_val}, knots={len(self.xk)})")


def calibrated_scorer(detector, calibrator: Calibrator, nan_to: Optional[float] = None):
    """
    Detector + Calibrator -> X (B, 92, L) -> p (B,) 함수 (SlidingWindowVerifier의 score_fn 형태).
    nan_to를 주면 NaN p를 그 값으로 바꿉니다 (기본은 NaN 유지: 검증기 쪽에서 None 처리).
    """
    def fn(X):
        p = calibrator(detector.score(X))
        if nan_to is not None:
            p = np.where(np.isnan(p), float(nan_to), p)
        return p
    return fn
