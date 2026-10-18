"""
"md": 손설계 특징(E0와 같은 windows.handcrafted)의 Mahalanobis 거리 (numpy, sklearn)

- 학습 정상 창의 특징을 표준화 (평균/표준편차, sd <= 1e-6이면 1)
- 공분산: sklearn LedoitWolf 수축 추정 (특징 24개, 상관이 강해 표본 공분산은 조건수가 나쁨)
- 점수 = 제곱 Mahalanobis 거리 (z - m)^T P (z - m), P = 수축 공분산의 역행렬
결정적이라 seed는 쓰지 않습니다 (인터페이스 통일용).
"""
import time
from typing import Any, Dict

import numpy as np

from training.anomaly.base import Detector, as_windows, no_input

__all__ = ["MahalanobisDetector", "fit_md"]


class MahalanobisDetector(Detector):
    kind = "md"

    def __init__(self, mu, sd, location, precision, shrinkage: float = float("nan")):
        super().__init__()
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.asarray(sd, dtype=np.float64)
        self.location = np.asarray(location, dtype=np.float64)
        self.precision = np.asarray(precision, dtype=np.float64)
        self.shrinkage = float(shrinkage)
        d = len(self.mu)
        self.n_params = int(3 * d + d * (d + 1) // 2)   # mu, sd, location + 대칭 precision

    def features(self, X) -> np.ndarray:
        from training.experiments.windows import handcrafted
        return (handcrafted(X).astype(np.float64) - self.mu) / self.sd

    def score(self, X) -> np.ndarray:
        X = as_windows(X)
        out = np.full(len(X), np.nan)
        ok = ~no_input(X)
        if ok.any():
            z = self.features(X[ok]) - self.location
            out[ok] = np.einsum("ni,ij,nj->n", z, self.precision, z)
        return out

    def state(self) -> Dict[str, Any]:
        return {"kind": self.kind, "mu": self.mu, "sd": self.sd, "location": self.location,
                "precision": self.precision, "shrinkage": self.shrinkage, "info": self.info}

    @classmethod
    def from_state(cls, st: Dict[str, Any]) -> "MahalanobisDetector":
        det = cls(st["mu"], st["sd"], st["location"], st["precision"], st.get("shrinkage", float("nan")))
        det.info = dict(st.get("info") or {})
        return det


def fit_md(X_tr, X_va=None, seed: int = 0, device: str = "cpu") -> MahalanobisDetector:
    """정상 학습 창 (N, 92, T)으로 적합. 검증 창이 있으면 평균 점수를 info에 기록 (선택에는 쓰지 않음)."""
    from sklearn.covariance import LedoitWolf

    from training.experiments.windows import handcrafted

    t0 = time.time()
    X_tr = as_windows(X_tr)
    F = handcrafted(X_tr).astype(np.float64)
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Z = (F - mu) / sd
    lw = LedoitWolf().fit(Z)
    det = MahalanobisDetector(mu, sd, lw.location_, lw.precision_, lw.shrinkage_)
    tr_s = det.score(X_tr)
    info: Dict[str, Any] = {"n_train": int(len(X_tr)), "n_features": int(F.shape[1]),
                            "shrinkage": float(lw.shrinkage_), "train_loss": float(np.nanmean(tr_s))}
    if X_va is not None and len(X_va):
        info["n_val"] = int(len(X_va))
        info["val_loss"] = float(np.nanmean(det.score(X_va)))
    info["time_s"] = round(time.time() - t0, 2)
    det.info = info
    return det
