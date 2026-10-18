"""
"ae": 사건 구간 v2 창의 1D conv 오토인코더 재구성 오차 (torch 필요)

구조 (입력 = 채널별 표준화한 v2 창 (92, 60), 표준화 통계는 학습 정상 창)
  인코더: Conv1d(92->64, k5, s2) BN ReLU -> Conv1d(64->32, k5, s2) BN ReLU   => 병목 (32, 15)
  디코더: ConvT1d(32->64, k4, s2) BN ReLU -> ConvT1d(64->92, k4, s2) (선형 출력) => (92, 60)
  길이가 4의 배수가 아니면 출력을 입력 길이로 자르거나 0으로 채움
손실/점수 (표준화 공간의 제곱 오차, 창별 가중 평균)
  w[c, t] = rel/vel 채널(0..67)은 그 관절의 유효(mask_j 이고 골반 유효) 여부, 나머지 채널은 1
  score = sum(w * (x_hat - x)^2) / sum(w)    (보이지 않는 관절은 0으로 채워진 값이라 재구성 대상에서 뺌)
학습: AdamW lr 1e-3 (wd 1e-4), batch 128, 최대 60 epoch, 검증 정상 손실 patience 8 early stopping, seed 고정
"""
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from training.anomaly._loop import EVAL_BATCH, train_unsupervised
from training.anomaly.base import Detector, Standardizer, as_windows, no_input

__all__ = ["ConvAE", "AutoencoderDetector", "fit_ae", "joint_weights", "AE_RECIPE"]

AE_RECIPE = dict(optimizer="adamw", lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=60, patience=8)

_N_REL_VEL = 68
_MASK0 = 75


def joint_weights(X_raw: np.ndarray) -> np.ndarray:
    """표준화 전 v2 창 (N, 92, T) -> 손실 가중치 (N, 92, T) float32"""
    m = X_raw[:, _MASK0:_MASK0 + 17] > 0.5                        # (N, 17, T)
    valid = m & (m[:, 11:12] | m[:, 12:13])
    w = np.ones(X_raw.shape, dtype=np.float32)
    v2 = np.repeat(valid, 2, axis=1).astype(np.float32)           # (N, 34, T): x, y
    w[:, 0:34] = v2
    w[:, 34:_N_REL_VEL] = v2
    return w


class ConvAE(nn.Module):
    def __init__(self, in_ch: int = 92, h1: int = 64, h2: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, h1, 5, stride=2, padding=2), nn.BatchNorm1d(h1), nn.ReLU(inplace=True),
            nn.Conv1d(h1, h2, 5, stride=2, padding=2), nn.BatchNorm1d(h2), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(h2, h1, 4, stride=2, padding=1), nn.BatchNorm1d(h1), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(h1, in_ch, 4, stride=2, padding=1),
        )

    def forward(self, x):
        t = x.shape[-1]
        y = self.dec(self.enc(x))
        if y.shape[-1] > t:
            y = y[..., :t]
        elif y.shape[-1] < t:
            y = nn.functional.pad(y, (0, t - y.shape[-1]))
        return y


def _ae_loss(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    """xb = (B, 2 * 92, T): 앞 92 = 표준화 입력, 뒤 92 = 가중치. 반환 (B,) 창별 가중 MSE"""
    c = xb.shape[1] // 2
    x, w = xb[:, :c], xb[:, c:]
    err = (model(x) - x) ** 2
    return (w * err).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1.0)


class AutoencoderDetector(Detector):
    kind = "ae"

    def __init__(self, model: ConvAE, std: Standardizer, device: str = "cpu"):
        super().__init__()
        self.model = model.eval()
        self.std = std
        self.device = torch.device(device)
        self.model.to(self.device)
        self.n_params = int(sum(p.numel() for p in model.parameters()))

    def to(self, device) -> "AutoencoderDetector":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def _pack(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([self.std(X), joint_weights(X)], axis=1).astype(np.float32))

    def score(self, X) -> np.ndarray:
        X = as_windows(X)
        out = np.full(len(X), np.nan)
        ok = ~no_input(X)
        if not ok.any():
            return out
        xt = self._pack(X[ok])
        self.model.eval()
        res = []
        with torch.no_grad():
            for i in range(0, len(xt), EVAL_BATCH):
                res.append(_ae_loss(self.model, xt[i:i + EVAL_BATCH].to(self.device)).double().cpu())
        out[ok] = torch.cat(res).numpy()
        return out

    def state(self) -> Dict[str, Any]:
        return {"kind": self.kind, "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "mu": self.std.mu, "sd": self.std.sd, "info": self.info}

    @classmethod
    def from_state(cls, st: Dict[str, Any], device: str = "cpu") -> "AutoencoderDetector":
        model = ConvAE(in_ch=len(st["mu"]))
        model.load_state_dict(st["state_dict"])
        det = cls(model, Standardizer(st["mu"], st["sd"]), device=device)
        det.info = dict(st.get("info") or {})
        return det


def fit_ae(X_tr, X_va, seed: int = 0, device: str = "cpu", verbose: bool = False, **overrides) -> AutoencoderDetector:
    """정상 학습 창 (N, 92, T), 검증 정상 창 (M, 92, T)으로 학습. overrides: AE_RECIPE 키 덮어쓰기 (점검용)"""
    from training.experiments.trainer import set_seed

    rc = dict(AE_RECIPE, **overrides)
    t0 = time.time()
    X_tr, X_va = as_windows(X_tr), as_windows(X_va)
    std = Standardizer.fit(X_tr)
    set_seed(int(seed))
    model = ConvAE(in_ch=X_tr.shape[1])
    det = AutoencoderDetector(model, std, device="cpu")
    dev = torch.device(device)
    hist = train_unsupervised(model, det._pack(X_tr), det._pack(X_va), _ae_loss, seed=int(seed), device=dev,
                              verbose=verbose, **rc)
    det.to(dev)
    hist["recipe"] = rc
    hist["time_s"] = round(time.time() - t0, 2)
    hist["device"] = str(dev)
    det.info = hist
    return det
