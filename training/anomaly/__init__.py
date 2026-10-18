"""training.anomaly - 정상 행동만으로 학습하는 포즈 시계열 이상 탐지 (단계 G, 오프라인 전용)

낙상 라벨 없이 정상 창(ADL, 낙상 이전 구간, ADL 진입 하드 네거티브, Figshare ADL)만으로 학습하고,
정상에서 벗어난 정도(이상 점수)를 검증 정상 창 분위수로 의사 p_fall로 바꿔 기존 verifier 자리에 넣습니다.

- base.py        : Detector 인터페이스, v2 창 표준화
- mahalanobis.py : "md" 손설계 특징(E0) + LedoitWolf 공분산 Mahalanobis 거리
- autoencoder.py : "ae" v2 창 1D conv 오토인코더, 관절 마스크 가중 재구성 오차
- segments.py    : v2 창 -> STG-NF 입력 포즈 세그먼트 (2, 24, 17)
- stgnf.py       : "nf" STG-NF (Hirschorn & Avidan, ICCV 2023) 직접 구현, 세그먼트 NLL의 창 내 최댓값
- calibrate.py   : Calibrator, 점수 -> 의사 p_fall (검증 정상 q=0.8 분위수 -> 0.4 = 억제 임계값)

공통 API
  fit_detector(kind, X_tr, X_va, seed, device) -> Detector      (X: (N, 92, 60) 정상 창)
  Detector.score(X) -> (B,) float64 (NaN = 쓸 입력 없음), Detector.state(), Detector.n_params, Detector.info
  load_detector(kind, state, device="cpu") -> Detector
  Calibrator.fit(val_scores, q=0.8, p_at_q=0.4) -> Calibrator, Calibrator(scores) -> p

패키지 import만으로는 numpy/torch를 불러오지 않습니다 (표준 라이브러리 테스트 환경 호환).
"""
from typing import Any, Dict

__all__ = ["KINDS", "fit_detector", "load_detector", "Calibrator", "Detector", "calibrated_scorer"]

KINDS = ("md", "ae", "nf")


def fit_detector(kind: str, X_tr, X_va, seed: int = 0, device: str = "cpu", **kwargs):
    """
    정상 창으로 이상 탐지기를 학습합니다. kind: "md" | "ae" | "nf".
    X_tr: 학습(fit 행) 정상 창 (N, 92, T), X_va: 검증 정상 창 (early stopping, md는 기록만).
    device: "cpu" | "cuda" | "auto". kwargs는 각 fit 함수로 전달 (점검용 레시피 덮어쓰기).
    """
    if kind == "md":
        from training.anomaly.mahalanobis import fit_md
        return fit_md(X_tr, X_va, seed=seed, device=device, **kwargs)
    dev = _device(device)
    if kind == "ae":
        from training.anomaly.autoencoder import fit_ae
        return fit_ae(X_tr, X_va, seed=seed, device=dev, **kwargs)
    if kind == "nf":
        from training.anomaly.stgnf import fit_nf
        return fit_nf(X_tr, X_va, seed=seed, device=dev, **kwargs)
    raise ValueError(f"알 수 없는 탐지기 종류: {kind} (md, ae, nf)")


def load_detector(kind: str, state: Dict[str, Any], device: str = "cpu"):
    """Detector.state()로 저장한 dict -> Detector"""
    if state.get("kind", kind) != kind:
        raise ValueError(f"state kind {state.get('kind')} != {kind}")
    if kind == "md":
        from training.anomaly.mahalanobis import MahalanobisDetector
        return MahalanobisDetector.from_state(state)
    if kind == "ae":
        from training.anomaly.autoencoder import AutoencoderDetector
        return AutoencoderDetector.from_state(state, device=_device(device))
    if kind == "nf":
        from training.anomaly.stgnf import STGNFDetector
        return STGNFDetector.from_state(state, device=_device(device))
    raise ValueError(f"알 수 없는 탐지기 종류: {kind} (md, ae, nf)")


def _device(device) -> str:
    if device is None or device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def __getattr__(name: str):
    # numpy가 필요한 클래스는 처음 쓸 때 import
    if name in ("Calibrator", "calibrated_scorer"):
        from training.anomaly import calibrate
        return getattr(calibrate, name)
    if name == "Detector":
        from training.anomaly.base import Detector
        return Detector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
