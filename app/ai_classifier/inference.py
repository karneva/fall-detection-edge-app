import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.ai_classifier.features import normalize_sequence_v1
from app.ai_classifier.model_meta import (
    V2_FEATURE_VERSION, V2_HZ, V2_IN_CHANNELS, check_compatible, legacy_v1_meta, load_meta,
)
from app.ai_classifier.weights_resolver import resolve_weights_path, sha256_8

logger = logging.getLogger(__name__)


def _numpy_safe_globals():
    """numpy 배열(mu/sd)이 든 체크포인트를 weights_only=True로 읽을 때 허용할 numpy 타입."""
    mods = [getattr(np, "_core", None), getattr(np, "core", None)]
    recon = None
    for m in mods:
        ma = getattr(m, "multiarray", None) if m is not None else None
        recon = getattr(ma, "_reconstruct", None) if ma is not None else None
        if recon is not None:
            break
    out = [np.ndarray, np.dtype, type(np.dtype(np.float32)), type(np.dtype(np.float64))]
    if recon is not None:
        out.append(recon)
    return out


def _load_checkpoint(torch, path: str, device):
    """
    torch.load(weights_only=True). numpy 배열이 든 v2 체크포인트도 numpy 타입만 허용해서 읽습니다
    (임의 코드 실행이 가능한 weights_only=False는 쓰지 않음). weights_only 인자가 없는 구버전 torch는 기본 로드.
    """
    try:
        safe = getattr(getattr(torch, "serialization", None), "safe_globals", None)
        if safe is not None:
            with safe(_numpy_safe_globals()):
                return torch.load(path, map_location=device, weights_only=True)
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _as_vec(v, n: int, name: str) -> np.ndarray:
    if v is None:
        raise ValueError(f"checkpoint has no {name}")
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} shape {arr.shape} (need ({n},), finite)")
    return arr


class FallClassifier:
    """
    2차 검증 분류기 래퍼. 가중치 옆 "<weights>.meta.json"의 feature_version으로 형식을 고릅니다.

    - v1 (legacy, 메타가 없으면 v1): Fall1DCNN, 엔진의 최근 30프레임 kp_seq -> predict_proba(seq)
    - v2 (배포): Fall1DCNNGAP, 사건 구간 v2 특징 4초 창 -> score_windows(X)
      엔진에 묶인 검증기(make_engine_verifier)가 ABNORMAL 진입부터 판정 시점까지 창들의 최대 p_fall을 냄
      (app/ai_classifier/event_window.SlidingWindowVerifier, 오프라인 E*max와 같은 구현).
      v2 가중치는 {"arch", "state_dict", "mu", "sd"} 체크포인트 (mu/sd = 채널 표준화, 창마다 적용).

    가중치가 없거나 로드에 실패하거나 메타가 이 런타임과 맞지 않으면 enabled=False가 되고
    모든 예측이 None을 반환합니다 (초기화되지 않은 모델로 알람을 억제하지 않도록 fail-open).
    """

    # v2 필드 기본값 (__new__로 만든 객체도 v1처럼 동작하도록 클래스 속성으로도 둠)
    feature_version: Optional[str] = None
    std = None
    window_s: Optional[float] = None
    step_s: Optional[float] = None
    mode: Optional[str] = None
    decision_threshold: Optional[float] = None
    _scorer = None

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.6, device=None):
        self.enabled = False
        self.model = None
        self.model_path: Optional[str] = None
        self.weights_hash: Optional[str] = None
        self.load_error: Optional[str] = None
        self.meta: Optional[Dict[str, Any]] = None
        self.threshold = threshold
        self.seq_len = 30
        self.device = None
        self._torch = None
        self._init_v2_fields()

        try:
            import torch
            from app.ai_classifier.model import Fall1DCNN, Fall1DCNNGAP
        except Exception as e:
            self.load_error = f"import failed: {e!r}"
            logger.error(f"[AI] classifier DISABLED ({self.load_error}) -> rule-based result stands")
            return

        path = resolve_weights_path(model_path)
        if path is None:
            self.load_error = f"weights not found (model_path={model_path!r})"
            logger.error(f"[AI] classifier DISABLED: {self.load_error} -> rule-based result stands")
            return

        try:
            meta = load_meta(path)
        except Exception as e:
            self.load_error = f"meta unreadable: {e}"
            logger.error(f"[AI] classifier DISABLED: {self.load_error} ({path}) -> rule-based result stands")
            return
        if meta is None:
            meta = legacy_v1_meta()
            logger.info(f"[AI] no weights meta for {path} -> assuming legacy v1")
        self.meta = meta
        reason = check_compatible(meta)
        if reason is not None:
            self.load_error = f"incompatible meta: {reason}"
            logger.error(f"[AI] classifier DISABLED: {self.load_error} ({path}) -> rule-based result stands")
            return
        version = meta.get("feature_version")

        try:
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if version == V2_FEATURE_VERSION:
                model = self._load_v2(torch, Fall1DCNNGAP, path, meta)
            else:
                model = Fall1DCNN().to(self.device)
                try:
                    state = torch.load(path, map_location=self.device, weights_only=True)
                except TypeError:
                    # weights_only 인자를 지원하지 않는 구버전 torch
                    state = torch.load(path, map_location=self.device)
                model.load_state_dict(state, strict=True)
            model.eval()
        except Exception as e:
            self._init_v2_fields()
            self.load_error = f"load failed: {e!r}"
            logger.error(f"[AI] classifier DISABLED: {self.load_error} ({path}) -> rule-based result stands")
            return

        self.model = model
        self._torch = torch
        self.model_path = path
        self.weights_hash = sha256_8(path)
        self.feature_version = version
        if version == V2_FEATURE_VERSION:
            from app.ai_classifier.scoring import nn_batch_scorer
            # 창마다 배치 1 (오프라인 E*max와 같은 경로: CUDA TF32에서 배치 크기에 따라 p가 1e-4 정도 달라짐)
            self._scorer = nn_batch_scorer(model, device=self.device, preprocess=self.std, batched=False)
        self.enabled = True
        extra = (f", protocol={self.protocol()}" if version == V2_FEATURE_VERSION else "")
        logger.info(f"[AI] Loaded classifier {path} (version={version}, sha256={self.weights_hash}, "
                    f"threshold={threshold}{extra})")

    def _init_v2_fields(self):
        self.feature_version = None
        self.std = None
        self.window_s: Optional[float] = None
        self.step_s: Optional[float] = None
        self.mode: Optional[str] = None
        self.decision_threshold: Optional[float] = None
        self._scorer = None

    def _load_v2(self, torch, arch_cls, path: str, meta: Dict[str, Any]):
        from app.ai_classifier.event_window import Standardizer

        ck = _load_checkpoint(torch, path, self.device)
        if isinstance(ck, dict) and "state_dict" in ck:
            state = ck["state_dict"]
            mu, sd = ck.get("mu", meta.get("mu")), ck.get("sd", meta.get("sd"))
        else:
            state, mu, sd = ck, meta.get("mu"), meta.get("sd")
        mu = _as_vec(mu, V2_IN_CHANNELS, "mu")
        sd = _as_vec(sd, V2_IN_CHANNELS, "sd")
        if not np.all(sd > 0):
            raise ValueError("sd must be > 0")
        model = arch_cls(V2_IN_CHANNELS).to(self.device)
        model.load_state_dict(state, strict=True)
        self.std = Standardizer(mu, sd)
        self.window_s = float(meta["window_s"])
        self.step_s = float(meta["step_s"])
        self.mode = str(meta["mode"])
        self.decision_threshold = float(meta["threshold"])
        return model

    @property
    def is_v2(self) -> bool:
        return self.enabled and self.feature_version == V2_FEATURE_VERSION

    def protocol(self) -> Optional[Dict[str, Any]]:
        """v2 판정 프로토콜 (창 길이, 간격, 모드, 임계값). v1/비활성이면 None."""
        if self.feature_version != V2_FEATURE_VERSION:
            return None
        return {"window_s": self.window_s, "step_s": self.step_s, "mode": self.mode,
                "threshold": self.decision_threshold, "batch": 1}

    def normalize_sequence(self, sequence):
        """시퀀스(30, 34)에 v1 골반 중심 정규화를 적용합니다. (학습 코드와 같은 구현 사용)"""
        return normalize_sequence_v1(sequence)

    def predict_proba(self, keypoint_sequence) -> Optional[float]:
        """
        v1: 최근 30프레임 시퀀스의 낙상 확률 p_fall(0~1).
        분류기가 비활성이거나, v2 모델이거나(30프레임 시퀀스로는 판정하지 않음, make_engine_verifier 사용),
        프레임이 부족하거나, 추론 중 오류가 나면 None.
        """
        if not self.enabled or self.feature_version == V2_FEATURE_VERSION:
            return None
        frames = list(keypoint_sequence)
        if len(frames) < self.seq_len:
            return None
        try:
            torch = self._torch
            seq_np = np.asarray(frames[-self.seq_len:], dtype=np.float32)
            seq_np = self.normalize_sequence(seq_np)  # (30, 34)
            # [30, 34] -> [1, 34, 30] (Batch, Channel, Length)
            seq_np = np.transpose(seq_np, (1, 0))
            seq_tensor = torch.from_numpy(np.ascontiguousarray(seq_np)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(seq_tensor), dim=1)
            return float(probs[0, 1].item())
        except Exception:
            logger.exception("[AI] predict_proba failed")
            return None

    def score_windows(self, X) -> np.ndarray:
        """v2: 특징 창 묶음 X (B, 92, L) -> p_fall (B,) (창마다 표준화 후 배치 1). v2가 아니면 RuntimeError."""
        if not self.is_v2 or self._scorer is None:
            raise RuntimeError("score_windows requires an enabled v2 classifier")
        return self._scorer(np.asarray(X, dtype=np.float32))

    def make_engine_verifier(self, model_id: Optional[str] = None):
        """
        v2: 엔진 하나에 묶어 쓸 검증기 (event_window.SlidingWindowVerifier, 메타의 창/간격/모드).
        호출 측이 bind_engine(engine)으로 엔진을 넘깁니다. v2가 아니거나 비활성이면 None.
        """
        if not self.is_v2:
            return None
        from app.ai_classifier.event_window import SlidingWindowVerifier
        return SlidingWindowVerifier(self.score_windows, mode=self.mode, step_s=self.step_s, win_s=self.window_s,
                                     model_id=model_id or self.weights_hash)

    def predict(self, keypoint_sequence) -> Tuple[Optional[int], float]:
        """
        기존 호환 API (v1).
        Returns:
            label: 1 for Fall, 0 for ADL, None if uncertain/unavailable
            confidence: float
        """
        p = self.predict_proba(keypoint_sequence)
        if p is None:
            return None, 0.0
        label = 1 if p >= 0.5 else 0
        conf = max(p, 1.0 - p)
        # 신뢰도 임계값 체크
        if conf < self.threshold:
            return None, conf  # 불확실하면 판단 보류 (Rule-based 따르기)
        return label, conf

    def warmup(self) -> Optional[float]:
        """첫 추론 지연을 미리 소모합니다. 소요 시간(ms)을 반환하고, 비활성이면 None."""
        if not self.enabled:
            return None
        t0 = time.perf_counter()
        if self.feature_version == V2_FEATURE_VERSION:
            try:
                n = int(round(float(self.window_s) * V2_HZ))
                self.score_windows(np.zeros((1, V2_IN_CHANNELS, n), dtype=np.float32))
            except Exception:
                logger.exception("[AI] v2 warmup failed")
        else:
            self.predict_proba([[0.0] * 34] * self.seq_len)
        return (time.perf_counter() - t0) * 1000.0

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "version": self.feature_version if self.enabled else None,
            "model_path": self.model_path,
            "weights_hash": self.weights_hash,
            "load_error": self.load_error,
            "threshold": self.threshold,
            "decision_threshold": self.decision_threshold,
            "protocol": self.protocol() if self.enabled else None,
            "meta": dict(self.meta) if isinstance(self.meta, dict) else None,
            "feature_version": self.meta.get("feature_version") if isinstance(self.meta, dict) else None,
        }


class _DisabledClassifier(FallClassifier):
    """FallClassifier 생성 자체가 실패했을 때 쓰는 비활성 대체 객체."""

    def __init__(self, reason: str):
        self.enabled = False
        self.model = None
        self.model_path = None
        self.weights_hash = None
        self.load_error = reason
        self.meta = None
        self.threshold = 0.6
        self.seq_len = 30
        self.device = None
        self._torch = None
        self._init_v2_fields()


# Singleton instance
_classifier = None
_classifier_lock = threading.Lock()


def get_classifier() -> FallClassifier:
    """분류기 싱글턴을 반환합니다. 예외를 던지지 않습니다. (스레드 안전, 모델은 한 번만 로드)"""
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                try:
                    from app.config import AI_MODEL_PATH
                    _classifier = FallClassifier(model_path=AI_MODEL_PATH or None)
                except Exception as e:
                    logger.exception("[AI] classifier init failed")
                    _classifier = _DisabledClassifier(f"init failed: {e!r}")
    return _classifier


def load_classifier(model_path: Optional[str]) -> FallClassifier:
    """
    싱글턴을 지정한 가중치로 다시 만듭니다 (오프라인 평가 --weights 등). 예외를 던지지 않습니다.
    model_path가 None/빈 문자열이면 기본 탐색 순서(get_classifier와 같음).
    """
    global _classifier
    with _classifier_lock:
        try:
            if model_path:
                _classifier = FallClassifier(model_path=model_path)
            else:
                from app.config import AI_MODEL_PATH
                _classifier = FallClassifier(model_path=AI_MODEL_PATH or None)
        except Exception as e:
            logger.exception("[AI] classifier init failed")
            _classifier = _DisabledClassifier(f"init failed: {e!r}")
    return _classifier


def peek_classifier() -> Optional[FallClassifier]:
    """이미 로드된 분류기를 반환합니다. 아직 없으면 None (로드하지 않음)."""
    return _classifier


def reset_classifier():
    """테스트용: 싱글턴을 초기화합니다."""
    global _classifier
    with _classifier_lock:
        _classifier = None
