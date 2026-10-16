import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.ai_classifier.features import normalize_sequence_v1
from app.ai_classifier.model_meta import check_compatible, legacy_v1_meta, load_meta
from app.ai_classifier.weights_resolver import resolve_weights_path, sha256_8

logger = logging.getLogger(__name__)


class FallClassifier:
    """
    2차 검증용 1D CNN 분류기 래퍼.

    가중치가 없거나 로드에 실패하면 enabled=False가 되고 모든 예측이 None을 반환합니다.
    (초기화되지 않은 랜덤 모델로 알람을 억제하는 일이 없도록 fail-open)

    가중치 옆 "<weights>.meta.json"이 이 런타임과 맞지 않는 형식(예: features v2 모델)이면
    역시 비활성화합니다. 메타 파일이 없으면 기존 v1 모델로 간주합니다.
    """

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

        try:
            import torch
            from app.ai_classifier.model import Fall1DCNN
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

        try:
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Fall1DCNN().to(self.device)
            try:
                state = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:
                # weights_only 인자를 지원하지 않는 구버전 torch
                state = torch.load(path, map_location=self.device)
            model.load_state_dict(state, strict=True)
            model.eval()
        except Exception as e:
            self.load_error = f"load failed: {e!r}"
            logger.error(f"[AI] classifier DISABLED: {self.load_error} ({path}) -> rule-based result stands")
            return

        self.model = model
        self._torch = torch
        self.model_path = path
        self.weights_hash = sha256_8(path)
        self.enabled = True
        logger.info(f"[AI] Loaded classifier {path} (sha256={self.weights_hash}, threshold={threshold})")

    def normalize_sequence(self, sequence):
        """시퀀스(30, 34)에 v1 골반 중심 정규화를 적용합니다. (학습 코드와 같은 구현 사용)"""
        return normalize_sequence_v1(sequence)

    def predict_proba(self, keypoint_sequence) -> Optional[float]:
        """
        낙상 확률 p_fall(0~1)을 반환합니다.
        분류기가 비활성이거나, 프레임이 부족하거나, 추론 중 오류가 나면 None.
        """
        if not self.enabled:
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

    def predict(self, keypoint_sequence) -> Tuple[Optional[int], float]:
        """
        기존 호환 API.
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
        self.predict_proba([[0.0] * 34] * self.seq_len)
        return (time.perf_counter() - t0) * 1000.0

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_path": self.model_path,
            "weights_hash": self.weights_hash,
            "load_error": self.load_error,
            "threshold": self.threshold,
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


def peek_classifier() -> Optional[FallClassifier]:
    """이미 로드된 분류기를 반환합니다. 아직 없으면 None (로드하지 않음)."""
    return _classifier


def reset_classifier():
    """테스트용: 싱글턴을 초기화합니다."""
    global _classifier
    with _classifier_lock:
        _classifier = None
