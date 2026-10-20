"""
실험용 검증기 (Level1Engine(params, verifier=...)에 주입)

- V1SeqVerifier: 운영 v1 경로와 같음. 엔진의 kp_seq(15Hz, EMA 스무딩된 최근 30프레임 x 34)를
  features.normalize_sequence_v1 -> (1, 34, 30) -> softmax[1] (app.ai_classifier.inference.FallClassifier.predict_proba와 동일)
- EventWindowVerifier: LEVEL1 판정 시점에 정지 구간 30프레임 대신 ABNORMAL 진입 사건 구간
  (engine.get_incident_snapshot()["event_window"], 원시 키포인트)을 v2 특징(92 x N)으로 바꿔 판정
- SlidingWindowVerifier (E*): 같은 v2 모델을 여러 4초 창에 적용 (재학습 없음)
  - mode "max": 진입 사건 구간(첫 창, EventWindowVerifier와 같음)부터 판정 시점 - 4초까지 step_s 간격 창들의 최대 p_fall
  - mode "last" (참고): 판정 시점 직전 [ts - 4, ts] 한 창. 정지 꼬리에서는 같은 프레임 반복이라 학습에 없던 입력

두 검증기 모두 판단할 수 없으면 None을 반환하고, 엔진은 이를 fail-open(규칙 결과 유지)으로 처리합니다.
모델은 생성 시 eval 모드로 바꾸고 지정한 장치로 옮깁니다.

SlidingWindowVerifier, event_window_to_v2, entry_window_starts, nn_batch_scorer는 운영 v2 분류기와 같은 구현
(app/ai_classifier/event_window.py, scoring.py)을 다시 내보냅니다. 단일 구현이라 오프라인 결과 = 운영 판정
(tests/test_ai_v2_runtime.py가 예전 오프라인 구현과의 비트 단위 동일성을 고정).
"""
from typing import Any, Callable, List, Optional

import numpy as np

from app.ai_classifier.event_window import SLIDE_STEP_S, SlidingWindowVerifier, entry_window_starts, event_window_to_v2
from app.ai_classifier.features import WINDOW_S, normalize_sequence_v1
from app.ai_classifier.scoring import nn_batch_scorer
from app.ai_classifier.scoring import resolve_device as _resolve_device
from app.ai_classifier.scoring import softmax_p1 as _softmax_p1

__all__ = ["V1SeqVerifier", "EventWindowVerifier", "SlidingWindowVerifier", "event_window_to_v2",
           "entry_window_starts", "nn_batch_scorer", "SLIDE_STEP_S"]


class _Base:
    def __init__(self, model, device=None, model_id: Optional[str] = None):
        self.device = _resolve_device(model, device)
        self.model = model.to(self.device).eval()
        self.model_id = model_id or type(model).__name__
        # 진단용 카운터
        self.calls = 0
        self.n_none = 0
        self.last_p: Optional[float] = None

    def _done(self, p: Optional[float]) -> Optional[float]:
        self.last_p = p
        if p is None:
            self.n_none += 1
        return p


class V1SeqVerifier(_Base):
    """운영 v1 검증기와 같은 계산: 최근 seq_len(30)프레임 -> 골반 중심 정규화 -> (1, 34, 30) -> p_fall."""

    seq_len = 30

    def __call__(self, seq: List[List[float]]) -> Optional[float]:
        self.calls += 1
        frames = list(seq)
        if len(frames) < self.seq_len:
            return self._done(None)
        x = normalize_sequence_v1(np.asarray(frames[-self.seq_len:], dtype=np.float32))  # (30, 34)
        return self._done(_softmax_p1(self.model, x.T, self.device))


class EventWindowVerifier(_Base):
    """
    ABNORMAL 진입 사건 구간(v2)으로 판정하는 검증기. 엔진이 넘겨주는 seq(정지 구간 30프레임)는 쓰지 않습니다.

    engine_ref: 엔진을 돌려주는 callable. replay.run_clip은 bind_engine(engine)을 자동 호출하므로
                run_clip으로 재생할 때는 None으로 둬도 됩니다 (engine_ref가 있으면 그것이 우선).
    fixed_len_s: event_window_to_v2 참고. preprocess: 특징 (92, N) -> 모델 입력 변환 (예: 채널 표준화), 선택.
    사건 구간이 없거나 비어 있으면 None (fail-open).
    """

    def __init__(self, model, device=None, engine_ref: Optional[Callable[[], Any]] = None,
                 model_id: Optional[str] = None, fixed_len_s: Optional[float] = WINDOW_S,
                 preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        super().__init__(model, device, model_id)
        self._engine_ref = engine_ref
        self._engine = None
        self.fixed_len_s = fixed_len_s
        self.preprocess = preprocess
        self.last_x: Optional[np.ndarray] = None

    def bind_engine(self, engine) -> None:
        self._engine = engine

    def __call__(self, seq: List[List[float]]) -> Optional[float]:
        self.calls += 1
        self.last_x = None
        engine = self._engine_ref() if self._engine_ref is not None else self._engine
        if engine is None:
            return self._done(None)
        snap = engine.get_incident_snapshot() or {}
        x = event_window_to_v2(snap.get("event_window"), self.fixed_len_s)
        if x is None:
            return self._done(None)
        if self.preprocess is not None:
            x = np.asarray(self.preprocess(x), dtype=np.float32)
        self.last_x = x
        return self._done(_softmax_p1(self.model, x, self.device))
