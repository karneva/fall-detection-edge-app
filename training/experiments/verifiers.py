"""
실험용 검증기 (Level1Engine(params, verifier=...)에 주입)

- V1SeqVerifier: 운영 v1 경로와 같음. 엔진의 kp_seq(15Hz, EMA 스무딩된 최근 30프레임 x 34)를
  features.normalize_sequence_v1 -> (1, 34, 30) -> softmax[1] (app.ai_classifier.inference.FallClassifier.predict_proba와 동일)
- EventWindowVerifier: LEVEL1 판정 시점에 정지 구간 30프레임 대신 ABNORMAL 진입 사건 구간
  (engine.get_incident_snapshot()["event_window"], 원시 키포인트)을 v2 특징(92 x N)으로 바꿔 판정

두 검증기 모두 판단할 수 없으면 None을 반환하고, 엔진은 이를 fail-open(규칙 결과 유지)으로 처리합니다.
모델은 생성 시 eval 모드로 바꾸고 지정한 장치로 옮깁니다.
"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from app.ai_classifier.features import NUM_KPS, V2_CHANNELS, WINDOW_S, normalize_sequence_v1, window_v2
from training.datasets.clipfmt import from_sidecar

__all__ = ["V1SeqVerifier", "EventWindowVerifier", "event_window_to_v2"]

# v2 관절 conf 마스크 채널 (mask_nose부터 17개)
_MASK0 = V2_CHANNELS.index("mask_nose")


def _resolve_device(model, device):
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _softmax_p1(model, x: np.ndarray, device) -> float:
    """x (C, L) -> softmax(model(x[None]))[0, 1]"""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)
    return float(probs[0, 1].item())


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


def event_window_to_v2(win: Optional[Dict[str, Any]], fixed_len_s: Optional[float] = WINDOW_S) -> Optional[np.ndarray]:
    """
    kp_buffer.window 결과(사건 구간) -> v2 특징 (92, N). 쓸 수 없으면 None.

    clipfmt.from_sidecar와 같은 규칙으로 배열을 만들고(missing 프레임 0 + present False, ts는 구간 첫 샘플 기준),
    meta의 window_start_s/window_end_s(여유 pad 제외 구간)를 features.window_v2에 넘깁니다.
    fixed_len_s가 있으면 [window_start_s, window_start_s + fixed_len_s]만 사용합니다 (기본 4초 = 60스텝).
      저자세 지속 진입은 구간 시작이 저자세 시작 시각 - pre_s라 구간이 4초보다 길 수 있는데,
      이때도 학습 창(onset -1.5 ~ +2.5초)과 같은 길이/정렬을 유지하기 위함입니다.
    None이 되는 경우: 구간 없음, ts 비어 있음, 사람이 있는 프레임 없음, 시작 시각 없음, 길이 0,
      창 안에 보이는 관절이 하나도 없음 (사람이 여유 pad 구간에만 있던 경우 등: 학습에 없는 입력이라 fail-open).
    """
    if not isinstance(win, dict) or not win.get("ts"):
        return None
    try:
        clip = from_sidecar({"snapshot": {"event_window": win}})
    except ValueError:
        return None
    if not np.any(clip["present"]):
        return None
    meta = clip["meta"]
    t_start = meta.get("window_start_s")
    t_end = meta.get("window_end_s")
    if t_start is None:
        return None
    if fixed_len_s is not None:
        t_end = float(t_start) + float(fixed_len_s)
    if t_end is None or float(t_end) <= float(t_start):
        return None
    x = window_v2(clip["ts"], clip["kps"], clip["bbox"], clip["present"], clip["frame_wh"],
                  float(t_start), float(t_end))
    if x.shape[1] == 0 or not np.any(x[_MASK0:_MASK0 + NUM_KPS] > 0):
        return None
    return x


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
