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
"""
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from app.ai_classifier.features import NUM_KPS, V2_CHANNELS, WINDOW_S, normalize_sequence_v1, window_v2
from training.datasets.clipfmt import from_sidecar

__all__ = ["V1SeqVerifier", "EventWindowVerifier", "SlidingWindowVerifier", "event_window_to_v2",
           "entry_window_starts", "nn_batch_scorer"]

# E* 창 이동 간격 (초)
SLIDE_STEP_S = 0.5

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


def entry_window_starts(start_s: float, ts: float, win_s: float = WINDOW_S, step_s: float = SLIDE_STEP_S
                        ) -> List[float]:
    """
    창 시작 시각 목록: start_s, start_s + step_s, ... (마지막 창 끝 <= ts).
    첫 값은 항상 start_s (구간이 창 하나보다 짧아도 그 창은 씀).
    """
    n = int(math.floor((float(ts) - float(win_s) - float(start_s)) / float(step_s) + 1e-6))
    return [float(start_s) + i * float(step_s) for i in range(max(n, 0) + 1)]


def nn_batch_scorer(model, device=None, preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                    batched: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """
    v2 모델 -> 묶음 점수 함수 X (B, 92, L) -> p_fall (B,).
    preprocess는 EventWindowVerifier와 같이 창 하나 (92, L)씩 적용합니다.
    batched=False(기본): 창마다 배치 1로 계산 (EventWindowVerifier의 _softmax_p1과 같은 경로라 첫 창 p가 E와 비트 단위로 같음).
      CUDA에서는 cuDNN(TF32) 알고리즘이 배치 크기마다 달라 B개를 한 번에 돌리면 p가 1e-4 정도 달라짐.
    batched=True: B개를 한 번에 (빠름, 위 차이 허용 시).
    """
    dev = _resolve_device(model, device)
    model = model.to(dev).eval()

    def score(X: np.ndarray) -> np.ndarray:
        xs = [np.asarray(preprocess(x), dtype=np.float32) if preprocess is not None else x for x in X]
        if not batched:
            return np.asarray([_softmax_p1(model, x, dev) for x in xs], dtype=np.float64)
        t = torch.from_numpy(np.ascontiguousarray(np.stack(xs), dtype=np.float32)).to(dev)
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1)[:, 1]
        return probs.cpu().numpy().astype(np.float64)

    return score


class SlidingWindowVerifier:
    """
    E*: 사건 구간 v2 모델을 여러 4초 창에 적용하는 검증기 (모델 종류와 무관하게 score_fn으로 받음).

    score_fn: X (B, 92, L) -> p_fall (B,) (nn_batch_scorer, 또는 고전 ML 특징 + predict_proba)
    mode "max": 첫 창 = 진입 사건 구간 (get_incident_snapshot()["event_window"], EventWindowVerifier와 같은 입력),
                이어서 (event_anchor_ts 또는 abnormal_start_ts) - event_window_pre_s부터 판정 시점 - win_s까지
                step_s 간격으로 kp_buffer.window(t, t + win_s, pad_s=event_window_pad_s) 창. 최대 p_fall을 반환.
                버퍼 앞이 잘린 창(truncated, 버퍼 보관 시간 초과)은 첫 창을 빼고 건너뜀.
    mode "last": 판정 시점 직전 [ts - win_s, ts] 한 창 (참고용).
    판정 시점 ts = 엔진의 마지막 kp_ts (AI 호출은 사람이 보이는 프레임에서만 일어남).
    쓸 수 있는 창이 없으면 None (fail-open).
    """

    def __init__(self, score_fn: Callable[[np.ndarray], np.ndarray], mode: str = "max",
                 step_s: float = SLIDE_STEP_S, win_s: float = WINDOW_S, model_id: Optional[str] = None,
                 engine_ref: Optional[Callable[[], Any]] = None):
        if mode not in ("max", "last"):
            raise ValueError(f"mode는 'max' 또는 'last': {mode!r}")
        self.score_fn = score_fn
        self.mode = mode
        self.step_s = float(step_s)
        self.win_s = float(win_s)
        self.model_id = model_id or f"sliding-{mode}"
        self._engine_ref = engine_ref
        self._engine = None
        # 진단용 카운터
        self.calls = 0
        self.n_none = 0
        self.n_windows = 0          # 점수를 낸 창 수 합
        self.n_windows_skipped = 0  # None/잘린 창 수 합
        self.n_max_later = 0        # 최대 p_fall이 첫 창이 아닌 창에서 나온 호출 수 (max)
        self.last_p: Optional[float] = None
        self.last_ps: Optional[List[float]] = None

    def bind_engine(self, engine) -> None:
        self._engine = engine

    def stats(self) -> Dict[str, Any]:
        return {"calls": self.calls, "none": self.n_none, "windows": self.n_windows,
                "windows_skipped": self.n_windows_skipped, "max_later": self.n_max_later,
                "mode": self.mode, "step_s": self.step_s}

    def _done(self, p: Optional[float]) -> Optional[float]:
        self.last_p = p
        if p is None:
            self.n_none += 1
        return p

    def _windows(self, engine) -> List[Optional[np.ndarray]]:
        kp_ts = getattr(engine, "kp_ts", None)
        ts = float(kp_ts[-1]) if kp_ts else None
        pad = float(engine.p.event_window_pad_s)
        if self.mode == "last":
            if ts is None:
                return []
            return [event_window_to_v2(engine.kp_buffer.window(ts - self.win_s, ts, pad_s=pad), self.win_s)]
        snap = engine.get_incident_snapshot() or {}
        xs = [event_window_to_v2(snap.get("event_window"), self.win_s)]
        start = engine.event_anchor_ts if engine.event_anchor_ts is not None else engine.abnormal_start_ts
        if ts is None or start is None:
            return xs
        for t in entry_window_starts(float(start) - float(engine.p.event_window_pre_s), ts,
                                     self.win_s, self.step_s)[1:]:
            win = engine.kp_buffer.window(t, t + self.win_s, pad_s=pad)
            xs.append(None if win.get("truncated") else event_window_to_v2(win, self.win_s))
        return xs

    def __call__(self, seq: List[List[float]]) -> Optional[float]:
        self.calls += 1
        self.last_ps = None
        engine = self._engine_ref() if self._engine_ref is not None else self._engine
        if engine is None:
            return self._done(None)
        xs = self._windows(engine)
        ok = [i for i, x in enumerate(xs) if x is not None]
        self.n_windows_skipped += len(xs) - len(ok)
        if not ok:
            return self._done(None)
        ps = np.asarray(self.score_fn(np.stack([xs[i] for i in ok])), dtype=np.float64).reshape(-1)
        self.n_windows += len(ok)
        self.last_ps = [float(p) for p in ps]
        j = int(np.argmax(ps))
        if self.mode == "max" and ok[j] != 0:
            self.n_max_later += 1
        return self._done(float(ps[j]) if self.mode == "max" else float(ps[0]))
