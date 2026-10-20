"""
사건 구간(v2) 판정 경로: 엔진의 원시 키포인트 버퍼 -> 4초 창 v2 특징 -> 여러 창의 최대 p_fall (numpy만 사용)

운영(app/core/level1._ClassifierVerifier)과 오프라인 실험(training/experiments/verifiers)이 같은 구현을 씁니다.
- event_window_to_v2: kp_buffer.window 결과 -> v2 특징 (92, 60)
  (training.datasets.clipfmt.from_sidecar와 같은 규칙으로 배열을 만든 뒤 features.window_v2)
- entry_window_starts: ABNORMAL 진입 사건 구간부터 판정 시점까지 step_s 간격 창 시작 시각
- SlidingWindowVerifier: 엔진에 묶인(bind_engine) 검증기. mode "max" = 배포 프로토콜 (E*max)
- Standardizer: 채널별 표준화 (학습 창 mu/sd, 체크포인트에 함께 저장됨)

점수 함수(score_fn: X (B, 92, L) -> p_fall (B,))는 밖에서 주입합니다 (torch 모델, 고전 ML 등).
"""
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.ai_classifier.features import NUM_KPS, V2_CHANNELS, WINDOW_S, window_v2

__all__ = ["SLIDE_STEP_S", "Standardizer", "event_window_to_v2", "entry_window_starts", "SlidingWindowVerifier"]

# 창 이동 간격 (초)
SLIDE_STEP_S = 0.5

# v2 관절 conf 마스크 채널 (mask_nose부터 17개)
_MASK0 = V2_CHANNELS.index("mask_nose")


class Standardizer:
    """채널별 표준화 (x - mu) / sd. x: (C, L) 창 하나 또는 (N, C, L). 학습(run_progression)과 같은 계산."""

    def __init__(self, mu, sd):
        self.mu = np.asarray(mu, dtype=np.float32).reshape(-1)
        self.sd = np.asarray(sd, dtype=np.float32).reshape(-1)
        if self.mu.shape != self.sd.shape:
            raise ValueError(f"mu {self.mu.shape} / sd {self.sd.shape} shape mismatch")

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            return (x - self.mu[:, None]) / self.sd[:, None]
        return (x - self.mu[None, :, None]) / self.sd[None, :, None]


def _default_frame_wh() -> List[int]:
    """해상도 기록이 없는 사건 구간용 기본값 (런타임 카메라 설정, clipfmt와 같음)."""
    try:
        from app.config import FRAME_H, FRAME_W
        return [int(FRAME_W), int(FRAME_H)]
    except Exception:
        return [1920, 1080]


def _num_or_none(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _window_arrays(win: Dict[str, Any]):
    """
    kp_buffer.window 결과 -> (ts (T,) 구간 첫 샘플 기준 초, kps (T,17,3), bbox (T,4), present (T,), frame_wh (2,), t0).
    missing 프레임은 0 + present False. 형식이 어긋나면 ValueError (clipfmt.from_sidecar + _assemble과 같은 규칙).
    """
    ts_abs = [float(t) for t in win["ts"]]
    t = len(ts_abs)
    kps_rows = win.get("kps") or [None] * t
    bboxes = win.get("bbox") or [None] * t
    if len(kps_rows) != t or len(bboxes) != t:
        raise ValueError("event_window ts/kps/bbox length mismatch")
    kps = np.zeros((t, NUM_KPS, 3), dtype=np.float32)
    bbox = np.zeros((t, 4), dtype=np.float32)
    present = np.zeros(t, dtype=bool)
    for i, (rows, b) in enumerate(zip(kps_rows, bboxes)):
        if rows is None:
            continue
        f = np.asarray(rows, dtype=np.float32)
        if f.shape != (NUM_KPS, 3):
            continue  # 잘못된 프레임은 missing
        kps[i] = f
        if b is not None and len(b) == 4:
            bbox[i] = np.asarray([float(v) for v in b], dtype=np.float32)
        present[i] = True
    t0 = ts_abs[0] if t else 0.0
    ts = np.asarray([x - t0 for x in ts_abs], dtype=np.float64)
    fwh = win.get("frame_wh")
    if fwh and len(fwh) == 2 and int(fwh[0]) > 0 and int(fwh[1]) > 0:
        frame_wh = np.asarray([int(fwh[0]), int(fwh[1])], dtype=np.int32)
    else:
        frame_wh = np.asarray(_default_frame_wh(), dtype=np.int32)
    if not np.all(np.isfinite(ts)):
        raise ValueError("ts has non-finite values")
    if t >= 2 and np.any(np.diff(ts) < 0):
        raise ValueError("ts must be non-decreasing")
    if not (np.all(np.isfinite(kps)) and np.all(np.isfinite(bbox))):
        raise ValueError("kps/bbox have non-finite values")
    return ts, kps, bbox, present, frame_wh, t0


def event_window_to_v2(win: Optional[Dict[str, Any]], fixed_len_s: Optional[float] = WINDOW_S) -> Optional[np.ndarray]:
    """
    kp_buffer.window 결과(사건 구간) -> v2 특징 (92, N). 쓸 수 없으면 None.

    배열은 clipfmt.from_sidecar와 같은 규칙(missing 프레임 0 + present False, ts는 구간 첫 샘플 기준)으로 만들고,
    구간의 t_start/t_end(여유 pad 제외)를 features.window_v2에 넘깁니다.
    fixed_len_s가 있으면 [t_start, t_start + fixed_len_s]만 사용합니다 (기본 4초 = 60스텝).
      저자세 지속 진입은 구간 시작이 저자세 시작 시각 - pre_s라 구간이 4초보다 길 수 있는데,
      이때도 학습 창(onset -1.5 ~ +2.5초)과 같은 길이/정렬을 유지하기 위함입니다.
    None이 되는 경우: 구간 없음, ts 비어 있음, 형식 오류, 사람이 있는 프레임 없음, 시작 시각 없음, 길이 0,
      창 안에 보이는 관절이 하나도 없음 (사람이 여유 pad 구간에만 있던 경우 등: 학습에 없는 입력이라 fail-open).
    """
    if not isinstance(win, dict) or not win.get("ts"):
        return None
    try:
        ts, kps, bbox, present, frame_wh, t0 = _window_arrays(win)
    except (ValueError, TypeError):
        return None
    if not np.any(present):
        return None
    t_start = _num_or_none(win.get("t_start"))
    t_end = _num_or_none(win.get("t_end"))
    if t_start is None:
        return None
    t_start -= t0
    if t_end is not None:
        t_end -= t0
    if fixed_len_s is not None:
        t_end = float(t_start) + float(fixed_len_s)
    if t_end is None or float(t_end) <= float(t_start):
        return None
    x = window_v2(ts, kps, bbox, present, frame_wh, float(t_start), float(t_end))
    if x.shape[1] == 0 or not np.any(x[_MASK0:_MASK0 + NUM_KPS] > 0):
        return None
    return x


def entry_window_starts(start_s: float, ts: float, win_s: float = WINDOW_S, step_s: float = SLIDE_STEP_S
                        ) -> List[float]:
    """
    창 시작 시각 목록: start_s, start_s + step_s, ... (마지막 창 끝 <= ts).
    첫 값은 항상 start_s (구간이 창 하나보다 짧아도 그 창은 씀).
    """
    n = int(math.floor((float(ts) - float(win_s) - float(start_s)) / float(step_s) + 1e-6))
    return [float(start_s) + i * float(step_s) for i in range(max(n, 0) + 1)]


class SlidingWindowVerifier:
    """
    사건 구간 v2 모델을 여러 4초 창에 적용하는 검증기 (모델 종류와 무관하게 score_fn으로 받음).

    score_fn: X (B, 92, L) -> p_fall (B,)
    mode "max" (배포 프로토콜): 첫 창 = 진입 사건 구간 (get_incident_snapshot()["event_window"]),
                이어서 (event_anchor_ts 또는 abnormal_start_ts) - event_window_pre_s부터 판정 시점 - win_s까지
                step_s 간격으로 kp_buffer.window(t, t + win_s, pad_s=event_window_pad_s) 창. 최대 p_fall을 반환.
                버퍼 앞이 잘린 창(truncated, 버퍼 보관 시간 초과)은 첫 창을 빼고 건너뜀.
    mode "last" (참고): 판정 시점 직전 [ts - win_s, ts] 한 창.
    판정 시점 ts = 엔진의 마지막 kp_ts (AI 호출은 사람이 보이는 프레임에서만 일어남).
    엔진은 bind_engine(engine) 또는 engine_ref(엔진을 돌려주는 callable, 있으면 우선)로 받습니다.
    엔진이 없거나 쓸 수 있는 창이 없으면 None (fail-open).
    """

    def __init__(self, score_fn: Callable[[np.ndarray], np.ndarray], mode: str = "max",
                 step_s: float = SLIDE_STEP_S, win_s: float = WINDOW_S, model_id: Optional[str] = None,
                 engine_ref: Optional[Callable[[], Any]] = None):
        if mode not in ("max", "last"):
            raise ValueError(f"mode는 'max' 또는 'last': {mode!r}")
        if not (float(step_s) > 0 and float(win_s) > 0):
            raise ValueError(f"step_s/win_s는 양수: {step_s!r}, {win_s!r}")
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
        # 마지막 호출에서 만든 창들의 시작 시각 (절대 ts, 점수 낸 창만; 첫 창은 사건 구간 시작)
        self.last_starts: Optional[List[float]] = None
        self._starts: List[Optional[float]] = []

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
            self._starts = [ts - self.win_s]
            return [event_window_to_v2(engine.kp_buffer.window(ts - self.win_s, ts, pad_s=pad), self.win_s)]
        snap = engine.get_incident_snapshot() or {}
        ew = snap.get("event_window")
        self._starts = [_num_or_none(ew.get("t_start")) if isinstance(ew, dict) else None]
        xs = [event_window_to_v2(ew, self.win_s)]
        start = engine.event_anchor_ts if engine.event_anchor_ts is not None else engine.abnormal_start_ts
        if ts is None or start is None:
            return xs
        for t in entry_window_starts(float(start) - float(engine.p.event_window_pre_s), ts,
                                     self.win_s, self.step_s)[1:]:
            win = engine.kp_buffer.window(t, t + self.win_s, pad_s=pad)
            xs.append(None if win.get("truncated") else event_window_to_v2(win, self.win_s))
            self._starts.append(t)
        return xs

    def __call__(self, seq: List[List[float]]) -> Optional[float]:
        self.calls += 1
        self.last_ps = None
        self.last_starts = None
        engine = self._engine_ref() if self._engine_ref is not None else self._engine
        if engine is None:
            return self._done(None)
        self._starts = []
        xs = self._windows(engine)
        ok = [i for i, x in enumerate(xs) if x is not None]
        self.n_windows_skipped += len(xs) - len(ok)
        if not ok:
            return self._done(None)
        ps = np.asarray(self.score_fn(np.stack([xs[i] for i in ok])), dtype=np.float64).reshape(-1)
        self.n_windows += len(ok)
        self.last_ps = [float(p) for p in ps]
        if len(self._starts) == len(xs):
            self.last_starts = [self._starts[i] for i in ok]
        j = int(np.argmax(ps))
        if self.mode == "max" and ok[j] != 0:
            self.n_max_later += 1
        return self._done(float(ps[j]) if self.mode == "max" else float(ps[0]))
