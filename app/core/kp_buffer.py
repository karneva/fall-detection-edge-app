"""
타임스탬프 기반 원시 키포인트 링버퍼 (표준 라이브러리만 사용)

- 프레임 수가 아니라 시간(horizon_s)으로 보관 범위를 정하므로 처리 FPS에 무관
- 사람이 없거나 품질이 낮은 프레임도 missing으로 기록해 시간축을 보존
- 사건 시점 기준 구간(예: ABNORMAL 진입 -1.5s ~ +2.5s)을 잘라내 학습/분석용으로 저장
- 원본 해상도(frame_wh)도 함께 보관해 정규화 좌표를 등방성 픽셀로 되돌릴 수 있게 함 (features v2)
"""
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

NUM_KPS = 17

Frame = Optional[List[List[float]]]  # 17 x [x, y, conf] 또는 None(missing)
FrameWH = Optional[Tuple[int, int]]  # (width, height) 픽셀, 모르면 None


def kps_to_frame(kps_list: Optional[List[Dict[str, Any]]]) -> Frame:
    """obs의 keypoints 목록을 17 x [x, y, conf] 고정 배열로 바꿉니다. 비어 있으면 None."""
    if not kps_list:
        return None
    frame = [[0.0, 0.0, 0.0] for _ in range(NUM_KPS)]
    for k in kps_list:
        kid = int(k.get("id", -1))
        if 0 <= kid < NUM_KPS:
            frame[kid] = [float(k.get("x", 0.0)), float(k.get("y", 0.0)), float(k.get("conf", 0.0))]
    return frame


def frame_wh_from_track(track: Optional[Dict[str, Any]]) -> FrameWH:
    """obs 트랙의 frame_shape (h, w[, c])를 (w, h)로 바꿉니다. 없거나 잘못된 값이면 None."""
    if not isinstance(track, dict):
        return None
    shape = track.get("frame_shape")
    try:
        h, w = int(shape[0]), int(shape[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (w, h)


def _normalize_wh(frame_wh: Any) -> FrameWH:
    """(w, h)를 양의 정수 튜플로 정리합니다. numpy 배열도 허용하고, 잘못된 값은 None (프레임 루프에서 예외 금지)."""
    if frame_wh is None:
        return None
    try:
        w, h = int(frame_wh[0]), int(frame_wh[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (w, h)


class TimedKeypointBuffer:
    def __init__(self, horizon_s: float = 20.0):
        self.horizon_s = float(horizon_s)
        # (ts, frame, bbox, frame_wh)
        self._buf: Deque[Tuple[float, Frame, Optional[List[float]], FrameWH]] = deque()

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self):
        self._buf.clear()

    def append(self, ts: float, frame: Frame, bbox: Optional[List[float]] = None, frame_wh: FrameWH = None):
        ts = float(ts)
        if self._buf and ts < self._buf[-1][0]:
            # 시간이 거꾸로 가면(리플레이 재시작 등) 이전 기록은 의미가 없음
            self._buf.clear()
        wh = _normalize_wh(frame_wh)
        self._buf.append((ts, frame, list(bbox) if bbox else None, wh))
        while self._buf and ts - self._buf[0][0] > self.horizon_s:
            self._buf.popleft()

    def window(self, t_start: float, t_end: float, pad_s: float = 0.0) -> Dict[str, Any]:
        """
        [t_start, t_end] 구간의 프레임을 반환합니다.
        pad_s: 앞뒤로 더 담을 여유 (초). 리샘플링이 구간 경계의 값을 보간하려면 경계 밖 샘플이 필요함.
               ts/kps/bbox에는 [t_start - pad_s, t_end + pad_s]가 담기고,
               n_frames/n_missing/fps/truncated 요약은 원래 구간 [t_start, t_end] 기준 (이벤트 요약 불변)
        frame_wh: 구간 안에서 마지막으로 기록된 원본 해상도 [w, h] (없으면 None)
        """
        pad = max(0.0, float(pad_s))
        picked = [(t, f, b) for t, f, b, _ in self._buf if t_start <= t <= t_end]
        context = [(t, f, b) for t, f, b, _ in self._buf if t_start - pad <= t <= t_end + pad]
        wh = None
        for t, _, _, w in self._buf:
            if t_start - pad <= t <= t_end + pad and w is not None:
                wh = w
        n = len(picked)
        span = (picked[-1][0] - picked[0][0]) if n >= 2 else 0.0
        return {
            "t_start": float(t_start),
            "t_end": float(t_end),
            "n_frames": n,
            "n_missing": sum(1 for _, f, _ in picked if f is None),
            "fps": round((n - 1) / span, 2) if span > 0 else None,
            # 요청 구간이 버퍼 범위를 벗어나 앞부분이 잘렸는지
            "truncated": bool(not self._buf or self._buf[0][0] > t_start),
            "pad_s": pad,
            "ts": [t for t, _, _ in context],
            "kps": [f for _, f, _ in context],
            "bbox": [b for _, _, b in context],
            "frame_wh": list(wh) if wh is not None else None,
        }


def window_summary(win: Dict[str, Any]) -> Dict[str, Any]:
    """이벤트에 넣을 가벼운 요약 (프레임 데이터 제외)."""
    return {k: win[k] for k in ("t_start", "t_end", "n_frames", "n_missing", "fps", "truncated")}
