"""
회전 TTA (numpy만 사용, 영상 회전은 호출하는 쪽에서 cv2.rotate)

누운 사람은 포즈 모델이 학습한 서 있는 자세와 방향이 달라 놓치기 쉽습니다. 영상을 90도 돌리면 누운 사람이
서 있는 모습이 되므로, 네 방향으로 돌려 검출한 뒤 결과를 원래 영상 좌표로 되돌려 가장 확실한 것을 고릅니다.
회전은 거울 반전이 아니므로 관절 좌우(왼쪽 어깨/오른쪽 어깨)는 바뀌지 않습니다.

좌표 규칙: 픽셀 좌표 (x, y), 원본 영상 크기 (W, H). cv2.rotate 기준
  90  (ROTATE_90_CLOCKWISE):         원본 (x, y) -> 회전 (H - 1 - y, x),   회전 영상 크기 (H, W)
  180 (ROTATE_180):                  원본 (x, y) -> 회전 (W - 1 - x, H - 1 - y)
  270 (ROTATE_90_COUNTERCLOCKWISE):  원본 (x, y) -> 회전 (y, W - 1 - x),   회전 영상 크기 (H, W)
"""
from typing import Tuple

import numpy as np

ANGLES = (0, 90, 180, 270)

__all__ = ["ANGLES", "rotated_size", "to_rotated", "from_rotated", "box_from_rotated"]


def rotated_size(w: int, h: int, angle: int) -> Tuple[int, int]:
    """회전한 영상의 (폭, 높이)"""
    if angle in (90, 270):
        return h, w
    if angle in (0, 180):
        return w, h
    raise ValueError(f"angle must be one of {ANGLES}: {angle}")


def to_rotated(xy: np.ndarray, w: int, h: int, angle: int) -> np.ndarray:
    """원본 픽셀 좌표 (..., 2) -> angle만큼 돌린 영상의 좌표"""
    xy = np.asarray(xy, dtype=np.float64)
    x, y = xy[..., 0], xy[..., 1]
    if angle == 0:
        out = (x, y)
    elif angle == 90:
        out = (h - 1 - y, x)
    elif angle == 180:
        out = (w - 1 - x, h - 1 - y)
    elif angle == 270:
        out = (y, w - 1 - x)
    else:
        raise ValueError(f"angle must be one of {ANGLES}: {angle}")
    return np.stack(out, axis=-1)


def from_rotated(xy: np.ndarray, w: int, h: int, angle: int) -> np.ndarray:
    """angle만큼 돌린 영상의 픽셀 좌표 (..., 2) -> 원본 좌표 (w, h는 원본 크기)"""
    xy = np.asarray(xy, dtype=np.float64)
    xr, yr = xy[..., 0], xy[..., 1]
    if angle == 0:
        out = (xr, yr)
    elif angle == 90:
        out = (yr, h - 1 - xr)
    elif angle == 180:
        out = (w - 1 - xr, h - 1 - yr)
    elif angle == 270:
        out = (w - 1 - yr, xr)
    else:
        raise ValueError(f"angle must be one of {ANGLES}: {angle}")
    return np.stack(out, axis=-1)


def box_from_rotated(box: np.ndarray, w: int, h: int, angle: int) -> np.ndarray:
    """회전 영상의 상자 [x1, y1, x2, y2] -> 원본 좌표 상자 (네 모서리를 되돌린 뒤 최소/최대)"""
    x1, y1, x2, y2 = (float(v) for v in box)
    corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    back = from_rotated(corners, w, h, angle)
    return np.array([back[:, 0].min(), back[:, 1].min(), back[:, 0].max(), back[:, 1].max()])
