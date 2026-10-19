"""
회전 폴백: 사람을 놓친 프레임에서만 영상을 90/270도로 돌려 다시 검출합니다 (numpy, 영상 회전만 cv2).

누운 사람은 포즈 모델이 학습한 서 있는 자세와 방향이 달라 놓치기 쉽습니다. 영상을 90도 돌리면 누운 사람이
서 있는 모습이 됩니다. 원래 방향에서 아무도 못 찾았고, 직전(recent_s초 이내)에 대상을 보고 있었을 때만
회전 영상으로 다시 찾고, 결과를 원래 좌표로 되돌려 직전 대상 위치 근처의 가장 확실한 사람을 고릅니다.
회전 추론은 놓친 프레임에서만 하므로 사람이 잘 보일 때의 연산과 결과는 그대로입니다.

운영(app/engine/live.py)과 학습용 추출기(training/datasets/extract_keypoints.py)가 같은 함수를 씁니다.
회전은 거울 반전이 아니므로 관절 좌우는 바뀌지 않습니다.

좌표 규칙: 픽셀 (x, y), 원본 크기 (W, H). cv2.rotate 기준
  90  (ROTATE_90_CLOCKWISE):         원본 (x, y) -> 회전 (H - 1 - y, x)
  180 (ROTATE_180):                  원본 (x, y) -> 회전 (W - 1 - x, H - 1 - y)
  270 (ROTATE_90_COUNTERCLOCKWISE):  원본 (x, y) -> 회전 (y, W - 1 - x)
"""
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

ANGLES = (0, 90, 180, 270)
FALLBACK_ANGLES = (90, 270)
# 직전 대상을 본 뒤 이 시간 안에서만 폴백 (빈 방에서 매 프레임 회전 추론을 하지 않도록)
FALLBACK_RECENT_S = 5.0
# 직전 대상 상자 중심에서 (두 상자 중 큰 변 길이 x near_scale) 안의 검출만 같은 사람으로 봄
FALLBACK_NEAR_SCALE = 1.0

__all__ = ["ANGLES", "FALLBACK_ANGLES", "FALLBACK_RECENT_S", "FALLBACK_NEAR_SCALE", "rotated_size", "to_rotated",
           "from_rotated", "box_from_rotated", "rotate_image", "near", "fallback_detect", "Detection"]

# (상자 [x1,y1,x2,y2] 원본 픽셀, 상자 신뢰도, 관절 (17, 3) 원본 픽셀 x, y, conf, 회전 각도)
Detection = Tuple[np.ndarray, float, np.ndarray, int]


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


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    import cv2
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return img if angle == 0 else cv2.rotate(img, codes[angle])


def near(box: np.ndarray, ref: np.ndarray, scale: float = FALLBACK_NEAR_SCALE) -> bool:
    """box 중심이 ref 중심에서 (두 상자 중 가장 긴 변 x scale) 안이면 True"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    rx, ry = (ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2
    diag = max(ref[2] - ref[0], ref[3] - ref[1], box[2] - box[0], box[3] - box[1])
    return float(np.hypot(cx - rx, cy - ry)) <= float(scale) * float(diag)


def fallback_detect(predict: Callable[[List[np.ndarray]], Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
                    img: np.ndarray, last_seen_bbox: np.ndarray, angles: Sequence[int] = FALLBACK_ANGLES,
                    near_scale: float = FALLBACK_NEAR_SCALE) -> Optional[Detection]:
    """
    img를 angles로 돌려 predict(회전 영상 목록) -> 영상마다 (boxes (N,4), confs (N,), kps (N,17,3)) 픽셀 좌표.
    결과를 원본 좌표로 되돌려 last_seen_bbox(원본 픽셀) 근처의 검출 중 상자 신뢰도가 가장 높은 하나를 반환.
    근처에 없으면 None. predict의 신뢰도 하한(conf)은 원래 방향 검출과 같게 맞춰 부르는 쪽에서 정함.
    """
    h, w = img.shape[:2]
    ref = np.asarray(last_seen_bbox, dtype=np.float64)
    best: Optional[Detection] = None
    for a, (boxes, confs, kps) in zip(angles, predict([rotate_image(img, a) for a in angles])):
        for b, c, k in zip(boxes, confs, kps):
            box = box_from_rotated(b, w, h, a)
            if not near(box, ref, near_scale):
                continue
            if best is None or float(c) > best[1]:
                k = np.asarray(k, dtype=np.float64).copy()
                k[:, :2] = from_rotated(k[:, :2], w, h, a)
                best = (box, float(c), k, int(a))
    return best
