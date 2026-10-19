"""
교사 모델 + 회전 TTA로 프레임 의사 라벨 만들기 (ultralytics, cv2 필요)

한 프레임을 네 방향(0/90/180/270도)으로 돌려 교사 모델(기본 yolov8x-pose)에 넣고, 검출을 원래 좌표로 되돌린 뒤
같은 사람을 가리키는 검출(원래 좌표 상자 IoU >= 0.5)끼리 묶어 사람마다 가장 확실한 방향의 결과를 고릅니다.
'확실함' = 상자 신뢰도 x 관절 신뢰도 평균. 결과는 원본 픽셀 좌표의 사람 목록.

품질 필터 (keep_person): 상자 신뢰도 >= box_th, 관절 신뢰도 평균 >= kp_th, 보이는 관절 수 >= min_vis
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from training.pose.rotation import ANGLES, box_from_rotated, from_rotated

__all__ = ["Person", "iou", "io_min", "detect_rotations", "merge_rotations", "keep_person", "TeacherTTA"]

Person = Dict[str, Any]   # {"box": (4,) 원본 픽셀, "conf": float, "kps": (17, 3) 원본 픽셀 x, y, conf, "angle": int}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """상자 [x1, y1, x2, y2] 두 개의 IoU"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def io_min(a: np.ndarray, b: np.ndarray) -> float:
    """교집합 / 작은 상자 넓이 (한 상자가 다른 상자 안에 거의 들어가면 1에 가까움)"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / small) if small > 0 else 0.0


def _score(p: Person) -> float:
    return float(p["conf"]) * float(np.mean(p["kps"][:, 2]))


def detect_rotations(predict, img: np.ndarray, angles: Sequence[int] = ANGLES) -> List[Person]:
    """
    predict(images) -> 이미지마다 (boxes (N,4) 픽셀, confs (N,), kps (N,17,3) 픽셀) 목록.
    img를 네 방향으로 돌려 한 번에 예측하고, 모든 검출을 원본 좌표로 되돌려 반환합니다.
    """
    import cv2
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    h, w = img.shape[:2]
    imgs = [img if a == 0 else cv2.rotate(img, rot[a]) for a in angles]
    out: List[Person] = []
    for a, (boxes, confs, kps) in zip(angles, predict(imgs)):
        for b, c, k in zip(boxes, confs, kps):
            k = np.asarray(k, dtype=np.float64).copy()
            k[:, :2] = from_rotated(k[:, :2], w, h, a)
            out.append({"box": box_from_rotated(b, w, h, a), "conf": float(c), "kps": k, "angle": int(a)})
    return out


def merge_rotations(dets: List[Person], iou_th: float = 0.5, contain_th: float = 0.7) -> List[Person]:
    """
    같은 사람 검출을 묶어 사람마다 점수가 가장 높은 하나만 남김 (점수순).
    같은 사람 = 원본 좌표 상자 IoU >= iou_th, 또는 한 상자가 다른 상자 안에 거의 들어감 (교집합/작은 상자 >= contain_th:
    회전한 방향에서 같은 사람의 상반신만 따로 잡히는 경우)
    """
    dets = sorted(dets, key=_score, reverse=True)
    kept: List[Person] = []
    for d in dets:
        if all(iou(d["box"], k["box"]) < iou_th and io_min(d["box"], k["box"]) < contain_th for k in kept):
            kept.append(d)
    return kept


def keep_person(p: Person, box_th: float = 0.5, kp_th: float = 0.5, min_vis: int = 10,
                vis_th: float = 0.5) -> bool:
    """의사 라벨로 쓸 만큼 확실한 사람인지"""
    k = p["kps"]
    return (float(p["conf"]) >= box_th and float(np.mean(k[:, 2])) >= kp_th
            and int(np.sum(k[:, 2] >= vis_th)) >= min_vis)


class TeacherTTA:
    """교사 모델 래퍼: __call__(img BGR) -> 회전 TTA로 합친 사람 목록 (원본 픽셀 좌표)"""

    def __init__(self, model_path: str, device: Optional[str] = "0", conf: float = 0.25, imgsz: int = 640,
                 angles: Sequence[int] = ANGLES):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.device = int(device) if isinstance(device, str) and device.isdigit() else device
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.angles = tuple(angles)

    def predict(self, imgs: List[np.ndarray]):
        res = self.model.predict(imgs, device=self.device, conf=self.conf, imgsz=self.imgsz, verbose=False)
        out = []
        for r in res:
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                out.append((np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))))
                continue
            kxy = r.keypoints.xy.cpu().numpy()
            kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
            out.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                        np.concatenate([kxy, kc[..., None]], axis=-1)))
        return out

    def __call__(self, img: np.ndarray) -> List[Person]:
        return merge_rotations(detect_rotations(self.predict, img, self.angles))
