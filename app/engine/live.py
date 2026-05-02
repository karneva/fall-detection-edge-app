import time
from typing import Optional, Dict, Any, Tuple

import cv2
import torch
from ultralytics import YOLO

from ..config import FRAME_W, FRAME_H, DEFAULT_CONF, USE_HALF
from ..core import build_observation
from .smoothing import ema_smooth_keypoints_inplace

# COCO 17개 키포인트 연결(엣지) 정의
COCO17_EDGES = [
    (0, 1), (0, 2),       # 코 -> 눈
    (1, 3), (2, 4),       # 눈 -> 귀
    (5, 6),               # 어깨 사이
    (5, 7), (7, 9),       # 왼쪽 팔 (어깨-팔꿈치-손목)
    (6, 8), (8, 10),      # 오른쪽 팔
    (5, 11), (6, 12),     # 어깨 -> 골반
    (11, 12),             # 골반 사이
    (11, 13), (13, 15),   # 왼쪽 다리 (골반-무릎-발목)
    (12, 14), (14, 16),   # 오른쪽 다리
]

def draw_bbox_norm(frame, bbox_norm, color=(0, 255, 0), thickness=2):
    """정규화된 좌표(0~1)를 바탕으로 프레임에 바운딩 박스를 그립니다."""
    if not bbox_norm:
        return
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_norm
    p1 = (int(x1 * w), int(y1 * h))
    p2 = (int(x2 * w), int(y2 * h))
    cv2.rectangle(frame, p1, p2, color, thickness)

def draw_skeleton_norm(frame, keypoints, conf_thr=0.3, edges=COCO17_EDGES,
                       color=(0, 255, 0), pt_radius=3, line_thickness=2):
    """정규화된 키포인트 좌표를 바탕으로 스켈레톤(관절)을 그립니다."""
    if not keypoints:
        return
    h, w = frame.shape[:2]
    pts = {}
    for kp in keypoints:
        c = float(kp.get("conf", 0.0))
        if c < conf_thr:
            continue
        i = int(kp["id"])
        px = int(float(kp["x"]) * w)
        py = int(float(kp["y"]) * h)
        pts[i] = (px, py)
        cv2.circle(frame, (px, py), pt_radius, color, -1)

    for a, b in edges:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, line_thickness)

def keep_top1_person(r):
    """추론 결과 중 신뢰도가 가장 높은 사람 한 명의 결과만 남깁니다."""
    if r.boxes is None or len(r.boxes) == 0:
        return r
    top = int(torch.argmax(r.boxes.conf).item())
    r.boxes = r.boxes[top:top + 1]
    if getattr(r, "keypoints", None) is not None and r.keypoints is not None:
        r.keypoints = r.keypoints[top:top + 1]
    return r

def calculate_iou(box1, box2):
    """두 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def select_best_person(results, last_bbox=None, iou_thresh=0.3):
    """
    이전 프레임의 위치 정보를 고려하여 추적할 최적의 대상을 선택합니다.
    Returns: (results, is_switched)
    """
    if results.boxes is None or len(results.boxes) == 0:
        return results, False

    best_idx = 0
    is_switched = False
    
    if last_bbox is not None:
        best_iou = -1.0
        boxes = results.boxes.xyxy.cpu().numpy()
        
        # 1. IoU 기반 가장 가까운 후보 찾기
        match_idx = -1
        for i, box in enumerate(boxes):
            iou = calculate_iou(last_bbox, box)
            if iou > best_iou:
                best_iou = iou
                match_idx = i
        
        # 2. 최고 신뢰도 후보 찾기
        top_conf_idx = int(torch.argmax(results.boxes.conf).item())
        
        # 3. 교체 결정 (IoU가 낮으면 스위칭 발생)
        if match_idx != -1 and best_iou >= iou_thresh:
            best_idx = match_idx
        else:
            best_idx = top_conf_idx
            is_switched = True
    else:
        best_idx = int(torch.argmax(results.boxes.conf).item())
        
    results.boxes = results.boxes[best_idx:best_idx + 1]
    if getattr(results, "keypoints", None) is not None and results.keypoints is not None:
        results.keypoints = results.keypoints[best_idx:best_idx + 1]
        
    return results, is_switched

def encode_jpeg(img, jpeg_quality: int) -> Optional[bytes]:
    """이미지 배열을 JPEG 데이터로 인코딩합니다."""
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return None
    return jpg.tobytes()

class LivePipeline:
    """
    실시간 영상 처리 파이프라인 클래스입니다.
    카메라 캡처 -> YOLO 추론 -> IoU 추적 -> EMA 스무딩 -> 시각화 단계를 수행합니다.
    """
    def __init__(self, model: YOLO, cap: cv2.VideoCapture, jpeg_quality: int, source_id: str = "cam0"):
        self.model = model
        self.cap = cap
        self.jpeg_quality = jpeg_quality
        self.source_id = source_id
        self.frame_index = 0
        
        # 실제 카메라 입력 해상도 정보를 저장 (첫 프레임 감지)
        self.actual_w = FRAME_W
        self.actual_h = FRAME_H
        self._first_frame = True
        
        # 객체 추적을 위한 이전 프레임 바운딩 박스 보관
        self.last_bbox = None

    def step(self, overlay: str = "smooth") -> Tuple[Optional[Dict[str, Any]], Optional[bytes], Optional[Any]]:
        """
        한 프레임을 읽어와 분석을 수행하고 결과를 반환합니다.
        
        Args:
            overlay: 시각화 옵션 ("raw" | "smooth" | "both")
            
        Returns:
            (관측데이터, 시각화된JPEG바이너리, 원본BGR프레임) 튜플
        """
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None
        
        # 실제 입력 해상도 동적 감지
        frame_h, frame_w = frame.shape[:2]
        if self._first_frame or frame_w != self.actual_w or frame_h != self.actual_h:
            self.actual_h = frame_h
            self.actual_w = frame_w
            self._first_frame = False

        # YOLO 모델을 이용한 객체 탐지 및 포즈 추정
        results = self.model.predict(
            frame,
            device=0,
            half=USE_HALF,
            conf=DEFAULT_CONF,
            verbose=False,
        )
        
        # 단일 대상 추적 (IoU 기반 대상 고정)
        r0, is_switched = select_best_person(results[0], last_bbox=self.last_bbox)
        
        # 다음 프레임 추적을 위한 위치 업데이트
        if r0.boxes is not None and len(r0.boxes) > 0:
            self.last_bbox = r0.boxes.xyxy[0].cpu().numpy()
        else:
            self.last_bbox = None

        ts = time.time()
        # 공통 데이터 구조인 Observation 생성 및 EMA 필터링 적용
        obs = build_observation(r0, self.actual_w, self.actual_h, ts, self.frame_index, 
                                source_id=self.source_id, target_switched=is_switched)
        obs = ema_smooth_keypoints_inplace(obs)

        self.frame_index += 1

        # 시각화 처리 (스켈레톤 및 박스 그리기)
        t0 = (obs.get("tracks") or [{}])[0]
        bbox_norm = t0.get("bbox")
        kps_raw = t0.get("keypoints_raw") or []
        kps_smooth = t0.get("keypoints_smooth") or t0.get("keypoints") or []

        annotated = frame.copy()
        draw_bbox_norm(annotated, bbox_norm)

        ov = (overlay or "smooth").lower().strip()
        if ov == "raw":
            draw_skeleton_norm(annotated, kps_raw, conf_thr=0.3, color=(0, 0, 255))
        elif ov == "both":
            draw_skeleton_norm(annotated, kps_raw, conf_thr=0.3, color=(0, 0, 255))
            draw_skeleton_norm(annotated, kps_smooth, conf_thr=0.3, color=(0, 255, 0))
        else:
            # 기본값: 스무딩된 스켈레톤 출력 (초록색)
            draw_skeleton_norm(annotated, kps_smooth, conf_thr=0.3, color=(0, 255, 0))

        jpg = encode_jpeg(annotated, self.jpeg_quality)
        if jpg is None:
            return None, None, None

        return obs, jpg, frame
