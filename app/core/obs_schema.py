from typing import Dict, Any, List, Optional
from ..config import DEFAULT_CONF

def clamp01(x: float) -> float:
    """값을 0.0과 1.0 사이의 범위로 제한합니다."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def build_observation(
    r0,
    frame_w: int,
    frame_h: int,
    ts: float,
    frame_idx: int,
    source_id: str = "cam0",
    target_switched: bool = False,
) -> Dict[str, Any]:
    """
    YOLO 추론 결과를 표준 관측(Observation) 데이터 구조로 변환합니다.
    
    Args:
        r0: YOLO 추론 결과 객체
        frame_w, frame_h: 현재 원본 프레임의 해상도
        ts: 프레임의 타임스탬프
        frame_idx: 프레임 번호
        source_id: 카메라 식별자
        
    Returns:
        표준화된 관측 데이터 딕셔너리
    """
    obs: Dict[str, Any] = {
        "schema_version": "1.0",
        "ts": float(ts),
        "frame_index": int(frame_idx),
        "source_id": source_id,
        "tracks": [],
        "target_switched": target_switched,
        "meta": {
            "conf_thres": float(DEFAULT_CONF),
        },
    }

    has_person = False
    bbox_norm: Optional[List[float]] = None
    box_conf = 0.0
    kps: List[Dict[str, Any]] = []

    # 1. 바운딩 박스 정보 추출 (가장 높은 신뢰도 1개 대상)
    if r0.boxes is not None and len(r0.boxes) > 0:
        b = r0.boxes.xyxy[0].cpu().numpy()
        box_conf = float(r0.boxes.conf[0].item())
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        # 좌표 정규화 (0.0 ~ 1.0)
        bbox_norm = [
            clamp01(x1 / frame_w),
            clamp01(y1 / frame_h),
            clamp01(x2 / frame_w),
            clamp01(y2 / frame_h),
        ]

    # 2. 키포인트(포즈) 정보 추출
    if r0.keypoints is not None and len(r0.keypoints) > 0:
        has_person = True
        xy = r0.keypoints.xy[0].cpu().numpy()
        cf = r0.keypoints.conf[0].cpu().numpy()
        for i in range(xy.shape[0]):
            kps.append(
                {
                    "id": int(i),
                    "x": clamp01(float(xy[i, 0]) / frame_w),
                    "y": clamp01(float(xy[i, 1]) / frame_h),
                    "conf": float(cf[i]),
                }
            )

    # 3. 객체의 전체적인 품질(신뢰도) 점수 계산
    quality = 0.0
    if kps:
        quality = float(sum(k["conf"] for k in kps) / len(kps))

    # 트랙 목록에 추가 (현재는 단일 객체 추적 기준)
    obs["tracks"].append(
        {
            "track_id": 0,
            "has_person": bool(has_person),
            "bbox": bbox_norm,
            "conf": float(box_conf),
            "keypoints_raw": kps,        # 원본 키포인트
            "keypoints_smooth": None,    # 스무딩 처리 후 채워질 필드
            "keypoints": kps,            # 기존 코드와의 호환성을 위한 기본 필드
            "quality_score": float(quality),
            "frame_shape": (frame_h, frame_w),  # 분석에 사용된 해상도 정보
        }
    )
    return obs
