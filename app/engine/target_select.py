"""
단일 추적 대상 선택 (표준 라이브러리만 사용)

런타임(app/engine/live.py)과 학습용 키포인트 추출기(training/datasets/extract_keypoints.py)가
같은 규칙으로 사람을 고르도록 선택 로직을 한 곳에 둡니다. 규칙이 달라지면 학습 데이터와 운영 입력의
분포가 어긋나므로 이 파일만 고치고 양쪽이 함께 따라가게 합니다.

규칙
- 후보가 없으면 (None, False)
- 이전 대상 박스가 없으면 신뢰도 최고 후보 (교체 아님)
- 이전 박스와 IoU가 가장 큰 후보의 IoU가 iou_thresh 이상이면 그 후보를 유지
- 아니면 신뢰도 최고 후보로 교체 (switched=True)
  단 policy="multi"이고 후보가 1명뿐이면 같은 사람으로 봄 (switched=False, 로드맵 R2)
- 동점이면 앞쪽 인덱스 (torch.argmax와 같음)

정책(policy)은 고르는 인덱스는 바꾸지 않고 '교체' 표시만 바꿉니다. 기본값은 "all"(기존 규칙)이고 런타임은
app.config.TARGET_SWITCH_POLICY를 넘깁니다. 학습용 추출기는 "all"로 기록하고(클립의 switched), 오프라인 재생이
n_cand로 정책을 적용합니다 (training/experiments/replay.effective_switched와 같은 결과).

박스/신뢰도 원소는 float 리스트나 numpy 배열 행 모두 받습니다. 런타임은 numpy(float32) 값을 그대로
넘겨 기존 계산과 비트 단위로 같은 결과를 냅니다.
"""
from typing import Optional, Sequence, Tuple


def calculate_iou(box1, box2) -> float:
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


def _argmax_first(values: Sequence) -> int:
    """최댓값의 첫 인덱스 (torch.argmax와 같은 동점 처리)."""
    best_i = 0
    best_v = values[0]
    for i in range(1, len(values)):
        if values[i] > best_v:
            best_v = values[i]
            best_i = i
    return best_i


def select_target_index(
    boxes_xyxy: Sequence,
    confs: Sequence,
    last_bbox=None,
    iou_thresh: float = 0.3,
    policy: str = "all",
) -> Tuple[Optional[int], bool]:
    """
    이전 프레임의 대상 위치를 고려해 추적할 후보의 인덱스를 고릅니다.

    Args:
        boxes_xyxy: 후보 박스 [[x1, y1, x2, y2], ...] (픽셀, last_bbox와 같은 좌표계)
        confs: 후보별 박스 신뢰도
        last_bbox: 이전 프레임에서 선택한 박스 (없으면 None)
        iou_thresh: 같은 대상으로 볼 최소 IoU
        policy: "all" = IoU가 낮으면 항상 교체 (기존), "multi" = 후보가 2명 이상일 때만 교체로 봄

    Returns:
        (선택 인덱스 또는 None, 대상 교체 여부)
    """
    if policy not in ("all", "multi"):
        raise ValueError(f"policy must be 'all' or 'multi': {policy!r}")
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None, False

    if last_bbox is None:
        return _argmax_first(confs), False

    # 1. IoU 기반 가장 가까운 후보 찾기
    best_iou = -1.0
    match_idx = -1
    for i, box in enumerate(boxes_xyxy):
        iou = calculate_iou(last_bbox, box)
        if iou > best_iou:
            best_iou = iou
            match_idx = i

    # 2. 교체 결정 (IoU가 낮으면 신뢰도 최고 후보로 스위칭)
    if match_idx != -1 and best_iou >= iou_thresh:
        return match_idx, False
    # 3. 후보가 1명뿐이면 다른 사람이 나타났다고 볼 근거가 없음 (자세가 크게 바뀐 같은 사람, multi 정책)
    if policy == "multi" and len(boxes_xyxy) == 1:
        return 0, False
    return _argmax_first(confs), True
