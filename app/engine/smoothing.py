from typing import Dict, Any, Optional
import math
import logging

logger = logging.getLogger(__name__)

from ..config import (
    KP_EMA_ENABLE,          # EMA 스무딩 전체 활성화 여부
    KP_EMA_ALPHA,           # 기본 EMA 알파값 (폴백용)
    KP_MIN_CONF_FOR_SMOOTH, # 이 임계치보다 낮은 신뢰도는 업데이트 건너뜀 (홀드)
    KP_ALPHA_BY_ID,         # 관절별(Id) 특화 알파 테이블
    KP_ALPHA_DEFAULT,       # 기본 알파
    KP_CONF_HOLD,           # 정점 유지 신뢰도 구간
    KP_CONF_MID,            # 중간 속도 신뢰도 구간
    KP_ALPHA_MUL_LOW,       # 저신뢰도 알파 보정 계수
    KP_ALPHA_MUL_MID,       # 중신뢰도 알파 보정 계수
    KP_ALPHA_MUL_HIGH,      # 고신뢰도 알파 보정 계수
    KP_JUMP_ENABLE,         # 급격한 이동(점프) 감찰 활성화
    KP_JUMP_RATIO,          # 일반 관절 점프 허용 비율
    KP_JUMP_RATIO_EXT,      # 손/발 등 가동 범위가 큰 관절의 점프 허용 비율
    KP_JUMP_ALPHA_MUL,      # 점프 발생 시 알파값 강화 계수 (지연 반영)
    KP_JUMP_HOLD_IF_LOWCONF, # 낮은 신뢰도에서 점프 발생 시 이전 프레임 값 유지
)

# ----------------------------
# 관절 특정 및 임계 상수 정의
# ----------------------------
# COCO 포즈 기준: 손(9,10), 발(15,16)은 가동 범위가 매우 큼
JOINT_IDS_EXTENDED_MOVEMENT = {9, 10, 15, 16}
CONFIDENCE_JUMP_THRESHOLD = 0.5

# -------------------------------------------------
# 이전 프레임의 추론 상태를 보관하는 전역 메모리
# 단일 인물 추적을 전제로 하여 단순 큐가 아닌 딕셔너리로 관리
# -------------------------------------------------
_prev_kp_ema: Optional[Dict[int, Dict[str, float]]] = None
_prev_frame_shape: Optional[tuple] = None
_prev_person_id: Optional[int] = None

def _reset_ema_state() -> None:
    """EMA 필터의 내부 상태를 완전히 초기화합니다."""
    global _prev_kp_ema, _prev_frame_shape, _prev_person_id
    _prev_kp_ema = None
    _prev_frame_shape = None
    _prev_person_id = None


def _check_person_changed(track_id: Optional[int]) -> bool:
    """트래킹 ID 변화를 감지하여 추적 대상이 교체되었는지 확인합니다."""
    global _prev_person_id
    if track_id is None:
        return False
    return _prev_person_id is not None and _prev_person_id != track_id


def _bbox_diag(t0: Dict[str, Any]) -> float:
    """바운딩 박스의 대각선 길이를 계산하여 관절 정규화의 기준 척도로 삼습니다."""
    b = t0.get("bbox")
    if not b or len(b) != 4:
        return 1.0
    
    try:
        x1, y1, x2, y2 = map(float, b)
    except (ValueError, TypeError):
        return 1.0
    
    # 정규화 좌표가 아닌 픽셀 기반일 경우 방어 로직
    if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
        return 1.0
    
    return max(1e-6, math.hypot(x2 - x1, y2 - y1))


def _alpha_for(kid: int, conf: float) -> float:
    """
    관절 부위와 현재 신뢰도를 기반으로 최적의 EMA 알파값을 결정합니다.
    낮은 신뢰도일수록 이전 값을 더 많이 신뢰하도록 가중치를 조정합니다.
    """
    base = float(
        KP_ALPHA_BY_ID.get(
          kid,
          KP_ALPHA_DEFAULT if KP_ALPHA_DEFAULT is not None else KP_EMA_ALPHA
        )
    )

    if conf < KP_CONF_HOLD:
        alpha = base * KP_ALPHA_MUL_LOW
    elif conf < KP_CONF_MID:
        alpha = base * KP_ALPHA_MUL_MID
    else:
        alpha = base * KP_ALPHA_MUL_HIGH
    
    return min(0.99, max(0.0, alpha))


def ema_smooth_keypoints_inplace(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    관측 데이터(Observation) 내의 키포인트에 지수 이동 평균(EMA) 필터를 적용합니다.
    데이터 떨림(Jitter)을 제거하고 부드러운 이동 궤적을 생성합니다.
    """
    global _prev_kp_ema, _prev_frame_shape, _prev_person_id

    if not KP_EMA_ENABLE:
        return obs

    tracks = obs.get("tracks") or []
    if not tracks:
        _reset_ema_state()
        return obs

    t0 = tracks[0]
    if not t0.get("has_person", False):
        _reset_ema_state()
        return obs

    # 사람 교체 또는 해상도 변화 시 상태 리셋
    current_track_id = t0.get("track_id")
    if _check_person_changed(current_track_id):
        _reset_ema_state()
    _prev_person_id = current_track_id

    frame_shape = t0.get("frame_shape")
    if frame_shape is not None:
        fs = tuple(frame_shape)
        if _prev_frame_shape is not None and fs != _prev_frame_shape:
            _reset_ema_state()
        _prev_frame_shape = fs

    kps_raw = t0.get("keypoints_raw") or t0.get("keypoints") or []
    if not kps_raw:
        _reset_ema_state()
        t0["keypoints_smooth"] = []
        return obs

    # 데이터의 타당성 검사 및 정제
    curr: Dict[int, Dict[str, float]] = {}
    for kp in kps_raw:
        kid = int(kp.get("id", -1))
        if kid < 0: continue
        curr[kid] = {
            "x": float(kp.get("x", 0.0)),
            "y": float(kp.get("y", 0.0)),
            "conf": float(kp.get("conf", 0.0)),
        }

    # 첫 프레임 처리: 스무딩 결과를 원본과 동일하게 초기화
    if _prev_kp_ema is None:
        _prev_kp_ema = {kid: {"x": v["x"], "y": v["y"], "jump_cnt": 0} for kid, v in curr.items()}
        smooth_list = sorted([{"id": k, **v, "conf": curr[k]["conf"]} for k, v in _prev_kp_ema.items() if k in curr], key=lambda x: x["id"])
        # 불필요 필드 제거 후 반영
        for s in smooth_list: s.pop("jump_cnt", None)
        t0["keypoints_smooth"] = smooth_list
        t0["keypoints"] = smooth_list
        return obs

    diag = _bbox_diag(t0)
    out_list = []

    # 관절별 필터링 루프
    for kid in sorted(curr.keys()):
        c = curr[kid]
        conf = c["conf"]

        if kid not in _prev_kp_ema:
            _prev_kp_ema[kid] = {"x": c["x"], "y": c["y"], "jump_cnt": 0}

        prev = _prev_kp_ema[kid]
        dist = math.hypot(c["x"] - prev["x"], c["y"] - prev["y"])
        
        thr_ratio = KP_JUMP_RATIO_EXT if kid in JOINT_IDS_EXTENDED_MOVEMENT else KP_JUMP_RATIO
        is_jump = (dist > thr_ratio * diag)
        
        # 이상 이동이 지속되면 실제 이동으로 판단하도록 카운터 적용
        if is_jump: prev["jump_cnt"] += 1
        else: prev["jump_cnt"] = 0
            
        force_update = (prev["jump_cnt"] >= 4)

        # 낮은 신뢰도에서는 필터 업데이트 건너뜀 (값 고정)
        if conf < KP_MIN_CONF_FOR_SMOOTH and not force_update:
            out_list.append({"id": kid, "x": prev["x"], "y": prev["y"], "conf": conf})
            continue

        alpha = _alpha_for(kid, conf)

        # 점프 게이팅: 갑작스러운 불연속적 이동 시 반응 속도 늦춤
        if KP_JUMP_ENABLE and is_jump and not force_update:
            if KP_JUMP_HOLD_IF_LOWCONF and conf < CONFIDENCE_JUMP_THRESHOLD:
                out_list.append({"id": kid, "x": prev["x"], "y": prev["y"], "conf": conf})
                continue
            alpha = min(0.99, alpha * KP_JUMP_ALPHA_MUL)
        
        # EMA 공식 적용: Smooth = alpha * Prev + (1 - alpha) * Curr
        sx = alpha * prev["x"] + (1.0 - alpha) * c["x"]
        sy = alpha * prev["y"] + (1.0 - alpha) * c["y"]

        prev["x"], prev["y"] = sx, sy
        out_list.append({"id": kid, "x": sx, "y": sy, "conf": conf})

    t0["keypoints_smooth"] = out_list
    t0["keypoints"] = out_list
    if out_list:
        t0["quality_score"] = float(sum(k["conf"] for k in out_list) / len(out_list))

    return obs
