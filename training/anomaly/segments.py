"""
v2 창 (92, T) -> STG-NF 입력 포즈 세그먼트 (C=2, T=24, V=17) (numpy만 사용)

좌표
- 프레임별 절대 좌표 = rel_j + (hip_dx, hip_dy): 몸통 길이 단위, 원점 = 창의 첫 유효 골반 중점
  (features.build_features_v2의 rel/hip_dx/hip_dy 정의를 되돌린 값)
- 관절 유효 = mask_j 이고 골반 중점 유효 (골반이 없는 프레임은 rel이 모두 0이라 관절 좌표를 알 수 없음)
- 골반 유효 = mask_l_hip | mask_r_hip (features._mid_point와 같은 규칙)

세그먼트
- 길이 24스텝(1.6초 @15Hz), 간격 6스텝: 60스텝 창 -> 시작 0, 6, ..., 36 의 7개
- 세그먼트 안에서만 보간: 무효 관절은 같은 관절의 세그먼트 안 유효 시점으로 선형 보간 (양 끝 밖은 가장 가까운 값),
  세그먼트 안에 한 번도 없으면 그 시점의 골반 중점 좌표로 채움 (골반도 같은 방식으로 보간)
- 중심화: 보간한 골반 궤적의 세그먼트 평균을 뺌 (세그먼트 안의 하강 같은 움직임은 남김)
- 사용 여부: 관절 유효 비율(17관절 x 24스텝 평균) >= 0.5 인 세그먼트만 학습/점수에 씀
"""
from typing import Tuple

import numpy as np

from app.ai_classifier import features as _F

__all__ = ["SEG_LEN", "SEG_STRIDE", "MIN_SEG_VALID", "COCO_EDGES", "coco_adjacency", "segment_starts",
           "joint_coords", "window_segments", "fill_linear"]

SEG_LEN = 24          # 1.6초 @15Hz
SEG_STRIDE = 6        # 0.4초
MIN_SEG_VALID = 0.5   # 세그먼트 관절 유효 비율 하한

NUM_KPS = _F.NUM_KPS
_REL0 = _F.V2_CHANNELS.index("rel_nose_x")      # 0
_HIP_DX = _F.V2_CHANNELS.index("hip_dx")        # 68
_HIP_DY = _F.V2_CHANNELS.index("hip_dy")        # 69
_MASK0 = _F.V2_CHANNELS.index("mask_nose")      # 75

# COCO-17 골격: COCO person_keypoints의 skeleton 19개 (0부터 센 번호)
# 머리와 몸통은 귀-어깨 (3, 5), (4, 6)로 이어짐. 이 연결이 없으면 그래프가 머리 {0..4}와 몸 {5..16}으로 나뉘어
# 결합층이 두 부분을 섞지 못함 (밀도가 p(머리) x p(몸)으로 분리됨)
COCO_EDGES = (
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),         # 얼굴
    (3, 5), (4, 6),                                 # 귀-어깨 (머리와 몸통 연결)
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),        # 어깨, 팔
    (5, 11), (6, 12), (11, 12),                     # 몸통
    (11, 13), (13, 15), (12, 14), (14, 16),         # 다리
)


def coco_adjacency(kind: str = "skeleton") -> np.ndarray:
    """
    (17, 17) 정규화 인접 행렬.
    skeleton: A + I 의 대칭 정규화 D^-1/2 (A + I) D^-1/2 (ST-GCN 'uniform' 1분할)
    full    : 모든 관절이 같은 비중 1/17 (논문 보충 자료의 'unit adjacency', 비교용)
    """
    if kind == "full":
        return np.full((NUM_KPS, NUM_KPS), 1.0 / NUM_KPS)
    if kind != "skeleton":
        raise ValueError(f"알 수 없는 인접 행렬: {kind}")
    a = np.eye(NUM_KPS)
    for i, j in COCO_EDGES:
        a[i, j] = a[j, i] = 1.0
    d = 1.0 / np.sqrt(a.sum(axis=1))
    return a * d[:, None] * d[None, :]


def segment_starts(t: int, seg_len: int = SEG_LEN, stride: int = SEG_STRIDE) -> np.ndarray:
    """길이 t 창의 세그먼트 시작 스텝 (t < seg_len이면 빈 배열)"""
    if t < seg_len:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, t - seg_len + 1, stride, dtype=np.int64)


def joint_coords(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    v2 창 (N, 92, T) -> 관절 절대 좌표 (N, 2, T, 17), 관절 유효 (N, T, 17) bool,
    골반 중점 좌표 (N, 2, T), 골반 유효 (N, T) bool. 좌표는 몸통 길이 단위, 원점 = 창의 첫 유효 골반.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[None]
    n, c, t = X.shape
    if c != len(_F.V2_CHANNELS):
        raise ValueError(f"channels {c} != {len(_F.V2_CHANNELS)}")
    rel = X[:, _REL0:_REL0 + 2 * NUM_KPS].reshape(n, NUM_KPS, 2, t).transpose(0, 2, 3, 1)  # (N, 2, T, 17)
    hip = X[:, [_HIP_DX, _HIP_DY]]                                                       # (N, 2, T)
    mask = X[:, _MASK0:_MASK0 + NUM_KPS] > 0.5                                           # (N, 17, T)
    hip_ok = mask[:, 11] | mask[:, 12]                                                   # (N, T)
    valid = (mask & hip_ok[:, None, :]).transpose(0, 2, 1)                               # (N, T, 17)
    coords = rel + hip[:, :, :, None]
    return coords, valid, hip, hip_ok


def fill_linear(vals: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    마지막 축(시간)을 따라 무효 값을 유효 값으로 선형 보간합니다. 양 끝 밖은 가장 가까운 유효 값.
    vals: (..., T), valid: (..., T) bool (vals와 같은 모양 또는 브로드캐스트 가능)
    반환: (filled (..., T), any_valid (...) bool). 유효 값이 없는 줄은 0.
    """
    vals = np.asarray(vals, dtype=np.float64)
    valid = np.broadcast_to(np.asarray(valid, dtype=bool), vals.shape)
    t = vals.shape[-1]
    ar = np.arange(t)
    prev = np.maximum.accumulate(np.where(valid, ar, -1), axis=-1)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(valid, ar, t), axis=-1), axis=-1), axis=-1)
    has_p, has_n = prev >= 0, nxt < t
    pc, nc = np.clip(prev, 0, t - 1), np.clip(nxt, 0, t - 1)
    vp = np.take_along_axis(vals, pc, axis=-1)
    vn = np.take_along_axis(vals, nc, axis=-1)
    span = np.maximum(nc - pc, 1)
    lin = vp + (vn - vp) * (ar - pc) / span
    out = np.where(has_p & has_n, lin, np.where(has_p, vp, np.where(has_n, vn, 0.0)))
    return out, valid.any(axis=-1)


def window_segments(X: np.ndarray, seg_len: int = SEG_LEN, stride: int = SEG_STRIDE,
                    min_valid: float = MIN_SEG_VALID) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    v2 창 (N, 92, T) -> 세그먼트 (N, S, 2, seg_len, 17) float32, 사용 여부 (N, S) bool, 유효 비율 (N, S) float.
    S = len(segment_starts(T)). 사용하지 않는 세그먼트도 값은 채워 둡니다 (호출 쪽에서 keep으로 거름).
    """
    coords, valid, hip, hip_ok = joint_coords(X)
    n, _, t, v = coords.shape
    starts = segment_starts(t, seg_len, stride)
    s = len(starts)
    if s == 0 or n == 0:
        return (np.zeros((n, s, 2, seg_len, v), dtype=np.float32), np.zeros((n, s), dtype=bool),
                np.zeros((n, s), dtype=np.float64))
    idx = starts[:, None] + np.arange(seg_len)[None, :]                 # (S, L)
    c_seg = coords[:, :, idx, :]                                        # (N, 2, S, L, 17)
    v_seg = valid[:, idx, :]                                            # (N, S, L, 17)
    h_seg = hip[:, :, idx]                                              # (N, 2, S, L)
    ho_seg = hip_ok[:, idx]                                             # (N, S, L)
    frac = v_seg.mean(axis=(2, 3))                                      # (N, S)

    # 골반 궤적 보간 -> 세그먼트 평균
    hip_f, _ = fill_linear(h_seg, ho_seg[:, None])                      # (N, 2, S, L)
    center = hip_f.mean(axis=-1, keepdims=True)                         # (N, 2, S, 1)

    # 관절: 시간축을 마지막으로 옮겨 보간 (N, 2, S, 17, L)
    cj = c_seg.transpose(0, 1, 2, 4, 3)
    vj = v_seg.transpose(0, 1, 3, 2)[:, None]                           # (N, 1, S, 17, L)
    filled, any_v = fill_linear(cj, vj)                                 # any_v (N, 2, S, 17)
    hip_b = np.broadcast_to(hip_f[:, :, :, None, :], filled.shape)
    filled = np.where(any_v[..., None], filled, hip_b)
    filled = filled - center[:, :, :, None, :]                          # 골반 평균 중심화
    seg = filled.transpose(0, 2, 1, 4, 3)                               # (N, S, 2, L, 17)
    keep = frac >= min_valid - 1e-9
    return np.ascontiguousarray(seg, dtype=np.float32), keep, frac
