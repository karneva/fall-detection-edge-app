"""
1D CNN 입력 특징 (학습과 운영이 공유하는 단일 구현, numpy만 사용)

v1: 프레임마다 골반 중심을 원점으로 하는 상대좌표 (기존 학습 가중치와 비트 단위 동일)
    - 이전에는 dataset_builder.py, video_to_data.py, inference.py 세 곳에 같은 코드가 복제되어 있었음
    - tests/test_features_parity.py가 기존 구현과의 np.array_equal 동일성을 고정
v2: 타임스탬프 기준 15Hz 리샘플링(0.2초 이하 공백만 보간, 나머지는 마스크), 등방 픽셀 + 몸통 길이 스케일 정규화,
    초당 속도, 전역 골반 궤적, 몸통 각도, bbox 종횡비, conf 마스크 (92채널, V2_CHANNELS 순서)
"""
import numpy as np

FEATURE_VERSION_V1 = "v1"

NUM_KPS = 17
L_HIP, R_HIP, NOSE = 11, 12, 0


def normalize_frame_v1(kps: np.ndarray) -> np.ndarray:
    """
    골반(Hip Center)을 원점으로 하는 상대 좌표계로 변환합니다.
    kps: (17, 2) [x, y] 정규화 좌표. 반환: (34,)
    """
    l_hip = kps[L_HIP]
    r_hip = kps[R_HIP]

    # 두 골반의 신뢰도가 0일 경우(미검출)를 대비해 평균값 계산
    if np.any(l_hip) and np.any(r_hip):
        origin = (l_hip + r_hip) / 2
    elif np.any(l_hip):
        origin = l_hip
    elif np.any(r_hip):
        origin = r_hip
    else:
        # 골반이 없으면 Nose, 그것도 없으면 화면 중앙(0.5, 0.5)
        origin = kps[NOSE] if np.any(kps[NOSE]) else np.array([0.5, 0.5])

    return (kps - origin).flatten()


def normalize_sequence_v1(sequence) -> np.ndarray:
    """
    시퀀스의 각 프레임에 골반 중심 정규화를 적용합니다.
    sequence: (T, 34) 배열. 반환: (T, 34) float32
    """
    return np.array([normalize_frame_v1(frame.reshape(NUM_KPS, 2)) for frame in sequence], dtype=np.float32)


# ---------------------------------------------------------------------------
# v2
# ---------------------------------------------------------------------------

FEATURE_VERSION_V2 = "v2"
TARGET_HZ = 15.0
WINDOW_S = 4.0
MIN_CONF = 0.3
MAX_GAP_S = 0.2

# 부동소수점 타임스탬프 비교 허용 오차 (초). 예: 36/30 - 30/30 = 0.2000000000000002
_TS_EPS = 1e-6

COCO_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
# 좌우 대칭 관절 쌍 (1-2, 3-4, ..., 15-16)
COCO_FLIP_PAIRS = [(i, i + 1) for i in range(1, NUM_KPS, 2)]
L_SHOULDER, R_SHOULDER = 5, 6

# bbox 종횡비 상한 (bbox 폭이 거의 0인 프레임에서 값이 폭주하지 않도록)
_MAX_ASPECT = 10.0


def _v2_channel_names():
    names = []
    names += [f"rel_{n}_{a}" for n in COCO_NAMES for a in ("x", "y")]   # 34
    names += [f"vel_{n}_{a}" for n in COCO_NAMES for a in ("x", "y")]   # 34
    names += ["hip_dx", "hip_dy", "hip_vx", "hip_vy"]                   # 4
    names += ["torso_angle", "torso_angvel"]                            # 2
    names += ["bbox_aspect"]                                            # 1
    names += [f"mask_{n}" for n in COCO_NAMES]                          # 17
    return names


V2_CHANNELS = _v2_channel_names()
assert len(V2_CHANNELS) == 92


def _interp_valid(ts, values, valid, grid, max_gap_s):
    """
    valid 샘플만 사용해 grid 시각에 선형 보간합니다.
    grid 점이 샘플과 (허용 오차 내에서) 일치하거나, 양옆 valid 샘플 간격이 max_gap_s 이하일 때만 유효.
    ts: (T,) 오름차순, values: (T, D), valid: (T,) bool, grid: (N,)
    반환: (out (N, D) float64 [무효는 0], ok (N,) bool)
    """
    n_grid = len(grid)
    out = np.zeros((n_grid, values.shape[1]), dtype=np.float64)
    ok = np.zeros(n_grid, dtype=bool)
    if n_grid == 0 or not np.any(valid):
        return out, ok

    tv, first = np.unique(ts[valid], return_index=True)   # 중복 타임스탬프는 첫 샘플 사용
    vv = values[valid][first]
    n = len(tv)

    right = np.searchsorted(tv, grid, side="right")       # tv[right-1] <= g < tv[right]
    left = right - 1
    has_l = left >= 0
    has_r = right < n
    lc = np.clip(left, 0, n - 1)
    rc = np.clip(right, 0, n - 1)
    exact = (has_l & (grid - tv[lc] <= _TS_EPS)) | (has_r & (tv[rc] - grid <= _TS_EPS))
    bracket = has_l & has_r & (tv[rc] - tv[lc] <= max_gap_s + _TS_EPS)
    ok = exact | bracket
    if not np.any(ok):
        return out, ok

    for d in range(values.shape[1]):
        out[ok, d] = np.interp(grid[ok], tv, vv[:, d])
    return out, ok


def _make_grid(ts, hz, t_start, t_end):
    if (t_start is None) != (t_end is None):
        raise ValueError("t_start와 t_end는 함께 지정해야 합니다")
    if t_start is None:
        # 클립 전체: 유한한 첫 시각부터 마지막 시각까지 (끝 포함).
        # NaN/inf 타임스탬프 하나 때문에 클립 전체가 사라지지 않도록 유한한 값만으로 범위를 정합니다.
        tf = ts[np.isfinite(ts)]
        if len(tf) == 0:
            return np.zeros(0, dtype=np.float64)
        t0, t1 = float(tf.min()), float(tf.max())
        n = int(np.floor((t1 - t0) * hz + _TS_EPS)) + 1
        return t0 + np.arange(max(n, 0), dtype=np.float64) / hz
    # 창: [t_start, t_end) 고정 길이
    n = int(round((float(t_end) - float(t_start)) * hz))
    return float(t_start) + np.arange(max(n, 0), dtype=np.float64) / hz


def resample_clip(ts, kps, bbox, present, hz=TARGET_HZ, max_gap_s=MAX_GAP_S, min_conf=MIN_CONF,
                  *, t_start=None, t_end=None):
    """
    가변 fps 키포인트를 타임스탬프 기준 고정 hz 격자로 리샘플링합니다.

    ts: (T,) 초, kps: (T, 17, 3) [x_norm, y_norm, conf], bbox: (T, 4) 정규화 xyxy (None이면 0),
    present: (T,) bool (None이면 모두 True)
    관절별로 conf >= min_conf 이고 present인 샘플만 써서 선형 보간하며, 격자 점의 양옆 유효 샘플 간격이
    max_gap_s보다 크거나 유효 범위 밖이면 mask 0, 값 0 입니다. (샘플과 일치하는 격자 점은 그 샘플 값)
    격자: 기본은 ts[0]부터 ts[-1]까지(끝 포함), t_start/t_end를 주면 [t_start, t_end) 고정 길이
          (round((t_end - t_start) * hz)개).

    반환: (t_grid (N,) float64, kps_r (N, 17, 2) float32, conf_mask (N, 17) float32 {0,1},
           bbox_r (N, 4) float32, present_r (N,) bool)
    """
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    kps = np.asarray(kps, dtype=np.float64)
    t = len(ts)
    if kps.shape != (t, NUM_KPS, 3):
        raise ValueError(f"kps shape {kps.shape} != ({t}, {NUM_KPS}, 3)")
    bbox = np.zeros((t, 4)) if bbox is None else np.asarray(bbox, dtype=np.float64).reshape(t, 4)
    present = np.ones(t, dtype=bool) if present is None else np.asarray(present, dtype=bool).reshape(t)

    order = np.argsort(ts, kind="stable")
    ts, kps, bbox, present = ts[order], kps[order], bbox[order], present[order]

    grid = _make_grid(ts, float(hz), t_start, t_end)
    n = len(grid)

    kps_r = np.zeros((n, NUM_KPS, 2), dtype=np.float32)
    conf_mask = np.zeros((n, NUM_KPS), dtype=np.float32)
    pres_valid = present & np.isfinite(ts)
    for j in range(NUM_KPS):
        xy = kps[:, j, :2]
        valid = pres_valid & (kps[:, j, 2] >= min_conf) & np.all(np.isfinite(xy), axis=1)
        vals, ok = _interp_valid(ts, xy, valid, grid, max_gap_s)
        kps_r[:, j] = vals
        conf_mask[:, j] = ok

    _, present_r = _interp_valid(ts, np.zeros((t, 1)), pres_valid, grid, max_gap_s)

    bbox_valid = pres_valid & np.all(np.isfinite(bbox), axis=1) & np.any(bbox != 0, axis=1)
    bbox_vals, _ = _interp_valid(ts, bbox, bbox_valid, grid, max_gap_s)
    bbox_r = bbox_vals.astype(np.float32)

    return grid, kps_r, conf_mask, bbox_r, present_r


def _mid_point(p, m, a, b):
    """두 관절 중점. 한쪽만 유효하면 그 관절, 둘 다 없으면 무효. 반환: (N, 2), (N,) bool"""
    ma, mb = m[:, a], m[:, b]
    both = ma & mb
    mid = np.zeros((p.shape[0], 2), dtype=np.float64)
    mid[both] = (p[both, a] + p[both, b]) / 2
    only_a = ma & ~mb
    only_b = mb & ~ma
    mid[only_a] = p[only_a, a]
    mid[only_b] = p[only_b, b]
    return mid, ma | mb


def _masked_gradient(x, valid, dt, wrap=False):
    """
    시간축(0번 축) 미분. 중앙 차분 우선, 이웃이 무효면 전진/후진 차분, 자기 자신이 무효면 0.
    x: (N, D), valid: (N, D) bool. wrap=True면 각도 차이를 [-pi, pi)로 감습니다.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return np.zeros_like(x)

    def _d(a, b):
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi if wrap else d

    step = _d(x[1:], x[:-1]) / dt
    step_ok = valid[1:] & valid[:-1]

    fwd = np.zeros_like(x); fwd_ok = np.zeros_like(valid)
    fwd[:-1], fwd_ok[:-1] = step, step_ok
    bwd = np.zeros_like(x); bwd_ok = np.zeros_like(valid)
    bwd[1:], bwd_ok[1:] = step, step_ok
    cen = np.zeros_like(x); cen_ok = np.zeros_like(valid)
    if n >= 3:
        cen[1:-1] = _d(x[2:], x[:-2]) / (2 * dt)
        cen_ok[1:-1] = valid[2:] & valid[:-2] & valid[1:-1]

    return np.where(cen_ok, cen, np.where(fwd_ok, fwd, np.where(bwd_ok, bwd, 0.0)))


def _body_scale(p, m, bbox_r, h):
    """몸통 길이 s (px): 유효 프레임의 |mid_shoulder - mid_hip| 중앙값 -> 0.3 * bbox 높이 중앙값 -> 1.0, 최소 1px"""
    sh, sh_ok = _mid_point(p, m, L_SHOULDER, R_SHOULDER)
    hip, hip_ok = _mid_point(p, m, L_HIP, R_HIP)
    ok = sh_ok & hip_ok
    s = np.nan
    if np.any(ok):
        s = float(np.median(np.linalg.norm(sh[ok] - hip[ok], axis=1)))
    if not np.isfinite(s) or s < 1.0:
        bh = (bbox_r[:, 3] - bbox_r[:, 1]) * h
        bh = bh[np.isfinite(bh) & (bh > 0)]
        if len(bh):
            s = 0.3 * float(np.median(bh))
    if not np.isfinite(s):
        s = 1.0
    return max(s, 1.0)


def build_features_v2(kps_r, conf_mask, bbox_r, frame_wh, hz=TARGET_HZ) -> np.ndarray:
    """
    리샘플링된 키포인트로 v2 특징을 만듭니다.

    kps_r: (N, 17, 2) 정규화 좌표, conf_mask: (N, 17) {0,1}, bbox_r: (N, 4) 정규화 xyxy (None이면 0),
    frame_wh: (W, H) 픽셀 (None이면 (1, 1)로 간주해 등방 보정 없음)
    반환: (92, N) float32, 채널 순서는 V2_CHANNELS
      - rel: (관절 - mid_hip_t) / s  (34)        s = 몸통 길이 (px, _body_scale)
      - vel: d(rel)/dt, 단위 s/초, 중앙 차분 (34)
      - hip_dx, hip_dy: (mid_hip_t - 첫 유효 mid_hip) / s, hip_vx, hip_vy: 그 속도 (4)
      - torso_angle: mid_hip -> mid_shoulder 벡터의 수직(위쪽) 대비 부호 있는 각도 (rad, 선 자세 0,
        어깨가 화면 오른쪽으로 기울면 +, 범위 (-pi, pi]), torso_angvel: rad/초 (2)
      - bbox_aspect: bbox 높이/폭 (등방 px, 0~10으로 제한) (1)
      - mask: conf_mask (17)
    중점은 한쪽 관절만 있으면 그 관절을 씁니다. 골반 중점이 없는 프레임은 rel/vel/hip 채널이 0,
    마스크된 관절도 0 입니다. 모든 값은 유한합니다.
    """
    kps_r = np.asarray(kps_r, dtype=np.float64).reshape(-1, NUM_KPS, 2)
    n = kps_r.shape[0]
    m = np.asarray(conf_mask, dtype=np.float64).reshape(n, NUM_KPS) > 0.5
    bbox_r = np.zeros((n, 4)) if bbox_r is None else np.asarray(bbox_r, dtype=np.float64).reshape(n, 4)
    bbox_r = np.where(np.isfinite(bbox_r), bbox_r, 0.0)
    w, h = (1.0, 1.0) if frame_wh is None else (float(frame_wh[0]), float(frame_wh[1]))
    dt = 1.0 / float(hz)

    m = m & np.all(np.isfinite(kps_r), axis=2)
    p = np.where(m[:, :, None], kps_r * np.array([w, h]), 0.0)

    s = _body_scale(p, m, bbox_r, h)
    hip, hip_ok = _mid_point(p, m, L_HIP, R_HIP)
    sh, sh_ok = _mid_point(p, m, L_SHOULDER, R_SHOULDER)

    # 관절 상대좌표와 속도
    rel_ok = m & hip_ok[:, None]                                  # (N, 17)
    rel = np.where(rel_ok[:, :, None], (p - hip[:, None, :]) / s, 0.0)
    rel_flat = rel.reshape(n, NUM_KPS * 2)
    vel_flat = _masked_gradient(rel_flat, np.repeat(rel_ok, 2, axis=1), dt)

    # 전역 골반 궤적
    glob = np.zeros((n, 2))
    if np.any(hip_ok):
        first = int(np.argmax(hip_ok))
        glob[hip_ok] = (hip[hip_ok] - hip[first]) / s
    glob_vel = _masked_gradient(glob, np.repeat(hip_ok[:, None], 2, axis=1), dt)

    # 몸통 각도 (영상 y축은 아래 방향이므로 위쪽은 -y)
    torso_ok = sh_ok & hip_ok
    v = sh - hip
    angle = np.where(torso_ok, np.arctan2(v[:, 0], -v[:, 1]), 0.0)
    ang_vel = _masked_gradient(angle[:, None], torso_ok[:, None], dt, wrap=True)[:, 0]

    # bbox 종횡비 (등방 px)
    bw = (bbox_r[:, 2] - bbox_r[:, 0]) * w
    bh = (bbox_r[:, 3] - bbox_r[:, 1]) * h
    good = (bw > 0) & (bh > 0)
    aspect = np.zeros(n)
    aspect[good] = bh[good] / bw[good]
    aspect = np.clip(np.nan_to_num(aspect, nan=0.0, posinf=_MAX_ASPECT, neginf=0.0), 0.0, _MAX_ASPECT)

    feats = np.concatenate([
        rel_flat,                           # 34
        vel_flat,                           # 34
        glob, glob_vel,                     # 4
        angle[:, None], ang_vel[:, None],   # 2
        aspect[:, None],                    # 1
        m.astype(np.float64),               # 17
    ], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(feats.T, dtype=np.float32)


def window_v2(ts, kps, bbox, present, frame_wh, t_start, t_end, hz=TARGET_HZ,
              max_gap_s=MAX_GAP_S, min_conf=MIN_CONF) -> np.ndarray:
    """
    [t_start, t_end) 창을 잘라 리샘플링하고 v2 특징을 만듭니다.
    반환: (92, round((t_end - t_start) * hz)) float32
    창 밖 max_gap_s 이내 샘플까지 보간에 쓰므로 창 경계의 kps/마스크는 클립 전체를 리샘플링한 값과 같습니다.
    (스케일 s와 hip_dx/dy 기준점은 창 안에서 계산)
    """
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    t = len(ts)
    kps = np.asarray(kps, dtype=np.float64).reshape(t, NUM_KPS, 3)
    sel = (ts >= t_start - max_gap_s - _TS_EPS) & (ts <= t_end + max_gap_s + _TS_EPS)
    bbox_s = None if bbox is None else np.asarray(bbox, dtype=np.float64).reshape(t, 4)[sel]
    present_s = None if present is None else np.asarray(present, dtype=bool).reshape(t)[sel]
    _, kps_r, conf_mask, bbox_r, _ = resample_clip(
        ts[sel], kps[sel], bbox_s, present_s,
        hz=hz, max_gap_s=max_gap_s, min_conf=min_conf, t_start=t_start, t_end=t_end)
    return build_features_v2(kps_r, conf_mask, bbox_r, frame_wh, hz=hz)


def flip_lr_coco(kps) -> np.ndarray:
    """
    좌우 반전: 대칭 관절 쌍(1-2, 3-4, ..., 15-16)을 맞바꾸고 x -> 1 - x.
    kps: (..., 17, C) 정규화 좌표, C >= 2 (C=3이면 conf는 관절과 함께 이동)
    모든 값이 0인 관절(미검출 표기)은 0으로 유지합니다 (v1의 np.any 결측 판정을 깨지 않도록).
    두 번 적용하면 원래 값입니다.
    """
    k = np.array(kps, copy=True)
    if k.ndim < 2 or k.shape[-2] != NUM_KPS or k.shape[-1] < 2:
        raise ValueError(f"kps shape {k.shape}: (..., {NUM_KPS}, C>=2)이어야 합니다")
    perm = list(range(NUM_KPS))
    for a, b in COCO_FLIP_PAIRS:
        perm[a], perm[b] = b, a
    k = k[..., perm, :]
    absent = ~np.any(k != 0, axis=-1)
    k[..., 0] = np.where(absent, k[..., 0], 1.0 - k[..., 0])
    return k


def flip_lr_bbox(bbox) -> np.ndarray:
    """bbox 좌우 반전: [x1, y1, x2, y2] -> [1 - x2, y1, 1 - x1, y2]. 모두 0인 bbox(부재)는 유지."""
    b = np.array(bbox, copy=True)
    absent = ~np.any(b != 0, axis=-1)
    x1, x2 = b[..., 0].copy(), b[..., 2].copy()
    b[..., 0] = np.where(absent, x1, 1.0 - x2)
    b[..., 2] = np.where(absent, x2, 1.0 - x1)
    return b
