"""
데이터 카드: 통합 manifest와 키포인트 클립(v1)으로 데이터셋별 통계를 계산해 docs/DATA_CARD.md를 만듭니다

  python -m training.datasets.data_card --manifest dataset/_manifest/manifest_v1.jsonl \
      --kp-dir dataset/_keypoints/v1 --out docs/DATA_CARD.md

계산 항목 (데이터셋별 + 전체)
- 구성: 클립, 낙상/ADL, event_group, 라벨 출처, 라이선스, 프레임 크기, fps(클립 meta), 클립 길이, 낙상 구간 길이
- 포즈 품질: 사람 검출 비율(원본 프레임), 관절 가시율(features v2와 같은 15Hz 리샘플링 conf 마스크, conf >= 0.3)
  전체/부위별/낙상 구간 [fall_start, fall_end + 2초]
- 하강 속도: 골반 중점의 아래 방향 속도 최댓값 (몸통 길이/초, 등방 픽셀 x*W, y*H, 15Hz, 공백은 건너뜀)
  낙상 구간 vs 하드 ADL 구간(lie_down, sit_down, kneel_down, squat_down)
- 종횡비: 규칙 엔진 임계값(정규화 bbox h/w)의 픽셀 환산 th * H / W, 서 있을 때(처음 1초)와 누운 뒤의 정규화 종횡비
- 알려진 라벨/데이터 문제 확인 (MCFD fps, URFD crop, 라벨 끝 초과, 중복 영상 등)

순수 계산 함수는 numpy 없이 import할 수 있고, 키포인트 통계(array 함수)는 numpy가 필요합니다.
"""
import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATASETS = ("le2i", "caucafall", "gmdcsa24", "mcfd", "urfd", "figshare_ursul29")
ALL = "전체"

HZ = 15.0                    # features v2 TARGET_HZ
MIN_CONF = 0.3               # features v2 MIN_CONF (conf 마스크 기준, tests에서 같은 값인지 확인)
MAX_GAP_S = 0.2              # features v2 MAX_GAP_S
FALL_POST_S = 2.0            # 낙상 구간 품질: [fall_start, fall_end + 2초]
ONSET_PRE_S, ONSET_POST_S = 1.5, 2.5   # DATASETS.md "낙상 시작 구간" (E 단계 창과 같은 범위)
VEL_PAD_S = 0.25             # 속도 측정 구간 여유 (구간 시작/끝 양쪽, 낙상과 하드 ADL 동일)
STAND_S = 1.0                # 서 있는 종횡비: 클립 처음 1초
LYING_END_S = 1.0            # 누운 뒤 구간 라벨이 없으면 클립 마지막 1초
LOW_VIS = 0.3                # 낙상 구간 가시율 하한 (windows.MIN_VISIBILITY와 같음)
LOW_PRESENT = 0.5            # 누운 뒤 검출 비율이 이보다 낮으면 "누운 사람 놓침"
_MASK0 = 75                  # features v2 마스크 채널 시작 (V2_CHANNELS.index("mask_nose"), tests에서 확인)

HARD_ADL_CLS = ("lie_down", "sit_down", "kneel_down", "squat_down")
POST_FALL_CLS = ("fallen", "lying")
RECOVER_CLS = ("stand_up", "standing", "walk")   # 낙상 뒤 이 구간이 오면 누운 채 끝나지 않음

# COCO-17 부위 묶음
JOINT_GROUPS = {
    "head": (0, 1, 2, 3, 4),
    "arms": (7, 8, 9, 10),
    "torso": (5, 6, 11, 12),
    "legs": (13, 14, 15, 16),
}
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

# 규칙 엔진의 정규화 bbox 종횡비 임계값 이름 (값은 Level1Params 기본값에서 읽음)
RULE_ASPECT_FIELDS = ("aspect_abs_th", "force_abnormal_aspect_th", "confirm_aspect_abs")


# ---------------------------------------------------------------------------
# 순수 계산 (표준 라이브러리만)
# ---------------------------------------------------------------------------

def _finite(values: Iterable[Any]) -> List[float]:
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def quantile(values: Iterable[Any], q: float) -> Optional[float]:
    """선형 보간 분위수 (numpy 기본 방식과 같음). 유한한 값이 없으면 None."""
    v = sorted(_finite(values))
    if not v:
        return None
    pos = (len(v) - 1) * min(max(float(q), 0.0), 1.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def describe(values: Iterable[Any]) -> Dict[str, Any]:
    """{n, median, p10, p90, min, max} (값이 없으면 n=0, 나머지 None)."""
    v = _finite(values)
    if not v:
        return {"n": 0, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {"n": len(v), "median": quantile(v, 0.5), "p10": quantile(v, 0.1), "p90": quantile(v, 0.9),
            "min": min(v), "max": max(v)}


def share(values: Iterable[Any], pred) -> Optional[float]:
    """유한한 값 중 pred를 만족하는 비율 (값이 없으면 None)."""
    v = _finite(values)
    return sum(1 for x in v if pred(x)) / len(v) if v else None


def pixel_threshold(th: float, w: float, h: float) -> float:
    """
    정규화 bbox 종횡비 임계값 -> 픽셀 종횡비 임계값.
    정규화 h/w = (h_px / H) / (w_px / W) = 픽셀 h/w * W / H 이므로 픽셀 기준은 th * H / W.
    """
    return float(th) * float(h) / float(w)


def normalized_aspect(px_aspect: float, w: float, h: float) -> float:
    """픽셀 종횡비 -> 해당 프레임 크기에서의 정규화 종횡비 (pixel_threshold의 역)."""
    return float(px_aspect) * float(w) / float(h)


def to_frame(norm_aspect: float, src_wh: Sequence[float], dst_wh: Sequence[float]) -> float:
    """src 프레임의 정규화 종횡비를 같은 픽셀 자세로 dst 프레임에서 보았을 때의 정규화 종횡비로 바꿉니다."""
    px = pixel_threshold(norm_aspect, src_wh[0], src_wh[1])
    return normalized_aspect(px, dst_wh[0], dst_wh[1])


def segments_of(row: Dict[str, Any], classes: Sequence[str]) -> List[Tuple[str, float, float]]:
    """행의 구간 중 classes에 속하는 것: [(cls, start_s, end_s)] (시작 순)."""
    segs = []
    for s in row.get("segments") or []:
        if s.get("cls") in classes:
            segs.append((s["cls"], float(s["start_s"]), float(s["end_s"])))
    return sorted(segs, key=lambda x: (x[1], x[2]))


def lying_interval(row: Dict[str, Any], clip_end: float) -> Tuple[Optional[float], Optional[float], str]:
    """
    낙상 클립의 누운 뒤 구간: 첫 fall 구간 이후 처음 시작하는 fallen/lying 구간,
    없고 낙상 뒤 일어나는 구간(stand_up/standing/walk)이 있으면 누운 구간 없음 (None, None, "recovered"),
    그 밖(라벨 없음, 낙상으로 끝남)은 클립 마지막 LYING_END_S초. 반환 (start, end, 출처 "label" | "clip_end" | "recovered").
    """
    fs, fe = row.get("fall_start_s"), row.get("fall_end_s")
    if fs is not None:
        for _, a, b in segments_of(row, POST_FALL_CLS):
            if a >= float(fs):
                return a, min(b, clip_end), "label"
        after = float(fe if fe is not None else fs)
        if any(a >= after - 0.05 for _, a, _ in segments_of(row, RECOVER_CLS)):
            return None, None, "recovered"
    return max(0.0, clip_end - LYING_END_S), clip_end, "clip_end"


def label_overrun_s(row: Dict[str, Any], clip_end: float) -> Optional[float]:
    """구간 라벨 끝이 클립 끝을 넘는 길이 (초, 넘지 않으면 0). 구간이 없으면 None."""
    segs = row.get("segments") or []
    if not segs:
        return None
    return max(0.0, max(float(s["end_s"]) for s in segs) - float(clip_end))


def overlap(fall_v: Iterable[Any], adl_v: Iterable[Any]) -> Dict[str, Optional[float]]:
    """
    낙상/하드 ADL 하강 속도 분포 겹침.
    adl_gt_fall_median: 낙상 중앙값보다 빠른 하드 ADL 비율, adl_gt_fall_p10: 낙상 p10보다 빠른 하드 ADL 비율,
    fall_lt_adl_p90: 하드 ADL p90보다 느린 낙상 비율.
    """
    f, a = _finite(fall_v), _finite(adl_v)
    fm, f10, a90 = quantile(f, 0.5), quantile(f, 0.1), quantile(a, 0.9)
    return {
        "adl_gt_fall_median": share(a, lambda x: x > fm) if fm is not None else None,
        "adl_gt_fall_p10": share(a, lambda x: x > f10) if f10 is not None else None,
        "fall_lt_adl_p90": share(f, lambda x: x < a90) if a90 is not None else None,
    }


def rule_aspect_thresholds() -> Dict[str, float]:
    """Level1Params 기본값에서 종횡비 임계값을 읽습니다 (app.core.level1은 numpy 없이 import 가능)."""
    from app.core.level1 import Level1Params
    p = Level1Params()
    return {k: float(getattr(p, k)) for k in RULE_ASPECT_FIELDS}


def deploy_frame_wh() -> Tuple[int, int]:
    """배포 카메라 해상도 (app.config FRAME_W/FRAME_H, 환경변수 기본값 1920x1080)."""
    from app.config import FRAME_H, FRAME_W
    return int(FRAME_W), int(FRAME_H)


# ---------------------------------------------------------------------------
# 배열 계산 (numpy)
# ---------------------------------------------------------------------------

def mid_point(xy, mask, a: int, b: int):
    """두 관절 중점 (N,2)과 유효 여부 (N,). 한쪽만 보이면 그 관절 (features v2 _mid_point와 같은 규칙)."""
    import numpy as np
    xy = np.asarray(xy, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    ma, mb = m[:, a], m[:, b]
    both = ma & mb
    mid = np.full((xy.shape[0], 2), np.nan)
    mid[both] = (xy[both, a] + xy[both, b]) / 2
    mid[ma & ~mb] = xy[ma & ~mb, a]
    mid[mb & ~ma] = xy[mb & ~ma, b]
    return mid, ma | mb


def torso_length(xy_px, mask) -> float:
    """몸통 길이 (px): 어깨 중점-골반 중점 거리의 중앙값 (둘 다 보이는 프레임). 없으면 nan."""
    import numpy as np
    sh, sh_ok = mid_point(xy_px, mask, L_SHOULDER, R_SHOULDER)
    hip, hip_ok = mid_point(xy_px, mask, L_HIP, R_HIP)
    ok = sh_ok & hip_ok
    if not np.any(ok):
        return float("nan")
    return float(np.median(np.linalg.norm(sh[ok] - hip[ok], axis=1)))


def down_velocity(t, y, ok, scale: float):
    """
    아래 방향(+y) 속도 / scale (몸통 길이/초). 중앙 차분이며 i-1, i, i+1이 모두 유효할 때만 값이 있고
    나머지는 nan (공백을 건너뛰어 잇지 않음). t: (N,), y: (N,), ok: (N,) bool.
    """
    import numpy as np
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.asarray(ok, dtype=bool)
    v = np.full(len(t), np.nan)
    if len(t) < 3 or not (scale > 0 and math.isfinite(scale)):
        return v
    good = ok[2:] & ok[1:-1] & ok[:-2]
    dt = t[2:] - t[:-2]
    good &= dt > 0
    v[1:-1][good] = (y[2:][good] - y[:-2][good]) / dt[good] / scale
    return v


def max_in(t, v, t0: float, t1: float) -> Optional[float]:
    """[t0, t1] 안의 유한한 v 최댓값 (없으면 None)."""
    import numpy as np
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    sel = (t >= t0) & (t <= t1) & np.isfinite(v)
    return float(v[sel].max()) if np.any(sel) else None


def mean_in(t, x, t0: float, t1: float) -> Optional[float]:
    """[t0, t1] 안의 x 평균 (x: (N,) 또는 (N, K)는 전체 평균, 없으면 None)."""
    import numpy as np
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    sel = (t >= t0) & (t <= t1)
    return float(x[sel].mean()) if np.any(sel) else None


def visibility(mask, joints: Optional[Sequence[int]] = None) -> Optional[float]:
    """관절 가시율: (N,17) 마스크의 평균 (joints를 주면 그 관절만). 프레임이 없으면 None."""
    import numpy as np
    m = np.asarray(mask, dtype=np.float64)
    if m.shape[0] == 0:
        return None
    if joints is not None:
        m = m[:, list(joints)]
    return float(m.mean())


def bbox_aspect_norm(bbox, present):
    """정규화 bbox 높이/폭 (Level1 bbox_aspect와 같은 정의). 사람이 없거나 bbox가 비면 nan."""
    import numpy as np
    b = np.asarray(bbox, dtype=np.float64).reshape(-1, 4)
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    ok = np.asarray(present, dtype=bool) & (w > 0) & (h > 0)
    out = np.full(len(b), np.nan)
    out[ok] = h[ok] / w[ok]
    return out


def _median_in(t, x, t0: float, t1: float) -> Optional[float]:
    import numpy as np
    t = np.asarray(t, dtype=np.float64)
    sel = (t >= t0) & (t <= t1) & np.isfinite(x)
    return float(np.median(x[sel])) if np.any(sel) else None


def clip_stats(row: Dict[str, Any], clip: Dict[str, Any]) -> Dict[str, Any]:
    """클립 하나의 통계 (manifest 행 + clipfmt.load_clip 결과)."""
    import numpy as np
    from app.ai_classifier import features as F

    ts = np.asarray(clip["ts"], dtype=np.float64)
    present = np.asarray(clip["present"], dtype=bool)
    w, h = (int(v) for v in clip["frame_wh"])
    meta = clip.get("meta") or {}
    clip_end = float(ts[-1]) if len(ts) else 0.0
    out: Dict[str, Any] = {
        "clip_id": row["clip_id"], "dataset": row["dataset"], "label": row["label"], "cam": row.get("cam"),
        "frame_wh": (w, h), "native_fps": meta.get("native_fps"), "fps_source": meta.get("fps_source"),
        "container_fps": meta.get("container_fps"), "n_frames": int(len(ts)), "duration_s": clip_end,
        "present_ratio": float(present.mean()) if len(ts) else None,
        "n_target_switch": int(meta.get("n_target_switch") or 0),
        "n_multi_person_frames": int(meta.get("n_multi_person_frames") or 0),
        "crop_px": meta.get("crop_px"), "overrun_s": label_overrun_s(row, clip_end),
    }

    # features v2와 같은 15Hz 리샘플링 (conf >= MIN_CONF, 0.2초 이하 공백만 보간)
    tg, kps_r, mask, _, _ = F.resample_clip(ts, clip["kps"], clip["bbox"], present,
                                            hz=HZ, max_gap_s=MAX_GAP_S, min_conf=MIN_CONF)
    mask = mask > 0.5
    out["vis"] = visibility(mask)
    for g, js in JOINT_GROUPS.items():
        out[f"vis_{g}"] = visibility(mask, js)

    fs, fe = row.get("fall_start_s"), row.get("fall_end_s")
    if fs is not None and fe is not None:
        a, b = float(fs), float(fe) + FALL_POST_S
        out["fall_present"] = mean_in(ts, present, a, b)
        sel = (tg >= a) & (tg <= b)
        out["fall_vis"] = visibility(mask[sel]) if np.any(sel) else None
        for g, js in JOINT_GROUPS.items():
            out[f"fall_vis_{g}"] = visibility(mask[sel], js) if np.any(sel) else None
        # 시작 구간은 E 단계 창과 똑같이 잘라 계산 (window_v2의 마스크 채널 평균 = windows.visibility)
        xw = F.window_v2(ts, clip["kps"], clip["bbox"], present, (w, h),
                         float(fs) - ONSET_PRE_S, float(fs) + ONSET_POST_S)
        out["onset_vis"] = float(xw[_MASK0:].mean()) if xw.shape[1] else None

    # 하강 속도 (몸통 길이/초)
    xy_px = kps_r.astype(np.float64) * np.array([w, h], dtype=np.float64)
    scale = torso_length(xy_px, mask)
    hip, hip_ok = mid_point(xy_px, mask, L_HIP, R_HIP)
    sw = clip.get("switched")
    if sw is not None and len(tg):
        # 프레임별 대상 전환이 있으면 (재추출 클립) 전환 시각 1/HZ 이내 격자점은 속도 계산에서 뺌
        t_sw = ts[np.asarray(sw, dtype=bool).reshape(-1)[:len(ts)]]
        if len(t_sw):
            near = np.abs(tg[:, None] - t_sw[None, :]).min(axis=1) <= 1.0 / HZ
            hip_ok = hip_ok & ~near
    v = down_velocity(tg, hip[:, 1], hip_ok, scale)
    out["torso_px"] = scale if math.isfinite(scale) else None
    out["fall_vel"] = [max_in(tg, v, s - VEL_PAD_S, e + VEL_PAD_S) for _, s, e in segments_of(row, ("fall",))]
    out["hard_vel"] = [(c, max_in(tg, v, s - VEL_PAD_S, e + VEL_PAD_S)) for c, s, e in segments_of(row, HARD_ADL_CLS)]
    out["clip_vel"] = max_in(tg, v, -1.0, clip_end + 1.0)

    # bbox 종횡비 (원본 프레임, 정규화 좌표)
    asp = bbox_aspect_norm(clip["bbox"], present)
    t_first = float(ts[0]) if len(ts) else 0.0
    if row["label"] == "fall":
        # 서 있음: 낙상 클립 처음 STAND_S초 (낙상 시작 전까지만)
        t_stand = t_first + STAND_S if fs is None else min(t_first + STAND_S, float(fs))
        out["stand_aspect"] = _median_in(ts, asp, t_first, t_stand)
        a, b, src = lying_interval(row, clip_end)
        out["lying_aspect"] = _median_in(ts, asp, a, b) if a is not None else None
        out["lying_present"] = mean_in(ts, present, a, b) if a is not None else None
        out["lying_src"] = src
    return out


def load_clip_for(row: Dict[str, Any], kp_dir: Path) -> Dict[str, Any]:
    from training.datasets.clipfmt import load_clip
    from training.datasets.extract_keypoints import clip_out_path
    return load_clip(clip_out_path(Path(kp_dir), row["clip_id"]))


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------

def _vals(stats: Sequence[Dict[str, Any]], key: str, label: Optional[str] = None) -> List[Any]:
    return [s.get(key) for s in stats if label is None or s["label"] == label]


def aggregate(rows: Sequence[Dict[str, Any]], stats: Sequence[Dict[str, Any]], thresholds: Dict[str, float],
              deploy_wh: Tuple[int, int]) -> Dict[str, Any]:
    """행/클립 통계를 데이터셋별 + 전체로 모읍니다. stats는 키포인트가 있는 클립만."""
    rows_by = defaultdict(list)
    stats_by = defaultdict(list)
    for r in rows:
        rows_by[r["dataset"]].append(r)
        rows_by[ALL].append(r)
    for s in stats:
        stats_by[s["dataset"]].append(s)
        stats_by[ALL].append(s)
    names = [d for d in DATASETS if d in rows_by] + sorted(d for d in rows_by if d not in DATASETS and d != ALL)
    out = {"datasets": names, "per": {}}
    for d in names + [ALL]:
        rs, ss = rows_by[d], stats_by[d]
        fall_ss = [s for s in ss if s["label"] == "fall"]
        fall_vel = [v for s in ss for v in s["fall_vel"]]
        hard_adl = [v for s in ss if s["label"] == "adl" for _, v in s["hard_vel"]]
        hard_fallclip = [v for s in ss if s["label"] == "fall" for _, v in s["hard_vel"]]
        by_cls = defaultdict(list)
        for s in ss:
            if s["label"] == "adl":
                for c, v in s["hard_vel"]:
                    by_cls[c].append(v)
        fall_vis = _vals(fall_ss, "fall_vis")
        onset_vis = _vals(fall_ss, "onset_vis")
        # 낙상 클립 종횡비를 배포 프레임으로 환산
        lying_dep = [to_frame(s["lying_aspect"], s["frame_wh"], deploy_wh) for s in fall_ss
                     if s.get("lying_aspect") is not None]
        stand_dep = [to_frame(s["stand_aspect"], s["frame_wh"], deploy_wh) for s in fall_ss
                     if s.get("stand_aspect") is not None]
        out["per"][d] = {
            "clips": len(rs), "fall": sum(r["label"] == "fall" for r in rs),
            "adl": sum(r["label"] == "adl" for r in rs),
            "event_groups": len({r["event_group"] for r in rs}),
            "label_source": Counter(r.get("label_source") for r in rs),
            "license": Counter(r.get("license") for r in rs),
            "with_kp": len(ss),
            "frames": sum(s["n_frames"] for s in ss),
            "fps_range": (min(_finite(_vals(ss, "native_fps")), default=None),
                          max(_finite(_vals(ss, "native_fps")), default=None)),
            "frame_wh": Counter(f"{s['frame_wh'][0]}x{s['frame_wh'][1]}" for s in ss),
            "fps": Counter(_fmt_num(round(float(s["native_fps"])) if s["native_fps"] else None) for s in ss),
            "fps_source": Counter(s["fps_source"] for s in ss),
            "duration": describe(_vals(ss, "duration_s")),
            "fall_seg_dur": describe([e - a for r in rs for _, a, e in segments_of(r, ("fall",))]),
            "n_fall_segments": sum(len(segments_of(r, ("fall",))) for r in rs),
            "present": describe(_vals(ss, "present_ratio")),
            "fall_present": describe(_vals(fall_ss, "fall_present")),
            "lying_present": describe(_vals(fall_ss, "lying_present")),
            "lying_present_low": sum(1 for v in _finite(_vals(fall_ss, "lying_present")) if v < LOW_PRESENT),
            "lying_present_n": len(_finite(_vals(fall_ss, "lying_present"))),
            "vis": describe(_vals(ss, "vis")),
            "vis_groups": {g: describe(_vals(ss, f"vis_{g}"))["median"] for g in JOINT_GROUPS},
            "fall_vis": describe(fall_vis),
            "fall_vis_groups": {g: describe(_vals(fall_ss, f"fall_vis_{g}"))["median"] for g in JOINT_GROUPS},
            "fall_vis_low": sum(1 for v in _finite(fall_vis) if v < LOW_VIS),
            "fall_vis_n": len(_finite(fall_vis)),
            "onset_vis": describe(onset_vis),
            "onset_vis_low": sum(1 for v in _finite(onset_vis) if v < LOW_VIS),
            "onset_vis_mean": sum(_finite(onset_vis)) / len(_finite(onset_vis)) if _finite(onset_vis) else None,
            "multi_person_clips": sum(1 for s in ss if s["n_multi_person_frames"] > 0),
            "switch_clips": sum(1 for s in ss if s["n_target_switch"] > 0),
            "fall_vel": describe(fall_vel),
            "fall_vel_missing": sum(1 for v in fall_vel if v is None),
            "fall_vel_noswitch": describe([v for s in ss if s["n_target_switch"] == 0 for v in s["fall_vel"]]),
            "fall_vel_switch": describe([v for s in ss if s["n_target_switch"] > 0 for v in s["fall_vel"]]),
            "hard_adl_vel": describe(hard_adl),
            "hard_adl_vel_missing": sum(1 for v in hard_adl if v is None),
            "hard_fallclip_vel": describe(hard_fallclip),
            "hard_by_cls": {c: describe(v) for c, v in sorted(by_cls.items())},
            "overlap": overlap(fall_vel, hard_adl),
            "clip_vel_fall": describe(_vals(ss, "clip_vel", "fall")),
            "clip_vel_adl": describe(_vals(ss, "clip_vel", "adl")),
            "clip_overlap": overlap(_vals(ss, "clip_vel", "fall"), _vals(ss, "clip_vel", "adl")),
            "stand_aspect": describe(_vals(fall_ss, "stand_aspect")),
            "lying_aspect": describe(_vals(fall_ss, "lying_aspect")),
            "lying_src": Counter(s.get("lying_src") for s in fall_ss),
            "stand_dep": describe(stand_dep),
            "lying_dep": describe(lying_dep),
            "stand_below_abs": share(_vals(fall_ss, "stand_aspect"), lambda x: x < thresholds["aspect_abs_th"]),
            "lying_below_confirm": share(_vals(fall_ss, "lying_aspect"),
                                         lambda x: x < thresholds["confirm_aspect_abs"]),
            "lying_below_force": share(_vals(fall_ss, "lying_aspect"),
                                       lambda x: x < thresholds["force_abnormal_aspect_th"]),
            "lying_dep_below_confirm": share(lying_dep, lambda x: x < thresholds["confirm_aspect_abs"]),
            "lying_dep_below_force": share(lying_dep, lambda x: x < thresholds["force_abnormal_aspect_th"]),
        }
    out["frame_sizes"] = _frame_size_table(stats, thresholds, deploy_wh)
    out["mcfd_cams"] = _per_cam(stats, "mcfd")
    out["issues"] = check_issues(rows, stats)
    out["le2i_subjects"] = le2i_subjects(rows)
    out["segment_classes"] = Counter(s.get("cls") for r in rows for s in r.get("segments") or [])
    out["thresholds"] = thresholds
    out["deploy_wh"] = deploy_wh
    return out


def _frame_size_table(stats, thresholds, deploy_wh) -> List[Dict[str, Any]]:
    by = defaultdict(lambda: Counter())
    for s in stats:
        by[tuple(s["frame_wh"])][s["dataset"]] += 1
    rows = []
    for wh in sorted(by, key=lambda x: (x[0] * x[1], x)):
        rows.append({"wh": wh, "datasets": by[wh], "deploy": False,
                     "px": {k: pixel_threshold(v, *wh) for k, v in thresholds.items()}})
    rows.append({"wh": tuple(deploy_wh), "datasets": Counter(), "deploy": True,
                 "px": {k: pixel_threshold(v, *deploy_wh) for k, v in thresholds.items()}})
    return rows


def _per_cam(stats, dataset: str) -> Dict[str, Dict[str, Any]]:
    by = defaultdict(list)
    for s in stats:
        if s["dataset"] == dataset:
            by[s.get("cam")].append(s)
    out = {}
    for cam in sorted(by, key=lambda c: (len(str(c)), str(c))):
        ss = by[cam]
        fss = [s for s in ss if s["label"] == "fall"]
        fv = _vals(fss, "fall_vis")
        out[cam] = {"clips": len(ss), "present": describe(_vals(ss, "present_ratio"))["median"],
                    "fall_vis": describe(fv)["median"], "onset_vis": describe(_vals(fss, "onset_vis"))["median"],
                    "lying_present": describe(_vals(fss, "lying_present"))["median"],
                    "fall_vis_low": sum(1 for v in _finite(fv) if v < LOW_VIS), "n_fall": len(_finite(fv))}
    return out


def check_issues(rows: Sequence[Dict[str, Any]], stats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """알려진 라벨/데이터 문제를 데이터로 확인합니다 (값이 있는 것만)."""
    st = {s["clip_id"]: s for s in stats}
    iss: Dict[str, Any] = {}

    # MCFD: 헤더 fps 120 vs 실제 30 (라벨 끝이 프레임수/30과 맞는지)
    m = [(r, st[r["clip_id"]]) for r in rows if r["dataset"] == "mcfd" and r["clip_id"] in st]
    if m:
        def last_end(r):
            return max((float(x["end_s"]) for x in r.get("segments") or []), default=None)
        fit30 = fit120 = 0
        for r, s in m:
            e = last_end(r)
            if e is None:
                continue
            fit30 += e <= s["n_frames"] / 30.0 * 1.05 + 0.5
            fit120 += e <= s["n_frames"] / 120.0 * 1.05 + 0.5
        iss["mcfd_fps"] = {"clips": len(m), "container_fps": Counter(_fmt_num(s["container_fps"]) for _, s in m),
                           "native_fps": Counter(_fmt_num(s["native_fps"]) for _, s in m),
                           "fps_source": Counter(s["fps_source"] for _, s in m),
                           "labels_fit_30": fit30, "labels_fit_120": fit120}
    # URFD: 깊이|RGB 합성 영상의 RGB 절반 crop
    u = [(r, st[r["clip_id"]]) for r in rows if r["dataset"] == "urfd" and r["clip_id"] in st]
    if u:
        iss["urfd_crop"] = {"clips": len(u),
                            "row_crop": Counter(json.dumps(r.get("crop_px")) for r, _ in u),
                            "meta_crop": Counter(json.dumps(s.get("crop_px")) for _, s in u),
                            "frame_wh": Counter(f"{s['frame_wh'][0]}x{s['frame_wh'][1]}" for _, s in u)}
    # URFD 낙상 구간 길이 (상태 0 프레임 수가 모두 같은지)
    ud = [round(e - a, 3) for r in rows if r["dataset"] == "urfd" for _, a, e in segments_of(r, ("fall",))]
    if ud:
        iss["urfd_fall_len"] = Counter(ud)
    # GMDCSA24 Subject_2/Fall/03: Fall 폴더지만 OmniFall 라벨은 눕기 -> ADL
    g = [r for r in rows if r["clip_id"] == "gmdcsa24/Subject_2/Fall/03"]
    if g:
        iss["gmdcsa24_s2_fall03"] = {"label": g[0]["label"],
                                    "segments": [s["cls"] for s in g[0].get("segments") or []]}
    iss["gmdcsa24_fall_folder_adl"] = sorted(r["clip_id"] for r in rows if r["dataset"] == "gmdcsa24"
                                             and "/Fall/" in r["clip_id"] and r["label"] == "adl")
    iss["gmdcsa24_adl_folder_fall"] = sorted(r["clip_id"] for r in rows if r["dataset"] == "gmdcsa24"
                                             and "/ADL/" in r["clip_id"] and r["label"] == "fall")
    # 라벨 끝이 클립 끝을 넘는 클립 (데이터셋별)
    over = defaultdict(list)
    for s in stats:
        if s.get("overrun_s") is not None and s["overrun_s"] > 1.0 / max(float(s["native_fps"] or 30.0), 1.0):
            over[s["dataset"]].append(s["overrun_s"])
    iss["label_overrun"] = {d: {"n": len(v), "max": max(v), "median": quantile(v, 0.5)} for d, v in over.items()}
    # 낙상 시작이 클립 끝 이후인 클립 (구간이 통째로 영상 밖)
    fs = {r["clip_id"]: r.get("fall_start_s") for r in rows}
    iss["fall_after_end"] = sorted(s["clip_id"] for s in stats
                                   if fs.get(s["clip_id"]) is not None and fs[s["clip_id"]] >= s["duration_s"])
    # Figshare: 폴더 라벨만, 짧은 클립
    f = [s for s in stats if s["dataset"] == "figshare_ursul29"]
    if f:
        iss["figshare"] = {"clips": len(f), "duration": describe(_vals(f, "duration_s")),
                           "with_segments": sum(1 for r in rows if r["dataset"] == "figshare_ursul29"
                                                and r.get("segments"))}
    # 중복/다시점 event_group
    grp = defaultdict(list)
    for r in rows:
        grp[(r["dataset"], r["event_group"])].append(r["clip_id"])
    multi = defaultdict(list)
    for (d, gname), ids in grp.items():
        if len(ids) > 1:
            multi[d].append(sorted(ids))
    iss["multi_clip_groups"] = {d: sorted(v) for d, v in multi.items()}
    iss["kp_missing"] = sorted(r["clip_id"] for r in rows if r["clip_id"] not in st)
    return iss


def le2i_subjects(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Le2i subject 값 분포: 장면(폴더)별 subject 수, subject별 클립/낙상 수, cam과 장면의 대응."""
    le = [r for r in rows if r["dataset"] == "le2i"]
    by_scene = defaultdict(Counter)
    cam_scene = defaultdict(set)
    for r in le:
        scene = r["clip_id"].split("/")[1]
        by_scene[scene][r.get("subject")] += 1
        cam_scene[r.get("cam")].add(scene)
    subj = Counter(r.get("subject") for r in le)
    subj_fall = Counter(r.get("subject") for r in le if r["label"] == "fall")
    subj_scenes = {s: sorted({r["clip_id"].split("/")[1] for r in le if r.get("subject") == s}) for s in subj}
    return {"by_scene": {k: dict(v) for k, v in sorted(by_scene.items())},
            "subjects": dict(subj), "subject_fall": dict(subj_fall), "subject_scenes": subj_scenes,
            "cam_scene": {k: sorted(v) for k, v in cam_scene.items()}}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt_num(v, nd: int = 2) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "-"
    txt = f"{f:.{nd}f}"
    if float(txt) == round(f) and abs(f) < 1e6:
        return str(int(round(f)))
    return txt


def _pct(v) -> str:
    return "-" if v is None else f"{100.0 * float(v):.0f}%"


def _counter(c: Counter, top: int = 4) -> str:
    items = c.most_common()
    s = ", ".join(f"{k} ({n})" for k, n in items[:top])
    if len(items) > top:
        s += f" 외 {len(items) - top}"
    return s


def _dist(d: Dict[str, Any], nd: int = 2) -> str:
    """중앙값 (p10-p90)."""
    if not d or not d["n"]:
        return "-"
    return f"{_fmt_num(d['median'], nd)} ({_fmt_num(d['p10'], nd)}-{_fmt_num(d['p90'], nd)})"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(lines)


def render(agg: Dict[str, Any], manifest: str, kp_dir: str, elapsed_s: float) -> str:
    per, names = agg["per"], agg["datasets"]
    order = names + [ALL]
    th, dwh = agg["thresholds"], agg["deploy_wh"]
    iss = agg["issues"]
    out: List[str] = []
    a = out.append

    a("# 데이터 카드 (manifest v1, 키포인트 v1)")
    a("")
    a(HEADER_NOTE)
    a("")
    a(f"생성: `python -m training.datasets.data_card --manifest {manifest} --kp-dir {kp_dir} --out docs/DATA_CARD.md` "
      f"({time.strftime('%Y-%m-%d')}, {elapsed_s:.0f}초). 표는 모두 이 명령이 다시 만듭니다.")
    a("")

    a("## 1. 구성")
    a("")
    a(md_table(
        ["데이터셋", "클립", "낙상 / ADL", "event_group", "라벨 출처", "프레임 수", "프레임 크기 (클립 수)",
         "fps 반올림 (클립 수) [최소-최대]", "클립 길이 초: 중앙값 (p10-p90) [최소-최대]", "낙상 구간 초 (구간 수)"],
        [[d, per[d]["clips"], f"{per[d]['fall']} / {per[d]['adl']}", per[d]["event_groups"],
          _counter(per[d]["label_source"]), f"{per[d]['frames']:,}", _counter(per[d]["frame_wh"], 3),
          f"{_counter(per[d]['fps'], 3)} [{_fmt_num(per[d]['fps_range'][0])}-{_fmt_num(per[d]['fps_range'][1])}]",
          f"{_dist(per[d]['duration'], 1)} [{_fmt_num(per[d]['duration']['min'], 1)}-"
          f"{_fmt_num(per[d]['duration']['max'], 1)}]",
          f"{_dist(per[d]['fall_seg_dur'])} ({per[d]['n_fall_segments']})"] for d in order]))
    a("")
    a(md_table(["데이터셋", "라이선스 (manifest)"], [[d, _counter(per[d]["license"], 2)] for d in names]))
    a("")
    a("- fps는 클립 meta의 native_fps (출처: "
      + ", ".join(f"{d} {_counter(per[d]['fps_source'], 2)}" for d in names) + ")")
    if iss.get("kp_missing"):
        a(f"- 키포인트 파일 없음 {len(iss['kp_missing'])}개 (예: {', '.join(iss['kp_missing'][:3])}), 아래 키포인트 통계에서 제외")
    a("- 구간 클래스 (manifest 전체): " + _counter(agg["segment_classes"], 20))
    a("")

    a("## 2. 포즈 품질")
    a("")
    a(f"검출 비율 = 원본 프레임 중 대상이 선택된 비율. 가시율 = features v2와 같은 15Hz conf 마스크 평균 "
      f"(conf >= {MIN_CONF}, {MAX_GAP_S}초 이하 공백만 보간, 사람 없는 시점은 0). "
      f"낙상 구간 = [fall_start, fall_end + {FALL_POST_S:g}초], 시작 구간 = [fall_start - {ONSET_PRE_S:g}, "
      f"fall_start + {ONSET_POST_S:g}) (E 단계 창과 같은 window_v2), 누운 뒤 = 4절과 같은 구간. "
      f"값은 클립별 값의 중앙값 (p10-p90). 시작 구간 평균은 DATASETS.md 추출 결과 표와 같은 정의.")
    a("")
    a(md_table(
        ["데이터셋", "검출 비율 (클립 전체)", "검출 비율 (낙상 구간)", "가시율 (클립 전체)", "가시율 (낙상 구간)",
         f"낙상 구간 가시율 < {LOW_VIS:g}", "시작 구간 가시율 평균", f"시작 구간 가시율 < {LOW_VIS:g}",
         "검출 비율 (누운 뒤)",
         f"누운 뒤 검출 < {LOW_PRESENT:g}", "여러 명 프레임 있는 클립", "대상 전환 있는 클립"],
        [[d, _dist(per[d]["present"]), _dist(per[d]["fall_present"]), _dist(per[d]["vis"]),
          _dist(per[d]["fall_vis"]),
          f"{per[d]['fall_vis_low']} / {per[d]['fall_vis_n']}" if per[d]["fall_vis_n"] else "-",
          _fmt_num(per[d]["onset_vis_mean"]),
          f"{per[d]['onset_vis_low']} / {per[d]['onset_vis']['n']}" if per[d]["onset_vis"]["n"] else "-",
          _dist(per[d]["lying_present"]),
          f"{per[d]['lying_present_low']} / {per[d]['lying_present_n']}" if per[d]["lying_present_n"] else "-",
          f"{per[d]['multi_person_clips']} / {per[d]['with_kp']}",
          f"{per[d]['switch_clips']} / {per[d]['with_kp']}"] for d in order]))
    a("")
    a("부위별 가시율 (클립 중앙값, 클립 전체 / 낙상 구간). head = 코·눈·귀, arms = 팔꿈치·손목, "
      "torso = 어깨·골반, legs = 무릎·발목")
    a("")
    a(md_table(["데이터셋"] + list(JOINT_GROUPS),
               [[d] + [f"{_fmt_num(per[d]['vis_groups'][g])} / {_fmt_num(per[d]['fall_vis_groups'][g])}"
                       for g in JOINT_GROUPS] for d in order]))
    a("")
    if agg["mcfd_cams"]:
        a("MCFD 카메라별 (같은 24개 시나리오를 8대가 동시에 촬영)")
        a("")
        a(md_table(["cam", "클립", "검출 비율 중앙값", "시작 구간 가시율 중앙값", "낙상 구간 가시율 중앙값",
                    f"낙상 구간 가시율 < {LOW_VIS:g}", "누운 뒤 검출 비율 중앙값"],
                   [[c, v["clips"], _fmt_num(v["present"]), _fmt_num(v["onset_vis"]), _fmt_num(v["fall_vis"]),
                     f"{v['fall_vis_low']} / {v['n_fall']}", _fmt_num(v["lying_present"])]
                    for c, v in agg["mcfd_cams"].items()]))
        a("")

    a("## 3. 하강 속도: 낙상 vs 하드 ADL")
    a("")
    a(f"골반 중점의 아래 방향 속도 최댓값 (몸통 길이/초). 등방 픽셀(x*W, y*H), 15Hz 리샘플링, 중앙 차분이고 "
      f"앞뒤 시점이 모두 보일 때만 계산 (공백은 잇지 않음). 몸통 길이 = 클립 전체에서 어깨 중점-골반 중점 거리의 "
      f"중앙값. 측정 구간 = 라벨 구간 ± {VEL_PAD_S:g}초. 하드 ADL = " + ", ".join(HARD_ADL_CLS)
      + " 구간 (manifest에 실제로 있는 것: "
      + ", ".join(f"{c} {agg['segment_classes'].get(c, 0)}" for c in HARD_ADL_CLS) + ").")
    a("")
    a(md_table(
        ["데이터셋", "낙상 구간 (n)", "하드 ADL, ADL 클립 (n)", "하드 ADL, 낙상 클립 (n)",
         "낙상 중앙값보다 빠른 하드 ADL", "낙상 p10보다 빠른 하드 ADL", "하드 ADL p90보다 느린 낙상"],
        [[d, f"{_dist(per[d]['fall_vel'])} ({per[d]['fall_vel']['n']})",
          f"{_dist(per[d]['hard_adl_vel'])} ({per[d]['hard_adl_vel']['n']})",
          f"{_dist(per[d]['hard_fallclip_vel'])} ({per[d]['hard_fallclip_vel']['n']})",
          _pct(per[d]["overlap"]["adl_gt_fall_median"]), _pct(per[d]["overlap"]["adl_gt_fall_p10"]),
          _pct(per[d]["overlap"]["fall_lt_adl_p90"])] for d in order if d != "figshare_ursul29"]))
    a("")
    hc = per[ALL]["hard_by_cls"]
    if hc:
        a("ADL 클립의 하드 ADL 클래스별 (전체): "
          + ", ".join(f"{c} {_dist(v)} (n={v['n']})" for c, v in hc.items()))
        a("")
    if "figshare_ursul29" in per:
        f = per["figshare_ursul29"]
        a(f"Figshare는 구간 라벨이 없어 클립 전체 최댓값으로 비교: 낙상 폴더 {_dist(f['clip_vel_fall'])} "
          f"(n={f['clip_vel_fall']['n']}), ADL 폴더 {_dist(f['clip_vel_adl'])} (n={f['clip_vel_adl']['n']}), "
          f"낙상 중앙값보다 빠른 ADL {_pct(f['clip_overlap']['adl_gt_fall_median'])}, "
          f"ADL p90보다 느린 낙상 {_pct(f['clip_overlap']['fall_lt_adl_p90'])}.")
        a("")
    a("최댓값이라 키포인트 튐과 대상 전환(다른 사람으로 바뀜)에 민감합니다. 대상 전환이 있는 클립과 없는 클립의 낙상 구간 값: "
      + "; ".join(f"{d} {_dist(per[d]['fall_vel_switch'])} (n={per[d]['fall_vel_switch']['n']}) vs "
                  f"{_dist(per[d]['fall_vel_noswitch'])} (n={per[d]['fall_vel_noswitch']['n']})"
                  for d in order if per[d]["fall_vel_switch"]["n"] >= 5)
      + ". v1 키포인트에는 프레임별 전환 표시(switched)가 없어 클립 단위로만 나눴습니다 (재추출 클립은 전환 시점을 빼고 계산).")
    a("")
    missing = sum(per[d]["fall_vel_missing"] for d in names)
    if missing:
        a(f"- 낙상 구간 중 {missing}개는 골반이 연속 3시점 보이지 않아 속도를 못 구함 (n에서 제외)")
        a("")

    a("## 4. 프레임 종횡비와 규칙 엔진 임계값")
    a("")
    a("규칙 엔진(`app/core/level1.py`)의 종횡비는 정규화 bbox의 h/w입니다. 정규화 h/w = 픽셀 h/w x W/H 이므로 "
      "같은 임계값이 프레임 크기마다 다른 픽셀 자세를 뜻합니다 (픽셀 환산 = 임계값 x H/W). "
      + ", ".join(f"{k} = {_fmt_num(v)}" for k, v in th.items())
      + f", 배포 카메라 = app/config.py FRAME_W x FRAME_H = {dwh[0]}x{dwh[1]}.")
    a("")
    a(md_table(["프레임 (W x H)", "데이터셋 (클립)", "W/H"] + [f"{k} 픽셀 환산" for k in th],
               [[f"{r['wh'][0]}x{r['wh'][1]}" + (" (배포)" if r["deploy"] else ""),
                 _counter(r["datasets"], 3) if r["datasets"] else "-",
                 _fmt_num(r["wh"][0] / r["wh"][1])] + [_fmt_num(r["px"][k]) for k in th]
                for r in agg["frame_sizes"]]))
    a("")
    a(f"낙상 클립의 서 있을 때(클립 처음 {STAND_S:g}초, 낙상 시작 전까지)와 누운 뒤(첫 fall 뒤 fallen/lying 구간, "
      f"없으면 마지막 {LYING_END_S:g}초, 낙상 뒤 일어나는 라벨(stand_up/standing/walk)이 있으면 제외)의 "
      f"정규화 bbox 종횡비 중앙값 (p10-p90). '배포 환산' = 같은 픽셀 자세를 {dwh[0]}x{dwh[1]}에서 본 정규화 값. "
      "처음 1초가 서 있는 자세라는 보장은 없습니다 (앉거나 걷기 시작 등).")
    a("")
    a(md_table(
        ["데이터셋", "서 있음", "서 있음 (배포 환산)", f"서 있는데 < {_fmt_num(th['aspect_abs_th'])}",
         "누운 뒤", "누운 뒤 (배포 환산)", f"누운 뒤 < {_fmt_num(th['confirm_aspect_abs'])} (원본 / 배포 환산)",
         f"누운 뒤 < {_fmt_num(th['force_abnormal_aspect_th'])} (원본 / 배포 환산)", "누운 뒤 기준 (라벨/끝/일어남)"],
        [[d, _dist(per[d]["stand_aspect"]), _dist(per[d]["stand_dep"]), _pct(per[d]["stand_below_abs"]),
          _dist(per[d]["lying_aspect"]), _dist(per[d]["lying_dep"]),
          f"{_pct(per[d]['lying_below_confirm'])} / {_pct(per[d]['lying_dep_below_confirm'])}",
          f"{_pct(per[d]['lying_below_force'])} / {_pct(per[d]['lying_dep_below_force'])}",
          f"{per[d]['lying_src'].get('label', 0)}/{per[d]['lying_src'].get('clip_end', 0)}/"
          f"{per[d]['lying_src'].get('recovered', 0)}"] for d in order]))
    a("")

    a("## 5. 알려진 라벨/데이터 문제 (데이터로 확인한 결과)")
    a("")
    issue_rows = []
    if "mcfd_fps" in iss:
        m = iss["mcfd_fps"]
        issue_rows.append(["MCFD 헤더 fps 120, 실제 30",
                           f"헤더 fps {_counter(m['container_fps'])}, 사용 fps {_counter(m['native_fps'])} "
                           f"(출처 {_counter(m['fps_source'])}). 라벨 끝이 클립 길이 안에 드는 클립: 30fps 기준 "
                           f"{m['labels_fit_30']}/{m['clips']}, 120fps 기준 {m['labels_fit_120']}/{m['clips']}",
                           "manifest fps 30 우선 (build_manifest FPS_OVERRIDE)"])
    if "urfd_crop" in iss:
        u = iss["urfd_crop"]
        issue_rows.append(["URFD cam0 = 깊이 | RGB 합성 (640x240)",
                           f"manifest crop {_counter(u['row_crop'])}, 추출 meta crop {_counter(u['meta_crop'])}, "
                           f"키포인트 프레임 {_counter(u['frame_wh'])}",
                           "RGB 오른쪽 절반만 추출, frame_wh는 crop 크기"])
    if "urfd_fall_len" in iss:
        ul = iss["urfd_fall_len"]
        issue_rows.append(["URFD 낙상 구간 길이 고정",
                           f"낙상 구간 길이(초) {_counter(ul)}: 상태 0(넘어지는 중) 프레임 수가 시퀀스마다 같음",
                           "실측 낙상 길이가 아님. 낙상 구간 길이 통계와 '제때' 기준 끝 시각은 참고용"])
    if "gmdcsa24_s2_fall03" in iss:
        g = iss["gmdcsa24_s2_fall03"]
        others = [c for c in iss["gmdcsa24_fall_folder_adl"] if c != "gmdcsa24/Subject_2/Fall/03"]
        issue_rows.append(["GMDCSA24 Subject_2/Fall/03",
                           f"label {g['label']}, 구간 {' > '.join(g['segments'])}. Fall 폴더인데 ADL인 다른 클립 "
                           f"{len(others)}개, ADL 폴더인데 낙상인 클립 {len(iss['gmdcsa24_adl_folder_fall'])}개",
                           "OmniFall 라벨(눕기)을 따름"])
    lo = iss.get("label_overrun", {})
    if lo:
        issue_rows.append(["라벨 끝 > 클립 끝",
                           "; ".join(f"{d} {v['n']}개 (중앙값 {_fmt_num(v['median'])}초, 최대 {_fmt_num(v['max'])}초)"
                                     for d, v in sorted(lo.items())),
                           "추출기 허용치(길이 x 1.05 + 0.5초) 안. 재생은 마지막 프레임을 반복하므로 끝 부분 라벨은 정지 꼬리에 걸림"])
    if iss.get("fall_after_end"):
        issue_rows.append(["낙상 시작이 클립 끝 이후", f"{len(iss['fall_after_end'])}개: "
                           + ", ".join(iss["fall_after_end"][:5]), "평가에서 제외 검토"])
    if "figshare" in iss:
        f = iss["figshare"]
        issue_rows.append(["Figshare 클립 단위 라벨", f"구간 있는 클립 {f['with_segments']}/{f['clips']}, 길이 "
                           f"{_dist(f['duration'])}초", "낙상 시점 모름: 클립 끝 기준 창, 시간 지표 계산 불가"])
    mg = iss.get("multi_clip_groups", {})
    if mg:
        desc = []
        for d, groups in sorted(mg.items()):
            sizes = Counter(len(g) for g in groups)
            ex = [g for g in groups if len(g) <= 3][:3]
            desc.append(f"{d}: {len(groups)}개 그룹 (크기 {_counter(sizes)})"
                        + (f" 예: {'; '.join(' = '.join('`' + x.split('/', 1)[1] + '`' for x in g) for g in ex)}" if ex
                           and d != "mcfd" else ""))
        issue_rows.append(["여러 클립이 한 event_group", " / ".join(desc),
                           "MCFD는 시나리오 단위(8시점), 나머지는 바이트가 같은 중복 영상. 분할은 event_group 단위"])
    a(md_table(["문제", "확인 결과", "처리"], issue_rows))
    a("")
    if "mcfd" in per:
        a(f"MCFD 카메라 편차는 2절 카메라별 표 참조 (낙상 구간 가시율 {LOW_VIS:g} 미만 "
          f"{per['mcfd']['fall_vis_low']}/{per['mcfd']['fall_vis_n']}).")
        a("")

    a("## 6. 요약")
    a("")
    for line in findings(agg):
        a(f"- {line}")
    a("")

    a("## 7. Le2i subject 값")
    a("")
    ls = agg["le2i_subjects"]
    if ls["subjects"]:
        subs = sorted(ls["subjects"], key=lambda s: (len(str(s)), str(s)))
        a(md_table(["장면 (cam)"] + [f"s{s}" for s in subs],
                   [[f"{sc} ({','.join(c for c, v in ls['cam_scene'].items() if sc in v)})"]
                    + [ls["by_scene"][sc].get(s, "") for s in subs] for sc in ls["by_scene"]]
                   + [["합계 (낙상)"] + [f"{ls['subjects'][s]} ({ls['subject_fall'].get(s, 0)})" for s in subs]]))
        a("")
    a(LE2I_NOTE)
    a("")
    a(CAVEATS_NOTE)
    a("")
    return "\n".join(out)


def findings(agg: Dict[str, Any]) -> List[str]:
    """표에서 뽑은 핵심 사실 (숫자는 모두 집계값)."""
    per, th, dwh = agg["per"], agg["thresholds"], agg["deploy_wh"]
    out = []
    seg = [d for d in agg["datasets"] if per[d]["fall_vis_n"]]
    if seg:
        lp = ", ".join(f"{d} {per[d]['lying_present_low']}/{per[d]['lying_present_n']}" for d in seg)
        n_on = sum(per[d]["onset_vis"]["n"] for d in seg)
        low_on = {d: per[d]["onset_vis_low"] for d in seg if per[d]["onset_vis_low"]}
        out.append(f"**누운 사람 검출 실패**: 낙상 시작 구간 가시율 < {LOW_VIS:g}인 낙상 클립은 "
                   f"{sum(low_on.values())}/{n_on}"
                   + (f" ({', '.join(f'{d} {n}' for d, n in low_on.items())})" if low_on else "")
                   + f"뿐이지만, 누운 뒤 검출 비율 < {LOW_PRESENT:g}인 낙상 클립은 {lp}입니다. "
                   "낙상 동작은 보이는데 바닥에 누운 뒤 사람을 놓치는 경우가 많습니다. Le2i 누락 8개를 영상으로 보니 6개는 "
                   "매트 위에 누운 사람이 보이는데 놓쳤고(어두운 옷, 어두운 조명 포함), 2개는 탁자·담요에 상당 부분 "
                   "가려져 있었습니다. 포즈 모델 개선(P)의 목표 장면은 누운 자세입니다.")
    if agg["mcfd_cams"]:
        cams = sorted(agg["mcfd_cams"].items(),
                      key=lambda kv: kv[1]["fall_vis"] if kv[1]["fall_vis"] is not None else 9.0)
        (wc, wv), (bc, bv) = cams[0], cams[-1]
        out.append(f"**MCFD 시점 편차**: 같은 사건인데 낙상 구간 가시율 중앙값이 cam {wc} {_fmt_num(wv['fall_vis'])}, "
                   f"cam {bc} {_fmt_num(bv['fall_vis'])}이고 누운 뒤 검출 비율은 {_fmt_num(wv['lying_present'])} vs "
                   f"{_fmt_num(bv['lying_present'])}입니다. 시점별 결과를 따로 보고, 8시점 클립을 독립 표본으로 세지 않습니다.")
    o = per[ALL]["overlap"]
    out.append(f"**하강 속도 겹침**: 낙상 구간 최대 하강 속도 {_dist(per[ALL]['fall_vel'])} vs ADL 클립의 하드 ADL "
               f"{_dist(per[ALL]['hard_adl_vel'])} (몸통 길이/초). 하드 ADL 중 낙상 중앙값보다 빠른 것 "
               f"{_pct(o['adl_gt_fall_median'])}, 낙상 p10보다 빠른 것 {_pct(o['adl_gt_fall_p10'])}. "
               "중앙값은 갈리지만 꼬리가 겹쳐 속도 하나로는 구분되지 않고, 데이터셋마다 차이가 큽니다 (3절).")
    sizes = [r for r in agg["frame_sizes"] if not r["deploy"]]
    if sizes:
        k = "aspect_abs_th"
        lo = max(sizes, key=lambda r: r["px"][k])
        le = per.get("le2i", {})
        out.append(f"**종횡비 임계값의 뜻이 프레임마다 다름**: {k} {_fmt_num(th[k])}는 {lo['wh'][0]}x{lo['wh'][1]}에서 "
                   f"픽셀 h/w {_fmt_num(lo['px'][k])}, 배포 {dwh[0]}x{dwh[1]}에서 "
                   f"{_fmt_num(pixel_threshold(th[k], *dwh))}입니다. 같은 누운 자세를 배포 프레임으로 환산하면 "
                   f"confirm_aspect_abs {_fmt_num(th['confirm_aspect_abs'])} 미만 비율이 전체 "
                   f"{_pct(per[ALL]['lying_below_confirm'])} -> {_pct(per[ALL]['lying_dep_below_confirm'])}, Le2i "
                   f"{_pct(le.get('lying_below_confirm'))} -> {_pct(le.get('lying_dep_below_confirm'))}로 줄어듭니다. "
                   "4:3 공개 데이터 재생은 배포 카메라보다 자세 조건을 쉽게 통과합니다.")
    return out


HEADER_NOTE = """\
학습/평가에 쓰는 공개 데이터 6종(통합 manifest v1)의 구성, 포즈 품질, 동작 분포, 알려진 문제를 정리합니다.
데이터 수집 경위와 라이선스 원문은 [DATASETS.md](DATASETS.md), 실험 프로토콜은 [MODEL_STUDY.md](MODEL_STUDY.md)를 봅니다.
키포인트는 운영과 같은 yolov8s-pose(FP32, conf 0.5, imgsz 640)로 추출했습니다. 원본 영상과 키포인트는 git 제외입니다."""

LE2I_NOTE = """\
결론: Le2i의 subject 값(0~8)은 자리표시자가 아니라 실제 배우 번호로 봅니다.

- 출처: manifest의 subject/cam은 OmniFall `labels__le2i.csv` 값을 그대로 씁니다 (build_manifest `read_omnifall`).
  Le2i 원본에는 배우 번호가 없고, OmniFall README는 Le2i를 "9 subjects, 6 different rooms"로 적고 있어 0~8의 9개 값과 맞습니다.
  cam은 장면 폴더와 1:1입니다 (cam 1~6 = Coffee_room_01, Coffee_room_02, Home_01, Home_02, Lecture_room, Office).
- 육안 확인: subject별로 장면이 다른 클립 6~16개의 낙상 직전 프레임을 겹쳐 보았을 때 s0(가장 많은 67개), s1, s7은
  장면이 바뀌어도 같은 사람으로 보였습니다 (옷은 장면마다 다름). 표본 확인이라 모든 클립을 검증한 것은 아닙니다.
- 결과: 모든 주요 subject가 여러 장면에 나오고 s0 한 명이 Le2i의 약 35%입니다. 그래서
  - 교차 피험자 분할(OmniFall cs: 학습 s0,1,3,4,6,8 / 검증 s5 / 테스트 s2,7)은 가능하지만 장면(배경, 카메라)은 학습과 테스트에 함께 들어갑니다.
    피험자 일반화와 장면 일반화를 따로 보려면 cs와 cv(장면 단위)를 둘 다 써야 합니다.
  - Coffee_room_01과 Coffee_room_02는 같은 방, 같은 시점으로 보입니다 (cam 번호만 다름). 장면 단위 분할에서도 둘은 같은 쪽에 두는 것이 안전합니다
    (OmniFall cv 분할도 둘 다 학습: 학습 Coffee_room_01/02, Lecture_room / 검증 Office / 테스트 Home_01/02).
  - 지금 progression 실험의 fold는 event_group 단위라 같은 배우의 다른 클립이 학습과 테스트에 나뉩니다. 사건 누수는 없지만
    피험자 일반화 수치는 아닙니다. 다른 데이터셋과 subject 번호 체계가 달라 데이터셋 간 subject 비교는 의미가 없습니다."""

CAVEATS_NOTE = """\
## 이 데이터로 할 수 없는 것 / 주의

- **실제 낙상 성능 추정**: 대부분 젊은 배우의 연출 낙상(매트, 실험실·연출된 방)입니다. 고령자, 보조기구, 느린 붕괴, 가구에 가려진 낙상은 거의 없습니다.
- **배포 카메라 조건 재현**: 1920x1080은 Figshare뿐이고(16:9는 GMDCSA24 1280x720, Le2i 일부 320x180 포함),
  나머지는 4:3/3:2 저해상도입니다. 카메라 높이와 각도도 데이터셋마다 다르고 배포 설치(벽 모서리 2m 이상)와 같다는 보장이 없습니다.
  규칙 엔진의 종횡비 임계값은 정규화 좌표 기준이라 4절의 표처럼 프레임 비율에 따라 실제 뜻이 달라집니다.
  공개 데이터로 맞춘 임계값을 배포 카메라에 그대로 옮기면 안 됩니다.
- **시간당 오탐(FA/h)**: ADL은 짧은 클립 모음이라 연속 시간이 부족합니다. 클립 단위 오탐률만 낼 수 있습니다.
- **침대·소파 눕기 하드 네거티브**: 하드 ADL 구간이 수십 개뿐이고 kneel_down/squat_down은 없습니다(OmniFall staged는 10클래스만 사용).
  장소(바닥/침대/소파) 정보도 라벨에 없습니다.
- **Figshare 낙상 시점**: 클립 단위 라벨만 있어 시간 기준 지표(제때 탐지)나 낙상 구간 품질을 낼 수 없습니다. 클립 수가 가장 많아 전체 합계를 지배하므로 데이터셋별로 봅니다.
- **MCFD 독립성**: 184개 낙상 클립은 낙상 시나리오 23개 x 8시점입니다 (OmniFall 기준 낙상 구간은 시나리오당 1~2개, 나머지 1개 시나리오 8클립이 ADL).
  배우도 1명이라 독립 표본은 클립 수보다 훨씬 적습니다. 클립 수로 신뢰구간을 계산하면 안 됩니다.
- **라벨 기준 차이**: OmniFall fall/fallen, URFD 깊이 기반 상태(넘어지는 중/누움), Figshare 폴더 라벨은 낙상 시작·끝 정의가 다릅니다. 데이터셋 간 지연 비교는 참고용입니다.
- **포즈 품질 지표의 한계**: 가시율은 YOLO conf 기준이라, conf가 높지만 골격이 틀린 경우(누운 자세에서 좌우·상하 혼동)는 잡히지 않습니다.
- **라이선스**: Le2i, URFD와 OmniFall 구간 라벨(CAUCAFall, GMDCSA24 라벨 포함)은 비상업(NC)이고 MCFD는 명시가 없습니다.
  상업 용도로 보수적으로 가려면 영상과 라벨이 모두 CC BY/MIT인 것(Figshare 폴더 라벨, CAUCAFall·GMDCSA24 원본 라벨)만으로 다시 학습해야 합니다.
  지금 manifest의 CAUCAFall·GMDCSA24 구간은 OmniFall 라벨이라 해당하지 않습니다."""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def compute(rows: Sequence[Dict[str, Any]], kp_dir: Path, progress: bool = True) -> List[Dict[str, Any]]:
    stats = []
    for i, r in enumerate(rows, 1):
        try:
            clip = load_clip_for(r, kp_dir)
        except FileNotFoundError:
            continue
        stats.append(clip_stats(r, clip))
        if progress and i % 500 == 0:
            print(f"  {i}/{len(rows)}", file=sys.stderr)
    return stats


def main(argv=None) -> int:
    from training.datasets.manifest import load_manifest
    from training.paths import REPO_ROOT

    ap = argparse.ArgumentParser(description="Compute dataset statistics and write docs/DATA_CARD.md")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "dataset" / "_manifest" / "manifest_v1.jsonl")
    ap.add_argument("--kp-dir", type=Path, default=REPO_ROOT / "dataset" / "_keypoints" / "v1")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "DATA_CARD.md")
    ap.add_argument("--json", type=Path, default=None, help="집계 결과를 JSON으로도 저장 (선택)")
    ap.add_argument("--datasets", nargs="+", default=None, choices=DATASETS)
    args = ap.parse_args(argv)

    t0 = time.time()
    rows = load_manifest(args.manifest)
    if args.datasets:
        rows = [r for r in rows if r["dataset"] in args.datasets]
    stats = compute(rows, args.kp_dir)
    agg = aggregate(rows, stats, rule_aspect_thresholds(), deploy_frame_wh())
    md = render(agg, args.manifest.as_posix(), args.kp_dir.as_posix(), time.time() - t0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(agg, f, ensure_ascii=False, indent=1, default=lambda o: dict(o) if isinstance(o, Counter)
                      else list(o) if isinstance(o, (tuple, set)) else str(o))
    print(f"rows={len(rows)} clips_with_keypoints={len(stats)} ({time.time() - t0:.0f}s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
