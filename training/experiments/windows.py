"""
진행 단계 연구(progression study)의 학습 창(window) 생성기 (numpy만 사용, 재생이 필요한 단계는 replay.py 사용)

단계별 창 (docs/MODEL_STUDY.md 프로토콜)
- B/C windows_original_urfd: 원래 방식 재현 (git show 9e40e48:app/ai_classifier/dataset_builder.py)
    원본 fps 그대로 30프레임 창, 간격 5, 영상 전체 라벨 (낙상 영상의 모든 창 = 1).
    프레임마다 골반 중심 상대좌표(features.normalize_frame_v1), 사람이 없는 프레임은 직전 프레임 반복
    (첫 검출 전에는 0), 30프레임보다 짧은 영상은 마지막 프레임으로 채운 창 1개.
    데이터셋 필터는 하지 않음: B는 URFD 행만, C는 모든 데이터셋 행을 넘김.
- D windows_runtime_v1: 운영이 보는 그대로의 v1 입력
    replay.clip_to_obs (15Hz, EMA, 정지 꼬리)를 Level1Engine.kp_seq와 같은 규칙으로 쌓은 30프레임 창.
    양성 = 창 끝이 낙상 후 누운 구간(fall 다음의 fallen/lying), 음성 = ADL 클립의 모든 창(눕기/앉기 포함) +
    낙상 클립의 낙상 시작 전 창. 추가로 학습 클립을 규칙 엔진으로 재생해 LEVEL1 시점의 kp_seq를 클립 라벨로 넣음.
- E windows_event_v2: ABNORMAL 진입 사건 구간 분류 (features.window_v2, 92채널 x 60스텝 = 4초 @15Hz)
    양성 = 낙상 시작 기준 [fall_start - 1.5, fall_start + 2.5] (±0.5초 이동, 낙상 구간마다 여러 개),
    음성 = ADL 슬라이딩 창(1초 간격) + 낙상 시작 전 창 + ADL 클립을 규칙 엔진으로 재생해 ABNORMAL 진입 때 잡힌 창.
    낙상 클립에서 낙상 근처의 ABNORMAL 진입 창도 양성으로 넣음 (운영 입력 분포 그대로, fall_entry_positives).
    관절 가시율(마스크 채널 평균) 0.3 미만 창 제외, augment=True면 좌우 반전/시간 늘이기/관절 dropout.
- E0 handcrafted: v2 창에서 손으로 만든 통계 특징 (HistGradientBoosting 비교군)

입력 형식
- rows: manifest 행 목록 (training/datasets/manifest.py), clips: {clip_id: clipfmt.load_clip 결과}
  (load_clips로 읽음). clips에 없는 행은 건너뜀.
- 모든 함수는 return_meta=True면 창마다 {"clip_id","dataset","label","kind","y"} 목록을 세 번째로 반환합니다
  (데이터셋/종류별 개수 보고용, count_windows).

운영 입력과 다른 점
- 재생 기반 창(D/E)은 사건 이후 쿨다운(25초) 동안 엔진이 kp_seq/kp_buffer를 갱신하지 않는 점은 따라 하지 않음
  (창 위치는 라벨 구간으로 정하므로 영향은 LEVEL1 이후 구간뿐).
- Figshare 클립(약 1.9초, 구간 라벨 없음)은 낙상 시점을 몰라서 클립 끝 기준 창을 씀 (낙상/ADL 동일 규칙).
"""
import dataclasses
import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from app.ai_classifier import features as _F

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

SEQ_LEN = 30                 # v1 창 길이 (프레임), Fall1DCNN 입력 (34, 30)
ORIGINAL_STEP = 5            # 원래 방식 학습 창 간격 (원본 fps 프레임)
RUNTIME_STEP = 3             # D 창 간격 (15Hz 프레임, 0.2초)

V2_PRE_S = 1.5               # 사건 구간: 기준 시각 - 1.5초 ~ + 2.5초 (Level1Params.event_window_pre/post_s 기본값)
V2_POST_S = 2.5
V2_WINDOW_S = V2_PRE_S + V2_POST_S
V2_STEPS = int(round(V2_WINDOW_S * _F.TARGET_HZ))   # 60
V2_CHANNELS = len(_F.V2_CHANNELS)                   # 92
_MASK0 = _F.V2_CHANNELS.index("mask_nose")          # 마스크 채널 시작 (75)

MIN_VISIBILITY = 0.3         # 창의 관절 가시율(마스크 평균) 하한
ONSET_SHIFT_S = 0.5          # 양성 onset 창 이동 범위 (±)
N_ONSET = 5                  # 낙상 구간당 양성 창 수
NEG_STEP_S = 1.0             # 음성 슬라이딩 창 간격
PRE_FALL_MARGIN_S = 0.5      # 낙상 시작 전 음성 창은 끝이 fall_start - 0.5초 이전
TAIL_USE_S = 2.0             # 정지 꼬리 중 창에 쓰는 길이 (꼬리는 마지막 프레임의 단순 반복이라 길게 쓸 이유가 없음)
FALL_ENTRY_BEFORE_S = 1.0    # 낙상 클립의 ABNORMAL 진입 창을 양성으로 볼 기준 시각 범위 [fall_start - 1, fall_end + 1.5]
FALL_ENTRY_AFTER_S = 1.5

POST_FALL_CLS = ("fallen", "lying")   # 낙상 직후 누운 상태 (OmniFall fallen, URFD lying)
_SEG_GAP_S = 0.5             # 이어진 구간으로 보는 최대 간격

AUG_FLIP_P = 0.5
AUG_WARP = (0.8, 1.25)
AUG_DROPOUT_P = 0.5          # 창마다 관절 dropout을 적용할 확률
AUG_JOINT_DROP_P = 0.1       # 관절 전체를 지울 확률 (관절마다)
AUG_POINT_DROP_P = 0.05      # (프레임, 관절) 단위로 지울 확률

_T0 = 1000.0                 # 재생 obs 시각 기준 (obs ts - _T0 = 클립 시각)
_EPS = 1e-6

__all__ = [
    "SEQ_LEN", "V2_STEPS", "V2_WINDOW_S", "MIN_VISIBILITY", "HANDCRAFTED_NAMES",
    "load_clips", "original_sequence", "windows_original_urfd",
    "runtime_v1_stream", "windows_runtime_v1",
    "obs_event_arrays", "model_window_bounds", "visibility", "augment_window_arrays", "windows_event_v2",
    "handcrafted", "count_windows", "clear_cache",
]


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------

def load_clips(rows: Iterable[Dict[str, Any]], kp_dir=None, missing_ok: bool = True) -> Dict[str, Dict[str, Any]]:
    """manifest 행들의 키포인트 클립을 {clip_id: clip}으로 읽습니다. 파일이 없으면 건너뜀 (missing_ok=False면 예외)."""
    from training.experiments.replay import load_row_clip
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        try:
            out[r["clip_id"]] = load_row_clip(r, kp_dir)
        except FileNotFoundError:
            if not missing_ok:
                raise
    return out


def _label(row: Dict[str, Any]) -> int:
    return 1 if row.get("label") == "fall" else 0


def _meta(row: Dict[str, Any], kind: str, y: int, **extra) -> Dict[str, Any]:
    m = {"clip_id": row.get("clip_id"), "dataset": row.get("dataset"), "label": row.get("label"),
         "kind": kind, "y": int(y)}
    m.update(extra)
    return m


def _has_segments(row: Dict[str, Any]) -> bool:
    return bool(row.get("segments"))


def _clip_level(row: Dict[str, Any]) -> bool:
    """
    시점 라벨 없이 클립 단위 라벨만 있는 행 (Figshare 폴더 라벨, 구간 없는 낙상 클립).
    구간이 없는 ADL 클립(URFD ADL)은 클립 전체가 ADL이므로 여기에 해당하지 않음.
    """
    if row.get("label_source") == "folder":
        return True
    return _label(row) == 1 and not _fall_segments(row)


def _fall_segments(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    segs = sorted(row.get("segments") or [], key=lambda s: (float(s["start_s"]), float(s["end_s"])))
    return [s for s in segs if s.get("cls") == "fall"]


def _post_fall_intervals(row: Dict[str, Any], clip_end: float, tail_use_s: float) -> List[Tuple[float, float]]:
    """
    낙상 구간마다 낙상 직후 누운 구간 [fall_end, 이어진 fallen/lying 구간의 끝].
    누운 구간이 없으면 [fall_end, fall_end + 1초]. 누운 채로 클립이 끝나면 정지 꼬리 tail_use_s까지 연장.
    """
    segs = sorted(row.get("segments") or [], key=lambda s: (float(s["start_s"]), float(s["end_s"])))
    out = []
    for i, s in enumerate(segs):
        if s.get("cls") != "fall":
            continue
        start = float(s["end_s"])
        end = start
        for nxt in segs[i + 1:]:
            # 바로 이어지는 fallen/lying만 (다른 동작이나 다음 낙상이 나오면 끝)
            if nxt.get("cls") not in POST_FALL_CLS or float(nxt["start_s"]) > end + _SEG_GAP_S:
                break
            end = max(end, float(nxt["end_s"]))
        if end <= start:
            end = start + 1.0
        if end >= clip_end - 0.1:
            end = clip_end + tail_use_s
        out.append((start, end))
    return out


def _clip_end(clip: Dict[str, Any]) -> float:
    ts = np.asarray(clip["ts"], dtype=np.float64).reshape(-1)
    return float(ts[-1]) if len(ts) else 0.0


def _empty_x(c: int, t: int) -> np.ndarray:
    return np.zeros((0, c, t), dtype=np.float32)


def _finish(xs: List[np.ndarray], ys: List[int], metas: List[Dict[str, Any]], c: int, t: int, return_meta: bool):
    X = np.stack(xs).astype(np.float32, copy=False) if xs else _empty_x(c, t)
    y = np.asarray(ys, dtype=np.int64)
    return (X, y, metas) if return_meta else (X, y)


def count_windows(meta: Sequence[Dict[str, Any]], by: Sequence[str] = ("dataset", "kind", "y")) -> Dict[str, int]:
    """meta 목록을 by 키 조합별로 셉니다. 반환: {"le2i/onset/1": 520, ...} (정렬됨)"""
    c = Counter("/".join(str(m.get(k)) for k in by) for m in meta)
    return dict(sorted(c.items()))


# 재생 결과 캐시: (clip_id, params repr) -> dict. fold마다 같은 클립을 다시 재생하지 않도록
_REPLAY_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def clear_cache():
    """재생 결과 캐시를 비웁니다."""
    _REPLAY_CACHE.clear()


# ---------------------------------------------------------------------------
# B/C: 원래 방식 (URFD 전용 dataset_builder 재현)
# ---------------------------------------------------------------------------

def original_sequence(clip: Dict[str, Any]) -> np.ndarray:
    """
    9e40e48 dataset_builder.build_dataset의 프레임 시퀀스를 클립 파일로 재현합니다. 반환: (T, 34) float32
    - 사람이 있으면 normalize_frame_v1(x, y) (원래 extract_keypoints가 첫 검출자의 xyn을 정규화한 것과 같음)
    - 없으면 직전 프레임 반복, 첫 검출 전이면 0
    """
    kps = np.asarray(clip["kps"], dtype=np.float32)
    present = np.asarray(clip["present"], dtype=bool).reshape(-1)
    seq = np.zeros((len(present), 34), dtype=np.float32)
    last = None
    for i in range(len(present)):
        if present[i]:
            last = _F.normalize_frame_v1(kps[i, :, :2]).astype(np.float32)
        if last is not None:
            seq[i] = last
    return seq


def windows_original_urfd(rows: Iterable[Dict[str, Any]], clips: Dict[str, Dict[str, Any]],
                          step: int = ORIGINAL_STEP, seq_len: int = SEQ_LEN, return_meta: bool = False):
    """
    원래 학습 방식의 창 (stage B: URFD 행만, stage C: 모든 데이터셋 행).
    원본 fps 그대로 seq_len 프레임 창을 step 간격으로, 라벨은 영상 전체 라벨 (fall=1, adl=0).
    짧은 영상은 마지막 프레임으로 채운 창 1개 (원래 코드와 같음, 빈 영상은 건너뜀).
    반환: X (N, 34, seq_len) float32 (이미 normalize_frame_v1 적용), y (N,) int64
    """
    xs, ys, metas = [], [], []
    for row in rows:
        clip = clips.get(row["clip_id"])
        if clip is None:
            continue
        seq = original_sequence(clip)
        if len(seq) == 0:
            continue
        y = _label(row)
        if len(seq) < seq_len:
            pad = np.repeat(seq[-1:], seq_len - len(seq), axis=0)
            wins = [np.concatenate([seq, pad], axis=0)]
        else:
            wins = [seq[i:i + seq_len] for i in range(0, len(seq) - seq_len + 1, step)]
        for w in wins:
            xs.append(w.T)
            ys.append(y)
            metas.append(_meta(row, "whole_video", y))
    return _finish(xs, ys, metas, 34, seq_len, return_meta)


# ---------------------------------------------------------------------------
# 재생 (D, E 공용): obs 스트림 -> v1 시퀀스 스트림 / 사건 구간 배열, 규칙 엔진 이벤트
# ---------------------------------------------------------------------------

def _is_lost(t0: Optional[Dict[str, Any]], min_quality: float) -> bool:
    """Level1Engine.step의 is_lost 판정과 같음."""
    if not isinstance(t0, dict) or not t0.get("has_person", False):
        return True
    return float(t0.get("quality_score") or 0.0) < min_quality


def runtime_v1_stream(obs_list: Sequence[Dict[str, Any]], min_quality: float, t0: float = _T0):
    """
    Level1Engine.kp_seq에 들어가는 프레임만 골라 (ts, seq, seg)를 반환합니다.
    - lost(사람 없음 또는 quality < min_quality) 프레임은 건너뜀 (kp_seq에 쌓이지 않음)
    - bbox가 없거나 잘못되면 엔진이 reset_all로 kp_seq를 비우므로 seg 번호를 올려 창이 넘어가지 않게 함
    - 값: keypoints_smooth(없으면 keypoints)의 x, y (관절 순서, conf 미포함, 없는 관절 0) = 엔진의 curr_kp_flat
    반환: ts (n,) 클립 기준 초, seq (n, 34) float32, seg (n,) int
    """
    ts, seq, seg = [], [], []
    cur = 0
    for obs in obs_list:
        tracks = obs.get("tracks") or []
        t0r = tracks[0] if tracks else None
        if _is_lost(t0r, min_quality):
            continue
        bbox = t0r.get("bbox")
        if not bbox or len(bbox) != 4:
            cur += 1
            continue
        flat = [0.0] * 34
        for k in (t0r.get("keypoints_smooth") or t0r.get("keypoints") or []):
            kid = int(k.get("id", -1))
            if 0 <= kid < 17:
                flat[kid * 2] = float(k.get("x", 0.0))
                flat[kid * 2 + 1] = float(k.get("y", 0.0))
        ts.append(float(obs["ts"]) - t0)
        seq.append(flat)
        seg.append(cur)
    return (np.asarray(ts, dtype=np.float64), np.asarray(seq, dtype=np.float32).reshape(-1, 34),
            np.asarray(seg, dtype=np.int64))


def obs_event_arrays(obs_list: Sequence[Dict[str, Any]], min_quality: float, t0: float = _T0,
                     t_max: Optional[float] = None) -> Dict[str, Any]:
    """
    obs 스트림 -> 엔진 kp_buffer(사건 구간 원본)와 같은 배열 (clipfmt.from_sidecar 규칙).
    - lost 프레임: kps/bbox 0, present False
    - 나머지: keypoints_raw(없으면 keypoints) [x, y, conf] (스무딩 전 원시값), bbox
    - bbox가 잘못된 프레임은 엔진이 버퍼에 넣지 않으므로 제외
    t_max: 클립 기준 이 시각 이후 obs는 버림 (정지 꼬리를 필요한 만큼만 보관)
    반환: {"ts" (T,) 클립 기준 초, "kps" (T,17,3) float32, "bbox" (T,4) float32, "present" (T,), "frame_wh" (2,) 또는 None}
    """
    ts, kps, bbox, present = [], [], [], []
    wh = None
    for obs in obs_list:
        t = float(obs["ts"]) - t0
        if t_max is not None and t > t_max + _EPS:
            break
        tracks = obs.get("tracks") or []
        t0r = tracks[0] if tracks else None
        if isinstance(t0r, dict):
            shape = t0r.get("frame_shape")
            if shape is not None and len(shape) >= 2 and int(shape[0]) > 0 and int(shape[1]) > 0:
                wh = (int(shape[1]), int(shape[0]))
        frame = np.zeros((17, 3), dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        ok = False
        if not _is_lost(t0r, min_quality):
            bb = t0r.get("bbox")
            if not bb or len(bb) != 4:
                continue
            for k in (t0r.get("keypoints_raw") or t0r.get("keypoints") or []):
                kid = int(k.get("id", -1))
                if 0 <= kid < 17:
                    frame[kid] = (float(k.get("x", 0.0)), float(k.get("y", 0.0)), float(k.get("conf", 0.0)))
            b = np.asarray([float(v) for v in bb], dtype=np.float32)
            ok = True
        ts.append(t)
        kps.append(frame)
        bbox.append(b)
        present.append(ok)
    return {
        "ts": np.asarray(ts, dtype=np.float64),
        "kps": np.asarray(kps, dtype=np.float32).reshape(-1, 17, 3),
        "bbox": np.asarray(bbox, dtype=np.float32).reshape(-1, 4),
        "present": np.asarray(present, dtype=bool),
        "frame_wh": wh,
    }


def model_window_bounds(window_start_s: float, window_end_s: Optional[float] = None) -> Tuple[float, float]:
    """
    사건 구간 [window_start_s, window_end_s] -> 모델 입력 4초 창 [start, start + 4).
    저자세 지속 진입은 구간 시작이 저자세 시작 - 1.5초라 4초보다 길 수 있음: 앞쪽 4초(하강 동작 포함)를 씀.
    학습 창(양성 onset 창과 같은 기준)과 EventWindowVerifier가 같은 규칙을 써야 합니다.
    """
    s = float(window_start_s)
    return s, s + V2_WINDOW_S


def _replay(row: Dict[str, Any], clip: Dict[str, Any], params, use_cache: bool = True) -> Dict[str, Any]:
    """
    클립을 운영 obs 스트림으로 바꿔 규칙 엔진(AI 끔)으로 재생합니다.
    반환: {"clip_end", "v1": (ts, seq, seg), "arrays": obs_event_arrays, "level1_seqs": [(ts, (30,34))],
           "entries": [{"anchor_s","start_s","reason"}]}
    """
    p = dataclasses.replace(params, ai_mode="off")
    key = (str(row["clip_id"]), repr(p))
    if use_cache and key in _REPLAY_CACHE:
        return _REPLAY_CACHE[key]
    from training.experiments import replay as _rp

    obs = _rp.clip_to_obs(clip, t0=_T0)
    level1_seqs: List[Tuple[float, np.ndarray]] = []
    entries: List[Dict[str, Any]] = []

    def on_event(ev, engine):
        typ = ev.get("type")
        if typ == "abnormal_enter":
            start = engine.abnormal_start_ts
            anchor = engine.event_anchor_ts if engine.event_anchor_ts is not None else start
            if start is not None:
                entries.append({"anchor_s": float(anchor) - _T0, "start_s": float(start) - _T0,
                                "reason": engine.enter_reason})
        elif typ == "level1" and len(engine.kp_seq) >= SEQ_LEN:
            level1_seqs.append((float(ev.get("ts", 0.0)) - _T0,
                                np.asarray(list(engine.kp_seq)[-SEQ_LEN:], dtype=np.float32)))

    _rp.run_clip(obs, p, verifier=None, on_event=on_event)
    clip_end = _clip_end(clip)
    # v2 창은 진입 기준 +2.5초(+여유)까지만 보므로 꼬리는 마지막 진입 이후 몇 초만 보관
    t_keep = clip_end + TAIL_USE_S + V2_WINDOW_S + 1.0
    for e in entries:
        t_keep = max(t_keep, e["start_s"] + V2_POST_S + 1.0)
    out = {
        "clip_end": clip_end,
        "v1": runtime_v1_stream(obs, p.min_quality),
        "arrays": obs_event_arrays(obs, p.min_quality, t_max=t_keep),
        "level1_seqs": level1_seqs,
        "entries": entries,
    }
    if use_cache:
        _REPLAY_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# D: 운영 정렬 v1 창
# ---------------------------------------------------------------------------

def _in_any(t: float, intervals: Sequence[Tuple[float, float]]) -> bool:
    return any(a - _EPS <= t <= b + _EPS for a, b in intervals)


def windows_runtime_v1(rows: Iterable[Dict[str, Any]], clips: Dict[str, Dict[str, Any]], params,
                       step: int = RUNTIME_STEP, tail_use_s: float = TAIL_USE_S,
                       include_level1: bool = True, use_cache: bool = True, return_meta: bool = False):
    """
    운영 입력과 같은 v1 창 (stage D). params: Level1Params (min_quality, 재생용 규칙 파라미터; AI는 끄고 재생).
    창 = kp_seq 규칙으로 쌓은 15Hz EMA 프레임 30개 (step 간격), 창 끝 시각 t_end로 라벨:
      - 구간 라벨 있는 낙상 클립: t_end가 낙상 후 누운 구간 -> 1 (post_fall), t_end <= 첫 fall_start -> 0 (pre_fall),
        그 밖(낙상 동작 중, 일어난 뒤)은 버림
      - 클립 단위 라벨만 있는 클립(Figshare, 약 1.9초): 낙상/ADL 같은 규칙으로 t_end가 [클립 끝, 클립 끝 + tail_use_s]인
        창만 -> 클립 라벨 (clip_end). 낙상 시점을 모르므로 클립 전체 + 마지막 자세가 들어간 창만 씀
      - 그 밖의 ADL 클립(구간 없는 URFD ADL 포함): t_end <= 클립 끝 + tail_use_s 인 모든 창 -> 0 (adl)
    include_level1: 규칙 엔진 재생에서 LEVEL1이 날 때의 kp_seq(검증기가 실제로 받는 입력)를 클립 라벨로 추가 (level1)
    반환: X (N, 34, 30) float32 (normalize_sequence_v1 적용 후 전치), y (N,)
    """
    xs, ys, metas = [], [], []
    for row in rows:
        clip = clips.get(row["clip_id"])
        if clip is None or len(clip["ts"]) == 0:
            continue
        rep = _replay(row, clip, params, use_cache=use_cache)
        ts, seq, seg = rep["v1"]
        clip_end = rep["clip_end"]
        lab = _label(row)
        clip_level = _clip_level(row)
        if lab == 1 and not clip_level:
            falls = _fall_segments(row)
            pos_iv = _post_fall_intervals(row, clip_end, tail_use_s)
            first_fall = float(falls[0]["start_s"])
        else:
            pos_iv, first_fall = [], -math.inf
        for i in range(SEQ_LEN - 1, len(ts), step):
            if seg[i - SEQ_LEN + 1] != seg[i]:
                continue
            t_end = float(ts[i])
            if t_end > clip_end + tail_use_s + _EPS:
                continue
            if clip_level:
                if t_end < clip_end - _EPS:
                    continue
                y, kind = lab, "clip_end"
            elif lab == 0:
                y, kind = 0, "adl"
            elif _in_any(t_end, pos_iv):
                y, kind = 1, "post_fall"
            elif t_end <= first_fall + _EPS:
                y, kind = 0, "pre_fall"
            else:
                continue
            xs.append(_F.normalize_sequence_v1(seq[i - SEQ_LEN + 1:i + 1]).T)
            ys.append(y)
            metas.append(_meta(row, kind, y, t_end=round(t_end, 3)))
        if include_level1:
            for t_ev, s in rep["level1_seqs"]:
                xs.append(_F.normalize_sequence_v1(s).T)
                ys.append(lab)
                metas.append(_meta(row, "level1", lab, t_end=round(t_ev, 3)))
    return _finish(xs, ys, metas, 34, SEQ_LEN, return_meta)


# ---------------------------------------------------------------------------
# E: ABNORMAL 진입 사건 구간 v2 창
# ---------------------------------------------------------------------------

def visibility(X_v2: np.ndarray) -> np.ndarray:
    """v2 창 (N, 92, T) 또는 (92, T)의 관절 가시율 (마스크 채널 평균)."""
    X = np.asarray(X_v2)
    return X[..., _MASK0:_MASK0 + 17, :].mean(axis=(-2, -1))


def augment_window_arrays(kps: np.ndarray, bbox: np.ndarray, rng: np.random.Generator):
    """
    원시 배열 증강 (window_v2 이전): 좌우 반전(관절 swap, x -> 1 - x), 관절 dropout(conf 0 -> 마스크 0).
    반환: (kps, bbox) 복사본. 시간 늘이기는 시각 축에서 처리 (_v2_from_arrays).
    """
    kps = np.array(kps, dtype=np.float32, copy=True)
    bbox = np.array(bbox, dtype=np.float32, copy=True)
    if rng.random() < AUG_FLIP_P:
        kps = _F.flip_lr_coco(kps).astype(np.float32)
        bbox = _F.flip_lr_bbox(bbox).astype(np.float32)
    if rng.random() < AUG_DROPOUT_P and len(kps):
        drop = rng.random(17) < AUG_JOINT_DROP_P
        drop = drop[None, :] | (rng.random(kps.shape[:2]) < AUG_POINT_DROP_P)
        kps[..., 2] = np.where(drop, 0.0, kps[..., 2])
    return kps, bbox


def _v2_from_arrays(arr: Dict[str, Any], anchor_s: float, rng: Optional[np.random.Generator] = None,
                    augment: bool = False) -> np.ndarray:
    """
    기준 시각 anchor_s의 4초 창 [anchor - 1.5, anchor + 2.5)를 v2 특징 (92, 60)으로 만듭니다.
    augment면 시간 늘이기(기준 시각 고정, 배율 0.8~1.25 로그 균등) + augment_window_arrays.
    """
    ts = arr["ts"]
    k = 1.0
    if augment and rng is not None:
        k = float(math.exp(rng.uniform(math.log(AUG_WARP[0]), math.log(AUG_WARP[1]))))
    # 원래 시각 기준으로 필요한 범위만 자름 (배율 하한 0.8에서도 창 + 보간 여유가 들어가도록)
    lo = anchor_s - V2_PRE_S / AUG_WARP[0] - 0.5
    hi = anchor_s + V2_POST_S / AUG_WARP[0] + 0.5
    sel = (ts >= lo) & (ts <= hi)
    kps, bbox, present = arr["kps"][sel], arr["bbox"][sel], arr["present"][sel]
    if augment and rng is not None:
        kps, bbox = augment_window_arrays(kps, bbox, rng)
    # 늘이기: 시각을 기준 시각 중심으로 k배 (k > 1이면 느린 동작)
    ts_w = anchor_s + (ts[sel] - anchor_s) * k
    a = anchor_s - V2_PRE_S
    return _F.window_v2(ts_w, kps, bbox, present, arr["frame_wh"], a, a + V2_WINDOW_S)


def _onset_shifts(n: int, rng: Optional[np.random.Generator], augment: bool) -> np.ndarray:
    if n <= 0:
        return np.zeros(0)
    if augment and rng is not None:
        return rng.uniform(-ONSET_SHIFT_S, ONSET_SHIFT_S, n)
    return np.linspace(-ONSET_SHIFT_S, ONSET_SHIFT_S, n) if n > 1 else np.zeros(1)


def windows_event_v2(rows: Iterable[Dict[str, Any]], clips: Dict[str, Dict[str, Any]], params,
                     rng: Optional[np.random.Generator] = None, augment: bool = False,
                     n_onset: int = N_ONSET, neg_step_s: float = NEG_STEP_S,
                     min_visibility: float = MIN_VISIBILITY, include_entries: bool = True,
                     fall_entry_positives: bool = True, tail_use_s: float = TAIL_USE_S,
                     onset_offset_s: float = 0.0,
                     use_cache: bool = True, return_meta: bool = False):
    """
    사건 구간 v2 창 (stage E). 입력 배열은 운영 kp_buffer와 같은 15Hz 원시 키포인트 (obs_event_arrays).
    창 = 기준 시각 anchor의 [anchor - 1.5, anchor + 2.5) (모델 입력 4초, model_window_bounds와 같음)
      - onset (1): 낙상 구간마다 anchor = fall_start + onset_offset_s + 이동(±0.5초) n_onset개
                   (augment면 무작위 이동, 아니면 -0.5~+0.5 등간격)
                   onset_offset_s 기본 0 (스펙). 실측(규칙 엔진 재생, figshare 제외 5개 데이터셋)에서 ABNORMAL 진입
                   시각은 라벨 fall_start보다 중앙값 약 1.0초 늦음 (데이터셋별 0.9~1.2초, IQR 약 0.75~2초):
                   운영 창 위치에 맞추려면 1.0을 줌
      - figshare (Figshare 클립, 구간 없음): anchor = 클립 끝 - 0.5 + 이동 n_onset개, 라벨 = 클립 라벨
                   (창 = [끝 - 2, 끝 + 2): 약 1.9초 클립 전체 + 마지막 자세 2초)
      - adl (0): ADL 클립의 슬라이딩 창 (neg_step_s 간격, 창이 정지 꼬리에 tail_use_s까지 걸칠 수 있음)
      - pre_fall (0): 낙상 클립에서 창 끝 <= 첫 fall_start - 0.5초 인 슬라이딩 창
      - entry_adl (0): ADL 클립을 규칙 엔진으로 재생해 ABNORMAL 진입 때의 사건 구간 (하드 네거티브)
      - entry_fall (1): 낙상 클립 재생의 ABNORMAL 진입 중 기준 시각이 [fall_start - 1, fall_end + 1.5]인 것
                   (fall_entry_positives; 구간 없는 Figshare 낙상 클립은 모든 진입), 진입 구간이 낙상 시작 전에
                   끝나면 entry_pre_fall (0), 그 밖(일어난 뒤 등)은 버림
    관절 가시율 < min_visibility 창은 버림. augment=True면 모든 창에 증강(반전/늘이기/dropout)을 적용.
    tail_use_s는 TAIL_USE_S 이하만 허용 (재생 캐시가 정지 꼬리를 그만큼만 보관하므로, 더 길면 창이 조용히 잘림).
    반환: X (N, 92, 60) float32, y (N,)
    """
    if tail_use_s > TAIL_USE_S + _EPS:
        raise ValueError(f"tail_use_s {tail_use_s} > TAIL_USE_S {TAIL_USE_S}: 재생 캐시가 보관하는 꼬리보다 깁니다")
    if augment and rng is None:
        rng = np.random.default_rng(0)
    xs, ys, metas = [], [], []

    def add(row, arr, anchor, y, kind):
        x = _v2_from_arrays(arr, anchor, rng, augment)
        if x.shape != (V2_CHANNELS, V2_STEPS):
            return
        vis = float(visibility(x))
        if vis < min_visibility:
            return
        xs.append(x)
        ys.append(int(y))
        metas.append(_meta(row, kind, y, anchor_s=round(float(anchor), 3), vis=round(vis, 3)))

    for row in rows:
        clip = clips.get(row["clip_id"])
        if clip is None or len(clip["ts"]) == 0:
            continue
        rep = _replay(row, clip, params, use_cache=use_cache)
        arr, clip_end = rep["arrays"], rep["clip_end"]
        lab = _label(row)
        falls = _fall_segments(row)

        if _clip_level(row):
            # 시점 라벨 없음 (Figshare): 낙상/ADL 같은 규칙
            for sh in _onset_shifts(n_onset, rng, augment):
                add(row, arr, clip_end - 0.5 + float(sh), lab, "figshare")
        elif lab == 1:
            for f in falls:
                for sh in _onset_shifts(n_onset, rng, augment):
                    add(row, arr, float(f["start_s"]) + float(onset_offset_s) + float(sh), 1, "onset")
            last_end = float(falls[0]["start_s"]) - PRE_FALL_MARGIN_S
            a = 0.0
            while a + V2_WINDOW_S <= last_end + _EPS:
                add(row, arr, a + V2_PRE_S, 0, "pre_fall")
                a += neg_step_s
        elif lab == 0:
            a = 0.0
            last_start = max(0.0, clip_end + tail_use_s - V2_WINDOW_S)
            while a <= last_start + _EPS:
                add(row, arr, a + V2_PRE_S, 0, "adl")
                a += neg_step_s

        if include_entries:
            for e in rep["entries"]:
                anchor = float(e["anchor_s"])
                if lab == 0:
                    add(row, arr, anchor, 0, "entry_adl")
                elif not falls:
                    # 클립 단위 낙상 라벨 (약 1.9초 Figshare 낙상 클립): 진입은 그 낙상으로 봄
                    if fall_entry_positives:
                        add(row, arr, anchor, 1, "entry_fall")
                elif falls:
                    near = any(float(f["start_s"]) - FALL_ENTRY_BEFORE_S <= anchor
                               <= float(f["end_s"]) + FALL_ENTRY_AFTER_S for f in falls)
                    if near:
                        if fall_entry_positives:
                            add(row, arr, anchor, 1, "entry_fall")
                    elif float(e["start_s"]) + V2_POST_S < float(falls[0]["start_s"]) - PRE_FALL_MARGIN_S:
                        add(row, arr, anchor, 0, "entry_pre_fall")
    return _finish(xs, ys, metas, V2_CHANNELS, V2_STEPS, return_meta)


# ---------------------------------------------------------------------------
# E0: 손으로 만든 창 통계 특징
# ---------------------------------------------------------------------------

HANDCRAFTED_NAMES = [
    "hip_vy_max", "hip_vy_min", "hip_vy_absmean",
    "hip_dy_range", "hip_dy_change",
    "torso_abs_start", "torso_abs_end", "torso_abs_change", "torso_abs_max", "torso_angvel_absmax",
    "aspect_start", "aspect_end", "aspect_min", "aspect_max", "aspect_change",
    "height_start", "height_end", "height_ratio",
    "joint_speed_mean", "joint_speed_max",
    "vis_mean", "vis_first_half", "vis_second_half", "hip_valid_frac",
]


def _mstat(v: np.ndarray, ok: np.ndarray, fn: str) -> np.ndarray:
    """마스크된 시간축 통계 (N, T) -> (N,). 유효 스텝이 없으면 0."""
    any_ok = ok.any(axis=1)
    if fn == "max":
        r = np.where(ok, v, -np.inf).max(axis=1)
    elif fn == "min":
        r = np.where(ok, v, np.inf).min(axis=1)
    else:
        cnt = np.maximum(ok.sum(axis=1), 1)
        r = np.where(ok, v, 0.0).sum(axis=1) / cnt
    return np.where(any_ok, r, 0.0)


def handcrafted(X_v2: np.ndarray) -> np.ndarray:
    """
    v2 창 (N, 92, T)에서 고전 ML용 통계 특징 (N, len(HANDCRAFTED_NAMES)) float32.
    - 골반 하강 속도(hip_vy, 아래가 +)의 최대/최소/절대 평균, 골반 높이 변화(hip_dy 범위, 끝 1초 - 처음 1초)
    - 몸통 각도 |angle| 처음/끝 1초 평균과 변화, 최대, 최대 각속도
    - bbox 종횡비(h/w) 처음/끝/최소/최대/변화, 몸 세로 길이(관절 rel_y 범위) 처음/끝/비율
    - 관절 속도 크기 평균/최대, 가시율 전체/전반/후반, 골반 유효 비율
    유효하지 않은 스텝(마스크 0)은 통계에서 제외합니다. 값은 모두 유한합니다.
    """
    X = np.asarray(X_v2, dtype=np.float64)
    if X.ndim == 2:
        X = X[None]
    n, c, t = X.shape
    if c != V2_CHANNELS:
        raise ValueError(f"X_v2 channels {c} != {V2_CHANNELS}")
    if n == 0:
        return np.zeros((0, len(HANDCRAFTED_NAMES)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(_F.V2_CHANNELS)}
    mask = X[:, _MASK0:_MASK0 + 17, :] > 0.5                     # (N, 17, T)
    hip_ok = mask[:, 11] | mask[:, 12]
    torso_ok = hip_ok & (mask[:, 5] | mask[:, 6])
    q = max(1, min(t, int(round(_F.TARGET_HZ))))                 # 처음/끝 1초
    first = np.zeros(t, dtype=bool); first[:q] = True
    last = np.zeros(t, dtype=bool); last[t - q:] = True

    hip_vy = X[:, idx["hip_vy"]]
    hip_dy = X[:, idx["hip_dy"]]
    ang = np.abs(X[:, idx["torso_angle"]])
    angvel = np.abs(X[:, idx["torso_angvel"]])
    aspect = X[:, idx["bbox_aspect"]]
    asp_ok = aspect > 0

    rel_y = X[:, 1:34:2, :]                                      # (N, 17, T) rel_*_y
    rel_ok = mask & hip_ok[:, None, :]
    y_max = np.where(rel_ok, rel_y, -np.inf).max(axis=1)
    y_min = np.where(rel_ok, rel_y, np.inf).min(axis=1)
    h_ok = rel_ok.sum(axis=1) >= 2
    height = np.where(h_ok, y_max - y_min, 0.0)

    vx = X[:, 34:68:2, :]
    vy = X[:, 35:68:2, :]
    speed_j = np.sqrt(vx ** 2 + vy ** 2)
    cnt = rel_ok.sum(axis=1)
    speed = np.where(cnt > 0, (speed_j * rel_ok).sum(axis=1) / np.maximum(cnt, 1), 0.0)
    sp_ok = cnt > 0

    torso_start = _mstat(ang, torso_ok & first, "mean")
    torso_end = _mstat(ang, torso_ok & last, "mean")
    asp_start = _mstat(aspect, asp_ok & first, "mean")
    asp_end = _mstat(aspect, asp_ok & last, "mean")
    h_start = _mstat(height, h_ok & first, "mean")
    h_end = _mstat(height, h_ok & last, "mean")
    half = t // 2
    vis = mask.mean(axis=1)                                      # (N, T)

    feats = [
        _mstat(hip_vy, hip_ok, "max"), _mstat(hip_vy, hip_ok, "min"), _mstat(np.abs(hip_vy), hip_ok, "mean"),
        _mstat(hip_dy, hip_ok, "max") - _mstat(hip_dy, hip_ok, "min"),
        _mstat(hip_dy, hip_ok & last, "mean") - _mstat(hip_dy, hip_ok & first, "mean"),
        torso_start, torso_end, torso_end - torso_start, _mstat(ang, torso_ok, "max"), _mstat(angvel, torso_ok, "max"),
        asp_start, asp_end, _mstat(aspect, asp_ok, "min"), _mstat(aspect, asp_ok, "max"), asp_end - asp_start,
        h_start, h_end, np.where(h_start > 1e-6, h_end / np.maximum(h_start, 1e-6), 0.0),
        _mstat(speed, sp_ok, "mean"), _mstat(speed, sp_ok, "max"),
        vis.mean(axis=1), vis[:, :half].mean(axis=1) if half else vis.mean(axis=1), vis[:, half:].mean(axis=1),
        hip_ok.mean(axis=1),
    ]
    out = np.stack(feats, axis=1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)
