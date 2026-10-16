"""
실패 원인 분석: 규칙 엔진(단계 A)이 놓친 낙상, 제때가 아닌 탐지, ADL 오탐을 원인별로 셉니다.

각 평가 클립을 progression과 같은 방식(replay.clip_to_obs, 15Hz, EMA, 정지 꼬리)으로 재생하면서
프레임마다 엔진 내부 상태(누적 타이머, 진입 신호, 사람 검출 여부)를 기록하고, 라벨 구간과 비교해 원인을 정합니다.

원인 묶음 (group)
  pose  : 포즈 검출 때문에 규칙이 판단할 입력이 없음. 세부 묶음(subgroup)으로 나눔:
          no_detection = 운영 임계값(conf 0.5, 품질 0.15)에서 사람 미검출,
          engine_lost_handling = 검출이 간헐적이라 엔진의 미검출 처리(워밍업 리셋, 미검출 중 판정 정지)에 걸림
  rule  : 입력은 있었지만 규칙 조건을 넘지 못함 (진입 신호 부족, 짧은 신호, bbox 비율, 부분 신체, 붕괴 미확정)
  label : 라벨상 일어남 등 규칙 설계상 알람이 없어야 맞는 경우, 라벨 구간이 클립 밖, 라벨이 끝난 뒤 해제
  replay: 재생 방식 때문에 생긴 경우 (마지막 실제 프레임만 미검출이라 정지 꼬리 20초 동안 판정이 멈춤)
낙상이 여러 번인 클립(MCFD)의 놓친 낙상은 첫 낙상 구간 기준으로 봅니다.
자동 분류는 신뢰도가 높지만 골격이 틀린 경우("자신 있게 틀린" 포즈)를 rule로 셉니다. 무작위 표본을 영상에 겹쳐
직접 확인해 보정합니다 (render_sheet).

엔진은 수정하지 않고 step() 뒤에 공개 속성(누적 타이머 등)을 읽습니다. 부분 신체 판정은 엔진의
_check_partial_body를 그대로 호출합니다 (같은 파라미터, 상태 없음).

실행:
  python -m training.experiments.failure_analysis --out dataset/_experiments/failure_v1
  python -m training.experiments.failure_analysis --out dataset/_experiments/failure_v1 --render-missed 30 --render-adl all
"""
import argparse
import collections
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from training.experiments import replay as R

logger = logging.getLogger(__name__)

EVAL_DATASETS = ("le2i", "caucafall", "gmdcsa24", "mcfd", "urfd")

# 낙상 구간 판정 창: [fall_start, fall_end + FALL_POST_S]
FALL_POST_S = 2.0
# 프레임 비율 기준 (창 안 프레임 중)
LOST_FRAC_TH = 0.5
WARMUP_FRAC_TH = 0.5
PARTIAL_FRAC_TH = 0.5
# 마지막 자세(누운 자세) bbox 비율을 볼 구간 (클립 끝 기준, 초)
END_POSE_S = 1.0
# 낙상 후 일어남으로 보는 라벨 클래스
GETUP_CLASSES = ("stand_up", "standing", "walk")
# 회복 해제 시각과 일어남 라벨 구간을 맞춰 볼 때의 여유 (MCFD 카메라별 라벨 시각 차이 흡수)
GETUP_TOL_S = 1.0
# 진입 후 실제 프레임에서 사람이 이 시간 이상 연속으로 사라져야 lost_after_fall (마지막 프레임 하나만 없는 경우와 구분)
LOST_RUN_S = 1.0

CAUSE_GROUP = {
    "pose_lost_during_fall": "pose",
    "detection_flicker": "pose",
    "lost_after_fall": "pose",
    "tail_last_frame_lost": "replay",
    "unlabeled_exit": "label",
    "partial_body": "rule",
    "signals_too_brief": "rule",
    "lying_bbox_not_flat": "rule",
    "weak_motion_signals": "rule",
    "false_recovery": "rule",
    "timeout_unconfirmed": "rule",
    "other": "rule",
    "got_up_per_label": "label",
    "label_outside_clip": "label",
}

# pose 묶음의 세부 묶음
POSE_SUBGROUP = {
    "pose_lost_during_fall": "no_detection",
    "detection_flicker": "engine_lost_handling",
    "lost_after_fall": "engine_lost_handling",
}

CAUSE_DESC = {
    "pose_lost_during_fall": "낙상 구간 프레임의 절반 이상에서 사람 미검출 또는 품질 미달 (운영 임계값 conf 0.5 기준)",
    "detection_flicker": "검출이 끊겼다 이어져 워밍업(12프레임)이 반복, 진입 판정을 못 함",
    "lost_after_fall": "ABNORMAL 진입 후 실제 영상에서 사람이 1초 이상 사라진 채 끝나 정지 누적·타임아웃이 진행되지 않음",
    "tail_last_frame_lost": "진입 후 사람이 대부분 보였지만 마지막 실제 프레임만 미검출이라 정지 꼬리 동안 판정이 멈춤 (재생 방식)",
    "unlabeled_exit": "회복 해제가 라벨 구간이 끝난 뒤라 일어났는지 라벨로 알 수 없음",
    "partial_body": "화면 가장자리/하단에 걸린 부분 신체로 판정되어 진입 조건이 달라짐",
    "signals_too_brief": "진입 신호 2개 이상이 잠깐 있었지만 0.35초 연속을 못 채움",
    "lying_bbox_not_flat": "누운 뒤에도 정규화 bbox 세로/가로 비율이 자세 임계값(1.6) 이상",
    "weak_motion_signals": "자세 변화 외의 진입 신호(급하강, 어깨 하강, 높이 감소)가 부족",
    "false_recovery": "ABNORMAL 진입 후 일어나지 않았는데 회복으로 판정되어 해제",
    "timeout_unconfirmed": "ABNORMAL 유지 중 붕괴 확정(bbox 비율 0.65배 또는 1.0 미만)이 안 되어 타임아웃 미발동",
    "other": "위 범주에 해당하지 않음",
    "got_up_per_label": "라벨상 낙상 뒤 일어남 (규칙 설계상 회복 해제가 맞음)",
    "label_outside_clip": "낙상 구간이 실제 프레임 범위 밖",
}

__all__ = [
    "EVAL_DATASETS", "CAUSE_GROUP", "CAUSE_DESC",
    "POSE_SUBGROUP", "trace_clip", "episodes", "segment_class_at", "classify_missed", "classify_entry_window",
    "classify_untimely", "classify_adl_alarm", "analyze_clip", "summarize_records", "render_markdown",
    "render_sheet", "main",
]


# ---------------------------------------------------------------------------
# 재생 + 프레임별 엔진 상태
# ---------------------------------------------------------------------------

def _is_lost(obs: Dict[str, Any], min_quality: float) -> bool:
    tracks = obs.get("tracks") or []
    if not tracks or not tracks[0].get("has_person", False):
        return True
    return float(tracks[0].get("quality_score") or 0.0) < min_quality


def trace_clip(obs_list: Sequence[Dict[str, Any]], params, t0: float = R.DEFAULT_T0
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    obs를 새 엔진(규칙만)에 넣으며 (이벤트 목록, 프레임별 기록)을 반환합니다.
    프레임 기록 (step 직후 상태):
      t (클립 시각), tail, lost, state, streak (present_streak), aspect (정규화 bbox h/w, lost면 None),
      partial, hits ({drop, posture, shoulder, height} 신호가 지속 기준을 넘었는지, NORMAL 상태에서만),
      n_hits, enter_acc, low_cnt, still_acc, confirmed, recover_acc
    """
    from app.core.level1 import Level1Engine, bbox_aspect

    p = params
    engine = Level1Engine(p, verifier=R.NullVerifier())
    events: List[Dict[str, Any]] = []
    frames: List[Dict[str, Any]] = []
    for obs in obs_list:
        state_before = engine.state
        ev = engine.step(obs)
        if ev is not None:
            events.append(ev)
        lost = _is_lost(obs, p.min_quality)
        tr = (obs.get("tracks") or [{}])[0]
        bbox = tr.get("bbox")
        kps = tr.get("keypoints_smooth") or tr.get("keypoints") or []
        aspect = None if lost else bbox_aspect(bbox)
        partial = (not lost) and bool(bbox) and engine._check_partial_body(bbox, kps)
        hits = None
        if not lost and state_before == "NORMAL":
            # 진입 판정 직후에는 누적값이 0으로 초기화되므로, 진입 프레임은 진입 신호가 충분했던 것으로 기록
            entered = ev is not None and ev.get("type") == "abnormal_enter"
            hits = {
                "drop": bool(entered and (ev.get("signals") or {}).get("drop")) or (
                    sum(1 for _, h in engine.drop_hits if h) >= p.drop_hits_k),
                "posture": entered or engine.posture_acc_s >= p.posture_persist_s,
                "shoulder": engine.shoulder_acc_s >= p.shoulder_drop_persist_s,
                "height": engine.height_acc_s >= p.height_persist_s,
            }
        frames.append({
            "t": float(obs["ts"]) - t0,
            "tail": bool((obs.get("meta") or {}).get("replay_tail")),
            "lost": lost,
            "state": engine.state,
            "streak": int(engine.present_streak),
            "aspect": aspect,
            "partial": bool(partial),
            "hits": hits,
            "n_hits": None if hits is None else sum(1 for v in hits.values() if v),
            "enter_acc": float(engine.enter_acc_s),
            "low_cnt": int(engine.low_posture_cnt),
            "still_acc": float(engine.still_acc_s),
            "confirmed": bool(engine.abnormal_confirmed),
            "recover_acc": float(engine.recover_acc_s),
        })
    return events, frames


def episodes(events: Sequence[Dict[str, Any]], t0: float = R.DEFAULT_T0) -> List[Dict[str, Any]]:
    """
    ABNORMAL 에피소드 목록: {"enter_s", "end_s", "end": "level1" | "recovered" | None (끝까지 유지), "reason"}.
    """
    eps: List[Dict[str, Any]] = []
    cur = None
    for e in events:
        typ = e.get("type")
        t = float(e.get("ts", 0.0)) - t0
        if typ == "abnormal_enter":
            cur = {"enter_s": t, "end_s": None, "end": None,
                   "reason": (e.get("values") or {}).get("reason")}
            eps.append(cur)
        elif cur is not None and cur["end"] is None and typ in ("level1", "abnormal_exit"):
            cur["end_s"] = t
            cur["end"] = "level1" if typ == "level1" else "recovered"
    return eps


def segment_class_at(row: Dict[str, Any], t: Optional[float]) -> Optional[str]:
    """라벨 구간에서 시각 t의 클래스 (구간 사이/밖이면 가장 가까운 앞 구간, 없으면 None)."""
    if t is None:
        return None
    segs = sorted(row.get("segments") or [], key=lambda s: float(s["start_s"]))
    hit = [s for s in segs if float(s["start_s"]) <= t <= float(s["end_s"])]
    if hit:
        return hit[-1]["cls"]
    prev = [s for s in segs if float(s["end_s"]) < t]
    return prev[-1]["cls"] if prev else None


def _window(frames: Sequence[Dict[str, Any]], lo: float, hi: float) -> List[Dict[str, Any]]:
    return [f for f in frames if not f["tail"] and lo - 1e-9 <= f["t"] <= hi + 1e-9]


def _frac(items: Sequence[Dict[str, Any]], pred) -> float:
    return (sum(1 for f in items if pred(f)) / len(items)) if items else 0.0


def _median(vals: Sequence[float]) -> Optional[float]:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def _end_aspect(frames: Sequence[Dict[str, Any]]) -> Optional[float]:
    main = [f for f in frames if not f["tail"]]
    if not main:
        return None
    t_end = main[-1]["t"]
    return _median([f["aspect"] for f in main if f["t"] >= t_end - END_POSE_S])


def _window_stats(frames: Sequence[Dict[str, Any]], lo: float, hi: float, warmup: int) -> Dict[str, Any]:
    w = _window(frames, lo, hi)
    seen = [f for f in w if not f["lost"]]
    normal = [f for f in seen if f["hits"] is not None]
    sig = collections.Counter()
    for f in normal:
        for k, v in (f["hits"] or {}).items():
            if v:
                sig[k] += 1
    return {
        "n": len(w),
        "lost_frac": _frac(w, lambda f: f["lost"]),
        # 사람이 보이는 NORMAL 프레임 중 워밍업 미달 비율 (lost 프레임마다 streak가 0으로 돌아감)
        "warmup_frac": _frac(normal, lambda f: f["streak"] < warmup),
        "partial_frac": _frac(seen, lambda f: f["partial"]),
        "max_hits": max((f["n_hits"] for f in normal), default=0),
        "max_enter_acc": max((f["enter_acc"] for f in normal), default=0.0),
        "signal_frames": dict(sig),
        "min_aspect": min((f["aspect"] for f in seen if f["aspect"] is not None), default=None),
    }


# ---------------------------------------------------------------------------
# 분류
# ---------------------------------------------------------------------------

def classify_entry_window(frames: Sequence[Dict[str, Any]], lo: float, hi: float, params) -> Tuple[str, Dict[str, Any]]:
    """
    [lo, hi] 동안 ABNORMAL에 들어가지 못한 이유 (놓친 낙상, 늦은 진입 공용). (원인, 근거 통계)
    순서: 프레임 없음 -> 미검출 -> 검출 끊김 -> 부분 신체 -> 짧은 신호 -> 누운 bbox 비율 -> 약한 신호
    """
    st = _window_stats(frames, lo, hi, params.warmup_frames)
    st["end_aspect"] = _end_aspect(frames)
    if st["n"] == 0:
        return "label_outside_clip", st
    if st["lost_frac"] >= LOST_FRAC_TH:
        return "pose_lost_during_fall", st
    if st["warmup_frac"] >= WARMUP_FRAC_TH:
        return "detection_flicker", st
    if st["partial_frac"] >= PARTIAL_FRAC_TH:
        return "partial_body", st
    if st["max_hits"] >= params.enter_confirm_need:
        return "signals_too_brief", st
    if st["end_aspect"] is not None and st["end_aspect"] >= params.aspect_abs_th:
        return "lying_bbox_not_flat", st
    return "weak_motion_signals", st


def _label_end(row: Dict[str, Any]) -> Optional[float]:
    segs = row.get("segments") or []
    return max((float(s["end_s"]) for s in segs), default=None)


def _got_up_near(row: Dict[str, Any], enter_s: float, exit_s: float, recover_s: float,
                 tol: float = GETUP_TOL_S) -> bool:
    """
    회복 해제(exit_s) 무렵에 라벨상 일어나는 중이었는지: 일어남 클래스 구간이 해제 시각 + tol 이전에 시작하고
    max(진입, 해제 - 회복 판정 시간) 이후에 끝나며, 그 구간과 해제 사이에 새 낙상이 시작하지 않음.
    """
    segs = sorted(row.get("segments") or [], key=lambda s: float(s["start_s"]))
    lo = max(enter_s, exit_s - recover_s)
    for s in segs:
        a, b = float(s["start_s"]), float(s["end_s"])
        if s["cls"] in GETUP_CLASSES and a <= exit_s + tol and b >= lo - tol:
            if not any(x["cls"] == "fall" and a < float(x["start_s"]) <= exit_s for x in segs):
                return True
    return False


def _trailing_lost_s(frames: Sequence[Dict[str, Any]]) -> float:
    """실제 프레임(정지 꼬리 제외) 끝에서 연속으로 미검출인 시간 (초)."""
    main = [f for f in frames if not f["tail"]]
    if not main or not main[-1]["lost"]:
        return 0.0
    k = len(main) - 1
    while k > 0 and main[k - 1]["lost"]:
        k -= 1
    first_lost = main[k]["t"]
    prev_seen = main[k - 1]["t"] if k > 0 else first_lost
    return main[-1]["t"] - prev_seen


def classify_missed(row: Dict[str, Any], events, frames, params, t0: float = R.DEFAULT_T0) -> Dict[str, Any]:
    """
    놓친 낙상(level1 없음) 하나의 원인. 낙상이 여러 번이면 첫 낙상 구간 기준.
    관련 에피소드: 진입이 [fall_start - 2, fall_end + 3] 안이거나, 그 전에 진입해 fall_start에 아직 이어지던 것.
    여러 개면 낙상과 겹치는 첫 에피소드. 없으면 낙상 구간 동안 진입하지 못한 이유(classify_entry_window).
    """
    falls = R.fall_intervals(row)
    fs, fe = falls[0] if falls else (float(row["fall_start_s"]), float(row["fall_end_s"]))
    lo, hi = fs - R.TIMELY_PRE_S, fe + R.TIMELY_POST_S
    eps = episodes(events, t0)
    rel = [e for e in eps if (lo <= e["enter_s"] <= hi)
           or (e["enter_s"] < lo and (e["end_s"] is None or e["end_s"] >= fs))]
    out: Dict[str, Any] = {"n_episodes": len(eps), "entered": bool(rel), "fall_s": [fs, fe]}
    if not rel:
        cause, st = classify_entry_window(frames, fs, fe + FALL_POST_S, params)
        out.update(cause=cause, stats=st)
        return out
    ep = rel[0]
    out["episode"] = ep
    real_after = [f for f in frames if f["t"] >= ep["enter_s"] and not f["tail"]]
    if ep["end"] == "recovered":
        exit_s = float(ep["end_s"])
        lab_end = _label_end(row)
        if _got_up_near(row, ep["enter_s"], exit_s, params.recover_s):
            cause = "got_up_per_label"
        elif lab_end is not None and exit_s > lab_end + GETUP_TOL_S:
            cause = "unlabeled_exit"
        else:
            cause = "false_recovery"
        out.update(cause=cause, stats={"exit_s": exit_s, "label_end_s": lab_end})
        return out
    last = frames[-1] if frames else None
    trailing = _trailing_lost_s(frames)
    out["stats"] = {
        "lost_frac_after": _frac(real_after, lambda f: f["lost"]),
        "trailing_lost_s": trailing,
        "confirmed": bool(last and last["confirmed"]),
        "max_still_acc": max((f["still_acc"] for f in real_after), default=0.0),
    }
    if last is not None and last["lost"]:
        if trailing >= LOST_RUN_S or out["stats"]["lost_frac_after"] >= LOST_FRAC_TH:
            out["cause"] = "lost_after_fall"
        else:
            out["cause"] = "tail_last_frame_lost"
    elif last is not None and not last["confirmed"]:
        out["cause"] = "timeout_unconfirmed"
    else:
        out["cause"] = "other"
    return out


def classify_untimely(row: Dict[str, Any], outcome: Dict[str, Any], events, frames, params) -> Dict[str, Any]:
    """제때가 아닌 탐지: 진입 시점의 라벨 클래스, 진입 사유, 늦은 진입이면 낙상 중 진입하지 못한 이유."""
    enter = outcome.get("level1_enter_s")
    eps = [e for e in episodes(events) if enter is not None and abs(e["enter_s"] - enter) < 1e-6]
    out = {
        "untimely": outcome.get("untimely"),
        "enter_s": enter,
        "level1_s": outcome.get("level1_s"),
        "enter_reason": _reason_kind(eps[0]["reason"] if eps else None),
        "class_at_enter": segment_class_at(row, enter),
    }
    ref = outcome.get("ref_fall_s") or [float(row["fall_start_s"]), float(row["fall_end_s"])]
    out["ref_fall_s"] = ref
    if outcome.get("untimely") == "enter_late":
        cause, st = classify_entry_window(frames, float(ref[0]), float(ref[1]) + FALL_POST_S, params)
        out.update(cause=cause, stats=st)
    return out


def _reason_kind(reason: Optional[str]) -> Optional[str]:
    if not reason:
        return None
    m = re.match(r"[a-z_]+", str(reason))
    return m.group(0).rstrip("_") if m else str(reason)


def classify_adl_alarm(row: Dict[str, Any], outcome: Dict[str, Any], events, frames) -> Dict[str, Any]:
    """ADL 오탐: 진입 사유, 진입/알람/클립 끝 시점의 라벨 클래스, 정지 꼬리 여부, 마지막 자세 bbox 비율."""
    enter = outcome.get("level1_enter_s")
    eps = [e for e in episodes(events) if enter is not None and abs(e["enter_s"] - enter) < 1e-6]
    main = [f for f in frames if not f["tail"]]
    t_end = main[-1]["t"] if main else None
    return {
        "enter_s": enter,
        "level1_s": outcome.get("level1_s"),
        "in_tail": outcome.get("in_tail"),
        "enter_reason": _reason_kind(eps[0]["reason"] if eps else None),
        "class_at_enter": segment_class_at(row, enter),
        "class_at_level1": segment_class_at(row, outcome.get("level1_s")),
        "class_at_end": segment_class_at(row, t_end),
        "classes": sorted({s["cls"] for s in (row.get("segments") or [])}),
        "end_aspect": _end_aspect(frames),
        "end_lost": bool(main and main[-1]["lost"]),
    }


def analyze_clip(row: Dict[str, Any], clip: Dict[str, Any], params) -> Dict[str, Any]:
    """클립 하나를 재생해 결과(clip_outcome)와 해당하는 실패 분석을 담은 레코드."""
    obs = R.clip_to_obs(clip)
    events, frames = trace_clip(obs, params)
    outcome = R.clip_outcome(row, events, end_s=R.clip_end_s(obs))
    rec: Dict[str, Any] = {
        "clip_id": row["clip_id"], "dataset": row["dataset"], "label": row["label"],
        "cam": row.get("cam"), "frame_wh": [int(v) for v in clip.get("frame_wh", [0, 0])],
        "outcome": {k: outcome[k] for k in ("alarm", "reasons", "n_abnormal", "level1_s", "level1_enter_s",
                                            "in_tail", "timely_strict", "timely_loose", "untimely", "ref_fall_s")},
        "presence": presence_around_fall(row, clip, params.min_quality),
        "clip_end_s": R.clip_end_s(obs),
        "fall_start_s": row.get("fall_start_s"), "fall_end_s": row.get("fall_end_s"),
        "kind": None, "analysis": None,
    }
    if row["label"] == "fall":
        if not outcome["alarm"]:
            rec["kind"] = "missed"
            rec["analysis"] = classify_missed(row, events, frames, params)
        elif not outcome["timely_strict"]:
            rec["kind"] = "untimely"
            rec["analysis"] = classify_untimely(row, outcome, events, frames, params)
        else:
            rec["kind"] = "timely"
    elif outcome["alarm"]:
        rec["kind"] = "adl_alarm"
        rec["analysis"] = classify_adl_alarm(row, outcome, events, frames)
    else:
        rec["kind"] = "adl_ok"
    if rec["kind"] in ("missed", "untimely") and rec["analysis"].get("cause"):
        rec["analysis"]["group"] = CAUSE_GROUP.get(rec["analysis"]["cause"], "rule")
        rec["analysis"]["subgroup"] = POSE_SUBGROUP.get(rec["analysis"]["cause"])
    return rec


PRESENCE_PRE_S = 3.0


def presence_around_fall(row: Dict[str, Any], clip: Dict[str, Any], min_quality: float) -> Optional[Dict[str, Any]]:
    """
    첫 낙상 전 3초와 fall_end 이후(클립 끝까지)의 사람 검출률 (원본 프레임, 사람 선택 + 관절 conf 평균 >= min_quality).
    낙상 클립이 아니면 None. 구간에 프레임이 없으면 그 값은 None.
    """
    falls = R.fall_intervals(row)
    if not falls:
        return None
    import numpy as np
    fs, fe = falls[0]
    ts = np.asarray(clip["ts"], dtype=np.float64)
    ok = np.asarray(clip["present"], dtype=bool) & (
        np.asarray(clip["kps"], dtype=np.float64)[:, :, 2].mean(axis=1) >= float(min_quality))

    def rate(lo, hi):
        m = (ts >= lo) & (ts <= hi)
        return float(ok[m].mean()) if m.any() else None
    return {"before": rate(fs - PRESENCE_PRE_S, fs), "after": rate(fe, float(ts[-1]) if len(ts) else fe)}


# ---------------------------------------------------------------------------
# 집계와 보고
# ---------------------------------------------------------------------------

def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """종류별 개수, 놓친 낙상 원인 (전체/데이터셋별), 묶음별 비율, 제때 아님 유형, ADL 오탐 속성 분포."""
    kinds = collections.Counter(r["kind"] for r in records)
    missed = [r for r in records if r["kind"] == "missed"]
    untimely = [r for r in records if r["kind"] == "untimely"]
    adl = [r for r in records if r["kind"] == "adl_alarm"]

    def by(items, key):
        c = collections.Counter(key(r) for r in items)
        return dict(c.most_common())

    datasets = sorted({r["dataset"] for r in records})
    missed_by_ds = {d: by([r for r in missed if r["dataset"] == d], lambda r: r["analysis"]["cause"])
                    for d in datasets}
    falls_by_ds = {d: sum(1 for r in records if r["dataset"] == d and r["label"] == "fall") for d in datasets}
    return {
        "kinds": dict(kinds),
        "n_fall": sum(1 for r in records if r["label"] == "fall"),
        "n_adl": sum(1 for r in records if r["label"] == "adl"),
        "missed_cause": by(missed, lambda r: r["analysis"]["cause"]),
        "missed_group": by(missed, lambda r: r["analysis"]["group"]),
        "missed_pose_subgroup": by([r for r in missed if r["analysis"]["group"] == "pose"],
                                   lambda r: r["analysis"]["subgroup"]),
        "presence": _presence_table(records),
        "missed_entered": by(missed, lambda r: "entered" if r["analysis"]["entered"] else "never_entered"),
        "missed_by_dataset": missed_by_ds,
        "falls_by_dataset": falls_by_ds,
        "missed_by_frame": by(missed, lambda r: "x".join(str(v) for v in r["frame_wh"])),
        "untimely_type": by(untimely, lambda r: r["analysis"]["untimely"]),
        "untimely_enter_reason": by(untimely, lambda r: r["analysis"]["enter_reason"]),
        "untimely_class_at_enter": by(untimely, lambda r: r["analysis"]["class_at_enter"]),
        "untimely_late_cause": by([r for r in untimely if r["analysis"]["untimely"] == "enter_late"],
                                  lambda r: r["analysis"]["cause"]),
        "adl_by_dataset": by(adl, lambda r: r["dataset"]),
        "adl_enter_reason": by(adl, lambda r: r["analysis"]["enter_reason"]),
        "adl_class_at_enter": by(adl, lambda r: r["analysis"]["class_at_enter"]),
        "adl_class_at_end": by(adl, lambda r: r["analysis"]["class_at_end"]),
        "adl_in_tail": by(adl, lambda r: r["analysis"]["in_tail"]),
        "adl_lying_classes": sum(1 for r in adl if {"lying", "lie_down"} & set(r["analysis"]["classes"])),
    }


def _presence_table(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """데이터셋 x 결과 종류(낙상 클립)별 낙상 전/후 사람 검출률 평균과 클립 수."""
    acc: Dict[str, Dict[str, List[float]]] = collections.defaultdict(lambda: {"before": [], "after": []})
    for r in records:
        pr = r.get("presence")
        if r["label"] != "fall" or not pr:
            continue
        for key in (f"{r['dataset']}:{r['kind']}", f"ALL:{r['kind']}", r["dataset"], "ALL"):
            for side in ("before", "after"):
                if pr.get(side) is not None:
                    acc[key][side].append(pr[side])
    out = {}
    for key, v in sorted(acc.items()):
        out[key] = {side: (sum(x) / len(x) if x else None) for side, x in v.items()}
        out[key]["n"] = max(len(v["before"]), len(v["after"]))
    return out


def _table(counter: Dict[Any, int], total: Optional[int] = None, desc: Optional[Dict[str, str]] = None,
           head: Tuple[str, str] = ("항목", "개수")) -> str:
    n = total if total is not None else sum(counter.values())
    lines = [f"| {head[0]} | {head[1]} | 비율 |" + (" 설명 |" if desc else ""),
             "| --- | --- | --- |" + (" --- |" if desc else "")]
    for k, v in counter.items():
        pct = f"{100.0 * v / n:.0f}%" if n else "-"
        extra = f" {desc.get(str(k), '')} |" if desc else ""
        lines.append(f"| {k} | {v} | {pct} |{extra}")
    return "\n".join(lines)


def render_markdown(summary: Dict[str, Any]) -> str:
    """요약 -> 마크다운 표 (자동 생성 부분)."""
    s = summary
    out = []
    out.append(f"평가 클립 {s['n_fall'] + s['n_adl']}개 (낙상 {s['n_fall']}, ADL {s['n_adl']}), 규칙만(A) 재생.\n")
    out.append(_table(s["kinds"], head=("결과", "클립")))
    n_missed = sum(s["missed_cause"].values())
    out.append(f"\n### 놓친 낙상 {n_missed}개: 원인 묶음\n")
    out.append(_table(s["missed_group"], n_missed, head=("묶음", "개수")))
    out.append(f"\n진입 여부: {s['missed_entered']}\n")
    out.append("pose 세부 묶음: " + ", ".join(f"{k} {v}" for k, v in s["missed_pose_subgroup"].items()) + "\n")
    out.append("\n### 놓친 낙상: 세부 원인\n")
    out.append(_table(s["missed_cause"], n_missed, CAUSE_DESC, head=("원인", "개수")))
    out.append("\n### 놓친 낙상: 데이터셋별 원인\n")
    causes = list(s["missed_cause"].keys())
    out.append("| 데이터셋 | 낙상 | 놓침 | " + " | ".join(causes) + " |")
    out.append("| --- | --- | --- | " + " | ".join("---" for _ in causes) + " |")
    for d, c in s["missed_by_dataset"].items():
        out.append(f"| {d} | {s['falls_by_dataset'][d]} | {sum(c.values())} | "
                   + " | ".join(str(c.get(k, 0)) for k in causes) + " |")
    out.append("\n### 놓친 낙상: 영상 크기별\n")
    out.append(_table(s["missed_by_frame"], n_missed, head=("WxH", "개수")))
    out.append("\n### 낙상 전 3초 / fall_end 이후 사람 검출률 (평균)\n")
    out.append("| 데이터셋:결과 | 클립 | 전 | 후 |")
    out.append("| --- | --- | --- | --- |")
    for k, v in s["presence"].items():
        def f(x):
            return "-" if x is None else f"{x:.2f}"
        out.append(f"| {k} | {v['n']} | {f(v['before'])} | {f(v['after'])} |")
    n_unt = sum(s["untimely_type"].values())
    out.append(f"\n### 제때 아닌 탐지 {n_unt}개\n")
    out.append(_table(s["untimely_type"], n_unt, head=("유형", "개수")))
    out.append("\n진입 사유: " + ", ".join(f"{k} {v}" for k, v in s["untimely_enter_reason"].items()))
    out.append("\n진입 시점 라벨 클래스: " + ", ".join(f"{k} {v}" for k, v in s["untimely_class_at_enter"].items()))
    if s["untimely_late_cause"]:
        out.append("\n늦은 진입(enter_late)이 낙상 중 진입하지 못한 이유: "
                   + ", ".join(f"{k} {v}" for k, v in s["untimely_late_cause"].items()))
    n_adl = sum(s["adl_by_dataset"].values())
    out.append(f"\n\n### ADL 오탐 {n_adl}개\n")
    out.append(_table(s["adl_by_dataset"], n_adl, head=("데이터셋", "개수")))
    out.append("\n진입 사유: " + ", ".join(f"{k} {v}" for k, v in s["adl_enter_reason"].items()))
    out.append("\n진입 시점 라벨 클래스: " + ", ".join(f"{k} {v}" for k, v in s["adl_class_at_enter"].items()))
    out.append("\n클립 끝 라벨 클래스: " + ", ".join(f"{k} {v}" for k, v in s["adl_class_at_end"].items()))
    out.append("\nLEVEL1이 정지 꼬리 안: " + ", ".join(f"{k} {v}" for k, v in s["adl_in_tail"].items()))
    out.append(f"\nlie_down/lying 구간이 있는 오탐 클립: {s['adl_lying_classes']}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 확인용 그림 (영상 프레임 + 키포인트)
# ---------------------------------------------------------------------------

COCO17_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def sheet_times(rec: Dict[str, Any]) -> List[Tuple[str, float]]:
    """그림에 넣을 시각 (이름, 클립 시각). 낙상: 시작 전후와 끝, ADL 오탐: 진입 전후와 끝."""
    end = float(rec.get("clip_end_s") or 0.0)

    def c(t):
        return max(0.0, min(end, t))
    if rec["label"] == "fall":
        fs, fe = float(rec["fall_start_s"]), float(rec["fall_end_s"])
        pts = [("fs-1", fs - 1.0), ("fall_start", fs), ("mid", 0.5 * (fs + fe)), ("fall_end", fe),
               ("fe+2", fe + 2.0), ("last", end)]
    else:
        en = rec["outcome"].get("level1_enter_s")
        en = float(en) if en is not None else end
        pts = [("enter-1", en - 1.0), ("enter", en), ("enter+1.5", en + 1.5), ("enter+4", en + 4.0),
               ("mid_end", 0.5 * (en + end)), ("last", end)]
    return [(n, c(t)) for n, t in pts]


def render_sheet(rec: Dict[str, Any], row: Dict[str, Any], clip: Dict[str, Any], out_png: Path,
                 repo_root: Path = R.REPO_ROOT, tile_w: int = 400) -> Optional[Path]:
    """
    영상에서 sheet_times 시각의 프레임을 읽어 키포인트(conf >= 0.3)와 bbox를 겹친 3x2 그림을 저장합니다.
    키포인트 좌표는 crop_px가 있으면 crop한 영상 기준. 영상을 열 수 없으면 None.
    """
    import cv2
    import numpy as np

    path = repo_root / row["video"]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("cannot open %s", path)
        return None
    ts = np.asarray(clip["ts"], dtype=np.float64)
    want = [(name, t, int(np.argmin(np.abs(ts - t)))) for name, t in sheet_times(rec)]
    need = sorted({i for _, _, i in want})
    frames: Dict[int, Any] = {}
    idx = 0
    try:
        while need and idx <= need[-1]:
            ok, img = cap.read()
            if not ok:
                break
            if idx in need:
                frames[idx] = img
            idx += 1
    finally:
        cap.release()
    crop = row.get("crop_px")
    tiles = []
    for name, t, i in want:
        img = frames.get(i)
        if img is None:
            img = np.zeros((240, 320, 3), dtype=np.uint8)
        elif crop:
            x1, y1, x2, y2 = (int(v) for v in crop)
            img = img[y1:y2, x1:x2]
        img = img.copy()
        h, w = img.shape[:2]
        if bool(clip["present"][i]):
            k = np.asarray(clip["kps"][i], dtype=np.float64)
            pts = [(int(k[j, 0] * w), int(k[j, 1] * h), float(k[j, 2])) for j in range(17)]
            for a, b in COCO17_EDGES:
                if pts[a][2] >= 0.3 and pts[b][2] >= 0.3:
                    cv2.line(img, pts[a][:2], pts[b][:2], (0, 255, 0), 2)
            for x, y, cf in pts:
                if cf >= 0.3:
                    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            bx = np.asarray(clip["bbox"][i], dtype=np.float64)
            cv2.rectangle(img, (int(bx[0] * w), int(bx[1] * h)), (int(bx[2] * w), int(bx[3] * h)), (255, 200, 0), 1)
            conf = float(np.mean(k[:, 2]))
            asp = (bx[3] - bx[1]) / max(1e-6, bx[2] - bx[0])
            info = f"conf {conf:.2f} asp {asp:.2f}"
        else:
            info = "no person"
        scale = tile_w / float(w)
        img = cv2.resize(img, (tile_w, max(1, int(round(h * scale)))))
        cv2.rectangle(img, (0, 0), (tile_w, 34), (0, 0, 0), -1)
        cv2.putText(img, f"{name} t={t:.2f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(img, info, (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(img)
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), grid)
    return out_png


def _safe_name(clip_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", clip_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_rows(manifest: Path, datasets: Iterable[str]) -> List[Dict[str, Any]]:
    keep = set(datasets)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["dataset"] in keep]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="규칙 엔진(단계 A) 실패 원인 분석")
    ap.add_argument("--manifest", default=str(R.REPO_ROOT / "dataset" / "_manifest" / "manifest_v1.jsonl"))
    ap.add_argument("--kp-dir", default=str(R.DEFAULT_KP_DIR))
    ap.add_argument("--out", default=str(R.REPO_ROOT / "dataset" / "_experiments" / "failure_v1"))
    ap.add_argument("--render-missed", default="0", help="놓친 낙상 그림 수 (정수, 무작위 표본) 또는 all")
    ap.add_argument("--render-adl", default="0", help="ADL 오탐 그림 수 (정수) 또는 all")
    ap.add_argument("--render-untimely", default="0", help="제때 아닌 탐지 그림 수 (정수) 또는 all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    from training.datasets.clipfmt import load_clip

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(Path(args.manifest), EVAL_DATASETS)
    params = R.study_params("off")
    records, clips = [], {}
    for row in rows:
        clip = load_clip(R.clip_path(row["clip_id"], args.kp_dir))
        clips[row["clip_id"]] = clip
        records.append(analyze_clip(row, clip, params))
    summary = summarize_records(records)
    (out / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    md = render_markdown(summary)
    (out / "summary.md").write_text(md, encoding="utf-8", newline="\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.write(md)

    by_id = {r["clip_id"]: r for r in rows}
    rng = random.Random(args.seed)
    picked: Dict[str, List[Dict[str, Any]]] = {}
    for kind, spec in (("missed", args.render_missed), ("adl_alarm", args.render_adl),
                       ("untimely", args.render_untimely)):
        pool = [r for r in records if r["kind"] == kind]
        n = len(pool) if spec == "all" else min(len(pool), int(spec))
        sample = sorted(rng.sample(pool, n), key=lambda r: r["clip_id"]) if n else []
        picked[kind] = sample
        for rec in sample:
            png = out / "sheets" / kind / f"{_safe_name(rec['clip_id'])}.png"
            rec["sheet"] = str(render_sheet(rec, by_id[rec["clip_id"]], clips[rec["clip_id"]], png) or "")
    if any(picked.values()):
        (out / "sheets.json").write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
