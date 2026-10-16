"""
클립(.npz) -> 운영(Jetson)과 같은 obs 스트림 -> Level1Engine 재생 -> 클립 단위 결과와 요약

운영 경로와 맞춘 점
- 처리 속도: Jetson 실측 처리 속도(약 15fps)에 맞춰 1/15초 격자에서 가장 가까운 원본 프레임을 고름 (보간 없음,
  운영도 카메라 프레임 중 처리된 것만 엔진에 들어감). 원본 fps가 hz보다 낮아 같은 프레임이 연달아 뽑히면 한 번만
  넣음 (운영은 같은 카메라 프레임을 두 번 처리하지 않음).
- obs ts는 고른 프레임의 원본 시각 (t0 + clip ts). 운영 ts(처리 시점 time.time())도 프레임 내용의 실제 시각을
  따르므로, 격자 시각을 쓰면 25/20fps 원본에서 내용 시각과 최대 반 프레임 어긋나 속도(vy, 어깨 하강, v2 속도 채널)가
  ±50%까지 흔들립니다. 또 obs ts - t0 = 클립 시각이 항상 성립합니다 (라벨 구간과 같은 시간축).
- obs 형식: app.core.obs_schema.build_observation 출력과 같은 키 (tracks[0]: has_person, bbox 정규화,
  keypoints/keypoints_raw = [{id,x,y,conf}] 17개, quality_score = conf 평균, frame_shape (h, w)).
  사람이 없는 프레임은 has_person False, bbox None, 키포인트 [] (YOLO 결과가 비었을 때와 같음).
- 스무딩: app.engine.smoothing.ema_smooth_keypoints_inplace를 프레임 순서대로 적용해 keypoints_smooth를 채움
  (클립마다 _reset_ema_state()로 초기화, 운영 LivePipeline.step과 같은 호출).

운영과 다른 점 (결과 해석 시 주의)
- 정지 꼬리(FREEZE TAIL): 공개 데이터 클립은 대부분 낙상 후 몇 초 안에 끝나 엔진의 10초 정지 판정(sustain_s) 전에 끝납니다.
  클립 끝에 마지막 프레임을 tail_s(기본 20초) 동안 반복해 붙입니다 ("마지막 자세가 계속 유지된다"는 가정).
  낙상/ADL 클립에 똑같이 적용하므로, 마지막에 누워 있는 ADL(침대 눕기 등)도 정지 판정을 받습니다.
  반복 프레임은 키포인트가 완전히 같아 실제 영상보다 '더 정지'해 있습니다.
- 대상 전환(target_switched): 클립 파일에는 프레임별 전환 여부가 없어 항상 False (엔진의 1초 유예 미적용).
- bbox conf(tracks[0].conf): 클립에 없어서 키포인트 conf 평균으로 대신함 (엔진은 사용하지 않음).
- 원본 fps 30에서 15Hz로 고를 때 추출기의 IoU 대상 추적은 원본 fps에서 이루어진 결과를 그대로 씀.

EMA 상태는 smoothing 모듈 전역 변수이므로 clip_to_obs는 스레드 안전하지 않습니다 (프로세스 병렬은 무방).
run_clip은 obs를 바꾸지 않으므로 clip_to_obs 결과를 여러 단계/검증기에서 재사용할 수 있습니다.
"""
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from app.config import DEFAULT_CONF
from app.core.level1 import Level1Engine, Level1Params
from app.engine import smoothing as _smoothing

logger = logging.getLogger(__name__)

RUNTIME_HZ = 15.0
FREEZE_TAIL_S = 20.0

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KP_DIR = REPO_ROOT / "dataset" / "_keypoints" / "v1"

NUM_KPS = 17
_TS_EPS = 1e-6

__all__ = [
    "RUNTIME_HZ", "FREEZE_TAIL_S", "DEFAULT_KP_DIR",
    "study_params", "resample_indices", "clip_to_obs", "run_clip", "clip_outcome", "summarize",
    "NullVerifier", "clip_path", "load_row_clip", "replay_rows",
]


# ---------------------------------------------------------------------------
# 클립 -> obs 스트림
# ---------------------------------------------------------------------------

def resample_indices(ts, hz: float = RUNTIME_HZ) -> np.ndarray:
    """
    ts[0]부터 ts[-1]까지(끝 포함) 1/hz 격자의 각 시각에 가장 가까운 원본 프레임 번호를 반환합니다.
    거리가 같으면 앞 프레임. ts는 오름차순이어야 합니다 (clipfmt가 보장). 빈 ts는 빈 배열.
    """
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    if len(ts) == 0:
        return np.zeros(0, dtype=np.int64)
    n = int(math.floor((float(ts[-1]) - float(ts[0])) * float(hz) + _TS_EPS)) + 1
    grid = float(ts[0]) + np.arange(max(n, 1), dtype=np.float64) / float(hz)
    right = np.clip(np.searchsorted(ts, grid, side="left"), 0, len(ts) - 1)
    left = np.clip(right - 1, 0, len(ts) - 1)
    # ts[left] < g <= ts[right]: 더 가까운 쪽, 같으면 앞 프레임
    pick_left = (grid - ts[left]) <= (ts[right] - grid) + _TS_EPS
    return np.where(pick_left, left, right).astype(np.int64)


_EMA_WARNED = False


def _warn_if_ema_disabled() -> None:
    """KP_EMA_ENABLE=0 환경이면 keypoints_smooth가 비어 운영과 다른 입력이 되므로 한 번 경고합니다."""
    global _EMA_WARNED
    from app import config as _cfg
    if not getattr(_cfg, "KP_EMA_ENABLE", True) and not _EMA_WARNED:
        _EMA_WARNED = True
        logger.warning("[replay] KP_EMA_ENABLE=0: EMA 스무딩 없이 재생합니다 (운영 기본값과 다름)")


def study_params(ai_mode: str = "off", **overrides: Any) -> Level1Params:
    """
    실험용 Level1Params. 환경변수(ABNORMAL_TIMEOUT_S, AI_* 등)에 따라 바뀌는 기본값을 코드 기본값으로 고정해
    어느 PC/셸에서 돌려도 같은 규칙으로 재생되게 합니다 (tests/test_level1_golden.py와 같은 값).
    overrides로 개별 값 변경 (예: ai_suppress_pfall_th=0.3).
    """
    base: Dict[str, Any] = dict(
        ai_mode=ai_mode,
        abnormal_timeout_s=10.0,
        ghost_timeout_s=10.0,
        ai_suppress_pfall_th=0.4,
        ai_escalate_s=60.0,
        ai_escalate_requires_drop=True,
    )
    base.update(overrides)
    return Level1Params(**base)


def _frame_shape(frame_wh) -> Optional[Tuple[int, int]]:
    try:
        w, h = int(frame_wh[0]), int(frame_wh[1])
    except (TypeError, ValueError, IndexError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (h, w)


def _make_obs(ts: float, frame_index: int, kps_row: np.ndarray, bbox_row: np.ndarray, present: bool,
              frame_shape: Optional[Tuple[int, int]], src_index: int, tail: bool) -> Dict[str, Any]:
    """build_observation과 같은 구조의 obs 하나 (키포인트 목록은 매번 새로 만듦: EMA가 제자리 수정하므로)."""
    kps: List[Dict[str, Any]] = []
    bbox_norm: Optional[List[float]] = None
    if present:
        kps = [
            {"id": j, "x": float(kps_row[j, 0]), "y": float(kps_row[j, 1]), "conf": float(kps_row[j, 2])}
            for j in range(NUM_KPS)
        ]
        bbox_norm = [float(v) for v in bbox_row]
    quality = float(sum(k["conf"] for k in kps) / len(kps)) if kps else 0.0
    return {
        "schema_version": "1.0",
        "ts": float(ts),
        "frame_index": int(frame_index),
        "source_id": "replay",
        "tracks": [{
            "track_id": 0,
            "has_person": bool(present),
            "bbox": bbox_norm,
            # 클립에는 bbox conf가 없어 키포인트 conf 평균으로 대신 (엔진은 사용하지 않음)
            "conf": quality,
            "keypoints_raw": kps,
            "keypoints_smooth": None,
            "keypoints": kps,
            "quality_score": quality,
            "frame_shape": frame_shape,
        }],
        "target_switched": False,
        "meta": {
            "conf_thres": float(DEFAULT_CONF),
            "replay_src_index": int(src_index),
            "replay_tail": bool(tail),
        },
    }


def clip_to_obs(clip: Dict[str, Any], hz: float = RUNTIME_HZ, tail_s: float = FREEZE_TAIL_S,
                t0: float = 1000.0) -> List[Dict[str, Any]]:
    """
    클립 dict(clipfmt.load_clip 결과: ts, kps (T,17,3), bbox (T,4), present (T,), frame_wh)를
    운영과 같은 obs 목록으로 바꿉니다 (EMA 스무딩 적용, 클립마다 EMA 초기화).

    - 1/hz 격자에서 가장 가까운 원본 프레임 (resample_indices, 연속 중복 제거), obs ts = t0 + 원본 ts
    - 이어서 마지막 원본 프레임을 round(tail_s * hz)개 반복 (정지 꼬리, ts는 1/hz 간격, tail_s <= 0이면 없음)
    - obs["meta"]["replay_src_index"]: 원본 프레임 번호, ["replay_tail"]: 꼬리 프레임 여부
    - frame_index는 0부터 연속
    """
    ts = np.asarray(clip["ts"], dtype=np.float64).reshape(-1)
    t = len(ts)
    if t == 0:
        return []
    kps = np.asarray(clip["kps"], dtype=np.float64).reshape(t, NUM_KPS, 3)
    bbox = np.asarray(clip["bbox"], dtype=np.float64).reshape(t, 4)
    present = np.asarray(clip["present"], dtype=bool).reshape(t)
    shape = _frame_shape(clip.get("frame_wh"))

    _warn_if_ema_disabled()
    idx = resample_indices(ts, hz)
    if len(idx) > 1:
        idx = idx[np.concatenate(([True], np.diff(idx) != 0))]
    n_tail = int(round(float(tail_s) * float(hz))) if tail_s and tail_s > 0 else 0
    sources = [int(i) for i in idx] + [int(idx[-1])] * n_tail
    n_main = len(idx)
    t_last = float(ts[idx[-1]])
    times = [float(t0) + float(ts[i]) for i in idx] + [
        float(t0) + t_last + (j + 1) / float(hz) for j in range(n_tail)]

    _smoothing._reset_ema_state()
    out: List[Dict[str, Any]] = []
    try:
        for k, i in enumerate(sources):
            obs = _make_obs(times[k], k, kps[i], bbox[i], bool(present[i]), shape,
                            src_index=int(i), tail=k >= n_main)
            _smoothing.ema_smooth_keypoints_inplace(obs)
            out.append(obs)
    finally:
        # 다음 사용자(다른 클립, 운영 코드)가 이전 클립의 EMA 상태를 물려받지 않도록
        _smoothing._reset_ema_state()
    return out


# ---------------------------------------------------------------------------
# 엔진 재생
# ---------------------------------------------------------------------------

class NullVerifier:
    """판단 불가(None)만 반환하는 검증기. verifier를 주지 않았을 때 운영 가중치를 불러오지 않도록 사용 (규칙만)."""

    model_id = "null"

    def __call__(self, seq):
        return None


def run_clip(obs_list: Iterable[Dict[str, Any]], params: Level1Params, verifier=None,
             on_event: Optional[Callable[[Dict[str, Any], Level1Engine], None]] = None
             ) -> Tuple[List[Dict[str, Any]], Level1Engine]:
    """
    새 Level1Engine(params, verifier)에 obs를 순서대로 넣고 (이벤트 목록, 엔진)을 반환합니다.

    - verifier가 None이면 NullVerifier (운영 기본 검증기처럼 가중치 파일을 찾지 않음, 항상 fail-open)
    - verifier에 bind_engine(engine)이 있으면 재생 전에 호출 (EventWindowVerifier가 사건 구간을 읽도록)
    - on_event(event, engine): 이벤트가 나올 때마다 바로 호출 (그 순간의 engine.kp_seq,
      engine.get_incident_snapshot()을 가져갈 수 있음)
    """
    v = verifier if verifier is not None else NullVerifier()
    engine = Level1Engine(params, verifier=v)
    bind = getattr(v, "bind_engine", None)
    if callable(bind):
        bind(engine)
    events: List[Dict[str, Any]] = []
    for obs in obs_list:
        ev = engine.step(obs)
        if ev is not None:
            events.append(ev)
            if on_event is not None:
                on_event(ev, engine)
    return events, engine


# ---------------------------------------------------------------------------
# 클립 결과와 요약
# ---------------------------------------------------------------------------

def _ai_p_fall(ev: Dict[str, Any]) -> Optional[float]:
    ai = (ev.get("values") or {}).get("ai")
    if isinstance(ai, dict):
        p = ai.get("p_fall")
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            return float(p)
    return None


def clip_outcome(row: Dict[str, Any], events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    클립 하나의 결과.

    alarm: level1 이벤트가 하나라도 있음 (사유 무관: 정지 지속, 타임아웃, AI 억제 후 승격)
    reasons: level1 사유 목록, suppressed: ai_suppressed 이벤트가 있음,
    reached_ai: LEVEL1 판정 지점까지 도달함 (level1 또는 ai_suppressed). AI는 거부권만 있으므로
                이것이 규칙 엔진이 허용하는 탐지 상한입니다.
    추가: n_abnormal (ABNORMAL 진입 수), alarm_ts (첫 level1 시각),
          p_fall (AI 호출마다 한 번씩의 p_fall. 억제 후 승격 이벤트는 억제 당시 판정을 다시 싣고 있으므로 제외)
    """
    level1 = [e for e in events if e.get("type") == "level1"]
    sup = [e for e in events if e.get("type") == "ai_suppressed"]
    ai_calls = [e for e in events if e.get("type") in ("level1", "ai_suppressed")
                and e.get("reason") != "escalated_after_ai_suppression"]
    p_fall = [p for p in (_ai_p_fall(e) for e in ai_calls) if p is not None]
    return {
        "clip_id": row.get("clip_id"),
        "dataset": row.get("dataset"),
        "label": row.get("label"),
        "alarm": bool(level1),
        "reasons": [e.get("reason") for e in level1],
        "suppressed": bool(sup),
        "reached_ai": bool(level1 or sup),
        "n_abnormal": sum(1 for e in events if e.get("type") == "abnormal_enter"),
        "p_fall": p_fall,
        "alarm_ts": float(level1[0]["ts"]) if level1 else None,
    }


def _ratio(a: int, b: int) -> Optional[float]:
    return (a / b) if b > 0 else None


def _hmean(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return 0.0 if a + b <= 0 else 2.0 * a * b / (a + b)


def _stats(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    falls = [o for o in outcomes if o.get("label") == "fall"]
    adls = [o for o in outcomes if o.get("label") == "adl"]
    n_fall, n_adl = len(falls), len(adls)
    n_det = sum(1 for o in falls if o["alarm"])
    n_fa = sum(1 for o in adls if o["alarm"])
    n_fall_reached = sum(1 for o in falls if o["reached_ai"])
    n_adl_reached = sum(1 for o in adls if o["reached_ai"])
    recall = _ratio(n_det, n_fall)
    fa_rate = _ratio(n_fa, n_adl)
    spec = None if fa_rate is None else 1.0 - fa_rate
    precision = _ratio(n_det, n_det + n_fa)
    return {
        "n_fall": n_fall,
        "n_adl": n_adl,
        "n_detected": n_det,
        "n_adl_alarm": n_fa,
        "recall": recall,
        "adl_fa_rate": fa_rate,
        # 낙상 recall과 ADL 특이도(1 - 오탐률)의 조화평균 (F1과 같은 형태, 클래스 비율에 무관)
        "balanced": _hmean(recall, spec),
        # AI가 억제했고 끝내 알람이 없었던 낙상 클립 (AI 때문에 놓친 낙상)
        "suppressed_true_falls": sum(1 for o in falls if o["suppressed"] and not o["alarm"]),
        # AI 억제가 한 번이라도 있었던 클립 수 (승격으로 결국 알람이 난 경우 포함)
        "n_fall_suppressed": sum(1 for o in falls if o["suppressed"]),
        "n_adl_suppressed": sum(1 for o in adls if o["suppressed"]),
        # 규칙 엔진이 LEVEL1 판정 지점까지 간 비율 (AI는 거부권만 있으므로 recall 상한)
        "rules_ceiling": _ratio(n_fall_reached, n_fall),
        "n_fall_reached": n_fall_reached,
        # AI가 없었다면 났을 ADL 오탐 비율 (AI가 줄일 수 있는 오탐의 출발점)
        "adl_rules_fa_rate": _ratio(n_adl_reached, n_adl),
        "n_adl_reached": n_adl_reached,
        "precision": precision,
        "f1": _hmean(recall, precision),
    }


def summarize(outcomes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    clip_outcome 목록 -> 전체 및 데이터셋별 지표.
    recall = 탐지된 낙상 클립 / 낙상 클립, adl_fa_rate = 알람이 난 ADL 클립 / ADL 클립,
    balanced = recall과 (1 - adl_fa_rate)의 조화평균, rules_ceiling = LEVEL1 판정에 도달한 낙상 클립 비율.
    분모가 0이면 비율은 None.
    """
    outcomes = list(outcomes)
    out = _stats(outcomes)
    datasets = sorted({str(o.get("dataset")) for o in outcomes})
    out["by_dataset"] = {d: _stats([o for o in outcomes if str(o.get("dataset")) == d]) for d in datasets}
    return out


# ---------------------------------------------------------------------------
# manifest 행 단위 편의 함수
# ---------------------------------------------------------------------------

def clip_path(clip_id: str, kp_dir=None) -> Path:
    """clip_id -> <kp_dir>/<clip_id>.npz (추출기 extract_keypoints.clip_out_path와 같은 규칙)."""
    from training.datasets.extract_keypoints import clip_out_path
    return clip_out_path(Path(kp_dir) if kp_dir is not None else DEFAULT_KP_DIR, clip_id)


def load_row_clip(row: Dict[str, Any], kp_dir=None) -> Dict[str, Any]:
    """manifest 행의 키포인트 클립을 읽습니다 (clipfmt.load_clip)."""
    from training.datasets.clipfmt import load_clip
    return load_clip(clip_path(row["clip_id"], kp_dir))


def replay_rows(rows: Iterable[Dict[str, Any]], params: Level1Params, verifier=None, kp_dir=None,
                obs_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                on_event: Optional[Callable[[Dict[str, Any], Dict[str, Any], Level1Engine], None]] = None,
                hz: float = RUNTIME_HZ, tail_s: float = FREEZE_TAIL_S) -> List[Dict[str, Any]]:
    """
    manifest 행마다 clip_to_obs -> run_clip -> clip_outcome. 결과 목록을 행 순서대로 반환합니다.
    obs_cache(dict)를 주면 clip_id별 obs 목록을 저장/재사용합니다 (같은 hz/tail_s에서만 재사용할 것).
    on_event(row, event, engine): 행 정보를 함께 받는 이벤트 콜백.
    키포인트 파일이 없는 행은 FileNotFoundError.
    """
    outcomes = []
    for row in rows:
        cid = row["clip_id"]
        obs = obs_cache.get(cid) if obs_cache is not None else None
        if obs is None:
            obs = clip_to_obs(load_row_clip(row, kp_dir), hz=hz, tail_s=tail_s)
            if obs_cache is not None:
                obs_cache[cid] = obs
        cb = None
        if on_event is not None:
            cb = (lambda ev, eng, _row=row: on_event(_row, ev, eng))
        events, _ = run_clip(obs, params, verifier=verifier, on_event=cb)
        outcomes.append(clip_outcome(row, events))
    return outcomes
