"""
포트폴리오/README용 데모 이미지 (venv 전용: numpy, opencv, torch, matplotlib, ultralytics)

이미지에 적는 값(시각, 엔진 상태, ABNORMAL 진입, LEVEL1, p_fall)은 모두 연구와 같은 재생에서 나온 값입니다.
손으로 적은 값은 없습니다.

- 재생: training.experiments.replay (15Hz, EMA, 20초 정지 꼬리), 앞단은 운영과 같은 v3_fb 키포인트,
  대상 전환 정책 multi, 회전 폴백 켬, lost_grace_s 0.5, 래치 끔 (study_params("veto"), 억제 임계값 0.4)
- AI: E1max 판정 (SlidingWindowVerifier mode max). 기본은 그 클립이 test fold였던 progression 모델
  (dataset/_experiments/p8_full/fold{k}/models/E1_s0.pt)이라 p_fall은 학습에 쓰지 않은 클립의 값입니다.
  배포 가중치(fall_cnn_gap_v2.pt)는 모든 클립으로 학습해 데모 클립도 학습에 들어 있으므로 기본으로 쓰지 않습니다
  (--verifier deployed로 바꿀 수 있고, 그때는 연구 결과와의 대조를 건너뜀).
- 대조: 재생 결과(알람, 사유, ABNORMAL 진입 시각, LEVEL1 시각, p_fall)를 연구 결과 파일
  (p8_full/fold{k}/E1max.json, seed 0)의 같은 클립 결과와 비교하고, 다르면 이미지를 만들지 않고 멈춤.
- 회전 폴백 그림: 저장된 키포인트의 fallback 플래그로 프레임을 고른 뒤 같은 포즈 모델(클립 메타의 모델, conf, imgsz)을
  그 프레임에 다시 돌려 원래 방향 검출이 0명이고 app.engine.rotate_fallback.fallback_detect가 사람을 찾는 것을 확인.
- 화면 문구는 OpenCV가 한글을 그리지 못해 영어(ASCII)로 적습니다. 클립 설명은 데이터셋 원본 표기
  (CAUCAFall 폴더 이름, GMDCSA24 ADL.csv/Fall.csv의 Description)만 씁니다.

사용 (저장소 루트, dataset/에 영상과 v3_fb 키포인트, p8_full 결과가 있어야 함):
  C:/fdv/Scripts/python -m training.demo.render_demo --out docs/assets
  C:/fdv/Scripts/python -m training.demo.render_demo --only strip timeline --fall-clip caucafall/forward/FallForwardS10

순수 함수(trace_clip 결과 고르기, 연구 결과 대조, 폴백 프레임 고르기, 좌표 변환)는 numpy만 쓰며
tests/test_demo_render.py가 확인합니다. cv2/torch/matplotlib/ultralytics는 함수 안에서만 불러옵니다.
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
KP_DIR = REPO_ROOT / "dataset" / "_keypoints" / "v3_fb"
MANIFEST = REPO_ROOT / "dataset" / "_manifest" / "manifest_v1.jsonl"
FOLDS = REPO_ROOT / "dataset" / "_experiments" / "folds_v1.json"
STUDY_DIR = REPO_ROOT / "dataset" / "_experiments" / "p8_full"
POSE_DIR = REPO_ROOT / "dataset" / "_models"
DEFAULT_OUT = REPO_ROOT / "docs" / "assets"

# 기본 클립 (CC BY 4.0 데이터셋만: CAUCAFall, GMDCSA24)
FALL_CLIP = "caucafall/backwards/FallBackwardsS9"
ADL_CLIP = "gmdcsa24/Subject_2/ADL/09"
ROT_CLIP = "gmdcsa24/Subject_2/Fall/06"
ALLOWED_DATASETS = ("caucafall", "gmdcsa24")

# 연구(p8_full)와 같은 앞단 설정
FRONTEND = {"switch_policy": "multi", "use_fallback": True, "lost_grace_s": 0.5}
STUDY_SEED = 0
VETO_TH = 0.4
# 연구 결과와 대조할 때 p_fall 허용 오차 (같은 가중치, GPU/CPU 차이 정도)
P_TOL = 1e-4
T_TOL = 1e-6
# 관절 표시 하한 (그림 전용, 엔진 입력과 무관)
DRAW_KP_CONF = 0.3

# COCO-17 골격 연결
COCO_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)

# 상태별 색 (BGR, OpenCV)
STATE_BGR = {"NORMAL": (70, 160, 60), "ABNORMAL": (0, 140, 245), "LEVEL1": (40, 40, 215), "VETO": (190, 110, 30)}

__all__ = [
    "COCO_EDGES", "FALL_CLIP", "ADL_CLIP", "ROT_CLIP", "kps_to_pixels", "bbox_to_pixels", "trace_clip",
    "decision_event", "pick_strip", "compare_outcome", "fallback_candidates", "ascii_text", "main",
]


# ---------------------------------------------------------------------------
# 순수 함수 (numpy만)
# ---------------------------------------------------------------------------

def kps_to_pixels(kps_norm, frame_wh) -> np.ndarray:
    """정규화 키포인트 (17, 3) [x/w, y/h, conf] -> 픽셀 (17, 3) [x, y, conf]"""
    k = np.asarray(kps_norm, dtype=np.float64).reshape(-1, 3).copy()
    w, h = float(frame_wh[0]), float(frame_wh[1])
    k[:, 0] *= w
    k[:, 1] *= h
    return k


def bbox_to_pixels(bbox_norm, frame_wh) -> np.ndarray:
    """정규화 상자 [x1, y1, x2, y2] -> 픽셀"""
    w, h = float(frame_wh[0]), float(frame_wh[1])
    return np.asarray(bbox_norm, dtype=np.float64).reshape(4) * np.array([w, h, w, h])


def ascii_text(s: Any) -> str:
    """OpenCV 글꼴로 그릴 수 있게 ASCII로 (그 밖의 문자는 '?')"""
    return str(s).encode("ascii", "replace").decode("ascii")


def trace_clip(obs_list: Sequence[Dict[str, Any]], params, verifier=None, t0: Optional[float] = None
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    replay.run_clip과 같은 재생(새 엔진, bind_engine, obs를 순서대로 step)을 하면서 프레임마다 엔진 상태를 기록합니다.
    반환: (이벤트 목록, trace). trace[i] = {t(클립 시각), src(원본 프레임 번호), tail, present, aspect(엔진이 쓰는
    정규화 상자 h/w), state(그 step 뒤 엔진 상태), event(그 step의 이벤트 종류 또는 None)}.
    """
    from app.core.level1 import Level1Engine, bbox_aspect
    from training.experiments import replay as R

    t0 = R.DEFAULT_T0 if t0 is None else float(t0)
    v = verifier if verifier is not None else R.NullVerifier()
    engine = Level1Engine(params, verifier=v)
    bind = getattr(v, "bind_engine", None)
    if callable(bind):
        bind(engine)
    events: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    for obs in obs_list:
        ev = engine.step(obs)
        if ev is not None:
            events.append(ev)
        tr = (obs.get("tracks") or [{}])[0]
        meta = obs.get("meta") or {}
        present = bool(tr.get("has_person"))
        trace.append({
            "t": float(obs["ts"]) - t0,
            "src": int(meta.get("replay_src_index", -1)),
            "tail": bool(meta.get("replay_tail")),
            "present": present,
            "aspect": bbox_aspect(tr.get("bbox")) if present else None,
            "state": str(engine.state),
            "event": ev.get("type") if ev is not None else None,
        })
    return events, trace


def decision_event(events: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """첫 AI 판정 지점 이벤트 (level1 또는 ai_suppressed, 억제 후 승격은 제외). 없으면 None."""
    for e in events:
        if e.get("type") in ("level1", "ai_suppressed") and e.get("reason") != "escalated_after_ai_suppression":
            return e
    return None


def _nearest(trace: Sequence[Dict[str, Any]], t: float, real_only: bool = True) -> int:
    """시각 t에 가장 가까운 trace 번호 (real_only면 정지 꼬리 제외, 없으면 전체)"""
    idx = [i for i, s in enumerate(trace) if not (real_only and s["tail"])] or list(range(len(trace)))
    return min(idx, key=lambda i: (abs(trace[i]["t"] - t), i))


def _index_at(trace: Sequence[Dict[str, Any]], ts_clip: float) -> int:
    """이벤트 시각(클립 시각)과 같은 trace 번호"""
    return min(range(len(trace)), key=lambda i: (abs(trace[i]["t"] - ts_clip), i))


def pick_strip(trace: Sequence[Dict[str, Any]], events: Sequence[Dict[str, Any]], t0: float,
               motion_s: float = 0.5, before_s: float = 1.5) -> List[Tuple[str, int]]:
    """
    연속 장면에 쓸 프레임 (이름, trace 번호) 목록. 모두 재생 결과로 정합니다.
      before   : ABNORMAL 진입 before_s초 전 (NORMAL)
      motion   : 진입 motion_s초 전 (넘어지는/자세를 낮추는 중, 아직 NORMAL)
      entry    : ABNORMAL 진입 프레임 (판정 이벤트 직전의 마지막 abnormal_enter)
      hold     : 진입과 판정 사이 가운데 (정지 꼬리면 마지막 실제 프레임 쪽으로)
      decision : AI 판정 프레임 (level1 또는 ai_suppressed). 정지 꼬리 안이면 그 프레임은 마지막 실제 프레임의 반복
    판정 이벤트가 없으면 [] (데모에 쓸 수 없는 클립).
    """
    dec = decision_event(events)
    if dec is None or not trace:
        return []
    t_dec = float(dec["ts"]) - t0
    enters = [float(e["ts"]) - t0 for e in events if e.get("type") == "abnormal_enter" and float(e["ts"]) - t0 <= t_dec]
    if not enters:
        return []
    t_ent = enters[-1]
    i_ent = _index_at(trace, t_ent)
    i_dec = _index_at(trace, t_dec)
    real = [i for i, s in enumerate(trace) if not s["tail"]]
    t_last_real = trace[real[-1]]["t"] if real else t_dec
    t_hold = (t_ent + min(t_dec, t_last_real)) / 2.0
    out = [("before", _nearest(trace, max(trace[0]["t"], t_ent - before_s))),
           ("motion", _nearest(trace, max(trace[0]["t"], t_ent - motion_s))),
           ("entry", i_ent),
           ("hold", _nearest(trace, t_hold)),
           ("decision", i_dec)]
    # 겹치는 프레임은 앞 이름을 남김 (클립이 짧을 때)
    seen, uniq = set(), []
    for name, i in out:
        if i not in seen:
            seen.add(i)
            uniq.append((name, i))
    return uniq


def compare_outcome(got: Dict[str, Any], ref: Dict[str, Any], p_tol: float = P_TOL, t_tol: float = T_TOL) -> List[str]:
    """
    재생 결과(replay.clip_outcome)와 연구 결과 파일의 같은 클립 결과 비교. 다른 항목 설명 목록 (같으면 []).
    비교: alarm, suppressed, reasons, n_abnormal, abnormal_s, level1_s, p_fall.
    """
    diffs: List[str] = []
    for key in ("alarm", "suppressed", "reasons", "n_abnormal"):
        if got.get(key) != ref.get(key):
            diffs.append(f"{key}: {got.get(key)!r} != {ref.get(key)!r}")

    def close_list(a, b, tol):
        a, b = list(a or []), list(b or [])
        return len(a) == len(b) and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))

    if not close_list(got.get("abnormal_s"), ref.get("abnormal_s"), t_tol):
        diffs.append(f"abnormal_s: {got.get('abnormal_s')} != {ref.get('abnormal_s')}")
    a, b = got.get("level1_s"), ref.get("level1_s")
    if (a is None) != (b is None) or (a is not None and abs(float(a) - float(b)) > t_tol):
        diffs.append(f"level1_s: {a} != {b}")
    if not close_list(got.get("p_fall"), ref.get("p_fall"), p_tol):
        diffs.append(f"p_fall: {got.get('p_fall')} != {ref.get('p_fall')}")
    return diffs


def fallback_candidates(fallback, present, fps: float, recent_s: float = 5.0) -> List[Tuple[int, int]]:
    """
    회전 폴백 그림 후보 (프레임 번호, 직전 대상 프레임 번호) 목록. 폴백 프레임이 연속으로 가장 길게 이어진 구간의
    가운데에 가까운 순. 직전 대상 프레임 = 그 앞에서 대상이 있던 마지막 프레임 (추출기의 last_seen과 같음),
    recent_s초보다 멀면 제외 (추출기도 그때는 폴백하지 않음).
    """
    fb = np.asarray(fallback, dtype=bool).reshape(-1)
    pr = np.asarray(present, dtype=bool).reshape(-1)
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < len(fb):
        if fb[i]:
            j = i
            while j + 1 < len(fb) and fb[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    out: List[Tuple[int, int]] = []
    for a, b in sorted(runs, key=lambda r: (-(r[1] - r[0]), r[0])):
        mid = (a + b) / 2.0
        for f in sorted(range(a, b + 1), key=lambda x: (abs(x - mid), x)):
            prev = [p for p in range(f - 1, -1, -1) if pr[p]]
            if prev and (f - prev[0]) / float(fps) <= recent_s:
                out.append((f, prev[0]))
    return out


# ---------------------------------------------------------------------------
# 데이터 읽기
# ---------------------------------------------------------------------------

def load_rows(path: Path = MANIFEST) -> Dict[str, Dict[str, Any]]:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["clip_id"]] = r
    return rows


def study_outcome(clip_id: str, fold: int, seed: int = STUDY_SEED, variant: str = "E1max") -> Dict[str, Any]:
    d = json.loads((STUDY_DIR / f"fold{fold}" / f"{variant}.json").read_text(encoding="utf-8"))
    fe = d.get("frontend") or {}
    for k, v in FRONTEND.items():
        if fe.get(k) != v:
            raise RuntimeError(f"study frontend mismatch {k}: {fe.get(k)!r} != {v!r}")
    run = next(r for r in d["runs"] if r.get("seed") == seed)
    return next(o for o in run["outcomes"] if o["clip_id"] == clip_id)


def clip_description(row: Dict[str, Any]) -> str:
    """데이터셋 원본 표기 (CAUCAFall: 폴더 이름, GMDCSA24: ADL.csv/Fall.csv의 Description)"""
    video = REPO_ROOT / row["video"]
    if row["dataset"] == "caucafall":
        return f"CAUCAFall {video.parent.parent.name.replace('.', ' ')}, '{video.parent.name}'"
    if row["dataset"] == "gmdcsa24":
        csv_path = video.parent.parent / f"{video.parent.name}.csv"
        desc = ""
        try:
            with open(csv_path, encoding="utf-8-sig", newline="") as f:
                for rec in csv.reader(f):
                    if rec and rec[0].strip() == video.name:
                        desc = rec[4].strip() if len(rec) > 4 else ""
                        break
        except OSError:
            pass
        head = f"GMDCSA24 {video.parent.parent.name}, {video.parent.name} {video.stem}"
        return f"{head}: '{desc}'" if desc else head
    return row["clip_id"]


def read_frames(video: Path, indices: Sequence[int]) -> Dict[int, np.ndarray]:
    """영상을 처음부터 순서대로 읽어 필요한 프레임만 반환 (추출기와 같은 순차 디코딩, 탐색 오차 없음)"""
    import cv2

    want = set(int(i) for i in indices)
    out: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    i = 0
    try:
        while want - set(out):
            ok, frame = cap.read()
            if not ok:
                break
            if i in want:
                out[i] = frame
            i += 1
    finally:
        cap.release()
    missing = want - set(out)
    if missing:
        raise RuntimeError(f"{video}: frames not decoded {sorted(missing)[:5]}")
    return out


# ---------------------------------------------------------------------------
# 재생
# ---------------------------------------------------------------------------

def _device(arg: str) -> str:
    if arg != "auto":
        return arg
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def replay_clip(row: Dict[str, Any], fold: int, device: str, verifier_kind: str = "oof") -> Dict[str, Any]:
    """
    연구와 같은 앞단/규칙으로 클립을 재생합니다. 반환: clip, obs, events, trace, outcome, fold, study(연구 결과 또는 None).
    replay 모듈 전역(SWITCH_POLICY, USE_FALLBACK, STUDY_OVERRIDES)은 재생 뒤 원래 값으로 되돌림.
    """
    from training.experiments import replay as R
    from training.experiments.verifiers import SlidingWindowVerifier

    if verifier_kind == "oof":
        from training.experiments.run_progression import _e_scorer, _load_e_model

        entry = _load_e_model(STUDY_DIR / f"fold{fold}" / "models", "E1", STUDY_SEED)
        if entry is None:
            raise FileNotFoundError(f"E1_s{STUDY_SEED}.pt not found for fold {fold}")
        verifier = SlidingWindowVerifier(_e_scorer(entry, device), mode="max", model_id=f"E1max-fold{fold}-s0")
    else:
        from app.ai_classifier.inference import load_classifier

        clf = load_classifier(str(REPO_ROOT / "app" / "ai_classifier" / "weights" / "fall_cnn_gap_v2.pt"))
        verifier = clf.make_engine_verifier()
        if verifier is None:
            raise RuntimeError("deployed v2 classifier not available")

    saved = (R.SWITCH_POLICY, R.USE_FALLBACK, dict(R.STUDY_OVERRIDES))
    try:
        R.SWITCH_POLICY = FRONTEND["switch_policy"]
        R.USE_FALLBACK = FRONTEND["use_fallback"]
        R.STUDY_OVERRIDES.clear()
        R.STUDY_OVERRIDES["lost_grace_s"] = FRONTEND["lost_grace_s"]
        params = R.study_params("veto")
        assert params.ai_suppress_pfall_th == VETO_TH and not params.post_alarm_latch
        clip = R.load_row_clip(row, KP_DIR)
        obs = R.clip_to_obs(clip)
        events, trace = trace_clip(obs, params, verifier)
        outcome = R.clip_outcome(row, events, end_s=R.clip_end_s(obs))
    finally:
        R.SWITCH_POLICY, R.USE_FALLBACK = saved[0], saved[1]
        R.STUDY_OVERRIDES.clear()
        R.STUDY_OVERRIDES.update(saved[2])
    study = study_outcome(row["clip_id"], fold) if verifier_kind == "oof" else None
    return {"clip": clip, "obs": obs, "events": events, "trace": trace, "outcome": outcome, "fold": fold,
            "study": study, "params": params, "verifier": verifier_kind}


# ---------------------------------------------------------------------------
# 그리기 (OpenCV)
# ---------------------------------------------------------------------------

def draw_pose(img: np.ndarray, kps_px: Optional[np.ndarray], bbox_px: Optional[np.ndarray], color,
              scale: float = 1.0, thick: int = 2) -> np.ndarray:
    """이미지 위에 상자와 골격 (좌표는 이미 img 기준 픽셀, scale은 좌표에 곱함)"""
    import cv2

    if bbox_px is not None:
        x1, y1, x2, y2 = (int(round(v * scale)) for v in bbox_px)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
    if kps_px is not None:
        k = np.asarray(kps_px, dtype=np.float64)
        pts = [(int(round(x * scale)), int(round(y * scale))) for x, y in k[:, :2]]
        ok = k[:, 2] >= DRAW_KP_CONF
        for a, b in COCO_EDGES:
            if ok[a] and ok[b]:
                cv2.line(img, pts[a], pts[b], (255, 255, 255), thick + 2, cv2.LINE_AA)
                cv2.line(img, pts[a], pts[b], color, thick, cv2.LINE_AA)
        for j, p in enumerate(pts):
            if ok[j]:
                cv2.circle(img, p, thick + 1, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, p, thick, color, -1, cv2.LINE_AA)
    return img


def _text(img, s, org, scale=0.48, color=(30, 30, 30), thick=1):
    import cv2
    cv2.putText(img, ascii_text(s), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _band(width: int, height: int, color=(250, 250, 250)) -> np.ndarray:
    return np.full((height, width, 3), color, np.uint8)


def _tile(frame: np.ndarray, tile_w: int) -> Tuple[np.ndarray, float]:
    import cv2
    s = tile_w / float(frame.shape[1])
    return cv2.resize(frame, (tile_w, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA), s


def _p_fall(ev: Optional[Dict[str, Any]]) -> Optional[float]:
    ai = ((ev or {}).get("values") or {}).get("ai")
    p = ai.get("p_fall") if isinstance(ai, dict) else None
    return float(p) if isinstance(p, (int, float)) and not isinstance(p, bool) else None


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def strip_captions(name: str, step: Dict[str, Any], t_ent: float, dec: Dict[str, Any]) -> Tuple[str, str, str]:
    """(1줄, 2줄, 상태색 키). 모든 값은 trace/이벤트에서."""
    t = step["t"]
    line1 = f"t={t:.2f}s  state={step['state']}"
    if name in ("before", "motion"):
        line2 = f"{t_ent - t:.1f}s before ABNORMAL entry"
        if not step["present"]:
            line2 += " (no person)"
        return line1, line2, step["state"]
    if name == "entry":
        return line1, "ABNORMAL entry (rules)", "ABNORMAL"
    if name == "hold":
        return line1, f"ABNORMAL for {t - t_ent:.1f}s", step["state"]
    p = _p_fall(dec)
    if dec.get("type") == "level1":
        return (f"t={t:.2f}s  LEVEL1 (alarm)", f"AI p_fall={_fmt_p(p)} >= {VETO_TH}: pass", "LEVEL1")
    return (f"t={t:.2f}s  AI veto (no alarm)", f"AI p_fall={_fmt_p(p)} < {VETO_TH}", "VETO")


def render_strip(res: Dict[str, Any], row: Dict[str, Any], out_path: Path, tile_w: int = 300,
                 names: Optional[Sequence[str]] = None) -> Path:
    import cv2

    from training.experiments import replay as R

    trace, events, clip = res["trace"], res["events"], res["clip"]
    picks = pick_strip(trace, events, R.DEFAULT_T0)
    if names is not None:
        picks = [p for p in picks if p[0] in names]
    if not picks:
        raise RuntimeError(f"{row['clip_id']}: no decision event, cannot draw strip")
    dec = decision_event(events)
    t_ent = trace[dict(picks)["entry"]]["t"] if "entry" in dict(picks) else float(dec["ts"]) - R.DEFAULT_T0
    frames = read_frames(REPO_ROOT / row["video"], [trace[i]["src"] for _, i in picks])
    wh = clip["frame_wh"]
    tiles = []
    for name, i in picks:
        st = trace[i]
        img, s = _tile(frames[st["src"]].copy(), tile_w)
        l1, l2, key = strip_captions(name, st, t_ent, dec)
        color = STATE_BGR.get(key, STATE_BGR["NORMAL"])
        if clip["present"][st["src"]]:
            draw_pose(img, kps_to_pixels(clip["kps"][st["src"]], wh), bbox_to_pixels(clip["bbox"][st["src"]], wh),
                      color, scale=s)
        if st["tail"]:
            cv2.rectangle(img, (0, 0), (232, 24), (40, 40, 40), -1)
            _text(img, "clip ended: last frame held", (8, 17), 0.45, (255, 255, 255), 1)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 4)
        cap = _band(img.shape[1], 50)
        cv2.rectangle(cap, (0, 0), (cap.shape[1] - 1, 5), color, -1)
        _text(cap, l1, (8, 24), 0.5, (20, 20, 20), 1)
        _text(cap, l2, (8, 43), 0.45, (60, 60, 60), 1)
        tiles.append(np.vstack([img, cap]))
    h = max(t.shape[0] for t in tiles)
    gap = np.full((h, 4, 3), 255, np.uint8)
    body = tiles[0]
    for t in tiles[1:]:
        body = np.hstack([body, gap, t])
    head = _band(body.shape[1], 30, (255, 255, 255))
    model = (f"AI: E1max, fold {res['fold']} model (clip held out)" if res["verifier"] == "oof"
             else "AI: deployed v2 weights")
    _text(head, f"{clip_description(row)}  |  15 Hz replay, rules + AI veto (th {VETO_TH})  |  {model}",
          (8, 20), 0.45, (20, 20, 20), 1)
    foot = credit_footer(body.shape[1], row["dataset"],
                         "Modified: resized and tiled; skeleton/box (YOLOv8s-pose keypoints used by the engine), "
                         "borders and text added.")
    out = np.vstack([head, body, foot])
    return _write(out_path, out)


def dataset_credit(dataset: str) -> str:
    if dataset == "caucafall":
        return "CAUCAFall (Eraso et al., Mendeley Data, doi:10.17632/7w7fccy7ky.4), CC BY 4.0"
    if dataset == "gmdcsa24":
        return "GMDCSA24 (Alam et al., Zenodo, doi:10.5281/zenodo.13354453), CC BY 4.0"
    raise ValueError(f"dataset not allowed for demo images: {dataset}")


def credit_footer(width: int, dataset: str, modified: str) -> np.ndarray:
    """출처, 라이선스 주소, 바꾼 점 두 줄 (CC BY 4.0 3(a)(1)). 이미지만 따로 써도 표시가 남도록 그림 안에 넣음"""
    foot = _band(width, 38, (255, 255, 255))
    _text(foot, f"Source: {dataset_credit(dataset)} (creativecommons.org/licenses/by/4.0).",
          (8, 15), 0.4, (90, 90, 90), 1)
    _text(foot, modified, (8, 31), 0.4, (90, 90, 90), 1)
    return foot


def _write(path: Path, img: np.ndarray) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    else:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    path.write_bytes(buf.tobytes())
    return path


# ---------------------------------------------------------------------------
# 시간축 그림 (matplotlib)
# ---------------------------------------------------------------------------

def render_timeline(res: Dict[str, Any], row: Dict[str, Any], out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from training.experiments import replay as R

    trace, events, params = res["trace"], res["events"], res["params"]
    dec = decision_event(events)
    t_dec = float(dec["ts"]) - R.DEFAULT_T0 if dec else None
    enters = [float(e["ts"]) - R.DEFAULT_T0 for e in events if e.get("type") == "abnormal_enter"]
    real = [s for s in trace if not s["tail"]]
    t_end = real[-1]["t"] if real else trace[-1]["t"]
    x_max = max(t_end, t_dec or 0.0) + 0.3
    # 실제 프레임은 실선, 정지 꼬리(마지막 프레임 반복)는 판정 시각까지만 점선
    xs = [s["t"] for s in real]
    ys = [s["aspect"] if s["aspect"] is not None else np.nan for s in real]
    tail = [s for s in trace if s["tail"] and s["t"] <= x_max]

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=120)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    # 엔진 상태 구간 (trace의 state)
    cur, start = None, None
    spans = []
    for s in trace:
        if s["t"] > x_max:
            break
        if s["state"] != cur:
            if cur is not None:
                spans.append((cur, start, s["t"]))
            cur, start = s["state"], s["t"]
    if cur is not None:
        spans.append((cur, start, min(x_max, trace[-1]["t"])))
    shade = {"ABNORMAL": "#fbe3c4", "LEVEL1": "#f6cdcd"}
    for st, a, b in spans:
        if st in shade:
            ax.axvspan(a, b, color=shade[st], lw=0, zorder=0)
    if t_dec is not None and t_dec > t_end:
        ax.axvspan(t_end, x_max, facecolor="none", hatch="///", edgecolor="#c9c8c3", lw=0, zorder=0)
        ax.text(t_end + 0.05, 0.02, "clip ended (last frame held)", transform=ax.get_xaxis_transform(),
                color=muted, fontsize=8)
    ax.plot(xs, ys, color="#2a78d6", lw=2, zorder=3)
    if tail:
        ax.plot([xs[-1]] + [s["t"] for s in tail],
                [ys[-1]] + [s["aspect"] if s["aspect"] is not None else np.nan for s in tail],
                color="#2a78d6", lw=1.5, ls="--", zorder=3)
    ax.axhline(params.aspect_abs_th, color=muted, lw=1, ls=":", zorder=2)
    ax.text(0.1, params.aspect_abs_th + 0.05, f"posture rule: h/w < {params.aspect_abs_th}", va="bottom",
            ha="left", color=muted, fontsize=8)
    for t in enters:
        ax.axvline(t, color="#c98500", lw=1.5, zorder=2)
        ax.text(t, 1.01, "ABNORMAL entry", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                color=ink, fontsize=8)
    if dec is not None:
        p = _p_fall(dec)
        lab = (f"LEVEL1 (alarm)\np_fall={_fmt_p(p)}" if dec.get("type") == "level1"
               else f"AI veto\np_fall={_fmt_p(p)}")
        ax.axvline(t_dec, color="#e34948" if dec.get("type") == "level1" else "#6250d6", lw=1.5, zorder=2)
        ax.text(t_dec - 0.1, 0.95, lab, transform=ax.get_xaxis_transform(), ha="right", va="top",
                color=ink, fontsize=8)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("clip time (s)", color=muted, fontsize=9)
    ax.set_ylabel("bbox h/w", color=muted, fontsize=9)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(axis="y", color=grid, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.set_title(ascii_text(f"{clip_description(row)}: engine replay (shaded = ABNORMAL / LEVEL1 state)"),
                 color=ink, fontsize=9, loc="left", pad=16)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 회전 폴백 그림
# ---------------------------------------------------------------------------

def rerun_fallback(row: Dict[str, Any], device: str, max_tries: int = 20) -> Dict[str, Any]:
    """
    저장된 fallback 플래그로 후보 프레임을 고르고 포즈 모델을 다시 돌려 확인합니다.
    조건: 원래 방향 검출 0명 (클립 메타의 conf 이상), fallback_detect가 직전 대상 근처에서 사람을 찾음.
    반환: frame, idx, base_n, det(상자, conf, 관절, 각도), stored_diff(저장된 키포인트와의 최대 차이, 정규화 좌표).
    """
    from app.engine import rotate_fallback as RF
    from training.datasets.extract_keypoints import make_yolo_predictor
    from training.experiments import replay as R

    clip = R.load_row_clip(row, KP_DIR)
    if clip.get("fallback") is None:
        raise RuntimeError(f"{row['clip_id']}: no fallback flags")
    meta = clip["meta"] if isinstance(clip["meta"], dict) else json.loads(clip["meta"])
    ex = meta["extractor"]
    fps = float(meta["native_fps"])
    model_path = POSE_DIR / ex["model"]
    predict = make_yolo_predictor(model_path, "0" if device == "cuda" else "cpu", float(ex["conf"]), int(ex["imgsz"]))
    cands = fallback_candidates(clip["fallback"], clip["present"], fps, RF.FALLBACK_RECENT_S)[:max_tries]
    if not cands:
        raise RuntimeError(f"{row['clip_id']}: no fallback frames")
    frames = read_frames(REPO_ROOT / row["video"], [f for f, _ in cands])
    wh = clip["frame_wh"]
    for f, prev in cands:
        img = frames[f]
        boxes, _, _, _ = predict([img])[0]
        if len(boxes) != 0:
            continue
        last_seen = bbox_to_pixels(clip["bbox"][prev], wh)
        det = RF.fallback_detect(lambda imgs: [(b, c, np.concatenate([kx, kc[..., None]], axis=-1))
                                               for b, c, kx, kc in predict(imgs)], img, last_seen)
        if det is None:
            continue
        k_norm = det[2][:, :2] / np.array([wh[0], wh[1]], np.float64)
        diff = float(np.max(np.abs(np.clip(k_norm, 0, 1) - clip["kps"][f][:, :2])))
        return {"frame": img, "idx": int(f), "t": float(clip["ts"][f]), "base_n": 0, "det": det,
                "stored_diff": diff, "conf": float(ex["conf"]), "model": ex["model"], "prev": int(prev)}
    raise RuntimeError(f"{row['clip_id']}: no candidate frame reproduced (tried {len(cands)})")


def render_rotation(fb: Dict[str, Any], row: Dict[str, Any], out_path: Path, panel_h: int = 300) -> Path:
    import cv2

    from app.engine import rotate_fallback as RF

    img = fb["frame"]
    h, w = img.shape[:2]
    box, conf, kps, angle = fb["det"]
    color = STATE_BGR["ABNORMAL"]

    def fit(im):
        s = panel_h / float(im.shape[0])
        return cv2.resize(im, (int(round(im.shape[1] * s)), panel_h), interpolation=cv2.INTER_AREA), s

    a, _ = fit(img.copy())
    rot = RF.rotate_image(img.copy(), angle)
    rk = kps.copy()
    rk[:, :2] = RF.to_rotated(kps[:, :2], w, h, angle)
    corners = RF.to_rotated(np.array([[box[0], box[1]], [box[2], box[3]]]), w, h, angle)
    rbox = np.array([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()])
    b, sb = fit(rot)
    draw_pose(b, rk, rbox, color, scale=sb)
    c, sc = fit(img.copy())
    draw_pose(c, kps, box, color, scale=sc)
    caps = [("1. Upright pass", f"{fb['model']}, conf>={fb['conf']}: 0 persons"),
            (f"2. Rotated {angle} deg", f"person found (box conf {conf:.2f})"),
            ("3. Mapped back", "target recovered, sent to engine")]
    tiles = []
    for im, (l1, l2) in zip((a, b, c), caps):
        cap = _band(im.shape[1], 46)
        _text(cap, l1, (8, 20), 0.5, (20, 20, 20), 1)
        _text(cap, l2, (8, 39), 0.42, (60, 60, 60), 1)
        tiles.append(np.vstack([im, cap]))
    gap = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
    body = np.hstack([tiles[0], gap, tiles[1], gap, tiles[2]])
    head = _band(body.shape[1], 30, (255, 255, 255))
    _text(head, f"Rotation fallback (app/engine/rotate_fallback.py) on {clip_description(row)}, "
                f"frame {fb['idx']} (t={fb['t']:.2f}s)", (8, 20), 0.45, (20, 20, 20), 1)
    foot = credit_footer(body.shape[1], row["dataset"],
                         "Modified: resized and tiled, middle tile rotated; skeleton/box (pose model re-run on this "
                         "frame) and text added.")
    return _write(out_path, np.vstack([head, body, foot]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _check_row(rows: Dict[str, Dict[str, Any]], clip_id: str) -> Dict[str, Any]:
    row = rows.get(clip_id)
    if row is None:
        raise SystemExit(f"clip not in manifest: {clip_id}")
    if row["dataset"] not in ALLOWED_DATASETS:
        raise SystemExit(f"{clip_id}: only {ALLOWED_DATASETS} (CC BY 4.0) are allowed for demo images")
    return row


def _replay_checked(row, folds, device, verifier_kind, no_check) -> Dict[str, Any]:
    fold = int(folds[row["clip_id"]])
    res = replay_clip(row, fold, device, verifier_kind)
    o = res["outcome"]
    print(f"[demo] {row['clip_id']} fold {fold}: alarm {o['alarm']} reasons {o['reasons']} "
          f"suppressed {o['suppressed']} abnormal_s {o['abnormal_s']} level1_s {o['level1_s']} p_fall {o['p_fall']}")
    if res["study"] is not None:
        diffs = compare_outcome(o, res["study"])
        if diffs and not no_check:
            raise SystemExit(f"[demo] replay differs from study result ({STUDY_DIR.name} fold {fold}): {diffs}")
        print(f"[demo]   matches {STUDY_DIR.name}/fold{fold}/E1max.json seed {STUDY_SEED}: {not diffs}")
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Render truthful demo images from study replays (CAUCAFall/GMDCSA24 only)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", nargs="+", default=["strip", "timeline", "rotation", "veto"],
                    choices=["strip", "timeline", "rotation", "veto"])
    ap.add_argument("--fall-clip", default=FALL_CLIP)
    ap.add_argument("--adl-clip", default=ADL_CLIP)
    ap.add_argument("--rot-clip", default=ROT_CLIP)
    ap.add_argument("--verifier", default="oof", choices=["oof", "deployed"],
                    help="oof = held-out fold E1 model from p8_full (default), deployed = fall_cnn_gap_v2.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="photo composites (timeline is always png)")
    ap.add_argument("--no-check", action="store_true", help="do not stop when the replay differs from the study file")
    args = ap.parse_args(argv)

    out = Path(args.out)
    rows = load_rows()
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    device = _device(args.device)
    written: List[Path] = []
    if "strip" in args.only or "timeline" in args.only:
        row = _check_row(rows, args.fall_clip)
        res = _replay_checked(row, folds, device, args.verifier, args.no_check)
        if not res["outcome"]["alarm"]:
            raise SystemExit(f"{args.fall_clip}: no alarm in replay, pick a correctly detected fall clip")
        if "strip" in args.only:
            written.append(render_strip(res, row, out / f"demo_fall_strip.{args.ext}"))
        if "timeline" in args.only:
            written.append(render_timeline(res, row, out / "demo_fall_timeline.png"))
    if "veto" in args.only:
        row = _check_row(rows, args.adl_clip)
        res = _replay_checked(row, folds, device, args.verifier, args.no_check)
        if res["outcome"]["alarm"] or not res["outcome"]["suppressed"]:
            raise SystemExit(f"{args.adl_clip}: not an AI-vetoed ADL clip in replay")
        written.append(render_strip(res, row, out / f"demo_adl_veto.{args.ext}",
                                    names=("before", "entry", "hold", "decision")))
    if "rotation" in args.only:
        row = _check_row(rows, args.rot_clip)
        fb = rerun_fallback(row, device)
        print(f"[demo] {row['clip_id']} frame {fb['idx']} t={fb['t']:.2f}s: upright 0 persons, rotated {fb['det'][3]} "
              f"conf {fb['det'][1]:.3f}, max |kp - stored| {fb['stored_diff']:.4f} (normalized)")
        written.append(render_rotation(fb, row, out / f"demo_rotation_fallback.{args.ext}"))
    for p in written:
        print(f"[demo] wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
