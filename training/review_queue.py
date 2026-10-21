"""
현장 클립 검수 큐 (표준 라이브러리만 사용)

운영 중 저장된 이벤트 sidecar(runs/events/*.json)와 클립(runs/clips/{fall,not_fall}/*.mp4)을 모아 검수 목록을 만들고,
사람이 붙인 라벨은 클립과 따로 append-only 기록(runs/review/labels.jsonl)에 쌓습니다.
클립과 sidecar는 고치거나 옮기지 않습니다.

  python -m training.review_queue list                       # 검수 전 항목 (우선순위, 시각 순)
  python -m training.review_queue list --status all --priority 1
  python -m training.review_queue show <id>
  python -m training.review_queue label <id> fall --fall-start 4.2 --fall-end 5.0 --note "침대 옆 바닥"
  python -m training.review_queue label <id> not_fall
  python -m training.review_queue stats
  python -m training.review_queue export --out dataset/_manifest/field_runs.jsonl

항목 id: sidecar 파일 이름(event_sidecar.sidecar_filename: 이벤트 시각과 종류)에서 .json을 뺀 값.
  sidecar 없이 클립만 있으면 "clip_<fall|not_fall>_<클립 파일 이름>". 목록 순서와 무관하게 늘 같음.
결과(outcome):
  level1               AI가 통과시킨 알람
  ai_suppressed        AI가 억제했고 이후 승격 기록이 없음 (사람이 회복했거나, AI가 실제 낙상을 숨겼거나)
  suppressed_escalated AI가 억제했지만 붕괴가 이어져 나중에 LEVEL1로 승격된 사건의 억제 기록
  escalated            그 사건의 승격 LEVEL1 기록 (reason "escalated_after_ai_suppression")
  unknown              sidecar 없는 클립, 알 수 없는 이벤트 종류
우선순위: 1 = AI 억제가 걸린 사건 (ai_suppressed, suppressed_escalated, escalated, sidecar 없는 not_fall 클립),
  2 = 일반 level1 (sidecar 없는 fall 클립 포함), 3 = 그 밖. AI가 실제 낙상을 숨겼을 수 있는 곳을 먼저 봅니다.
사건 연결: 같은 사건의 억제 기록과 승격 기록은 sidecar 스냅샷의 abnormal_start_ts가 같습니다. 스냅샷이 없으면
  승격 이벤트의 ts - values.suppressed_for_s가 억제 이벤트의 ts와 같은지로 찾습니다. 연결된 항목은 같은
  incident(manifest event_group)를 써서 학습/평가 분할에서 함께 움직입니다.
  sidecar 없는 클립끼리는 클립 이름의 저장 시각으로 추정합니다: not_fall 클립 저장 뒤 승격이 가능한 시간
  (CLIP_ONLY_LINK_MIN_S ~ CLIP_ONLY_LINK_MAX_S) 안에 저장된 fall 클립은 같은 사건으로 묶고 우선순위 1로 둡니다.
  추정이라 다른 사건을 묶을 수 있지만, 묶이면 아래 export 규칙으로 행이 줄거나 빠질 뿐 학습/평가에 섞이지는 않습니다.
라벨: fall / not_fall / unsure. 같은 id는 마지막 기록이 이깁니다. fall은 클립 기준 낙상 구간(--fall-start/--fall-end,
  클립 시작부터 초)이 있어야 내보냅니다. export는 검수된 fall/not_fall만 내보내고 unsure와 미검수는 내보내지 않습니다.
  라벨 기록에 읽지 못한 줄이 있으면 export(와 retrain_from_runs)는 멈춥니다. 그 줄이 어떤 id의 마지막 정정이었는지
  알 수 없어 이전 라벨로 학습할 수 있기 때문입니다. 줄을 고치거나 지운 뒤 다시 실행합니다.
"""
import argparse
import getpass
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    # python training/<script>.py 로 직접 실행한 경우에도 저장소 루트의 app/, training/을 찾도록
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import AI_ESCALATE_S
from training.datasets.manifest import make_row, validate_rows, write_manifest
from training.paths import REPO_ROOT, RUNS_DIR

DATASET = "field_runs"
LICENSE = "private field data (not for redistribution)"
LABEL_SOURCE = "review"
LABELS = ("fall", "not_fall", "unsure")
STATUSES = ("unreviewed",) + LABELS
CLIP_SUBDIRS = ("fall", "not_fall")
ESCALATED_REASON = "escalated_after_ai_suppression"
# 같은 사건 판정 허용 오차(초). 두 값 모두 같은 엔진 시각에서 나온 값이라 부동소수 오차만 있음
LINK_TOL_S = 0.01
PRIORITY = {"ai_suppressed": 1, "suppressed_escalated": 1, "escalated": 1, "level1": 2}
# sidecar 없는 클립의 사건 추정 범위(초): not_fall 클립 저장 시각부터 fall 클립 저장 시각까지.
# 승격은 억제 후 AI_ESCALATE_S가 지나야 나므로 아래 한도는 그보다 조금 짧게(클립 이름은 초 단위),
# 위 한도는 억제 뒤 붕괴 확인이 늦어지는 경우를 넉넉히 잡은 값
CLIP_ONLY_LINK_MIN_S = AI_ESCALATE_S - 5.0
CLIP_ONLY_LINK_MAX_S = AI_ESCALATE_S + 240.0
# label에서 낙상 구간 끝이 클립 길이(sidecar t_end - t_start)를 넘는지 볼 때 허용 오차(초)
CLIP_SPAN_TOL_S = 1.0
DEFAULT_EXPORT = REPO_ROOT / "dataset" / "_manifest" / "field_runs.jsonl"


def labels_path_for(runs_dir) -> Path:
    """runs 폴더의 검수 라벨 기록 위치 (training.paths.REVIEW_LABELS와 같은 규칙)."""
    return Path(runs_dir) / "review" / "labels.jsonl"


# ---------------------------------------------------------------------------
# 스캔: sidecar + 클립 -> 항목
# ---------------------------------------------------------------------------

def _num(v) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN 제외


def _local(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
    except (OverflowError, OSError, ValueError):
        return None


def _key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _rel(path: Path, root: Path = REPO_ROOT) -> str:
    """root 기준 상대 경로 (밖이면 절대 경로), 구분자는 '/'."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_clip(recorded, runs_dir: Path) -> Optional[Path]:
    """
    sidecar에 기록된 클립 경로를 실제 파일로 찾습니다.
    런타임은 실행 위치 기준 상대 경로(runs/clips/...)나 컨테이너 절대 경로를 기록하므로, 그대로 -> runs 폴더의
    부모 기준 -> runs/clips/<기록된 폴더 또는 fall, not_fall>/<파일 이름> 순으로 봅니다.
    """
    if not recorded or not isinstance(recorded, str):
        return None
    p = Path(recorded)
    cands = [p] if p.is_absolute() else [runs_dir.parent / p, Path.cwd() / p]
    for sub in dict.fromkeys((p.parent.name,) + CLIP_SUBDIRS):
        if sub:
            cands.append(runs_dir / "clips" / sub / p.name)
    for c in cands:
        if c.is_file():
            return c
    return None


def _clip_name_ts(stem: str) -> Optional[float]:
    """ClipRecorder 클립 이름(<prefix>_YYYYmmdd_HHMMSS)의 저장 시각."""
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    try:
        return time.mktime(time.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S"))
    except (ValueError, OverflowError):
        return None


def _sidecar_entry(path: Path, doc: Dict[str, Any], runs_dir: Path) -> Dict[str, Any]:
    ev = doc["event"]
    vals = ev.get("values") if isinstance(ev.get("values"), dict) else {}
    ai = vals.get("ai") if isinstance(vals.get("ai"), dict) else {}
    snap = doc.get("snapshot") if isinstance(doc.get("snapshot"), dict) else {}
    clip = doc.get("clip") if isinstance(doc.get("clip"), dict) else {}
    clip_path = _resolve_clip(clip.get("path"), runs_dir)
    ts = _num(ev.get("ts"))
    t0, t1 = _num(clip.get("t_start")), _num(clip.get("t_end"))
    abn = _num(snap.get("abnormal_start_ts"))
    return {
        "id": path.stem,
        "source": "sidecar",
        "event_type": ev.get("type"),
        "reason": ev.get("reason"),
        "ts": ts,
        "time": _local(ts),
        "p_fall": _num(ai.get("p_fall")),
        "ai_decision": ai.get("decision"),
        "device_id": doc.get("device_id"),
        "sidecar": _rel(path),
        "clip": _rel(clip_path) if clip_path else None,
        "clip_recorded": clip.get("path"),
        "folder": clip_path.parent.name if clip_path else None,
        "clip_span_s": (t1 - t0) if t0 is not None and t1 is not None else None,
        # 클립 안에서 ABNORMAL 진입 시각 (낙상 구간을 찾을 때 참고, 클립 첫 프레임이 t_start보다 늦으면 조금 어긋남)
        "entry_in_clip_s": (abn - t0) if abn is not None and t0 is not None else None,
        "abnormal_start_ts": abn,
        "suppressed_for_s": _num(vals.get("suppressed_for_s")),
        "linked": None,
    }


def _link_escalations(entries: List[Dict[str, Any]]) -> None:
    """승격 LEVEL1 기록을 같은 사건의 억제 기록과 연결합니다 (entries를 고침)."""
    supp = [e for e in entries if e["event_type"] == "ai_suppressed" and e["ts"] is not None]
    for e in entries:
        if not (e["event_type"] == "level1" and e["reason"] == ESCALATED_REASON and e["ts"] is not None):
            continue
        best = None
        for s in supp:
            if s["linked"] is not None or s["ts"] > e["ts"]:
                continue
            if e["device_id"] is not None and s["device_id"] is not None and e["device_id"] != s["device_id"]:
                continue
            same = (e["abnormal_start_ts"] is not None and s["abnormal_start_ts"] is not None
                    and abs(e["abnormal_start_ts"] - s["abnormal_start_ts"]) <= LINK_TOL_S)
            if not same and e["suppressed_for_s"] is not None:
                same = abs(e["ts"] - e["suppressed_for_s"] - s["ts"]) <= LINK_TOL_S
            if same and (best is None or s["ts"] > best["ts"]):
                best = s
        if best is not None:
            e["linked"], best["linked"] = best["id"], e["id"]


def _link_clip_only(entries: List[Dict[str, Any]]) -> None:
    """
    sidecar 없는 클립끼리 같은 사건을 추정해 묶습니다 (entries를 고침).
    fall 클립마다 그 전에 저장된 가장 최근 not_fall 클립을 보고, 저장 시각 차이가 승격 가능 범위
    (CLIP_ONLY_LINK_MIN_S ~ CLIP_ONLY_LINK_MAX_S) 안이면 같은 incident(먼저 난 not_fall 클립 id)로 둡니다.
    더 최근의 억제가 있었다면 그 전 사건은 이미 끝났으므로 가장 최근 것만 봅니다.
    """
    only = sorted((e for e in entries if e["source"] == "clip_only" and e["ts"] is not None),
                  key=lambda e: (e["ts"], e["id"]))
    last_nf = None
    for e in only:
        if e["folder"] == "not_fall":
            last_nf = e
            continue
        if last_nf is None or last_nf["linked"] is not None:
            continue
        if CLIP_ONLY_LINK_MIN_S <= e["ts"] - last_nf["ts"] <= CLIP_ONLY_LINK_MAX_S:
            e["linked"], last_nf["linked"] = last_nf["id"], e["id"]
            e["incident"] = last_nf["incident"]


def _outcome(e: Dict[str, Any]) -> str:
    if e["event_type"] == "ai_suppressed":
        return "suppressed_escalated" if e["linked"] else "ai_suppressed"
    if e["event_type"] == "level1":
        return "escalated" if e["reason"] == ESCALATED_REASON else "level1"
    return "unknown"


def _priority(e: Dict[str, Any]) -> int:
    if e["outcome"] in PRIORITY:
        return PRIORITY[e["outcome"]]
    if e["source"] == "clip_only" and e["linked"]:
        return 1  # 억제 클립과 묶인 클립 (승격 알람일 수 있음)
    # sidecar 없는 클립: not_fall 폴더에는 AI 억제 클립만 저장됨
    return {"not_fall": 1, "fall": 2}.get(e.get("folder"), 3)


def scan(runs_dir=RUNS_DIR) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    runs 폴더를 읽어 검수 항목 목록과 문제 목록(읽지 못한 sidecar 등)을 반환합니다.
    항목은 (우선순위, 시각, id) 순으로 정렬됩니다.
    """
    runs_dir = Path(runs_dir)
    entries: List[Dict[str, Any]] = []
    problems: List[str] = []
    events_dir = runs_dir / "events"
    for path in sorted(events_dir.glob("*.json")) if events_dir.is_dir() else []:
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            problems.append(f"{path.name}: unreadable ({type(e).__name__})")
            continue
        if not isinstance(doc, dict) or not isinstance(doc.get("event"), dict):
            problems.append(f"{path.name}: no event object")
            continue
        entries.append(_sidecar_entry(path, doc, runs_dir))

    _link_escalations(entries)
    linked_incident = {}
    for e in entries:
        # 연결된 사건은 먼저 난 억제 기록의 id를 사건 id로 씀
        if e["event_type"] == "ai_suppressed":
            linked_incident[e["id"]] = e["id"]
            if e["linked"]:
                linked_incident[e["linked"]] = e["id"]
    for e in entries:
        e["incident"] = linked_incident.get(e["id"], e["id"])

    referenced = {_key(REPO_ROOT / e["clip"]) for e in entries if e["clip"]}
    for sub in CLIP_SUBDIRS:
        d = runs_dir / "clips" / sub
        for clip in sorted(d.glob("*.mp4")) if d.is_dir() else []:
            if clip.name.endswith(".tmp.mp4") or _key(clip) in referenced:
                continue  # 변환 중 임시 파일, sidecar가 있는 클립
            ts = _clip_name_ts(clip.stem)
            cid = f"clip_{sub}_{clip.stem}"
            entries.append({
                "id": cid, "source": "clip_only", "event_type": None, "reason": None, "ts": ts,
                "time": _local(ts), "p_fall": None, "ai_decision": None, "device_id": None, "sidecar": None,
                "clip": _rel(clip), "clip_recorded": None, "folder": sub, "clip_span_s": None,
                "entry_in_clip_s": None, "abnormal_start_ts": None, "suppressed_for_s": None, "linked": None,
                "incident": cid,
            })
    _link_clip_only(entries)

    seen = Counter(e["id"] for e in entries)
    for k, n in seen.items():
        if n > 1:
            problems.append(f"{k}: duplicate id ({n} entries)")
    for e in entries:
        e["outcome"] = _outcome(e)
        e["priority"] = _priority(e)
    entries.sort(key=lambda e: (e["priority"], e["ts"] if e["ts"] is not None else float("inf"), e["id"]))
    return entries, problems


# ---------------------------------------------------------------------------
# 라벨 기록 (append-only JSONL, 같은 id는 마지막 줄이 이김)
# ---------------------------------------------------------------------------

def default_reviewer() -> str:
    name = os.getenv("REVIEWER", "").strip()
    if name:
        return name
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _check_record(entry_id, label, fall_start_s, fall_end_s) -> Tuple[Optional[float], Optional[float]]:
    """id, 라벨, 낙상 구간 검사 (make_label_record와 load_labels가 같은 규칙을 씀). 반환: 구간(float), 틀리면 ValueError."""
    if not entry_id or not isinstance(entry_id, str):
        raise ValueError("empty or non-string id")
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    if (fall_start_s is None) != (fall_end_s is None):
        raise ValueError("give both --fall-start and --fall-end, or neither")
    if fall_start_s is None:
        return None, None
    if label != "fall":
        raise ValueError("fall segment is only for label 'fall'")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (fall_start_s, fall_end_s)):
        raise ValueError("fall segment must be numbers")
    a, b = float(fall_start_s), float(fall_end_s)
    if not (math.isfinite(a) and math.isfinite(b) and 0.0 <= a < b):
        raise ValueError("fall segment must satisfy 0 <= start < end, finite (seconds from clip start)")
    return a, b


def make_label_record(entry_id: str, label: str, note: str = "", reviewer: Optional[str] = None,
                      fall_start_s=None, fall_end_s=None, now: Optional[float] = None) -> Dict[str, Any]:
    """검수 기록 한 줄을 만듭니다. 형식이 틀리면 ValueError."""
    fall_start_s, fall_end_s = _check_record(entry_id, label, fall_start_s, fall_end_s)
    return {
        "id": entry_id,
        "label": label,
        "fall_start_s": fall_start_s,
        "fall_end_s": fall_end_s,
        "note": note or "",
        "reviewer": reviewer or default_reviewer(),
        "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() if now is None else now)),
    }


def append_label(labels_path, record: Dict[str, Any]) -> None:
    """
    기록을 한 줄 덧붙입니다 (기존 줄은 고치지 않음).
    마지막 줄이 줄바꿈 없이 끝나 있으면(편집기가 끝 줄바꿈을 지웠거나, 이전 기록이 쓰다 끊긴 경우) 줄을 바꾼 뒤 씁니다.
    붙여 쓰면 새 기록까지 읽지 못하는 줄이 되기 때문입니다.
    """
    labels_path = Path(labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    with open(labels_path, "a+b") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                data = b"\n" + data
        f.write(data)


def load_labels(labels_path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    {id: 마지막 기록}, 문제 목록. 파일이 없으면 빈 dict.
    형식이 틀린 줄(JSON이 아님, make_label_record와 같은 검사에 걸림)은 건너뛰고 문제로 남김.
    """
    labels: Dict[str, Dict[str, Any]] = {}
    problems: List[str] = []
    labels_path = Path(labels_path)
    if not labels_path.is_file():
        return labels, problems
    with open(labels_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                problems.append(f"{labels_path.name}:{n}: invalid JSON")
                continue
            if not isinstance(rec, dict):
                problems.append(f"{labels_path.name}:{n}: bad record (not an object)")
                continue
            try:
                a, b = _check_record(rec.get("id"), rec.get("label"), rec.get("fall_start_s"), rec.get("fall_end_s"))
            except ValueError as err:
                problems.append(f"{labels_path.name}:{n}: bad record ({err})")
                continue
            rec["fall_start_s"], rec["fall_end_s"] = a, b
            labels[rec["id"]] = rec
    return labels, problems


def attach_labels(entries: List[Dict[str, Any]], labels: Dict[str, Dict[str, Any]]) -> List[str]:
    """항목에 최신 라벨(review)과 상태(status)를 붙입니다. 반환: 항목이 없는 라벨 id 목록 (클립이 지워진 경우 등)."""
    ids = set()
    for e in entries:
        rec = labels.get(e["id"])
        e["review"] = rec
        e["status"] = rec["label"] if rec else "unreviewed"
        ids.add(e["id"])
    return sorted(k for k in labels if k not in ids)


class LabelLogError(ValueError):
    """라벨 기록에 읽지 못한 줄이 있음 (strict 읽기에서)."""


def load_queue(runs_dir=RUNS_DIR, labels_path=None, strict: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    scan + 라벨. 반환: (항목, 문제 목록).
    strict이면 라벨 기록에 읽지 못한 줄이 있을 때 LabelLogError (학습용 export에서 사용: 그 줄이 어떤 id의
    마지막 정정이었는지 알 수 없어, 건너뛰면 이전 라벨로 학습할 수 있음).
    """
    runs_dir = Path(runs_dir)
    labels, lp = load_labels(labels_path if labels_path is not None else labels_path_for(runs_dir))
    if strict and lp:
        raise LabelLogError("; ".join(lp))
    entries, problems = scan(runs_dir)
    orphan = attach_labels(entries, labels)
    return entries, problems + lp + [f"label for unknown id {k}" for k in orphan]


# ---------------------------------------------------------------------------
# export: 검수된 fall/not_fall -> manifest 행
# ---------------------------------------------------------------------------

def export_rows(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    검수된 항목을 manifest 행(training/datasets/manifest.py)으로 바꿉니다. 반환: (행, 제외 목록 [{id, reason}]).
    - fall/not_fall만 내보냄 (unsure, 미검수는 대상 아님)
    - 클립 파일이 없으면 no_clip, fall인데 낙상 구간이 없으면 fall_no_segment
    - 한 사건(incident)의 클립들(억제 클립과 승격 클립은 같은 사건 시점에서 시작해 겹침)은 하나만 내보냄:
      그중 하나라도 unsure면 모두 incident_unsure (억제 클립이 승격 클립 구간을 모두 담으므로),
      라벨이 서로 다르면 모두 conflict, 같으면 가장 긴 클립 하나 (나머지는 duplicate_of:<id>)
    event_group = field_runs/<incident>, label not_fall -> "adl", label_source "review".
    video는 저장소 루트 기준 상대 경로(밖이면 절대 경로)로 extract_keypoints가 그대로 읽습니다.
    """
    skipped: List[Dict[str, Any]] = []
    by_incident: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        if e.get("status") not in LABELS:
            continue
        by_incident[e["incident"]].append(e)

    rows = []
    for inc in sorted(by_incident):
        group = [e for e in by_incident[inc] if e["status"] != "unsure"]
        if not group:
            continue
        if len(group) < len(by_incident[inc]):
            skipped += [{"id": e["id"], "reason": "incident_unsure"} for e in group]
            continue
        if len({e["status"] for e in group}) > 1:
            skipped += [{"id": e["id"], "reason": "conflict"} for e in group]
            continue
        ok = []
        for e in group:
            rv = e["review"]
            if not e.get("clip") or not (REPO_ROOT / e["clip"]).is_file():
                skipped.append({"id": e["id"], "reason": "no_clip"})
            elif e["status"] == "fall" and rv.get("fall_start_s") is None:
                skipped.append({"id": e["id"], "reason": "fall_no_segment"})
            else:
                ok.append(e)
        if not ok:
            continue
        ok.sort(key=lambda e: (-(e["clip_span_s"] or 0.0), e["id"]))
        best = ok[0]
        skipped += [{"id": e["id"], "reason": f"duplicate_of:{best['id']}"} for e in ok[1:]]
        rv = best["review"]
        segs = []
        if best["status"] == "fall":
            segs = [{"cls": "fall", "start_s": rv["fall_start_s"], "end_s": rv["fall_end_s"]}]
        row = make_row(DATASET, best["id"], best["clip"], segs, LABEL_SOURCE, LICENSE,
                       cam=best.get("device_id"), event_group=f"{DATASET}/{inc}",
                       label="fall" if best["status"] == "fall" else "adl")
        # fps는 넣지 않음: ClipRecorder가 실제 저장 fps를 컨테이너에 기록하고 추출기는 그 값을 씀
        row["review"] = {"id": best["id"], "outcome": best["outcome"], "p_fall": best["p_fall"],
                         "reviewer": rv.get("reviewer"), "labeled_at": rv.get("labeled_at"), "note": rv.get("note", "")}
        rows.append(row)
    errors = validate_rows(rows)
    if errors:
        raise ValueError("exported rows fail manifest validation: " + "; ".join(errors[:5]))
    return rows, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_p(p) -> str:
    return "-" if p is None else f"{p:.2f}"


def _filter(entries, status: str, priority: Optional[int], outcome: Optional[str]):
    out = []
    for e in entries:
        if status == "reviewed" and e["status"] == "unreviewed":
            continue
        if status not in ("all", "reviewed") and e["status"] != status:
            continue
        if priority is not None and e["priority"] > priority:
            continue
        if outcome and e["outcome"] != outcome:
            continue
        out.append(e)
    return out


def _print_problems(problems: List[str]) -> None:
    for p in problems:
        print(f"[warn] {p}", file=sys.stderr)


def _safe_console() -> None:
    """
    콘솔 인코딩(cp949 등)에 없는 문자(메모의 '—', '✓' 등)를 출력할 때 멈추지 않도록 역슬래시 이스케이프로 바꿔 씁니다.
    출력을 파일이나 파이프로 보내면 Windows에서 콘솔 코드페이지가 쓰이기 때문입니다.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass  # StringIO 등 reconfigure가 없는 스트림


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="현장 클립 검수 큐 (runs/events sidecar + runs/clips)")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR), help="운영 출력 폴더 (기본: 저장소 runs/)")
    ap.add_argument("--labels", default=None, help="검수 라벨 기록 JSONL (기본: <runs-dir>/review/labels.jsonl)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="항목 목록 (우선순위, 시각 순)")
    p_list.add_argument("--status", default="unreviewed", choices=("all", "reviewed") + STATUSES)
    p_list.add_argument("--priority", type=int, default=None, help="이 값 이하 우선순위만 (1 = AI 억제 사건)")
    p_list.add_argument("--outcome", default=None,
                        choices=("level1", "ai_suppressed", "suppressed_escalated", "escalated", "unknown"))
    p_list.add_argument("--json", action="store_true", help="JSON 한 줄씩 출력")

    p_show = sub.add_parser("show", help="항목 하나의 자세한 정보")
    p_show.add_argument("id")

    p_label = sub.add_parser("label", help="라벨 기록 (append-only, 같은 id는 마지막 기록이 이김)")
    p_label.add_argument("id")
    p_label.add_argument("label", choices=LABELS)
    p_label.add_argument("--fall-start", type=float, default=None, help="낙상 시작 (클립 시작부터 초, fall일 때)")
    p_label.add_argument("--fall-end", type=float, default=None, help="낙상 끝 (클립 시작부터 초, fall일 때)")
    p_label.add_argument("--note", default="")
    p_label.add_argument("--reviewer", default=None, help="검수자 (기본: REVIEWER 환경변수 또는 OS 사용자)")

    sub.add_parser("stats", help="상태/결과별 개수")

    p_exp = sub.add_parser("export", help="검수된 fall/not_fall을 manifest JSONL로 (unsure, 미검수 제외)")
    p_exp.add_argument("--out", default=str(DEFAULT_EXPORT))
    args = ap.parse_args(argv)
    _safe_console()

    runs_dir = Path(args.runs_dir)
    labels_path = Path(args.labels) if args.labels else labels_path_for(runs_dir)
    try:
        entries, problems = load_queue(runs_dir, labels_path, strict=args.cmd == "export")
    except LabelLogError as err:
        print(f"error: unreadable lines in {labels_path}: {err}\n"
              "Fix or delete those lines (a cut-off line may have been the latest label of an id) and export again.",
              file=sys.stderr)
        return 1
    by_id = {e["id"]: e for e in entries}

    if args.cmd == "list":
        _print_problems(problems)
        sel = _filter(entries, args.status, args.priority, args.outcome)
        for e in sel:
            if args.json:
                print(json.dumps(e, ensure_ascii=True))  # 기계가 읽는 출력이라 콘솔 인코딩과 무관하게 ASCII
            else:
                print(f"P{e['priority']} {e['status']:<10} {e['outcome']:<20} p={_fmt_p(e['p_fall']):<4} "
                      f"{e['time'] or '-':<19} {e['id']}  {e['clip'] or '(no clip)'}")
        if not args.json:
            print(f"{len(sel)} of {len(entries)} entries")
        return 0

    if args.cmd == "show":
        e = by_id.get(args.id)
        if e is None:
            print(f"unknown id: {args.id}", file=sys.stderr)
            return 2
        print(json.dumps(e, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "label":
        e = by_id.get(args.id)
        if e is None:
            print(f"unknown id: {args.id} (run 'list --status all')", file=sys.stderr)
            return 2
        try:
            rec = make_label_record(args.id, args.label, note=args.note, reviewer=args.reviewer,
                                    fall_start_s=args.fall_start, fall_end_s=args.fall_end)
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
        span = e.get("clip_span_s")
        if rec["fall_end_s"] is not None and span is not None and rec["fall_end_s"] > span + CLIP_SPAN_TOL_S:
            # 클립보다 긴 구간은 추출 단계(check_label_timebase)에서 거부되어 재학습 전체가 멈춤
            print(f"error: --fall-end {rec['fall_end_s']:g}s is past the clip end (clip is about {span:.1f}s)",
                  file=sys.stderr)
            return 2
        _print_problems(problems)
        append_label(labels_path, rec)
        print(f"labeled {args.id} = {args.label} -> {labels_path}")
        if args.label == "fall" and rec["fall_start_s"] is None:
            hint = e.get("entry_in_clip_s")
            hint = f" (ABNORMAL entry at about {hint:.1f}s in clip)" if hint is not None else ""
            print(f"[warn] no fall segment: not exported until relabeled with --fall-start/--fall-end{hint}")
        if not e.get("clip"):
            print("[warn] entry has no clip file: not exported")
        return 0

    if args.cmd == "stats":
        _print_problems(problems)
        by_status = Counter(e["status"] for e in entries)
        print(f"entries {len(entries)}: " + ", ".join(f"{s} {by_status.get(s, 0)}" for s in STATUSES))
        by_outcome: Dict[str, Counter] = defaultdict(Counter)
        for e in entries:
            by_outcome[e["outcome"]][e["status"]] += 1
        for oc in sorted(by_outcome, key=lambda o: (PRIORITY.get(o, 3), o)):
            c = by_outcome[oc]
            print(f"  {oc:<20} " + ", ".join(f"{s} {c.get(s, 0)}" for s in STATUSES))
        rows, skipped = export_rows(entries)
        n_fall = sum(r["label"] == "fall" for r in rows)
        print(f"exportable rows {len(rows)} (fall {n_fall}, adl {len(rows) - n_fall}), "
              f"skipped {dict(Counter(s['reason'].split(':')[0] for s in skipped))}")
        if load_labels(labels_path)[1]:
            print("[warn] export and retrain_from_runs refuse to run until the unreadable label lines are fixed",
                  file=sys.stderr)
        return 0

    if args.cmd == "export":
        _print_problems(problems)
        rows, skipped = export_rows(entries)
        write_manifest(args.out, rows)
        for s in skipped:
            print(f"[skip] {s['id']}: {s['reason']}")
        n_fall = sum(r["label"] == "fall" for r in rows)
        print(f"exported {len(rows)} rows (fall {n_fall}, adl {len(rows) - n_fall}) -> {args.out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
