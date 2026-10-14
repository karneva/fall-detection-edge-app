"""
이벤트 단위 평가 지표 (표준 라이브러리만 사용)

평가 단위는 '녹화 파일 1개 = 사건 0~1개'입니다.

- fall 파일: [fall_start - pre_tol_s, fall_end + max_latency_s] 안에 level1이 있으면 탐지(TP), 없으면 미탐(FN)
  창 밖의 level1은 오탐(FP)
- adl 파일: 모든 level1이 오탐(FP)

시간당 오탐(FA/h)은 adl 파일 길이 합으로 나눕니다.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

PRE_TOL_S = 2.0          # 라벨 시점보다 조금 이른 알람 허용
MAX_LATENCY_S = 120.0    # 이보다 늦은 알람은 미탐으로 간주


@dataclass
class Truth:
    """파일 하나의 정답. 시각은 obs ts와 같은 기준(절대 시각)입니다."""
    file: str
    label: str                          # "fall" | "adl"
    fall_start_ts: Optional[float] = None
    fall_end_ts: Optional[float] = None
    scenario: str = "unknown"
    source: str = "unknown"
    subject_id: Optional[str] = None
    session_id: Optional[str] = None


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def score_file(
    truth: Truth,
    events: Iterable[Dict[str, Any]],
    duration_s: float,
    pre_tol_s: float = PRE_TOL_S,
    max_latency_s: float = MAX_LATENCY_S,
) -> Dict[str, Any]:
    """파일 하나의 엔진 이벤트를 정답과 비교합니다."""
    events = list(events)
    level1 = [e for e in events if e.get("type") == "level1"]
    suppressed = [e for e in events if e.get("type") == "ai_suppressed"]

    rec: Dict[str, Any] = {
        "file": truth.file,
        "label": truth.label,
        "scenario": truth.scenario,
        "source": truth.source,
        "duration_s": round(float(duration_s), 3),
        "n_level1": len(level1),
        "n_ai_suppressed": len(suppressed),
        "n_abnormal_enter": sum(1 for e in events if e.get("type") == "abnormal_enter"),
        "detected": None,
        "latency_s": None,
        "detected_reason": None,
        "suppressed_in_window": False,
        "false_alarms": 0,
    }

    if truth.label == "fall":
        if truth.fall_start_ts is None:
            raise ValueError(f"{truth.file}: fall label requires fall_start_ts")
        start = truth.fall_start_ts
        end = truth.fall_end_ts if truth.fall_end_ts is not None else start
        lo, hi = start - pre_tol_s, end + max_latency_s
        in_win = [e for e in level1 if lo <= float(e["ts"]) <= hi]
        rec["false_alarms"] = len(level1) - len(in_win)
        rec["suppressed_in_window"] = any(lo <= float(e["ts"]) <= hi for e in suppressed)
        if in_win:
            first = min(in_win, key=lambda e: float(e["ts"]))
            rec["detected"] = True
            rec["latency_s"] = round(float(first["ts"]) - end, 3)
            rec["detected_reason"] = first.get("reason")
        else:
            rec["detected"] = False
    elif truth.label == "adl":
        rec["false_alarms"] = len(level1)
    else:
        raise ValueError(f"{truth.file}: unknown label {truth.label!r}")
    return rec


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """파일별 결과를 합쳐 요약 지표를 만듭니다."""
    falls = [r for r in records if r["label"] == "fall"]
    adls = [r for r in records if r["label"] == "adl"]
    tp = [r for r in falls if r["detected"]]
    latencies = [r["latency_s"] for r in tp if r["latency_s"] is not None]
    fp_total = sum(r["false_alarms"] for r in records)
    fp_adl = sum(r["false_alarms"] for r in adls)
    adl_hours = sum(r["duration_s"] for r in adls) / 3600.0

    return {
        "n_files": len(records),
        "n_fall": len(falls),
        "n_adl": len(adls),
        "tp": len(tp),
        "fn": len(falls) - len(tp),
        "recall": (len(tp) / len(falls)) if falls else None,
        "false_alarms": fp_total,
        "false_alarms_adl": fp_adl,
        "adl_files_with_alarm": sum(1 for r in adls if r["false_alarms"] > 0),
        "adl_hours": round(adl_hours, 4),
        "fa_per_hour": (fp_adl / adl_hours) if adl_hours > 0 else None,
        "latency_median_s": _median(latencies),
        "latency_p90_s": _quantile(latencies, 0.9),
        "suppressed_true_falls": sum(1 for r in falls if r["suppressed_in_window"]),
        "missed_after_suppression": sum(1 for r in falls if r["suppressed_in_window"] and not r["detected"]),
        "detected_by_escalation": sum(1 for r in tp if r["detected_reason"] == "escalated_after_ai_suppression"),
    }


def summarize_by(records: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[str(r.get(key, "unknown"))].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}
