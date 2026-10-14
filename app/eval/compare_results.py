"""
두 replay_eval 결과를 비교하고 채택 규칙을 검사합니다.

  python -m app.eval.compare_results eval/baselines/synthetic_v0.json runs/eval/synthetic_abc123.json

채택 규칙 (모드별):
  1. recall이 baseline보다 떨어지면 안 됨
  2. 하드 ADL 시간당 오탐(fa_per_hour)이 늘면 안 됨 (--allow-fa-increase로 완화)
위반이 있으면 종료 코드 1.
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

EPS = 1e-9
COLUMNS = (
    ("recall", "higher"),
    ("fa_per_hour", "lower"),
    ("false_alarms_adl", "lower"),
    ("latency_median_s", "lower"),
    ("latency_p90_s", "lower"),
    ("suppressed_true_falls", "lower"),
    ("missed_after_suppression", "lower"),
)


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return b - a


def check_rules(base: Dict[str, Any], cand: Dict[str, Any], allow_fa_increase: bool = False) -> List[str]:
    """채택 규칙 위반 목록을 반환합니다. (빈 목록이면 통과)"""
    violations = []
    for mode in sorted(set(base["modes"]) & set(cand["modes"])):
        b, c = base["modes"][mode]["summary"], cand["modes"][mode]["summary"]
        if b["recall"] is not None and c["recall"] is not None and c["recall"] < b["recall"] - EPS:
            violations.append(f"{mode}: recall dropped {b['recall']:.3f} -> {c['recall']:.3f}")
        if not allow_fa_increase and b["fa_per_hour"] is not None and c["fa_per_hour"] is not None:
            if c["fa_per_hour"] > b["fa_per_hour"] + EPS:
                violations.append(f"{mode}: fa_per_hour increased {b['fa_per_hour']:.3f} -> {c['fa_per_hour']:.3f}")
    return violations


def diff_rows(base: Dict[str, Any], cand: Dict[str, Any]) -> List[Tuple[str, str, Any, Any, Optional[float]]]:
    rows = []
    for mode in sorted(set(base["modes"]) | set(cand["modes"])):
        b = base["modes"].get(mode, {}).get("summary")
        c = cand["modes"].get(mode, {}).get("summary")
        for col, _ in COLUMNS:
            bv = b.get(col) if b else None
            cv = c.get(col) if c else None
            rows.append((mode, col, bv, cv, _delta(bv, cv)))
    return rows


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare two replay_eval results")
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--allow-fa-increase", action="store_true")
    args = ap.parse_args(argv)

    base, cand = load(args.baseline), load(args.candidate)
    if base.get("dataset") != cand.get("dataset"):
        print("[WARN] datasets differ; comparison may be meaningless", file=sys.stderr)

    print(f"baseline : {base.get('name')} @ {base.get('git_sha')}")
    print(f"candidate: {cand.get('name')} @ {cand.get('git_sha')}\n")
    print(f"{'mode':<12}{'metric':<26}{'baseline':>12}{'candidate':>12}{'delta':>12}")
    for mode, col, bv, cv, d in diff_rows(base, cand):
        mark = ""
        if d is not None and abs(d) > EPS:
            better = dict(COLUMNS)[col]
            mark = "  better" if (d > 0) == (better == "higher") else "  worse"
        print(f"{mode:<12}{col:<26}{_fmt(bv):>12}{_fmt(cv):>12}{_fmt(d):>12}{mark}")

    violations = check_rules(base, cand, args.allow_fa_increase)
    print()
    if violations:
        print("REJECT:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("ACCEPT: no recall drop" + ("" if args.allow_fa_increase else ", no FA/h increase"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
