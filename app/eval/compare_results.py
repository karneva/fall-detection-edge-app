"""
두 replay_eval 결과를 비교하고 채택 규칙을 검사합니다.

  python -m app.eval.compare_results eval/baselines/synthetic_v0.json runs/eval/synthetic_abc123.json

채택 규칙 (모드별):
  1. recall이 baseline보다 떨어지면 안 됨
  2. 하드 ADL 시간당 오탐(fa_per_hour)이 늘면 안 됨 (--allow-fa-increase로 완화)
  3. baseline에서 잡던 낙상 파일을 새로 놓치면 안 됨 (두 결과 모두 파일별 기록이 있을 때)

비교 자체가 무의미한 경우도 위반으로 처리합니다.
  - baseline에 있는 모드가 후보에 없음, 지표가 None으로 바뀜
  - 데이터셋 지문(fingerprint)이나 채점 기준(scoring)이 다름 (--allow-dataset-mismatch로 완화)
  - 분류기 로드 여부(enabled)가 다름
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


def _classifier_enabled(mode_result: Dict[str, Any]) -> Optional[bool]:
    clf = (mode_result.get("meta") or {}).get("classifier")
    return None if clf is None else bool(clf.get("enabled"))


def check_rules(
    base: Dict[str, Any],
    cand: Dict[str, Any],
    allow_fa_increase: bool = False,
    allow_dataset_mismatch: bool = False,
) -> List[str]:
    """채택 규칙 위반 목록을 반환합니다. (빈 목록이면 통과)"""
    violations = []

    if not allow_dataset_mismatch:
        bfp = (base.get("dataset") or {}).get("fingerprint")
        cfp = (cand.get("dataset") or {}).get("fingerprint")
        if bfp is not None and cfp is not None and bfp != cfp:
            violations.append(f"dataset fingerprint differs ({bfp} vs {cfp})")
        if base.get("scoring") is not None and cand.get("scoring") is not None and base["scoring"] != cand["scoring"]:
            violations.append(f"scoring differs ({base['scoring']} vs {cand['scoring']})")

    missing = sorted(set(base["modes"]) - set(cand["modes"]))
    for mode in missing:
        violations.append(f"{mode}: missing from candidate")
    if not set(base["modes"]) & set(cand["modes"]):
        violations.append("no common modes to compare")

    for mode in sorted(set(base["modes"]) & set(cand["modes"])):
        bm, cm = base["modes"][mode], cand["modes"][mode]
        b, c = bm["summary"], cm["summary"]
        for key in ("recall", "fa_per_hour"):
            if b.get(key) is not None and c.get(key) is None:
                violations.append(f"{mode}: {key} missing in candidate (baseline {b[key]:.3f})")
        if b["recall"] is not None and c["recall"] is not None and c["recall"] < b["recall"] - EPS:
            violations.append(f"{mode}: recall dropped {b['recall']:.3f} -> {c['recall']:.3f}")
        if not allow_fa_increase and b["fa_per_hour"] is not None and c["fa_per_hour"] is not None:
            if c["fa_per_hour"] > b["fa_per_hour"] + EPS:
                violations.append(f"{mode}: fa_per_hour increased {b['fa_per_hour']:.3f} -> {c['fa_per_hour']:.3f}")
        be, ce = _classifier_enabled(bm), _classifier_enabled(cm)
        if be is not None and ce is not None and be != ce:
            violations.append(f"{mode}: classifier enabled {be} -> {ce} (environment differs)")
        violations.extend(f"{mode}: {v}" for v in per_file_regressions(bm, cm))
    return violations


def per_file_regressions(base_mode: Dict[str, Any], cand_mode: Dict[str, Any]) -> List[str]:
    """baseline에서 탐지했던 낙상 파일을 후보가 놓쳤는지 확인합니다. (파일 기록이 없으면 확인 불가)"""
    bfiles, cfiles = base_mode.get("files"), cand_mode.get("files")
    if bfiles is None or cfiles is None:
        return []
    cand_by_file = {r["file"]: r for r in cfiles}
    out = []
    for r in bfiles:
        if r["label"] == "fall" and r["detected"]:
            c = cand_by_file.get(r["file"])
            if c is not None and not c["detected"]:
                out.append(f"newly missed fall: {r['file']}")
    return out


def param_diffs(base: Dict[str, Any], cand: Dict[str, Any]) -> List[str]:
    out = []
    for mode in sorted(set(base["modes"]) & set(cand["modes"])):
        bp = (base["modes"][mode].get("meta") or {}).get("params") or {}
        cp = (cand["modes"][mode].get("meta") or {}).get("params") or {}
        for k in sorted(set(bp) | set(cp)):
            if bp.get(k) != cp.get(k):
                out.append(f"{mode}: {k} {bp.get(k)!r} -> {cp.get(k)!r}")
        bh = ((base["modes"][mode].get("meta") or {}).get("classifier") or {}).get("weights_hash")
        ch = ((cand["modes"][mode].get("meta") or {}).get("classifier") or {}).get("weights_hash")
        if bh != ch:
            out.append(f"{mode}: weights {bh} -> {ch}")
    return out


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
    ap.add_argument("--allow-dataset-mismatch", action="store_true",
                    help="compare even if dataset fingerprint or scoring differ")
    args = ap.parse_args(argv)

    base, cand = load(args.baseline), load(args.candidate)

    print(f"baseline : {base.get('name')} @ {base.get('git_sha')}")
    print(f"candidate: {cand.get('name')} @ {cand.get('git_sha')}\n")
    print(f"{'mode':<12}{'metric':<26}{'baseline':>12}{'candidate':>12}{'delta':>12}")
    for mode, col, bv, cv, d in diff_rows(base, cand):
        mark = ""
        if d is not None and abs(d) > EPS:
            better = dict(COLUMNS)[col]
            mark = "  better" if (d > 0) == (better == "higher") else "  worse"
        print(f"{mode:<12}{col:<26}{_fmt(bv):>12}{_fmt(cv):>12}{_fmt(d):>12}{mark}")

    diffs = param_diffs(base, cand)
    if diffs:
        print("\nparameter / weights differences:")
        for d in diffs:
            print(f"  - {d}")
    if any(m.get("files") is None for r in (base, cand) for m in r["modes"].values()):
        print("\n[WARN] per-file records missing (--no-files); newly missed falls cannot be checked", file=sys.stderr)

    violations = check_rules(base, cand, args.allow_fa_increase, args.allow_dataset_mismatch)
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
