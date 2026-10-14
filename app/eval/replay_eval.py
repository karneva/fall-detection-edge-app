"""
기록된 obs JSONL을 Level1Engine에 오프라인으로 재생해 이벤트 단위 성능을 평가합니다.

실제 녹화:
  python -m app.eval.replay_eval --labels eval/labels.jsonl --modes rules_only,rules_ai

합성 시나리오 (데이터 없이 엔진 동작 확인):
  python -m app.eval.replay_eval --synthetic --out eval/baselines/synthetic_v0.json

모드:
  rules_only   AI 미사용 (AI_MODE=off)
  rules_ai     1D CNN veto (가중치 필요, 없으면 fail-open이라 rules_only와 같아짐)
  shadow       1D CNN 판정만 기록, 억제 안 함 (p_fall 분포 수집용)
  const_fall   항상 p_fall=0.9를 내는 가짜 AI
  const_adl    항상 p_fall=0.05를 내는 가짜 AI (AI가 전부 틀릴 때의 최악 recall 확인용)

라벨 파일(JSONL) 한 줄 예:
  {"file": "obs_20261014_101010.jsonl", "label": "fall", "fall_start_s": 12.3, "fall_end_s": 13.0,
   "scenario": "fast_fall", "source": "custom", "subject_id": "s01", "session_id": "sess01"}
  - file은 라벨 파일 기준 상대경로
  - 시각은 파일 첫 프레임 기준 초(fall_start_s) 또는 절대 ts(fall_start_ts) 중 하나
"""
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from app.eval.event_metrics import MAX_LATENCY_S, PRE_TOL_S, Truth, score_file, summarize, summarize_by

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA_VERSION = 1

ALL_MODES = ("rules_only", "rules_ai", "shadow", "const_fall", "const_adl")
SYNTH_FPS = (10.0, 15.0, 30.0)
SYNTH_SEEDS = (0, 1)
SYNTH_LIE_S = 90.0


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------
def load_obs_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obs = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} invalid JSON: {e}") from e
            if "ts" not in obs:
                raise ValueError(f"{path}:{lineno} obs without 'ts'")
            out.append(obs)
    out.sort(key=lambda o: float(o["ts"]))
    return out


def _abs_ts(row: Dict[str, Any], key: str, t0: float) -> Optional[float]:
    if row.get(f"{key}_ts") is not None:
        return float(row[f"{key}_ts"])
    if row.get(f"{key}_s") is not None:
        return t0 + float(row[f"{key}_s"])
    return None


def _sha8_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def dataset_fingerprint_labels(labels_path: Path) -> str:
    """라벨 파일과 그것이 가리키는 모든 obs 파일 내용의 해시. 내용이 바뀌면 결과를 비교하면 안 됨."""
    h = hashlib.sha256(_sha8_file(labels_path).encode())
    base = labels_path.parent
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                h.update(_sha8_file((base / json.loads(line)["file"]).resolve()).encode())
    return h.hexdigest()[:12]


def dataset_fingerprint_synthetic() -> str:
    """합성 생성기 코드와 생성 설정의 해시."""
    h = hashlib.sha256(_sha8_file(Path(__file__).with_name("synth.py")).encode())
    h.update(json.dumps([SYNTH_FPS, SYNTH_SEEDS, SYNTH_LIE_S]).encode())
    return h.hexdigest()[:12]


def load_labeled_files(labels_path: Path) -> List[Tuple[Truth, List[Dict[str, Any]]]]:
    """라벨 파일을 읽어 (정답, obs 목록) 목록을 반환합니다."""
    items = []
    base = labels_path.parent
    with open(labels_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            obs_path = (base / row["file"]).resolve()
            obs = load_obs_jsonl(obs_path)
            if not obs:
                raise ValueError(f"{labels_path}:{lineno} {row['file']} has no frames")
            t0 = float(obs[0]["ts"])
            truth = Truth(
                file=row["file"],
                label=row["label"],
                fall_start_ts=_abs_ts(row, "fall_start", t0),
                fall_end_ts=_abs_ts(row, "fall_end", t0),
                scenario=row.get("scenario", "unknown"),
                source=row.get("source", "unknown"),
                subject_id=row.get("subject_id"),
                session_id=row.get("session_id"),
            )
            if truth.label == "fall" and truth.fall_start_ts is None:
                raise ValueError(f"{labels_path}:{lineno} fall label needs fall_start_s or fall_start_ts")
            items.append((truth, obs))
    return items


def synthetic_files() -> List[Tuple[Truth, List[Dict[str, Any]]]]:
    from app.eval.synth import EVAL_SET, make_scenario

    items = []
    for name, label in EVAL_SET:
        for fps in SYNTH_FPS:
            for seed in SYNTH_SEEDS:
                obs, t = make_scenario(name, fps=fps, seed=seed, lie_s=SYNTH_LIE_S)
                truth = Truth(
                    file=f"synth/{name}_fps{int(fps)}_s{seed}",
                    label=label,
                    fall_start_ts=t["fall_start_ts"] if label == "fall" else None,
                    fall_end_ts=t["fall_end_ts"] if label == "fall" else None,
                    scenario=name,
                    source=f"synthetic_fps{int(fps)}",
                )
                items.append((truth, obs))
    return items


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
class _ConstVerifier:
    def __init__(self, p_fall: float):
        self.p_fall = p_fall
        self.model_id = f"const_{p_fall}"

    def __call__(self, seq):
        return self.p_fall


def _parse_param_overrides(pairs: Iterable[str]) -> Dict[str, Any]:
    from app.core.level1 import Level1Params

    fields = {f.name: f for f in dataclasses.fields(Level1Params)}
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        if key not in fields:
            raise SystemExit(f"unknown Level1Params field: {key}")
        default = fields[key].default
        if isinstance(default, bool):
            out[key] = raw.strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(default, int):
            out[key] = int(raw)
        elif isinstance(default, float):
            out[key] = float(raw)
        else:
            out[key] = raw
    return out


def _classifier_status() -> Dict[str, Any]:
    """1D CNN 분류기 상태. numpy/torch가 없어도 예외 없이 disabled로 보고합니다."""
    try:
        from app.ai_classifier.inference import get_classifier

        return get_classifier().status()
    except Exception as e:
        return {"enabled": False, "load_error": f"import failed: {e!r}"}


def build_mode(mode: str, overrides: Dict[str, Any],
               allow_disabled_ai: bool = False) -> Tuple[Callable[[], Any], Dict[str, Any]]:
    """모드 이름으로 (엔진 생성 함수, 모드 메타)를 만듭니다."""
    from app.core.level1 import Level1Engine, Level1Params

    verifier = None
    meta: Dict[str, Any] = {}
    if mode == "rules_only":
        ai_mode = "off"
    elif mode in ("rules_ai", "shadow"):
        ai_mode = "veto" if mode == "rules_ai" else "shadow"
        status = _classifier_status()
        meta["classifier"] = status
        if not status.get("enabled"):
            msg = f"{mode}: classifier disabled ({status.get('load_error')}); AI is fail-open, same as rules_only"
            if not allow_disabled_ai:
                raise SystemExit(f"[ERROR] {msg}. Pass --allow-disabled-ai to run anyway.")
            print(f"[WARN] {msg}", file=sys.stderr)
    elif mode == "const_fall":
        ai_mode, verifier = "veto", _ConstVerifier(0.9)
    elif mode == "const_adl":
        ai_mode, verifier = "veto", _ConstVerifier(0.05)
    else:
        raise SystemExit(f"unknown mode: {mode} (choose from {', '.join(ALL_MODES)})")

    kwargs = dict(overrides)
    kwargs["ai_mode"] = ai_mode
    params = Level1Params(**kwargs)
    meta["params"] = dataclasses.asdict(params)
    return (lambda: Level1Engine(params, verifier=verifier)), meta


def replay(make_engine: Callable[[], Any], obs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    engine = make_engine()
    events = []
    for o in obs:
        ev = engine.step(o)
        if ev is not None:
            events.append(ev)
    return events


def _ai_records(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for e in events:
        ai = (e.get("values") or {}).get("ai")
        if isinstance(ai, dict) and ai.get("decision") not in (None, "off"):
            out.append({
                "event": e["type"],
                "reason": e.get("reason"),
                "decision": ai.get("decision"),
                "p_fall": ai.get("p_fall"),
                "window_offset_from_abnormal_s": ai.get("window_offset_from_abnormal_s"),
            })
    return out


def evaluate(
    items: List[Tuple[Truth, List[Dict[str, Any]]]],
    modes: List[str],
    overrides: Dict[str, Any],
    pre_tol_s: float = PRE_TOL_S,
    max_latency_s: float = MAX_LATENCY_S,
    allow_disabled_ai: bool = False,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for mode in modes:
        make_engine, meta = build_mode(mode, overrides, allow_disabled_ai)
        records = []
        for truth, obs in items:
            events = replay(make_engine, obs)
            duration = float(obs[-1]["ts"]) - float(obs[0]["ts"]) if obs else 0.0
            rec = score_file(truth, events, duration, pre_tol_s=pre_tol_s, max_latency_s=max_latency_s)
            rec["ai"] = _ai_records(events)
            records.append(rec)
        results[mode] = {
            "meta": meta,
            "summary": summarize(records),
            "by_scenario": summarize_by(records, "scenario"),
            "by_source": summarize_by(records, "source"),
            "files": records,
        }
    return results


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_table(results: Dict[str, Any]) -> None:
    cols = ("recall", "tp", "fn", "false_alarms_adl", "fa_per_hour", "latency_median_s",
            "suppressed_true_falls", "missed_after_suppression", "detected_by_escalation")
    print(f"{'mode':<12}" + "".join(f"{c:>26}" for c in cols))
    for mode, r in results.items():
        s = r["summary"]
        print(f"{mode:<12}" + "".join(f"{_fmt(s[c]):>26}" for c in cols))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline event-level evaluation of Level1Engine")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels", help="labels JSONL (one recorded obs file per line)")
    src.add_argument("--synthetic", action="store_true", help="use built-in synthetic scenarios")
    ap.add_argument("--modes", default=None,
                    help=f"comma separated, from {','.join(ALL_MODES)} "
                         "(default: rules_only,rules_ai / synthetic: rules_only,const_fall,const_adl)")
    ap.add_argument("--param", action="append", default=[], help="Level1Params override key=value")
    ap.add_argument("--pre-tol-s", type=float, default=PRE_TOL_S)
    ap.add_argument("--max-latency-s", type=float, default=MAX_LATENCY_S)
    ap.add_argument("--name", default=None, help="result name (default: labels file stem or 'synthetic')")
    ap.add_argument("--out", default=None, help="output JSON path (default: runs/eval/<name>_<gitsha>.json)")
    ap.add_argument("--no-files", action="store_true",
                    help="omit per-file records (compare_results then cannot check per-file regressions)")
    ap.add_argument("--allow-disabled-ai", action="store_true",
                    help="run rules_ai/shadow even if the classifier failed to load")
    args = ap.parse_args(argv)

    if args.synthetic:
        items = synthetic_files()
        default_modes = "rules_only,const_fall,const_adl"
        name = args.name or "synthetic"
        dataset = {"type": "synthetic", "fps": list(SYNTH_FPS), "seeds": list(SYNTH_SEEDS), "lie_s": SYNTH_LIE_S,
                   "fingerprint": dataset_fingerprint_synthetic()}
    else:
        labels_path = Path(args.labels)
        items = load_labeled_files(labels_path)
        default_modes = "rules_only,rules_ai"
        name = args.name or labels_path.stem
        dataset = {"type": "labels", "labels": str(labels_path),
                   "fingerprint": dataset_fingerprint_labels(labels_path)}

    modes = [m.strip() for m in (args.modes or default_modes).split(",") if m.strip()]
    overrides = _parse_param_overrides(args.param)

    results = evaluate(items, modes, overrides, args.pre_tol_s, args.max_latency_s, args.allow_disabled_ai)
    if args.no_files:
        for r in results.values():
            r.pop("files", None)

    sha = _git_sha()
    doc = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "name": name,
        "git_sha": sha,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset,
        "scoring": {"pre_tol_s": args.pre_tol_s, "max_latency_s": args.max_latency_s},
        "param_overrides": overrides,
        "modes": results,
    }
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / "eval" / f"{name}_{sha or 'nogit'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print_table(results)
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "ERROR"))
    sys.exit(main())
