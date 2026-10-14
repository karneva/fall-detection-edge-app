import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REQUIRED_FIELDS = [
    "sample_id",
    "split",
    "source",
    "label",
    "scenario",
    "subject_id",
    "session_id",
    "file_path",
]

ALLOWED_SPLITS = {"train", "val", "test", "eval_fixed"}
ALLOWED_LABELS = {"fall", "adl"}

# 학습에 쓰는 split과 평가에 쓰는 split. 둘 사이에 세션/사람이 겹치면 누수
TRAIN_SPLITS = {"train", "val"}
HELDOUT_SPLITS = {"test", "eval_fixed"}


def load_manifest(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno} each line must be a JSON object")
            row["_lineno"] = lineno
            rows.append(row)
    return rows


def validate_rows(
    rows: Iterable[Dict],
    allow_leak: bool = False,
    check_files_root: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """
    manifest 행을 검증합니다.

    - 학습 split(train/val)과 평가 split(test/eval_fixed)에 같은 session_id나 subject_id가 있으면 error
      (allow_leak=True면 warning으로 완화)
    - train/val 사이 세션 공유는 warning
    - check_files_root가 주어지면 file_path 존재 여부를 확인 (상대경로는 이 디렉터리 기준, CLI 기본값은 현재 디렉터리)
    """
    errors: List[str] = []
    warnings: List[str] = []

    seen_ids = set()
    seen_paths = set()
    split_session_index = defaultdict(set)
    split_subject_index = defaultdict(set)

    for row in rows:
        where = f"line {row.get('_lineno', '?')}"

        for field in REQUIRED_FIELDS:
            if field not in row or row[field] in (None, ""):
                errors.append(f"{where}: missing required field '{field}'")

        split = row.get("split")
        if split and split not in ALLOWED_SPLITS:
            errors.append(f"{where}: invalid split '{split}'")

        label = row.get("label")
        if label and label not in ALLOWED_LABELS:
            errors.append(f"{where}: invalid label '{label}'")

        sample_id = row.get("sample_id")
        if sample_id:
            if sample_id in seen_ids:
                errors.append(f"{where}: duplicate sample_id '{sample_id}'")
            seen_ids.add(sample_id)

        file_path = row.get("file_path")
        if file_path:
            if file_path in seen_paths:
                warnings.append(f"{where}: duplicate file_path '{file_path}'")
            seen_paths.add(file_path)
            if check_files_root is not None:
                fp = Path(file_path)
                if not fp.is_absolute():
                    fp = check_files_root / fp
                if not fp.exists():
                    errors.append(f"{where}: file not found '{file_path}'")

        session_id = row.get("session_id")
        if split and session_id:
            split_session_index[session_id].add(split)
        subject_id = row.get("subject_id")
        if split and subject_id:
            split_subject_index[subject_id].add(split)

    for kind, index in (("session_id", split_session_index), ("subject_id", split_subject_index)):
        for key, splits in sorted(index.items()):
            if splits & TRAIN_SPLITS and splits & HELDOUT_SPLITS:
                msg = f"{kind} '{key}' leaks between training and held-out splits: {sorted(splits)}"
                (warnings if allow_leak else errors).append(msg)
            elif kind == "session_id" and len(splits) > 1:
                warnings.append(f"session_id '{key}' appears in multiple splits: {sorted(splits)}")

    return errors, warnings


def summarize_rows(rows: Iterable[Dict]) -> Dict[str, Counter]:
    counters = {
        "split": Counter(),
        "label": Counter(),
        "source": Counter(),
        "scenario": Counter(),
        "split_label": Counter(),
        "split_source": Counter(),
        "scenario_label": Counter(),
    }

    for row in rows:
        split = row.get("split", "unknown")
        label = row.get("label", "unknown")
        source = row.get("source", "unknown")
        scenario = row.get("scenario", "unknown")

        counters["split"][split] += 1
        counters["label"][label] += 1
        counters["source"][source] += 1
        counters["scenario"][scenario] += 1
        counters["split_label"][f"{split}:{label}"] += 1
        counters["split_source"][f"{split}:{source}"] += 1
        counters["scenario_label"][f"{scenario}:{label}"] += 1

    return counters


def print_summary(summary: Dict[str, Counter]) -> None:
    for key in ["split", "label", "source", "scenario", "split_label", "split_source", "scenario_label"]:
        print(f"[{key}]")
        for name, count in sorted(summary[key].items()):
            print(f"  {name}: {count}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation-set manifest validator and summarizer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Validate manifest format and split hygiene")
    p_validate.add_argument("--manifest", required=True, help="Path to JSONL manifest")
    p_validate.add_argument("--allow-leak", action="store_true",
                            help="Downgrade train/held-out session/subject overlap to a warning")
    p_validate.add_argument("--check-files", action="store_true",
                            help="Check that file_path exists (relative paths resolve against --root)")
    p_validate.add_argument("--root", default=".",
                            help="Base directory for relative file_path (default: current directory, i.e. repo root)")

    p_summary = sub.add_parser("summary", help="Print summary statistics for manifest")
    p_summary.add_argument("--manifest", required=True, help="Path to JSONL manifest")

    args = parser.parse_args()

    rows = load_manifest(args.manifest)

    if args.command == "validate":
        root = Path(args.root).resolve() if args.check_files else None
        errors, warnings = validate_rows(rows, allow_leak=args.allow_leak, check_files_root=root)
        if warnings:
            print("Warnings:")
            for item in warnings:
                print(f"  - {item}")
            print()
        if errors:
            print("Errors:")
            for item in errors:
                print(f"  - {item}")
            raise SystemExit(1)
        print(f"OK: manifest is valid ({len(rows)} rows)")
        return

    if args.command == "summary":
        summary = summarize_rows(rows)
        print_summary(summary)
        print(f"total_rows: {len(rows)}")


if __name__ == "__main__":
    main()
