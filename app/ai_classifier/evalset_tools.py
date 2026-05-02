import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


def validate_rows(rows: Iterable[Dict]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    seen_ids = set()
    seen_paths = set()
    split_session_index = defaultdict(set)

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

        session_id = row.get("session_id")
        if split and session_id:
            split_session_index[session_id].add(split)

    for session_id, splits in split_session_index.items():
        if len(splits) > 1:
            warnings.append(
                f"session_id '{session_id}' appears in multiple splits: {sorted(splits)}"
            )

    return errors, warnings


def summarize_rows(rows: Iterable[Dict]) -> Dict[str, Counter]:
    counters = {
        "split": Counter(),
        "label": Counter(),
        "source": Counter(),
        "scenario": Counter(),
        "split_label": Counter(),
        "split_source": Counter(),
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

    return counters


def print_summary(summary: Dict[str, Counter]) -> None:
    for key in ["split", "label", "source", "scenario", "split_label", "split_source"]:
        print(f"[{key}]")
        for name, count in sorted(summary[key].items()):
            print(f"  {name}: {count}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation-set manifest validator and summarizer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Validate manifest format and split hygiene")
    p_validate.add_argument("--manifest", required=True, help="Path to JSONL manifest")

    p_summary = sub.add_parser("summary", help="Print summary statistics for manifest")
    p_summary.add_argument("--manifest", required=True, help="Path to JSONL manifest")

    args = parser.parse_args()

    rows = load_manifest(args.manifest)

    if args.command == "validate":
        errors, warnings = validate_rows(rows)
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
