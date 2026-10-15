import tempfile
import unittest
from pathlib import Path

from app.eval.evalset_tools import load_manifest, summarize_rows, validate_rows

REPO_ROOT = Path(__file__).resolve().parents[1]


def row(i, split, session, subject, label="fall", file_path=None):
    return {
        "_lineno": i, "sample_id": f"s{i}", "split": split, "source": "custom", "label": label,
        "scenario": "fast_fall", "subject_id": subject, "session_id": session,
        "file_path": file_path or f"clip_{i}.mp4",
    }


class EvalsetToolsTest(unittest.TestCase):
    def test_example_manifest_is_valid(self):
        rows = load_manifest(str(REPO_ROOT / "eval" / "eval_manifest.example.jsonl"))
        errors, _ = validate_rows(rows)
        self.assertEqual(errors, [])

    def test_session_leak_train_to_eval_is_error(self):
        rows = [row(1, "train", "sess", "a"), row(2, "eval_fixed", "sess", "b")]
        errors, _ = validate_rows(rows)
        self.assertTrue(any("session_id 'sess' leaks" in e for e in errors))

    def test_subject_leak_is_error(self):
        rows = [row(1, "val", "s1", "alice"), row(2, "test", "s2", "alice")]
        errors, _ = validate_rows(rows)
        self.assertTrue(any("subject_id 'alice' leaks" in e for e in errors))

    def test_allow_leak_downgrades_to_warning(self):
        rows = [row(1, "train", "sess", "a"), row(2, "eval_fixed", "sess", "a")]
        errors, warnings = validate_rows(rows, allow_leak=True)
        self.assertEqual(errors, [])
        self.assertEqual(len([w for w in warnings if "leaks" in w]), 2)

    def test_train_val_session_share_is_warning_only(self):
        rows = [row(1, "train", "sess", "a"), row(2, "val", "sess", "a")]
        errors, warnings = validate_rows(rows)
        self.assertEqual(errors, [])
        self.assertTrue(any("multiple splits" in w for w in warnings))

    def test_check_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "exists.mp4").write_bytes(b"x")
            rows = [row(1, "train", "s1", "a", file_path="exists.mp4"),
                    row(2, "train", "s1", "a", file_path="missing.mp4")]
            errors, _ = validate_rows(rows, check_files_root=Path(d))
        self.assertEqual(len(errors), 1)
        self.assertIn("missing.mp4", errors[0])

    def test_scenario_label_crosstab(self):
        summary = summarize_rows([row(1, "train", "s", "a"), row(2, "train", "s", "a", label="adl")])
        self.assertEqual(summary["scenario_label"]["fast_fall:fall"], 1)
        self.assertEqual(summary["scenario_label"]["fast_fall:adl"], 1)


if __name__ == "__main__":
    unittest.main()
