import copy
import json
import tempfile
import unittest
from pathlib import Path

from app.eval import compare_results, replay_eval
from app.eval.event_metrics import Truth, score_file, summarize
from app.eval.synth import make_scenario


def ev(t, ts, reason=None):
    return {"type": t, "ts": ts, "reason": reason}


class ScoreFileTest(unittest.TestCase):
    def test_fall_detected_with_latency(self):
        truth = Truth("a", "fall", fall_start_ts=100.0, fall_end_ts=101.0)
        rec = score_file(truth, [ev("abnormal_enter", 100.5), ev("level1", 111.0, "still_for_sustain_s")], 60.0)
        self.assertTrue(rec["detected"])
        self.assertAlmostEqual(rec["latency_s"], 10.0)
        self.assertEqual(rec["false_alarms"], 0)

    def test_fall_window_bounds(self):
        truth = Truth("a", "fall", fall_start_ts=100.0, fall_end_ts=101.0)
        early = score_file(truth, [ev("level1", 97.9)], 300.0, pre_tol_s=2.0)
        self.assertFalse(early["detected"])
        self.assertEqual(early["false_alarms"], 1)
        late = score_file(truth, [ev("level1", 101.0 + 120.1)], 300.0, max_latency_s=120.0)
        self.assertFalse(late["detected"])
        edge = score_file(truth, [ev("level1", 98.0)], 300.0, pre_tol_s=2.0)
        self.assertTrue(edge["detected"])

    def test_suppressed_then_escalated(self):
        truth = Truth("a", "fall", fall_start_ts=100.0, fall_end_ts=101.0)
        rec = score_file(truth, [ev("ai_suppressed", 111.0),
                                 ev("level1", 171.0, "escalated_after_ai_suppression")], 200.0)
        self.assertTrue(rec["detected"])
        self.assertTrue(rec["suppressed_in_window"])
        s = summarize([rec])
        self.assertEqual(s["detected_by_escalation"], 1)
        self.assertEqual(s["missed_after_suppression"], 0)

    def test_adl_false_alarms_per_hour(self):
        recs = [
            score_file(Truth("a", "adl"), [ev("level1", 10.0), ev("level1", 50.0)], 1800.0),
            score_file(Truth("b", "adl"), [], 1800.0),
            score_file(Truth("c", "fall", fall_start_ts=0.0), [], 10.0),
        ]
        s = summarize(recs)
        self.assertEqual(s["false_alarms_adl"], 2)
        self.assertAlmostEqual(s["adl_hours"], 1.0)
        self.assertAlmostEqual(s["fa_per_hour"], 2.0)
        self.assertEqual(s["recall"], 0.0)
        self.assertEqual(s["adl_files_with_alarm"], 1)

    def test_fall_requires_start(self):
        with self.assertRaises(ValueError):
            score_file(Truth("a", "fall"), [], 1.0)

    def test_empty_summary(self):
        s = summarize([])
        self.assertIsNone(s["recall"])
        self.assertIsNone(s["fa_per_hour"])


class ReplayEvalTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_obs(self, name, obs):
        with open(self.dir / name, "w", encoding="utf-8") as f:
            for o in obs:
                f.write(json.dumps(o) + "\n")

    def test_labels_roundtrip_relative_times(self):
        fall_obs, fall_t = make_scenario("fast_fall", lie_s=20.0, t0=5000.0)
        adl_obs, _ = make_scenario("stand", t0=9000.0)
        self.write_obs("fall.jsonl", fall_obs)
        self.write_obs("adl.jsonl", adl_obs)
        rows = [
            {"file": "fall.jsonl", "label": "fall", "fall_start_s": 3.0, "fall_end_s": 3.5, "scenario": "fast_fall"},
            {"file": "adl.jsonl", "label": "adl", "scenario": "stand"},
        ]
        labels = self.dir / "labels.jsonl"
        labels.write_text("# comment\n" + "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        items = replay_eval.load_labeled_files(labels)
        self.assertAlmostEqual(items[0][0].fall_start_ts, fall_t["fall_start_ts"])
        res = replay_eval.evaluate(items, ["rules_only", "const_adl"], {})
        self.assertEqual(res["rules_only"]["summary"]["recall"], 1.0)
        self.assertEqual(res["rules_only"]["summary"]["false_alarms_adl"], 0)
        # 20초만 누워 있으므로 const_adl에서는 승격 전에 끝남 -> 억제된 실제 낙상으로 기록
        s = res["const_adl"]["summary"]
        self.assertEqual(s["suppressed_true_falls"], 1)
        self.assertEqual(s["missed_after_suppression"], 1)
        self.assertEqual(res["const_adl"]["files"][0]["ai"][0]["decision"], "adl")

    def test_fall_without_time_rejected(self):
        obs, _ = make_scenario("stand")
        self.write_obs("x.jsonl", obs[:10])
        labels = self.dir / "labels.jsonl"
        labels.write_text(json.dumps({"file": "x.jsonl", "label": "fall"}) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            replay_eval.load_labeled_files(labels)

    def test_param_override_types(self):
        o = replay_eval._parse_param_overrides(["ai_escalate_s=30", "warmup_frames=5",
                                                "ai_escalate_requires_drop=0", "ai_mode=shadow"])
        self.assertEqual(o, {"ai_escalate_s": 30.0, "warmup_frames": 5,
                             "ai_escalate_requires_drop": False, "ai_mode": "shadow"})
        with self.assertRaises(SystemExit):
            replay_eval._parse_param_overrides(["nope=1"])

    def test_disabled_classifier_rejected_unless_allowed(self):
        from unittest import mock

        items = replay_eval.synthetic_files()[:1]
        off = {"enabled": False, "load_error": "no weights"}
        with mock.patch.object(replay_eval, "_classifier_status", return_value=off):
            with self.assertRaises(SystemExit):
                replay_eval.evaluate(items, ["rules_ai"], {})
            res = replay_eval.evaluate(items, ["shadow"], {}, allow_disabled_ai=True)
        self.assertFalse(res["shadow"]["meta"]["classifier"]["enabled"])

    def test_cli_synthetic_writes_result(self):
        out = self.dir / "r.json"
        rc = replay_eval.main(["--synthetic", "--modes", "rules_only", "--out", str(out), "--no-files"])
        self.assertEqual(rc, 0)
        doc = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema_version"], replay_eval.RESULT_SCHEMA_VERSION)
        self.assertIn("rules_only", doc["modes"])
        self.assertNotIn("files", doc["modes"]["rules_only"])


class CompareResultsTest(unittest.TestCase):
    BASE = {"name": "b", "dataset": {}, "modes": {"rules_only": {"summary": {
        "recall": 0.9, "fa_per_hour": 2.0, "false_alarms_adl": 4, "latency_median_s": 10.0,
        "latency_p90_s": 12.0, "suppressed_true_falls": 0, "missed_after_suppression": 0}}}}

    def cand(self, **kw):
        c = copy.deepcopy(self.BASE)
        c["modes"]["rules_only"]["summary"].update(kw)
        return c

    def test_accept_when_equal_or_better(self):
        self.assertEqual(compare_results.check_rules(self.BASE, self.cand(recall=0.95, fa_per_hour=1.0)), [])

    def test_reject_recall_drop(self):
        v = compare_results.check_rules(self.BASE, self.cand(recall=0.8))
        self.assertEqual(len(v), 1)
        self.assertIn("recall", v[0])

    def test_reject_fa_increase_unless_allowed(self):
        c = self.cand(fa_per_hour=3.0)
        self.assertEqual(len(compare_results.check_rules(self.BASE, c)), 1)
        self.assertEqual(compare_results.check_rules(self.BASE, c, allow_fa_increase=True), [])

    def test_reject_missing_mode_and_no_overlap(self):
        c = {"name": "c", "dataset": {}, "modes": {"shadow": copy.deepcopy(self.BASE["modes"]["rules_only"])}}
        v = compare_results.check_rules(self.BASE, c)
        self.assertTrue(any("missing from candidate" in x for x in v))
        self.assertTrue(any("no common modes" in x for x in v))

    def test_reject_metric_becoming_none(self):
        v = compare_results.check_rules(self.BASE, self.cand(recall=None, fa_per_hour=None))
        self.assertEqual(len(v), 2)

    def test_reject_dataset_or_scoring_mismatch(self):
        b = copy.deepcopy(self.BASE)
        b.update(dataset={"fingerprint": "aaa"}, scoring={"pre_tol_s": 2.0, "max_latency_s": 120.0})
        c = copy.deepcopy(b)
        c.update(dataset={"fingerprint": "bbb"}, scoring={"pre_tol_s": 2.0, "max_latency_s": 300.0})
        self.assertEqual(len(compare_results.check_rules(b, c)), 2)
        self.assertEqual(compare_results.check_rules(b, c, allow_dataset_mismatch=True), [])

    def test_reject_newly_missed_fall_even_if_recall_same(self):
        b = copy.deepcopy(self.BASE)
        b["modes"]["rules_only"]["files"] = [
            {"file": "A", "label": "fall", "detected": True},
            {"file": "B", "label": "fall", "detected": False},
        ]
        c = copy.deepcopy(b)
        c["modes"]["rules_only"]["files"] = [
            {"file": "A", "label": "fall", "detected": False},
            {"file": "B", "label": "fall", "detected": True},
        ]
        v = compare_results.check_rules(b, c)
        self.assertEqual(v, ["rules_only: newly missed fall: A"])

    def test_reject_classifier_enabled_mismatch(self):
        b = copy.deepcopy(self.BASE)
        b["modes"]["rules_only"]["meta"] = {"classifier": {"enabled": True, "weights_hash": "x"}}
        c = copy.deepcopy(b)
        c["modes"]["rules_only"]["meta"] = {"classifier": {"enabled": False}}
        self.assertEqual(len(compare_results.check_rules(b, c)), 1)
        self.assertTrue(compare_results.param_diffs(b, c))


class SyntheticBaselineTest(unittest.TestCase):
    def test_committed_baseline_still_holds(self):
        """커밋된 합성 baseline 대비 recall 하락/FA 증가가 없어야 함 (엔진 회귀 감지)."""
        base_path = Path(__file__).resolve().parents[1] / "eval" / "baselines" / "synthetic_v0.json"
        if not base_path.exists():
            self.skipTest("baseline not generated")
        base = json.loads(base_path.read_text(encoding="utf-8"))
        modes = list(base["modes"])
        # 환경변수(AI_ESCALATE_S 등)에 따라 기본값이 바뀌지 않도록 baseline 당시 파라미터로 고정
        params = dict(base["modes"][modes[0]]["meta"]["params"])
        params.pop("ai_mode")
        cand = {
            "dataset": {"fingerprint": replay_eval.dataset_fingerprint_synthetic()},
            "scoring": base["scoring"],
            "modes": replay_eval.evaluate(replay_eval.synthetic_files(), modes, params),
        }
        self.assertEqual(compare_results.check_rules(base, cand), [])


if __name__ == "__main__":
    unittest.main()
