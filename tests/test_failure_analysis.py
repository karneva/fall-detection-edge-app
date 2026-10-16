"""
실패 원인 분석(training/experiments/failure_analysis) 테스트

합성 낙상 시나리오(app/eval/synth.py)를 클립 형식으로 바꿔 재생 기록과 원인 분류를 확인합니다.
numpy가 없으면 skip.
"""
import importlib.util
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None

if HAS_NUMPY:
    import numpy as np

    from app.eval.synth import make_scenario
    from training.datasets.clipfmt import from_obs_records
    from training.experiments import failure_analysis as FA
    from training.experiments import replay


def _clip(name="fast_fall", lie_s=15.0):
    obs, truth = make_scenario(name, lie_s=lie_s)
    clip = from_obs_records(obs, dataset="synth", clip_id=f"synth/{name}")
    t0 = float(obs[0]["ts"])
    fs = float(truth["fall_start_ts"]) - t0
    row = {"clip_id": f"synth/{name}", "dataset": "synth", "label": "fall",
           "fall_start_s": fs, "fall_end_s": fs + 1.0,
           "segments": [{"cls": "walk", "start_s": 0.0, "end_s": fs},
                        {"cls": "fall", "start_s": fs, "end_s": fs + 1.0},
                        {"cls": "fallen", "start_s": fs + 1.0, "end_s": float(clip["ts"][-1])}]}
    return row, clip


def _blank(clip, lo, hi):
    """[lo, hi] 클립 시각의 프레임에서 사람을 지움 (포즈 미검출 흉내)."""
    c = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in clip.items()}
    m = (c["ts"] >= lo) & (c["ts"] <= hi)
    c["present"][m] = False
    c["kps"][m] = 0.0
    c["bbox"][m] = 0.0
    return c


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class FailureAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.p = replay.study_params("off")

    def test_trace_matches_replay_events(self):
        row, clip = _clip()
        obs = replay.clip_to_obs(clip)
        events, frames = FA.trace_clip(obs, self.p)
        ref, _ = replay.run_clip(replay.clip_to_obs(clip), self.p)
        self.assertEqual([(e["type"], e["ts"]) for e in events], [(e["type"], e["ts"]) for e in ref])
        self.assertEqual(len(frames), len(obs))
        self.assertTrue(any(f["tail"] for f in frames))
        enter = [f for f in frames if f["state"] == "ABNORMAL"]
        self.assertTrue(enter)

    def test_detected_fall_is_timely(self):
        row, clip = _clip()
        rec = FA.analyze_clip(row, clip, self.p)
        self.assertEqual(rec["kind"], "timely")
        self.assertTrue(rec["outcome"]["alarm"])

    def test_pose_lost_during_fall(self):
        row, clip = _clip()
        fs = row["fall_start_s"]
        rec = FA.analyze_clip(row, _blank(clip, fs - 0.5, 1e9), self.p)
        self.assertEqual(rec["kind"], "missed")
        self.assertEqual(rec["analysis"]["cause"], "pose_lost_during_fall")
        self.assertEqual(rec["analysis"]["group"], "pose")
        self.assertFalse(rec["analysis"]["entered"])

    def test_lost_after_entry(self):
        row, clip = _clip()
        obs = replay.clip_to_obs(clip)
        events, _ = FA.trace_clip(obs, self.p)
        enter = [e for e in events if e["type"] == "abnormal_enter"][0]["ts"] - replay.DEFAULT_T0
        rec = FA.analyze_clip(row, _blank(clip, enter + 1.0, 1e9), self.p)
        self.assertEqual(rec["kind"], "missed")
        self.assertTrue(rec["analysis"]["entered"])
        self.assertEqual(rec["analysis"]["cause"], "lost_after_fall")

    def test_episodes_and_segments(self):
        ev = [{"type": "abnormal_enter", "ts": 1001.0, "values": {"reason": "confirm_enter hits=2 acc=0.40"}},
              {"type": "abnormal_exit", "ts": 1004.0},
              {"type": "abnormal_enter", "ts": 1006.0, "values": {"reason": "low_posture_sustained"}},
              {"type": "level1", "ts": 1016.0}]
        eps = FA.episodes(ev)
        self.assertEqual([(e["enter_s"], e["end"]) for e in eps], [(1.0, "recovered"), (6.0, "level1")])
        self.assertEqual(FA._reason_kind(eps[0]["reason"]), "confirm_enter")
        row = {"segments": [{"cls": "walk", "start_s": 0.0, "end_s": 2.0},
                            {"cls": "sit_down", "start_s": 3.0, "end_s": 4.0}]}
        self.assertEqual(FA.segment_class_at(row, 1.0), "walk")
        self.assertEqual(FA.segment_class_at(row, 2.5), "walk")     # 구간 사이 -> 앞 구간
        self.assertEqual(FA.segment_class_at(row, 9.0), "sit_down")
        self.assertIsNone(FA.segment_class_at({"segments": []}, 1.0))

    def test_entry_window_order(self):
        def f(t, **kw):
            base = {"t": t, "tail": False, "lost": False, "state": "NORMAL", "streak": 30, "aspect": 3.0,
                    "partial": False, "hits": {"drop": False, "posture": False, "shoulder": False, "height": False},
                    "n_hits": 0, "enter_acc": 0.0}
            base.update(kw)
            return base
        frames = [f(0.1 * i) for i in range(30)]
        self.assertEqual(FA.classify_entry_window(frames, 5.0, 6.0, self.p)[0], "label_outside_clip")
        self.assertEqual(FA.classify_entry_window(frames, 0.0, 2.9, self.p)[0], "lying_bbox_not_flat")
        flicker = [f(0.1 * i, streak=3) for i in range(30)]
        self.assertEqual(FA.classify_entry_window(flicker, 0.0, 2.9, self.p)[0], "detection_flicker")
        brief = [f(0.1 * i, n_hits=2 if i == 5 else 0) for i in range(30)]
        self.assertEqual(FA.classify_entry_window(brief, 0.0, 2.9, self.p)[0], "signals_too_brief")
        flat = [f(0.1 * i, aspect=0.9) for i in range(30)]
        self.assertEqual(FA.classify_entry_window(flat, 0.0, 2.9, self.p)[0], "weak_motion_signals")

    def test_summary_and_markdown(self):
        row, clip = _clip()
        fs = row["fall_start_s"]
        recs = [FA.analyze_clip(row, clip, self.p), FA.analyze_clip(row, _blank(clip, fs - 0.5, 1e9), self.p)]
        s = FA.summarize_records(recs)
        self.assertEqual(s["kinds"], {"timely": 1, "missed": 1})
        self.assertEqual(s["missed_group"], {"pose": 1})
        md = FA.render_markdown(s)
        self.assertIn("pose_lost_during_fall", md)
        self.assertTrue(set(FA.CAUSE_GROUP) <= set(FA.CAUSE_DESC))

    def test_sheet_times_clamped(self):
        rec = {"label": "fall", "fall_start_s": 0.5, "fall_end_s": 1.5, "clip_end_s": 2.0, "outcome": {}}
        ts = [t for _, t in FA.sheet_times(rec)]
        self.assertEqual(len(ts), 6)
        self.assertTrue(all(0.0 <= t <= 2.0 for t in ts))
        adl = {"label": "adl", "clip_end_s": 10.0, "outcome": {"level1_enter_s": 3.0}}
        self.assertEqual(FA.sheet_times(adl)[1], ("enter", 3.0))


if __name__ == "__main__":
    unittest.main()
