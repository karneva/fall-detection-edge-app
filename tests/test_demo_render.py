"""
데모 이미지 스크립트(training/demo/render_demo) 테스트

- 순수 함수: 좌표 변환, 연속 장면 프레임 고르기, 연구 결과 대조, 회전 폴백 후보 고르기
- trace_clip이 replay.run_clip과 같은 이벤트를 내는지 (합성 시나리오)
- 실제 데이터가 있으면 기본 낙상 클립 재생이 연구 결과 파일과 같은지 (없으면 skip, CI는 skip)
numpy가 없으면 전체 skip.
"""
import importlib.util
import unittest
from pathlib import Path

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = HAS_NUMPY and importlib.util.find_spec("torch") is not None
REPO_ROOT = Path(__file__).resolve().parents[1]

if HAS_NUMPY:
    import numpy as np

    from training.demo import render_demo as D

T0 = 1000.0


def _trace(n=60, hz=15.0, tail_from=None):
    out = []
    for i in range(n):
        out.append({"t": i / hz, "src": i, "tail": tail_from is not None and i >= tail_from, "present": True,
                    "aspect": 2.0, "state": "NORMAL", "event": None})
    return out


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class PureHelpersTest(unittest.TestCase):
    def test_pixels(self):
        k = np.zeros((17, 3))
        k[0] = [0.5, 0.25, 0.9]
        px = D.kps_to_pixels(k, [640, 480])
        self.assertEqual(px[0].tolist(), [320.0, 120.0, 0.9])
        self.assertEqual(k[0, 0], 0.5)  # 입력은 바꾸지 않음
        self.assertEqual(D.bbox_to_pixels([0.1, 0.2, 0.5, 1.0], [100, 50]).tolist(), [10.0, 10.0, 50.0, 50.0])

    def test_ascii_text(self):
        self.assertEqual(D.ascii_text("Muñoz p=0.4"), "Mu?oz p=0.4")

    def test_decision_event_skips_escalation(self):
        ev = [{"type": "abnormal_enter", "ts": 1.0},
              {"type": "level1", "ts": 5.0, "reason": "escalated_after_ai_suppression"},
              {"type": "ai_suppressed", "ts": 6.0}]
        self.assertEqual(D.decision_event(ev)["type"], "ai_suppressed")
        self.assertIsNone(D.decision_event([{"type": "abnormal_enter", "ts": 1.0}]))

    def test_pick_strip(self):
        tr = _trace(n=60)
        ev = [{"type": "abnormal_enter", "ts": T0 + 2.0}, {"type": "level1", "ts": T0 + 3.0}]
        picks = D.pick_strip(tr, ev, T0)
        names = [n for n, _ in picks]
        self.assertEqual(names, ["before", "motion", "entry", "hold", "decision"])
        got = {n: tr[i]["t"] for n, i in picks}
        self.assertAlmostEqual(got["before"], 0.5, delta=1 / 15.0)
        self.assertAlmostEqual(got["motion"], 1.5, delta=1 / 15.0)
        self.assertAlmostEqual(got["entry"], 2.0, delta=1e-6)
        self.assertAlmostEqual(got["hold"], 2.5, delta=1 / 15.0)
        self.assertAlmostEqual(got["decision"], 3.0, delta=1e-6)
        # 판정이 없으면 빈 목록
        self.assertEqual(D.pick_strip(tr, ev[:1], T0), [])

    def test_pick_strip_tail_decision(self):
        # 판정이 정지 꼬리 안: hold는 진입과 마지막 실제 프레임 사이, decision은 꼬리 프레임
        tr = _trace(n=60, tail_from=40)
        ev = [{"type": "abnormal_enter", "ts": T0 + 1.0}, {"type": "ai_suppressed", "ts": T0 + 50 / 15.0}]
        picks = dict(D.pick_strip(tr, ev, T0))
        self.assertFalse(tr[picks["hold"]]["tail"])
        self.assertAlmostEqual(tr[picks["hold"]]["t"], (1.0 + 39 / 15.0) / 2, delta=1 / 15.0)
        self.assertTrue(tr[picks["decision"]]["tail"])

    def test_compare_outcome(self):
        ref = {"alarm": True, "suppressed": False, "reasons": ["abnormal_timeout"], "n_abnormal": 1,
               "abnormal_s": [2.55], "level1_s": 12.55, "p_fall": [0.9329]}
        same = dict(ref, p_fall=[0.93291])
        self.assertEqual(D.compare_outcome(same, ref), [])
        diffs = D.compare_outcome(dict(ref, alarm=False, p_fall=[0.5], level1_s=None), ref)
        self.assertEqual(len(diffs), 3)
        self.assertTrue(any(d.startswith("p_fall") for d in diffs))
        self.assertTrue(D.compare_outcome(dict(ref, abnormal_s=[2.55, 9.0]), ref))

    def test_fallback_candidates(self):
        fb = np.zeros(20, bool)
        fb[3:5] = True      # 짧은 구간
        fb[10:15] = True    # 가장 긴 구간, 가운데 12
        present = np.ones(20, bool)
        present[9] = False  # 긴 구간 직전 프레임에 대상 없음 -> 그 앞(8)이 직전 대상
        c = D.fallback_candidates(fb, present, fps=10.0)
        self.assertEqual(c[0], (12, 11))
        self.assertIn((10, 8), c)
        self.assertEqual(len(c), 7)
        # 직전 대상이 recent_s보다 멀면 제외
        present2 = np.zeros(20, bool)
        present2[0] = True
        c2 = D.fallback_candidates(fb, present2, fps=10.0, recent_s=0.5)
        self.assertEqual(c2, [(3, 0), (4, 0)])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TraceMatchesRunClipTest(unittest.TestCase):
    def test_same_events_as_run_clip(self):
        from app.core.level1 import Level1Params
        from app.eval.synth import make_scenario
        from training.datasets.clipfmt import from_obs_records
        from training.experiments import replay as R

        params = Level1Params(ai_mode="off", sustain_s=10.0, abnormal_timeout_s=10.0, lost_grace_s=0.0)
        obs_raw, _ = make_scenario("fast_fall", lie_s=20.0)
        obs = R.clip_to_obs(from_obs_records(obs_raw, dataset="synth", clip_id="synth/fast_fall"))
        ref, _ = R.run_clip(obs, params)
        events, trace = D.trace_clip(obs, params)
        self.assertEqual(events, ref)
        self.assertTrue(any(e["type"] == "level1" for e in events))
        self.assertEqual(len(trace), len(obs))
        i = next(k for k, s in enumerate(trace) if s["event"] == "abnormal_enter")
        self.assertEqual(trace[i]["state"], "ABNORMAL")
        self.assertAlmostEqual(trace[i]["t"], float(obs[i]["ts"]) - R.DEFAULT_T0)


def _demo_data_ready() -> bool:
    if not HAS_TORCH:
        return False
    need = [REPO_ROOT / "dataset" / "_manifest" / "manifest_v1.jsonl",
            REPO_ROOT / "dataset" / "_experiments" / "folds_v1.json",
            REPO_ROOT / "dataset" / "_keypoints" / "v3_fb" / "caucafall" / "backwards" / "FallBackwardsS9.npz"]
    need += [REPO_ROOT / "dataset" / "_experiments" / "p8_full" / f"fold{k}" / "models" / "E1_s0.pt" for k in range(5)]
    return all(p.exists() for p in need)


@unittest.skipUnless(_demo_data_ready(), "demo data (dataset/ keypoints, p8_full models) not available")
class DemoReplayMatchesStudyTest(unittest.TestCase):
    def test_default_fall_clip(self):
        import json

        from training.experiments import replay as R

        rows = D.load_rows()
        folds = json.loads(D.FOLDS.read_text(encoding="utf-8"))["folds"]
        row = rows[D.FALL_CLIP]
        res = D.replay_clip(row, int(folds[D.FALL_CLIP]), "cpu", "oof")
        self.assertEqual(D.compare_outcome(res["outcome"], res["study"]), [])
        self.assertTrue(res["outcome"]["alarm"])
        self.assertEqual([n for n, _ in D.pick_strip(res["trace"], res["events"], R.DEFAULT_T0)][-1], "decision")


if __name__ == "__main__":
    unittest.main()
