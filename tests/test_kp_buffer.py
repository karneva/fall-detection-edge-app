import unittest

from app.core.kp_buffer import TimedKeypointBuffer, kps_to_frame, window_summary
from app.core.level1 import Level1Engine, Level1Params
from app.eval.synth import CountingVerifier, make_obs, make_scenario


def params(**kw):
    base = dict(sustain_s=10.0, abnormal_timeout_s=10.0, ai_mode="veto", ai_suppress_pfall_th=0.4,
                ai_escalate_s=60.0, ai_escalate_requires_drop=True)
    base.update(kw)
    return Level1Params(**base)


class BufferTest(unittest.TestCase):
    def test_kps_to_frame(self):
        self.assertIsNone(kps_to_frame([]))
        f = kps_to_frame([{"id": 3, "x": 0.1, "y": 0.2, "conf": 0.9}, {"id": 99, "x": 1, "y": 1}])
        self.assertEqual(len(f), 17)
        self.assertEqual(f[3], [0.1, 0.2, 0.9])
        self.assertEqual(f[0], [0.0, 0.0, 0.0])

    def test_horizon_by_time_not_frames(self):
        b = TimedKeypointBuffer(horizon_s=2.0)
        for i in range(100):
            b.append(i * 0.1, None)
        self.assertAlmostEqual(b._buf[-1][0] - b._buf[0][0], 2.0, places=6)

    def test_window_and_missing(self):
        b = TimedKeypointBuffer(horizon_s=10.0)
        for i in range(30):
            frame = None if i % 3 == 0 else [[0.0, 0.0, 1.0]] * 17
            b.append(100.0 + i * 0.1, frame, [0, 0, 1, 1])
        w = b.window(100.5, 101.5)
        self.assertEqual(w["n_frames"], 11)
        self.assertEqual(w["n_missing"], sum(1 for k in w["kps"] if k is None))
        self.assertAlmostEqual(w["fps"], 10.0, places=3)
        self.assertFalse(w["truncated"])
        self.assertTrue(b.window(99.0, 100.5)["truncated"])
        self.assertNotIn("kps", window_summary(w))

    def test_time_going_backwards_clears(self):
        b = TimedKeypointBuffer()
        b.append(10.0, None)
        b.append(5.0, None)
        self.assertEqual(len(b), 1)


class EngineEventWindowTest(unittest.TestCase):
    def run_until(self, obs, engine, types=("level1", "ai_suppressed")):
        for o in obs:
            ev = engine.step(o)
            if ev and ev["type"] in types:
                return ev
        return None

    def test_event_window_contains_fall_onset_at_any_fps(self):
        for fps in (10.0, 15.0, 30.0):
            obs, truth = make_scenario("fast_fall", fps=fps, lie_s=20.0)
            engine = Level1Engine(params(), verifier=CountingVerifier(0.9))
            ev = self.run_until(obs, engine)
            summ = ev["values"]["event_window"]
            self.assertLessEqual(summ["t_start"], truth["fall_start_ts"], fps)
            self.assertGreaterEqual(summ["t_end"], truth["fall_end_ts"], fps)
            self.assertAlmostEqual(summ["t_end"] - summ["t_start"], 4.0, places=6)
            self.assertAlmostEqual(summ["fps"], fps, delta=0.5)
            snap = engine.get_incident_snapshot()
            self.assertEqual(len(snap["event_window"]["kps"]), summ["n_frames"])
            self.assertEqual(len(snap["ai_input"]["kps"]), 30)
            self.assertEqual(snap["ai"]["decision"], "fall")

    def test_escalated_event_keeps_same_window(self):
        obs, truth = make_scenario("fast_fall", lie_s=90.0)
        engine = Level1Engine(params(), verifier=CountingVerifier(0.05))
        sup = self.run_until(obs, engine, ("ai_suppressed",))
        rest = obs[[o["ts"] for o in obs].index(sup["ts"]) + 1:]
        esc = self.run_until(rest, engine, ("level1",))
        self.assertEqual(esc["reason"], "escalated_after_ai_suppression")
        self.assertEqual(esc["values"]["event_window"], sup["values"]["event_window"])
        self.assertLessEqual(esc["values"]["event_window"]["t_start"], truth["fall_start_ts"])

    def test_lost_frames_recorded_as_missing(self):
        obs, truth = make_scenario("fast_fall", lie_s=20.0)
        # 낙상 동작 중 0.4초간 미검출
        for o in obs:
            if truth["fall_start_ts"] <= o["ts"] < truth["fall_start_ts"] + 0.4:
                o.update(make_obs(o["ts"], o["frame_index"], None))
        engine = Level1Engine(params(), verifier=CountingVerifier(0.9))
        ev = self.run_until(obs, engine)
        self.assertIsNotNone(ev)
        self.assertGreater(ev["values"]["event_window"]["n_missing"], 0)


if __name__ == "__main__":
    unittest.main()
