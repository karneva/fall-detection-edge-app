import importlib.util
import unittest

from app.core.kp_buffer import TimedKeypointBuffer, frame_wh_from_track, kps_to_frame, window_summary
from app.core.level1 import Level1Engine, Level1Params
from app.eval.synth import FRAME_H, FRAME_W, CountingVerifier, make_obs, make_scenario

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None


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

    def test_frame_wh_last_non_none_in_window(self):
        b = TimedKeypointBuffer(horizon_s=10.0)
        b.append(0.0, None, frame_wh=(320, 240))
        b.append(1.0, None)
        b.append(2.0, None, frame_wh=[640, 480])
        b.append(3.0, None)
        self.assertEqual(b.window(0.0, 3.0)["frame_wh"], [640, 480])
        self.assertEqual(b.window(0.0, 1.5)["frame_wh"], [320, 240])
        self.assertIsNone(b.window(2.5, 3.0)["frame_wh"])
        self.assertNotIn("frame_wh", window_summary(b.window(0.0, 3.0)))

    def test_append_frame_wh_never_raises(self):
        # 프레임 루프에서 호출되므로 잘못된 해상도 값은 무시(None)하고 예외를 내지 않음
        b = TimedKeypointBuffer(horizon_s=10.0)
        for i, bad in enumerate(((0, 480), (640,), "x", (None, 480), (-1, -1), {})):
            b.append(float(i), None, frame_wh=bad)
        self.assertIsNone(b.window(0.0, 10.0)["frame_wh"])
        b.append(10.0, None, frame_wh=(640.0, 480.0))
        self.assertEqual(b.window(0.0, 10.0)["frame_wh"], [640, 480])

    @unittest.skipUnless(_HAS_NUMPY, "numpy 필요")
    def test_append_frame_wh_numpy(self):
        import numpy as np

        b = TimedKeypointBuffer(horizon_s=10.0)
        b.append(0.0, None, frame_wh=np.array([1280, 720], dtype=np.int32))
        wh = b.window(0.0, 1.0)["frame_wh"]
        self.assertEqual(wh, [1280, 720])
        self.assertIs(type(wh[0]), int)  # json 직렬화 가능

    def test_frame_wh_from_track(self):
        self.assertEqual(frame_wh_from_track({"frame_shape": (480, 640)}), (640, 480))
        self.assertEqual(frame_wh_from_track({"frame_shape": [720, 1280, 3]}), (1280, 720))
        for bad in (None, {}, {"frame_shape": None}, {"frame_shape": (0, 640)}, {"frame_shape": "x"}, "track"):
            self.assertIsNone(frame_wh_from_track(bad), bad)

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

    def test_event_window_has_frame_wh_from_obs(self):
        obs, _ = make_scenario("fast_fall", lie_s=20.0)
        engine = Level1Engine(params(), verifier=CountingVerifier(0.9))
        ev = self.run_until(obs, engine)
        # 이벤트 요약에는 넣지 않고(골든 trace 유지) 전체 구간(sidecar)에만 들어감
        self.assertNotIn("frame_wh", ev["values"]["event_window"])
        snap = engine.get_incident_snapshot()
        self.assertEqual(snap["event_window"]["frame_wh"], [FRAME_W, FRAME_H])

    def test_event_window_frame_wh_none_without_frame_shape(self):
        obs, _ = make_scenario("fast_fall", lie_s=20.0)
        for o in obs:
            for t in o["tracks"]:
                t.pop("frame_shape", None)
        engine = Level1Engine(params(), verifier=CountingVerifier(0.9))
        self.assertIsNotNone(self.run_until(obs, engine))
        self.assertIsNone(engine.get_incident_snapshot()["event_window"]["frame_wh"])

    def test_escalated_event_keeps_same_window(self):
        obs, truth = make_scenario("fast_fall", lie_s=90.0)
        engine = Level1Engine(params(), verifier=CountingVerifier(0.05))
        sup = self.run_until(obs, engine, ("ai_suppressed",))
        rest = obs[[o["ts"] for o in obs].index(sup["ts"]) + 1:]
        esc = self.run_until(rest, engine, ("level1",))
        self.assertEqual(esc["reason"], "escalated_after_ai_suppression")
        self.assertEqual(esc["values"]["event_window"], sup["values"]["event_window"])
        self.assertLessEqual(esc["values"]["event_window"]["t_start"], truth["fall_start_ts"])

    def test_escalated_snapshot_keeps_ai_input_from_decision_time(self):
        obs, _ = make_scenario("fast_fall", lie_s=90.0)
        engine = Level1Engine(params(), verifier=CountingVerifier(0.05))
        sup = self.run_until(obs, engine, ("ai_suppressed",))
        sup_input = engine.get_incident_snapshot()["ai_input"]
        rest = obs[[o["ts"] for o in obs].index(sup["ts"]) + 1:]
        esc = self.run_until(rest, engine, ("level1",))
        snap = engine.get_incident_snapshot()
        self.assertEqual(esc["reason"], "escalated_after_ai_suppression")
        self.assertEqual(snap["ai_input"], sup_input)
        self.assertAlmostEqual(snap["ai_input"]["ts"][0], snap["ai"]["window_start_ts"])
        self.assertAlmostEqual(snap["ai_input"]["ts"][-1], snap["ai"]["window_end_ts"])

    def test_long_occlusion_after_entry_keeps_fall_window(self):
        # 진입 직후부터 25초간 가려져도(버퍼 20초) 낙상 구간이 남아 있어야 함
        obs, truth = make_scenario("fast_fall", lie_s=40.0)
        start = truth["fall_start_ts"] + 1.0
        for o in obs:
            if start <= o["ts"] < start + 25.0:
                o.update(make_obs(o["ts"], o["frame_index"], None))
        engine = Level1Engine(params(), verifier=CountingVerifier(0.9))
        self.run_until(obs, engine, ("level1",))
        win = engine.get_incident_snapshot()["event_window"]
        self.assertFalse(win["truncated"])
        self.assertLessEqual(win["t_start"], truth["fall_start_ts"])
        self.assertGreater(win["n_frames"] - win["n_missing"], 0)

    def test_low_posture_entry_anchors_window_at_posture_start(self):
        # LEVEL1 후 리셋되면 누운 자세로 저자세 지속(low_posture_sustained) 재진입이 발생
        obs, _ = make_scenario("fast_fall", lie_s=60.0)
        engine = Level1Engine(params(), verifier=CountingVerifier(0.9))
        it = iter(obs)
        for o in it:
            ev = engine.step(o)
            if ev and ev["type"] == "abnormal_enter" and "low_posture" in ev["values"]["reason"]:
                break
        else:
            self.fail("no low_posture_sustained entry")
        # (이 합성 시나리오는 몇 초 뒤 회복 판정이 나므로 진입 직후에 확인)
        snap = engine.get_incident_snapshot()
        self.assertEqual(snap["enter_reason"], "low_posture_sustained")
        self.assertLess(snap["event_anchor_ts"], snap["abnormal_start_ts"] - 2.0)
        self.assertAlmostEqual(snap["event_window"]["t_start"], snap["event_anchor_ts"] - 1.5)

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
