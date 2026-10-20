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
        padded = b.window(100.5, 101.5, pad_s=0.2)
        self.assertEqual(window_summary(padded), window_summary(w))  # 요약 불변
        self.assertEqual(len(padded["ts"]), 15)  # 100.3 ~ 101.7
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
            win = snap["event_window"]
            # 전체 구간에는 앞뒤 여유(pad_s)만큼 프레임이 더 담기고, 요약은 원래 구간 기준
            self.assertEqual(win["pad_s"], 0.25)
            self.assertGreater(len(win["kps"]), summ["n_frames"])
            self.assertLess(win["ts"][0], summ["t_start"])
            self.assertGreater(win["ts"][-1], summ["t_end"])
            self.assertEqual(sum(summ["t_start"] <= t <= summ["t_end"] for t in win["ts"]), summ["n_frames"])
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
        # v1 검증기(30프레임 seq): 기록은 v1_seq30이고 창 시각은 30프레임 버퍼 기준
        self.assertEqual(snap["ai_input"]["kind"], "v1_seq30")
        self.assertEqual(len(snap["ai_input"]["kps"]), 30)
        self.assertNotIn("window_starts", snap["ai"])
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
        # 이미 누운 뒤에 처음 보인 사람 -> 저자세 지속(low_posture_sustained)으로 진입
        # (LEVEL1 뒤의 재진입은 알람 뒤 래치가 막으므로 넘어진 뒤부터 재생)
        obs, truth = make_scenario("fast_fall", lie_s=60.0)
        obs = [o for o in obs if o["ts"] >= truth["fall_end_ts"] + 1.0]
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


class _FakeV2Windows:
    """v2 사건 구간 창 검증기 흉내 (_ClassifierVerifier와 같은 속성): 첫 창 = 사건 구간, 이어서 0.5초 간격 4초 창."""

    def __init__(self, ps_fn=lambda i: 0.2 + 0.1 * (i % 5), with_ps=True):
        self.ps_fn, self.with_ps = ps_fn, with_ps
        self.engine = None
        self.model_id = "fake-v2"
        self.last_window_starts = None
        self._ps = None
        self.calls = 0

    def bind_engine(self, engine):
        self.engine = engine

    @property
    def last_window_ps(self):
        if not self.with_ps:
            raise AttributeError("no ps")   # 속성이 없거나 오류여도 엔진은 fail-safe
        return self._ps

    last_window_s = 4.0

    def __call__(self, seq):
        self.calls += 1
        e = self.engine
        t0 = e.get_incident_snapshot()["event_window"]["t_start"]
        ts = e.kp_ts[-1]
        starts = [t0]
        t = t0 + 0.5
        while t + 4.0 <= ts:
            starts.append(t)
            t += 0.5
        self.last_window_starts = starts
        self._ps = [self.ps_fn(i) for i in range(len(starts))]
        return max(self._ps)


class AiInputRecordTest(unittest.TestCase):
    """sidecar/사건 스냅샷의 ai_input은 모델이 실제로 점수를 낸 입력 (v1: 30프레임, v2: 사건 구간 창들의 원시 구간)"""

    def _run(self, verifier, obs, types=("level1", "ai_suppressed")):
        engine = Level1Engine(params(), verifier=verifier)
        for o in obs:
            ev = engine.step(o)
            if ev and ev["type"] in types:
                return engine, ev
        self.fail("no decision event")

    def test_v2_records_scored_windows_and_raw_slice(self):
        for fps in (10.0, 15.0, 30.0):
            obs, truth = make_scenario("fast_fall", fps=fps, lie_s=20.0)
            v = _FakeV2Windows()
            engine, ev = self._run(v, obs)
            ai = ev["values"]["ai"]
            snap = engine.get_incident_snapshot()
            inp = snap["ai_input"]
            self.assertEqual(inp["kind"], "v2_event_windows", fps)
            self.assertEqual(inp["window_starts"], v.last_window_starts)
            self.assertEqual(inp["ps"], v._ps)
            self.assertEqual(inp["window_s"], 4.0)
            self.assertEqual(ai["window_starts"], v.last_window_starts)
            self.assertEqual(ai["window_ps"], v._ps)
            self.assertEqual(ai["n_windows"], len(v.last_window_starts))
            self.assertGreater(ai["n_windows"], 1)
            j = v._ps.index(max(v._ps))
            self.assertEqual(ai["argmax_window_start_ts"], v.last_window_starts[j])
            self.assertEqual(ai["p_fall"], max(v._ps))
            # 원시 구간: 첫 창 시작(pad 포함)부터 판정 시점까지, 낙상 동작 포함
            self.assertLessEqual(inp["ts"][0], ai["window_start_ts"])
            self.assertAlmostEqual(inp["t_start"], ai["window_start_ts"])
            self.assertAlmostEqual(inp["ts"][-1], engine.kp_ts[-1])
            self.assertGreaterEqual(inp["ts"][-1], v.last_window_starts[-1] + 4.0)
            self.assertLess(inp["ts"][0], truth["fall_start_ts"])
            self.assertEqual(len(inp["kps"]), len(inp["ts"]))
            self.assertEqual(len(inp["kps"][-1]), 17)   # 원시 프레임 (17 x [x, y, conf]), 평탄화 seq 아님
            self.assertEqual(inp["pad_s"], 0.25)
            self.assertFalse(inp["truncated"])
            self.assertIs(snap["ai"], ai)

    def test_v2_escalation_keeps_decision_time_input(self):
        obs, _ = make_scenario("fast_fall", lie_s=90.0)
        v = _FakeV2Windows(ps_fn=lambda i: 0.05)
        engine, sup = self._run(v, obs, ("ai_suppressed",))
        sup_input = engine.get_incident_snapshot()["ai_input"]
        rest = obs[[o["ts"] for o in obs].index(sup["ts"]) + 1:]
        for o in rest:
            ev = engine.step(o)
            if ev and ev["type"] == "level1":
                break
        else:
            self.fail("no escalation")
        self.assertEqual(ev["reason"], "escalated_after_ai_suppression")
        self.assertEqual(v.calls, 1)
        self.assertEqual(engine.get_incident_snapshot()["ai_input"], sup_input)
        self.assertEqual(sup_input["kind"], "v2_event_windows")

    def test_v2_without_ps_still_tagged(self):
        obs, _ = make_scenario("fast_fall", lie_s=20.0)
        engine, ev = self._run(_FakeV2Windows(with_ps=False), obs)
        ai = ev["values"]["ai"]
        inp = engine.get_incident_snapshot()["ai_input"]
        self.assertEqual(inp["kind"], "v2_event_windows")
        self.assertIsNone(inp["ps"])
        self.assertIsNone(ai["window_ps"])
        self.assertIsNone(ai["argmax_window_start_ts"])
        self.assertEqual(ai["decision"], "fall")

    @unittest.skipUnless(_HAS_NUMPY, "numpy not installed")
    def test_default_verifier_exposes_window_ps(self):
        # 기본 _ClassifierVerifier + 실제 SlidingWindowVerifier (점수 함수만 가짜): 창별 점수/길이가 기록까지 이어짐
        from unittest import mock

        import numpy as np

        from app.ai_classifier.event_window import SlidingWindowVerifier
        from app.core import level1

        def score(x):
            return np.linspace(0.1, 0.9, len(x))

        class _StubV2:
            enabled, feature_version, weights_hash = True, "v2", "stub-v2"

            def predict_proba(self, seq):
                raise AssertionError("v2 must not judge the 30-frame kp_seq")

            def make_engine_verifier(self):
                return SlidingWindowVerifier(score, mode="max", step_s=0.5, win_s=4.0)

        obs, truth = make_scenario("fast_fall", lie_s=20.0)
        clf = _StubV2()
        with mock.patch.object(level1._ClassifierVerifier, "_get", lambda self: clf):
            engine, ev = self._run(None, obs)
        ai = ev["values"]["ai"]
        inp = engine.get_incident_snapshot()["ai_input"]
        self.assertEqual(inp["kind"], "v2_event_windows")
        self.assertEqual(inp["window_s"], 4.0)
        self.assertEqual(len(inp["ps"]), ai["n_windows"])
        self.assertEqual(ai["window_ps"], inp["ps"])
        self.assertAlmostEqual(ai["p_fall"], 0.9)
        self.assertEqual(ai["argmax_window_start_ts"], ai["window_starts"][-1])
        self.assertLessEqual(inp["ts"][0], ai["window_start_ts"])
        self.assertLess(ai["window_start_ts"], truth["fall_start_ts"])


if __name__ == "__main__":
    unittest.main()
