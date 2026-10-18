"""
짧은 미검출 유예(Level1Params.lost_grace_s) 테스트. 표준 라이브러리만 사용 (합성 시나리오).

- 기본값 0: 한 프레임만 놓쳐도 워밍업 카운터가 0 (기존 동작, golden trace와 같음)
- 유예 0.5초: 짧은 끊김에는 카운터와 누적값 유지, 긴 끊김은 기존처럼 초기화
- ABNORMAL 중 짧은 끊김에도 타임아웃 판정이 이어짐, 긴 끊김에서는 기존처럼 멈춤
"""
import copy
import unittest

from app.core.level1 import Level1Engine, Level1Params
from app.eval.synth import CountingVerifier, make_scenario

BASE = dict(sustain_s=10.0, abnormal_timeout_s=10.0, ai_mode="off", ai_suppress_pfall_th=0.4, ai_escalate_s=60.0)


def _lost(o):
    o = copy.deepcopy(o)
    for t in o.get("tracks") or []:
        t["has_person"] = False
        t["keypoints"] = []
        t["keypoints_raw"] = []
        t["keypoints_smooth"] = None
        t["quality_score"] = 0.0
    return o


def _run(obs, **kw):
    eng = Level1Engine(Level1Params(**dict(BASE, **kw)), verifier=CountingVerifier(0.9))
    events = []
    for o in obs:
        ev = eng.step(o)
        if ev is not None:
            events.append(ev)
    return eng, events


class LostGraceTest(unittest.TestCase):
    def setUp(self):
        self.obs, self.truth = make_scenario("fast_fall", lie_s=30.0)
        self.t0 = float(self.obs[0]["ts"])

    def _with_gap(self, start_s, dur_s):
        out = []
        for o in self.obs:
            t = float(o["ts"]) - self.t0
            out.append(_lost(o) if start_s <= t < start_s + dur_s else o)
        return out

    def test_default_resets_on_single_lost_frame(self):
        obs = self.obs[:30] + [_lost(self.obs[30])]
        eng, _ = _run(obs)
        self.assertEqual(eng.present_streak, 0)
        eng_g, _ = _run(obs, lost_grace_s=0.5)
        self.assertGreater(eng_g.present_streak, 0)

    def test_long_gap_still_resets(self):
        obs = self.obs[:30] + [_lost(o) for o in self.obs[30:60]]   # 15fps 기준 약 2초
        eng, _ = _run(obs, lost_grace_s=0.5)
        self.assertEqual(eng.present_streak, 0)
        self.assertIsNotNone(eng.lost_since_ts)

    def test_default_matches_previous_events(self):
        a = [(e["type"], e["ts"]) for e in _run(self.obs)[1]]
        b = [(e["type"], e["ts"]) for e in _run(self.obs, lost_grace_s=0.0)[1]]
        self.assertEqual(a, b)
        self.assertIn("level1", [t for t, _ in a])

    def test_timeout_continues_during_short_gaps(self):
        """ABNORMAL 진입 후 0.4초 끊김이 반복되면, 기본값은 판정이 멈추는 프레임이 생기고 유예가 있으면 같거나 빨리 알람"""
        _, ev0 = _run(self.obs)
        enter = next(e["ts"] for e in ev0 if e["type"] == "abnormal_enter") - self.t0
        obs = list(self.obs)
        for k in range(30):
            obs = [(_lost(o) if enter + 1.0 + k * 1.0 <= float(o["ts"]) - self.t0 < enter + 1.4 + k * 1.0 else o)
                   for o in obs]
        _, ev_def = _run(obs)
        _, ev_g = _run(obs, lost_grace_s=0.5)
        l_def = [e["ts"] for e in ev_def if e["type"] == "level1"]
        l_g = [e["ts"] for e in ev_g if e["type"] == "level1"]
        self.assertTrue(l_g)
        if l_def:
            self.assertLessEqual(l_g[0], l_def[0])

    def test_person_leaves_for_good_no_new_alarm(self):
        """서 있다가 화면을 떠나 계속 미검출이면 유예가 있어도 알람 없음, 짧은 끊김이 반복되는 직립도 알람 없음"""
        stand, _ = make_scenario("stand")
        obs = stand[:60] + [_lost(o) for o in stand[60:400]]
        _, ev = _run(obs, lost_grace_s=0.5)
        self.assertFalse([e for e in ev if e["type"] in ("level1", "abnormal_enter")])
        flicker = [(_lost(o) if i % 10 == 0 else o) for i, o in enumerate(stand)]
        _, ev = _run(flicker, lost_grace_s=0.5)
        self.assertFalse([e for e in ev if e["type"] == "level1"])

    def test_safety_scenarios_unchanged(self):
        """회복/일어남/떠났다 돌아옴 시나리오는 유예가 있어도 알람 여부가 같음"""
        for name in ("fall_then_recover", "lie_on_bed_slow", "suppressed_leave_and_return", "fall_writhe"):
            obs, _ = make_scenario(name, lie_s=30.0)
            a = [e["type"] for e in _run(obs)[1] if e["type"] == "level1"]
            b = [e["type"] for e in _run(obs, lost_grace_s=0.5)[1] if e["type"] == "level1"]
            self.assertEqual(len(a), len(b), name)


if __name__ == "__main__":
    unittest.main()
