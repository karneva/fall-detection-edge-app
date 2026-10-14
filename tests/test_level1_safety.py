"""
Level1Engine + AI 게이트 안전성 회귀 테스트 (표준 라이브러리만 사용)

핵심 불변식: AI 분류기는 룰 기반 낙상 알람을 '조용히' 지울 수 없다.
- 검증기 비활성/예외/이상값 -> 룰 결과 유지 (fail-open)
- 억제는 p_fall <= 임계값일 때만, 사건당 1회, 쿨다운 미소모
- 억제 후에도 붕괴가 지속되면 승격(escalation)

실행: python -m unittest discover -s tests -t . -v
"""
import subprocess
import sys
import unittest
from pathlib import Path

from app.core.level1 import Level1Params
from app.eval.synth import CountingVerifier, RaisingVerifier, make_scenario, run_engine

REPO_ROOT = Path(__file__).resolve().parents[1]


def params(**overrides) -> Level1Params:
    """환경변수(ABNORMAL_TIMEOUT_S, AI_*)에 영향받지 않도록 기본값을 명시한 파라미터."""
    base = dict(
        sustain_s=10.0,
        abnormal_timeout_s=10.0,
        ai_mode="veto",
        ai_suppress_pfall_th=0.4,
        ai_escalate_s=60.0,
        ai_escalate_requires_drop=True,
    )
    base.update(overrides)
    return Level1Params(**base)


def of_type(events, *types):
    return [e for e in events if e["type"] in types]


class ImportTest(unittest.TestCase):
    def test_import_has_no_heavy_deps(self):
        code = (
            "import sys, app.core.level1, app.core;"
            "print(sorted({'torch', 'numpy', 'cv2'} & set(sys.modules)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(out, "[]")


class RuleBaselineTest(unittest.TestCase):
    def test_stand_no_alarm(self):
        obs, _ = make_scenario("stand")
        v = CountingVerifier(0.9)
        events = run_engine(obs, params(), v)
        self.assertEqual(of_type(events, "abnormal_enter", "level1", "ai_suppressed"), [])
        self.assertEqual(v.calls, 0)

    def test_fall_then_recover_no_alarm(self):
        obs, _ = make_scenario("fall_then_recover")
        events = run_engine(obs, params(), CountingVerifier(0.9))
        self.assertEqual(len(of_type(events, "abnormal_enter")), 1)
        self.assertEqual(len(of_type(events, "abnormal_exit")), 1)
        self.assertEqual(of_type(events, "level1", "ai_suppressed"), [])


class AiGateTest(unittest.TestCase):
    def setUp(self):
        self.obs, self.truth = make_scenario("fast_fall", lie_s=20.0)

    def test_fast_fall_fall_verdict(self):
        v = CountingVerifier(0.9)
        events = run_engine(self.obs, params(), v)
        enters = of_type(events, "abnormal_enter")
        level1 = of_type(events, "level1")
        self.assertEqual(len(level1), 1)
        self.assertEqual(v.calls, 1)
        ai = level1[0]["values"]["ai"]
        self.assertEqual(ai["decision"], "fall")
        self.assertAlmostEqual(ai["p_fall"], 0.9)
        self.assertEqual(ai["model_id"], "synthetic")
        fall_start = self.truth["fall_start_ts"]
        self.assertTrue(fall_start <= enters[0]["ts"] <= fall_start + 2.0)
        self.assertLessEqual(level1[0]["ts"] - fall_start, params().sustain_s + 5.0)

    def test_verifier_none_fails_open(self):
        events = run_engine(self.obs, params(), CountingVerifier(None))
        level1 = of_type(events, "level1")
        self.assertEqual(len(level1), 1)
        self.assertEqual(level1[0]["values"]["ai"]["decision"], "unavailable")
        self.assertEqual(of_type(events, "ai_suppressed"), [])

    def test_verifier_exception_fails_open(self):
        import logging
        logging.disable(logging.CRITICAL)
        try:
            events = run_engine(self.obs, params(), RaisingVerifier())
        finally:
            logging.disable(logging.NOTSET)
        level1 = of_type(events, "level1")
        self.assertEqual(len(level1), 1)
        self.assertEqual(level1[0]["values"]["ai"]["decision"], "error")

    def test_invalid_p_fall_fails_open(self):
        import logging
        logging.disable(logging.CRITICAL)
        try:
            for bad in (float("nan"), -0.1, 1.5, "abc", 10 ** 400, object()):
                events = run_engine(self.obs, params(), CountingVerifier(bad))
                level1 = of_type(events, "level1")
                self.assertEqual(len(level1), 1, bad)
                self.assertEqual(level1[0]["values"]["ai"]["decision"], "error", bad)
        finally:
            logging.disable(logging.NOTSET)

    def test_threshold_boundary_legacy_equiv(self):
        # 기존 규칙: argmax=ADL 이고 max-softmax >= 0.6 일 때 억제 <=> p_fall <= 0.4
        for p_fall, expect in ((0.41, "level1"), (0.40, "ai_suppressed"), (0.39, "ai_suppressed")):
            events = run_engine(self.obs, params(), CountingVerifier(p_fall))
            first = of_type(events, "level1", "ai_suppressed")[0]
            self.assertEqual(first["type"], expect, p_fall)

    def test_suppressed_event_keeps_legacy_keys(self):
        events = run_engine(self.obs, params(), CountingVerifier(0.05))
        sup = of_type(events, "ai_suppressed")[0]
        self.assertIn("ai_conf", sup["values"])
        self.assertIn("logic", sup["values"])
        self.assertAlmostEqual(sup["values"]["ai_conf"], 0.95)

    def test_shadow_mode_never_suppresses(self):
        v = CountingVerifier(0.05)
        events = run_engine(self.obs, params(ai_mode="shadow"), v)
        level1 = of_type(events, "level1")
        self.assertEqual(len(level1), 1)
        self.assertEqual(of_type(events, "ai_suppressed"), [])
        ai = level1[0]["values"]["ai"]
        self.assertEqual(ai["decision"], "adl")
        self.assertFalse(ai["suppress"])
        self.assertEqual(v.calls, 1)

    def test_off_mode_skips_verifier(self):
        v = CountingVerifier(0.05)
        events = run_engine(self.obs, params(ai_mode="off"), v)
        level1 = of_type(events, "level1")
        self.assertEqual(len(level1), 1)
        self.assertEqual(level1[0]["values"]["ai"]["decision"], "off")
        self.assertEqual(v.calls, 0)

    def test_unknown_mode_does_not_suppress(self):
        events = run_engine(self.obs, params(ai_mode="vetoo"), CountingVerifier(0.05))
        self.assertEqual(len(of_type(events, "level1")), 1)
        self.assertEqual(of_type(events, "ai_suppressed"), [])


class SuppressionWatchTest(unittest.TestCase):
    def test_suppression_does_not_consume_cooldown_and_escalates(self):
        from app.core.level1 import Level1Engine

        obs, _ = make_scenario("fast_fall", lie_s=90.0)
        v = CountingVerifier(0.05)
        engine = Level1Engine(params(), verifier=v)
        events = []
        supp_state = None
        for o in obs:
            ev = engine.step(o)
            if ev is None:
                continue
            events.append(ev)
            if ev["type"] == "ai_suppressed":
                supp_state = (engine.state, engine.cooldown_until_ts)

        self.assertEqual(supp_state, ("ABNORMAL", 0.0))
        supp_ts = of_type(events, "ai_suppressed")[0]["ts"]
        level1 = of_type(events, "level1")
        self.assertEqual(len(level1), 1)
        self.assertEqual(level1[0]["reason"], "escalated_after_ai_suppression")
        self.assertTrue(supp_ts + 60.0 <= level1[0]["ts"] <= supp_ts + 61.5)
        self.assertTrue(level1[0]["values"]["had_drop"])
        self.assertEqual(level1[0]["values"]["ai"]["decision"], "adl")
        # 사건당 AI 호출 1회, 억제 이후 enter/exit 반복 없음
        self.assertEqual(v.calls, 1)
        self.assertEqual(len(of_type(events, "abnormal_enter")), 1)
        self.assertEqual(of_type(events, "abnormal_exit"), [])

    def test_timeout_path_suppression_keeps_watch_and_escalates(self):
        # 바닥에서 계속 움직여 정지 판정이 안 되면 Case 2(abnormal_timeout)로 검증됨
        from app.core.level1 import Level1Engine

        obs, _ = make_scenario("fall_writhe", lie_s=90.0)
        v = CountingVerifier(0.05)
        engine = Level1Engine(params(), verifier=v)
        events, supp_state = [], None
        for o in obs:
            ev = engine.step(o)
            if ev is None:
                continue
            events.append(ev)
            if ev["type"] == "ai_suppressed":
                supp_state = (engine.state, engine.cooldown_until_ts)

        sup = of_type(events, "ai_suppressed")
        self.assertEqual(len(sup), 1)
        self.assertEqual(sup[0]["reason"], "abnormal_timeout")
        self.assertEqual(supp_state, ("ABNORMAL", 0.0))
        level1 = of_type(events, "level1")
        self.assertEqual(len(level1), 1)
        self.assertEqual(level1[0]["reason"], "escalated_after_ai_suppression")
        self.assertTrue(sup[0]["ts"] + 60.0 <= level1[0]["ts"] <= sup[0]["ts"] + 61.5)
        self.assertEqual(v.calls, 1)

    def test_no_escalation_when_person_returns_standing(self):
        # 억제 후 일어나 회복 판정 전에 화면 밖으로 나갔다가, 60초 넘게 지나 서서 돌아온 경우
        obs, _ = make_scenario("suppressed_leave_and_return")
        v = CountingVerifier(0.05)
        events = run_engine(obs, params(), v)
        self.assertEqual(len(of_type(events, "ai_suppressed")), 1)
        self.assertEqual(of_type(events, "level1"), [])
        self.assertEqual(of_type(events, "abnormal_exit")[-1]["reason"], "recovered")
        self.assertEqual(v.calls, 1)

    def test_escalation_can_be_tuned(self):
        obs, _ = make_scenario("fast_fall", lie_s=40.0)
        events = run_engine(obs, params(ai_escalate_s=20.0), CountingVerifier(0.05))
        supp_ts = of_type(events, "ai_suppressed")[0]["ts"]
        level1 = of_type(events, "level1")
        self.assertEqual(len(level1), 1)
        self.assertTrue(supp_ts + 20.0 <= level1[0]["ts"] <= supp_ts + 21.5)

    def test_slow_lie_on_bed_no_escalation(self):
        obs, _ = make_scenario("lie_on_bed_slow", lie_s=90.0)
        v = CountingVerifier(0.05)
        events = run_engine(obs, params(), v)
        # 억제 경로를 실제로 거친 뒤(급하강 없음) 승격되지 않아야 함
        self.assertEqual(len(of_type(events, "ai_suppressed")), 1)
        self.assertEqual(of_type(events, "level1"), [])
        self.assertEqual(v.calls, 1)

    def test_slow_lie_escalates_when_drop_not_required(self):
        obs, _ = make_scenario("lie_on_bed_slow", lie_s=90.0)
        events = run_engine(obs, params(ai_escalate_requires_drop=False), CountingVerifier(0.05))
        level1 = of_type(events, "level1")
        self.assertEqual(len(level1), 1)
        self.assertEqual(level1[0]["reason"], "escalated_after_ai_suppression")

    def test_get_up_after_suppression_exits(self):
        obs, _ = make_scenario("fall_lie_then_get_up")
        v = CountingVerifier(0.05)
        events = run_engine(obs, params(), v)
        types = [e["type"] for e in events]
        self.assertEqual(types, ["abnormal_enter", "ai_suppressed", "abnormal_exit"])
        self.assertEqual(v.calls, 1)


class KnownIssuesTest(unittest.TestCase):
    """
    알려진 결함을 기록하는 테스트. 수정되면 expectedFailure를 제거합니다.
    (예상대로 실패하는 동안은 'expected failure'로 통과 처리됨)
    """

    @unittest.expectedFailure
    def test_ai_window_contains_fall_onset(self):
        # 현재 CNN은 LEVEL1 시점(진입 후 sustain_s 경과)의 최근 30프레임만 봄.
        # 즉 넘어지는 동작이 아닌 '누워서 정지' 구간을 분류함. (로드맵 Day 5에서 해결)
        obs, truth = make_scenario("fast_fall", lie_s=20.0)
        events = run_engine(obs, params(), CountingVerifier(0.9))
        ai = of_type(events, "level1")[0]["values"]["ai"]
        self.assertLessEqual(ai["window_start_ts"], truth["fall_end_ts"])

    @unittest.expectedFailure
    def test_no_enter_exit_flapping_after_level1(self):
        # LEVEL1 후 reset_all 하면 누운 자세가 새 baseline이 되어
        # abnormal_enter -> abnormal_exit(recovered)가 몇 초 주기로 반복됨. (로드맵 Day 5에서 해결)
        obs, _ = make_scenario("fast_fall", lie_s=60.0)
        events = run_engine(obs, params(), CountingVerifier(0.9))
        self.assertEqual(of_type(events, "abnormal_exit"), [])


if __name__ == "__main__":
    unittest.main()
