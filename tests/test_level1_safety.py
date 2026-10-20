"""
Level1Engine + AI 게이트 안전성 회귀 테스트 (표준 라이브러리만 사용)

핵심 불변식: AI 분류기는 룰 기반 낙상 알람을 '조용히' 지울 수 없다.
- 검증기 비활성/예외/이상값 -> 룰 결과 유지 (fail-open)
- 억제는 p_fall < 임계값일 때만 (임계값 <= 0이면 억제 없음), 사건당 1회, 쿨다운 미소모
- 억제 후에도 붕괴가 지속되면 승격(escalation)
- LEVEL1 뒤 계속 누워 있으면 진입/회복을 반복하지 않고, 잠깐 일어났다 다시 넘어지면 다시 알람

실행: python -m unittest discover -s tests -t . -v
"""
import math
import random
import subprocess
import sys
import unittest
from pathlib import Path

from app.core.level1 import Level1Params
from app.eval import synth as S
from app.eval.synth import CountingVerifier, RaisingVerifier, make_scenario, run_engine

REPO_ROOT = Path(__file__).resolve().parents[1]


def params(**overrides) -> Level1Params:
    """환경변수(ABNORMAL_TIMEOUT_S, AI_*, LOST_GRACE_S)에 영향받지 않도록 기본값을 명시한 파라미터 (운영 기본값)."""
    base = dict(
        post_alarm_latch=True,   # 래치 기능 테스트용으로 켬 (운영 기본값은 꺼짐: POST_ALARM_LATCH)
        sustain_s=10.0,
        abnormal_timeout_s=10.0,
        ai_mode="veto",
        ai_suppress_pfall_th=0.4,
        ai_escalate_s=60.0,
        ai_escalate_requires_drop=True,
        lost_grace_s=0.5,
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

    def test_threshold_boundary_strict(self):
        # 억제 <=> p_fall < th (경계값은 알람 쪽)
        for p_fall, expect in ((0.41, "level1"), (0.40, "level1"), (0.39, "ai_suppressed")):
            events = run_engine(self.obs, params(), CountingVerifier(p_fall))
            first = of_type(events, "level1", "ai_suppressed")[0]
            self.assertEqual(first["type"], expect, p_fall)

    def test_threshold_zero_or_negative_disables_veto(self):
        for th in (0.0, -0.1):
            v = CountingVerifier(0.0)
            events = run_engine(self.obs, params(ai_suppress_pfall_th=th), v)
            self.assertEqual(of_type(events, "ai_suppressed"), [], th)
            level1 = of_type(events, "level1")
            self.assertEqual(len(level1), 1, th)
            self.assertFalse(level1[0]["values"]["ai"]["suppress"])
            self.assertEqual(v.calls, 1)

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


def _build(segs, fps=15.0, t0=1000.0, seed=0):
    """(지속시간, u -> (hip_y, angle) 또는 None) 구간 목록으로 합성 obs를 만듭니다 (synth.make_scenario와 같은 방식)."""
    rng = random.Random(seed)
    out, i, start = [], 0, 0.0
    for dur, fn in segs:
        n = max(1, int(round(dur * fps)))
        for j in range(n):
            pose = fn(j / max(1, n - 1))
            kps = None if pose is None else S.skeleton(0.5, pose[0], S.HEIGHT, pose[1], rng)
            out.append(S.make_obs(t0 + start + j / fps, i, kps))
            i += 1
        start += n / fps
    return out


def _stand(u):
    return S.STAND_HIP_Y, 0.0


def _lie(u):
    return S.LIE_HIP_Y, math.pi / 2


def _fall(u):
    return S._lerp(S.STAND_HIP_Y, S.LIE_HIP_Y, u), S._lerp(0.0, math.pi / 2, u)


def _up(u):
    return S._lerp(S.LIE_HIP_Y, S.STAND_HIP_Y, u), S._lerp(math.pi / 2, 0.0, u)


def _refall_segs(stand_s, fall2_s, lie2_s=60.0):
    """첫 낙상 -> 40초 누움(쿨다운 끝) -> 1.5초 일어남 -> stand_s초 섬 -> fall2_s초에 걸쳐 다시 넘어짐 -> 누움"""
    return [(3.0, _stand), (0.5, _fall), (40.0, _lie), (1.5, _up), (stand_s, _stand), (fall2_s, _fall), (lie2_s, _lie)]


def _seg_start_ts(obs, segs, k, fps):
    """k번째 구간 첫 프레임의 ts (_build와 같은 프레임 수 계산)."""
    return obs[sum(max(1, int(round(d * fps))) for d, _ in segs[:k])]["ts"]


class PostAlarmLatchTest(unittest.TestCase):
    """
    LEVEL1 뒤 래치: 누운 채로는 재진입하지 않고, 일어났다 다시 넘어지면 다시 알람.
    일어섬 = 넘어지기 전 자세 기준의 일어선 자세가 post_alarm_min_upright_s(0.4초) 이상 (감쇠 누적)
    """

    def test_writhing_after_level1_no_flapping_no_repeat(self):
        from app.core.level1 import Level1Engine

        for fps in (10.0, 15.0, 30.0):
            for seed in (0, 1):
                with self.subTest(fps=fps, seed=seed):
                    obs, _ = make_scenario("fall_writhe", fps=fps, seed=seed, lie_s=90.0)
                    eng = Level1Engine(params(), verifier=CountingVerifier(0.9))
                    events = [e for e in (eng.step(o) for o in obs) if e is not None]
                    self.assertEqual([e["type"] for e in events], ["abnormal_enter", "level1"])
                    self.assertTrue(eng.post_alarm_active)
                    self.assertFalse(eng.post_alarm_stood)

    def test_refall_after_getting_up_alarms_again(self):
        # 첫 알람의 쿨다운이 끝난 뒤 일어나 stand_s초 서 있다가 fall2_s초에 걸쳐 다시 넘어짐
        # (잠깐 섬 + 느린 붕괴도 포함; 10초 섬은 래치가 이미 풀린 경우)
        for fps in (10.0, 15.0, 30.0):
            for stand_s in (0.3, 1.0, 10.0):
                for fall2_s in (0.5, 2.0, 3.5, 5.0):
                    with self.subTest(fps=fps, stand_s=stand_s, fall2_s=fall2_s):
                        segs = _refall_segs(stand_s, fall2_s)
                        obs = _build(segs, fps=fps)
                        events = run_engine(obs, params(), CountingVerifier(0.9))
                        self.assertEqual([e["type"] for e in events], ["abnormal_enter", "level1"] * 2)
                        second_fall = _seg_start_ts(obs, segs, 5, fps)
                        enter2 = of_type(events, "abnormal_enter")[1]["ts"]
                        self.assertTrue(second_fall <= enter2 <= second_fall + fall2_s + 2.0, enter2 - second_fall)

    def test_refall_latch_state(self):
        # 래치 상태: 알람 -> (누움) 차단 -> 일어섬(stood) 진입 허용 -> 두 번째 알람에서 다시 차단으로
        from app.core.level1 import Level1Engine

        segs = _refall_segs(0.3, 3.5)
        obs = _build(segs, fps=15.0)
        get_up, fall2 = _seg_start_ts(obs, segs, 3, 15.0), _seg_start_ts(obs, segs, 5, 15.0)
        eng = Level1Engine(params(), verifier=CountingVerifier(0.9))
        seen = {}
        n_level1 = 0
        for o in obs:
            ev = eng.step(o)
            n_level1 += bool(ev and ev["type"] == "level1")
            if o["ts"] < get_up and n_level1 == 1:
                seen["lying"] = (eng.post_alarm_active, eng.post_alarm_stood)
            if o["ts"] < fall2:
                seen["before_fall2"] = (eng.post_alarm_active, eng.post_alarm_stood)
        self.assertEqual(seen["lying"], (True, False))
        self.assertEqual(seen["before_fall2"], (True, True))   # 0.3초 섬: 래치 해제(recover_s)까지는 아니지만 일어섬
        self.assertEqual(n_level1, 2)
        self.assertEqual((eng.post_alarm_active, eng.post_alarm_stood), (True, False))   # 두 번째 알람으로 다시 차단

    def test_refall_via_low_posture_path(self):
        # 다시 넘어짐이 복합 신호 없이 저자세 지속으로만 잡히는 경우도 일어선 뒤에는 진입해야 함
        # (enter_confirm_need=99로 복합 신호 경로를 끔. 30fps: 저자세 45프레임 = 1.5초 < 기준 창 3초)
        for stand_s in (0.3, 1.0):
            for fall2_s in (2.0, 5.0):
                with self.subTest(stand_s=stand_s, fall2_s=fall2_s):
                    obs = _build(_refall_segs(stand_s, fall2_s), fps=30.0)
                    events = run_engine(obs, params(enter_confirm_need=99), CountingVerifier(0.9))
                    self.assertEqual([e["type"] for e in events], ["abnormal_enter", "level1"] * 2)
                    reasons = [e["values"]["reason"] for e in of_type(events, "abnormal_enter")]
                    self.assertEqual(reasons, ["low_posture_sustained"] * 2)

    def test_isolated_upright_frames_while_lying_no_reentry(self):
        # 누운 채 bbox가 잠깐(1~3프레임, 0.4초 미만) 세로로 튀는 검출 잡음은 일어섬이 아님
        from app.core.level1 import Level1Engine

        for fps in (10.0, 15.0, 30.0):
            for k in (1, 2, 3):
                with self.subTest(fps=fps, k=k):
                    obs = _build([(3.0, _stand), (0.5, _fall), (120.0, _lie)], fps=fps)
                    rng = random.Random(k)
                    for i in range(int(45 * fps), len(obs), int(1.5 * fps)):
                        for j in range(i, min(i + k, len(obs))):
                            kps = S.skeleton(0.5, S.STAND_HIP_Y, S.HEIGHT, 0.0, rng)
                            obs[j] = S.make_obs(obs[j]["ts"], obs[j]["frame_index"], kps)
                    eng = Level1Engine(params(), verifier=CountingVerifier(0.9))
                    events = [e for e in (eng.step(o) for o in obs) if e is not None]
                    self.assertEqual([e["type"] for e in events], ["abnormal_enter", "level1"])
                    self.assertFalse(eng.post_alarm_stood)

    def test_upright_frame_after_long_loss_is_not_standing(self):
        # 오래 미검출된 뒤 첫 프레임의 긴 dt 한 번으로 일어섬/래치 해제가 되지 않음 (프레임당 누적 상한)
        from app.core.level1 import Level1Engine

        obs = _build([(3.0, _stand), (0.5, _fall), (40.0, _lie), (10.0, lambda u: None), (1 / 15.0, _stand),
                      (60.0, _lie)], fps=15.0)
        eng = Level1Engine(params(), verifier=CountingVerifier(0.9))
        events = [e for e in (eng.step(o) for o in obs) if e is not None]
        self.assertEqual([e["type"] for e in events], ["abnormal_enter", "level1"])
        self.assertTrue(eng.post_alarm_active)
        self.assertFalse(eng.post_alarm_stood)

    def test_latch_survives_reset_and_clears_on_switch(self):
        from app.core.level1 import Level1Engine

        obs, _ = make_scenario("fast_fall", lie_s=60.0)
        eng = Level1Engine(params(), verifier=CountingVerifier(0.9))
        for o in obs:
            eng.step(o)
        self.assertEqual(eng.state, "NORMAL")
        self.assertTrue(eng.post_alarm_active)
        eng.step(dict(obs[-1], ts=obs[-1]["ts"] + 0.1, target_switched=True))
        self.assertFalse(eng.post_alarm_active)

    def test_latch_off_restores_previous_behavior(self):
        obs, _ = make_scenario("fast_fall", lie_s=60.0)
        events = run_engine(obs, params(post_alarm_latch=False), CountingVerifier(0.9))
        self.assertGreater(len(of_type(events, "abnormal_exit")), 0)


class BindEngineTest(unittest.TestCase):
    def test_engine_aware_verifier_is_bound(self):
        from app.core.level1 import Level1Engine

        class Aware(CountingVerifier):
            def bind_engine(self, engine):
                self.engine = engine

        v = Aware(0.9)
        eng = Level1Engine(params(), verifier=v)
        self.assertIs(v.engine, eng)

    def test_bind_error_does_not_break_engine(self):
        import logging
        from app.core.level1 import Level1Engine

        class Broken(CountingVerifier):
            def bind_engine(self, engine):
                raise RuntimeError("boom")

        logging.disable(logging.CRITICAL)
        try:
            obs, _ = make_scenario("fast_fall", lie_s=20.0)
            eng = Level1Engine(params(), verifier=Broken(0.9))
            events = [e for e in (eng.step(o) for o in obs) if e is not None]
        finally:
            logging.disable(logging.NOTSET)
        self.assertEqual(len(of_type(events, "level1")), 1)


class KnownIssuesTest(unittest.TestCase):
    """
    알려진 결함을 기록하는 테스트. 수정되면 expectedFailure를 제거합니다.
    (예상대로 실패하는 동안은 'expected failure'로 통과 처리됨)
    """

    def test_ai_window_contains_fall_onset(self):
        # (Day 8 해결) 기본 검증기에 v2 분류기를 쓰면 AI 입력은 엔진에 묶인 사건 구간 창들이고,
        # 첫 창(진입 사건 구간)은 넘어지는 동작을 포함함: 창 시작 <= 낙상 끝.
        # v1(legacy, 메타 feature_version v1)은 여전히 LEVEL1 시점의 최근 30프레임(kp_seq)만 봐서
        # '누워서 정지' 구간을 분류함 (ai["window_start_ts"] > 낙상 끝). 아래 마지막 단언이 그 동작을 기록.
        from unittest import mock

        from app.core import level1

        class _Sliding:
            def __init__(self):
                self.engine = None
                self.starts = []

            def bind_engine(self, engine):
                self.engine = engine

            def __call__(self, seq):
                win = self.engine.get_incident_snapshot()["event_window"]
                self.starts.append(win["t_start"])
                self.last_starts = [win["t_start"]]   # event_window.SlidingWindowVerifier와 같은 속성
                return 0.9

        class _StubV2:
            enabled, feature_version, weights_hash = True, "v2", "stub-v2"

            def __init__(self):
                self.made = []

            def predict_proba(self, seq):
                raise AssertionError("v2 must not judge the 30-frame kp_seq")

            def make_engine_verifier(self):
                self.made.append(_Sliding())
                return self.made[-1]

        obs, truth = make_scenario("fast_fall", lie_s=20.0)
        clf = _StubV2()
        with mock.patch.object(level1._ClassifierVerifier, "_get", lambda self: clf):
            events = run_engine(obs, params(), None)   # 기본 검증기 (엔진 생성 시 bind_engine)
        lv = of_type(events, "level1")
        self.assertEqual(len(lv), 1)
        self.assertEqual(lv[0]["values"]["ai"]["p_fall"], 0.9)
        self.assertEqual(len(clf.made), 1)
        starts = clf.made[0].starts
        self.assertEqual(len(starts), 1)
        self.assertLessEqual(starts[0], truth["fall_end_ts"])
        self.assertLessEqual(starts[0], truth["fall_start_ts"])   # 낙상 시작 전부터 (진입 - pre_s)
        # 이벤트의 창 정보도 30프레임 버퍼가 아니라 v2 창 기준
        ai = lv[0]["values"]["ai"]
        self.assertEqual((ai["window_start_ts"], ai["n_windows"]), (starts[0], 1))
        self.assertLess(ai["window_offset_from_abnormal_s"], 0.0)

        # v1 경로: 최근 30프레임 창은 낙상 뒤 정지 구간 (알려진 한계, v1 호환용으로만 남김)
        events_v1 = run_engine(obs, params(), CountingVerifier(0.9))
        ai_v1 = of_type(events_v1, "level1")[0]["values"]["ai"]
        self.assertGreater(ai_v1["window_start_ts"], truth["fall_end_ts"])

    def test_no_enter_exit_flapping_after_level1_with_latch(self):
        # 알람 뒤 래치를 켜면 (POST_ALARM_LATCH=1) LEVEL1 뒤 누운 채 진입/회복이 반복되지 않음
        obs, _ = make_scenario("fast_fall", lie_s=60.0)
        events = run_engine(obs, params(post_alarm_latch=True), CountingVerifier(0.9))
        self.assertEqual(of_type(events, "abnormal_exit"), [])
        self.assertEqual(len(of_type(events, "abnormal_enter")), 1)

    @unittest.expectedFailure
    def test_no_enter_exit_flapping_after_level1_default(self):
        # 운영 기본값(래치 꺼짐)에서는 LEVEL1 뒤 누운 자세가 새 baseline이 되어 abnormal_enter -> abnormal_exit(recovered)가
        # 몇 초 주기로 반복 기록됨. 알림은 나가지 않아(로그만) 두 번째 낙상을 놓칠 수 있는 래치보다 이쪽을 택함 (Day 8)
        obs, _ = make_scenario("fast_fall", lie_s=60.0)
        events = run_engine(obs, params(post_alarm_latch=False), CountingVerifier(0.9))
        self.assertEqual(of_type(events, "abnormal_exit"), [])

    def test_latch_default_is_off(self):
        from app import config
        self.assertFalse(config.POST_ALARM_LATCH)
        self.assertEqual(Level1Params().post_alarm_latch, config.POST_ALARM_LATCH)


if __name__ == "__main__":
    unittest.main()
