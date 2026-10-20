"""
progression study 평가 도구(training/experiments: replay, verifiers, folds) 테스트

합성 시나리오(app/eval/synth.py)를 클립 형식으로 바꿔 운영 obs 재구성(15Hz, EMA, 정지 꼬리),
클립 결과/요약 계산, 검증기 fail-open, fold 배정을 확인합니다.
numpy가 없으면 전체 skip, torch/sklearn이 없으면 해당 테스트만 skip.
"""
import importlib.util
import json
import logging
import tempfile
import unittest
from pathlib import Path

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = HAS_NUMPY and importlib.util.find_spec("torch") is not None
HAS_SKLEARN = HAS_NUMPY and importlib.util.find_spec("sklearn") is not None

if HAS_NUMPY:
    import numpy as np

    from app.core.level1 import Level1Params
    from app.engine import smoothing
    from app.eval.synth import make_scenario
    from training.datasets.clipfmt import from_obs_records
    from training.experiments import replay

# 환경변수에 영향받지 않도록 판정 관련 값을 고정 (tests/test_level1_golden.py와 같은 값)
BASE_PARAMS = dict(
    sustain_s=10.0,
    abnormal_timeout_s=10.0,
    ai_suppress_pfall_th=0.4,
    ai_escalate_s=60.0,
    ai_escalate_requires_drop=True,
    lost_grace_s=0.0,
)


def _params(ai_mode="off"):
    return Level1Params(ai_mode=ai_mode, **BASE_PARAMS)


def _synth_clip(name="fast_fall", fps=30.0, lie_s=3.0, seed=0, drop_every=0):
    """합성 시나리오 obs -> 클립 dict (clipfmt.from_obs_records). drop_every > 0이면 그 간격마다 사람 없음 프레임."""
    obs, truth = make_scenario(name, fps=fps, seed=seed, lie_s=lie_s)
    if drop_every:
        for i in range(drop_every, len(obs), drop_every):
            t0 = obs[i]["tracks"][0]
            t0.update(has_person=False, bbox=None, keypoints=[], keypoints_raw=[], quality_score=0.0)
    return from_obs_records(obs, dataset="synth", clip_id=f"synth/{name}"), truth


class _T:
    """YOLO 결과 텐서 흉내 (.cpu().numpy(), .item(), 인덱싱)"""

    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def item(self):
        return float(self.a)

    def __getitem__(self, i):
        return _T(self.a[i])

    def __len__(self):
        return len(self.a)


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = _T(xyxy), _T(conf)

    def __len__(self):
        return len(self.xyxy)


class _Kps:
    def __init__(self, xy, conf):
        self.xy, self.conf = _T(xy), _T(conf)

    def __len__(self):
        return len(self.xy)


class _R0:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes, self.keypoints = boxes, keypoints


def _runtime_obs_stream(clip, hz=15.0):
    """운영 경로 그대로: 프레임마다 build_observation(YOLO 결과 흉내) -> ema_smooth_keypoints_inplace"""
    from app.core.obs_schema import build_observation

    w, h = int(clip["frame_wh"][0]), int(clip["frame_wh"][1])
    idx = replay.resample_indices(clip["ts"], hz)
    smoothing._reset_ema_state()
    out = []
    for k, i in enumerate(idx):
        if clip["present"][i]:
            px = clip["bbox"][i].astype(np.float64) * np.array([w, h, w, h])
            kxy = clip["kps"][i, :, :2].astype(np.float64) * np.array([w, h])
            r0 = _R0(_Boxes(px[None], [0.9]), _Kps(kxy[None], clip["kps"][i, :, 2][None]))
        else:
            r0 = _R0(_Boxes(np.zeros((0, 4)), np.zeros(0)), None)
        obs = build_observation(r0, w, h, 1000.0 + float(clip["ts"][i]), k)
        out.append(smoothing.ema_smooth_keypoints_inplace(obs))
    smoothing._reset_ema_state()
    return out


def _kp_array(kps_list):
    return np.array([[k["id"], k["x"], k["y"], k["conf"]] for k in kps_list], dtype=np.float64)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class ClipToObsTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_resample_nearest_grid(self):
        ts = np.arange(301) / 30.0  # 10초, 30fps
        idx = replay.resample_indices(ts, 15.0)
        self.assertEqual(len(idx), 151)
        np.testing.assert_array_equal(idx, np.arange(0, 301, 2))
        # 불규칙 시간축: 가장 가까운 프레임, 같은 거리면 앞 프레임
        ts2 = np.array([0.0, 0.05, 0.10, 0.14, 0.30])
        np.testing.assert_array_equal(replay.resample_indices(ts2, 10.0), [0, 2, 3, 4])
        self.assertEqual(len(replay.resample_indices(np.zeros(0))), 0)

    def test_obs_matches_runtime_schema_and_ema(self):
        clip, _ = _synth_clip("fast_fall", lie_s=2.0, drop_every=17)
        ref = _runtime_obs_stream(clip)
        got = replay.clip_to_obs(clip, tail_s=0.0)
        self.assertEqual(len(got), len(ref))
        for a, b in zip(got, ref):
            self.assertEqual(set(a) - {"meta"}, set(b) - {"meta"})
            self.assertTrue(set(b["meta"]) <= set(a["meta"]))
            self.assertAlmostEqual(a["ts"], b["ts"], places=9)
            self.assertEqual(a["frame_index"], b["frame_index"])
            ta, tb = a["tracks"][0], b["tracks"][0]
            self.assertEqual(set(ta), set(tb))
            self.assertEqual(ta["has_person"], tb["has_person"])
            self.assertEqual(tuple(ta["frame_shape"]), tuple(tb["frame_shape"]))
            self.assertAlmostEqual(ta["quality_score"], tb["quality_score"], places=6)
            if not tb["has_person"]:
                self.assertIsNone(ta["bbox"])
                self.assertEqual(ta["keypoints_raw"], [])
                continue
            np.testing.assert_allclose(ta["bbox"], tb["bbox"], atol=1e-6)
            self.assertEqual(len(ta["keypoints_raw"]), 17)
            for key in ("keypoints_raw", "keypoints_smooth", "keypoints"):
                np.testing.assert_allclose(_kp_array(ta[key]), _kp_array(tb[key]), atol=1e-6, err_msg=key)
        # 사람 없음 프레임이 실제로 섞여 있어야 의미 있는 비교
        self.assertTrue(any(not o["tracks"][0]["has_person"] for o in got))
        self.assertTrue(all(o["tracks"][0]["keypoints_smooth"] is not None
                            for o in got if o["tracks"][0]["has_person"]))

    def test_obs_ts_follow_source_frames(self):
        # 25fps 원본 + 시작 시각 0.5초: obs ts는 고른 원본 프레임의 실제 시각 (격자 시각이 아님)
        clip, _ = _synth_clip("fast_fall", lie_s=2.0)
        n = len(clip["ts"])
        c25 = dict(clip, ts=0.5 + np.arange(n) / 25.0)
        obs = replay.clip_to_obs(c25, tail_s=2.0)
        main = [o for o in obs if not o["meta"]["replay_tail"]]
        tail = [o for o in obs if o["meta"]["replay_tail"]]
        src = [o["meta"]["replay_src_index"] for o in main]
        np.testing.assert_array_equal(src, replay.resample_indices(c25["ts"]))
        np.testing.assert_allclose([o["ts"] - 1000.0 for o in main], c25["ts"][src], atol=1e-9)
        self.assertEqual(len(tail), 30)
        np.testing.assert_allclose([o["ts"] for o in tail],
                                   main[-1]["ts"] + np.arange(1, 31) / 15.0, atol=1e-9)
        # 원본 fps가 15Hz보다 낮으면 같은 프레임을 두 번 넣지 않음 (운영은 프레임마다 한 번 처리)
        c10 = dict(clip, ts=np.arange(n) / 10.0)
        obs10 = replay.clip_to_obs(c10, tail_s=0.0)
        src10 = [o["meta"]["replay_src_index"] for o in obs10]
        self.assertEqual(src10, list(range(n)))
        self.assertTrue(np.all(np.diff([o["ts"] for o in obs10]) > 0))
        self.assertEqual([o["frame_index"] for o in obs10], list(range(n)))

    def test_study_params_fixed(self):
        p = replay.study_params("veto", sustain_s=8.0)
        self.assertEqual((p.ai_mode, p.sustain_s, p.abnormal_timeout_s, p.ai_suppress_pfall_th,
                          p.ai_escalate_s, p.ai_escalate_requires_drop), ("veto", 8.0, 10.0, 0.4, 60.0, True))
        self.assertEqual(replay.study_params().ai_mode, "off")

    def test_study_params_pins_lost_grace_and_overrides(self):
        # 운영 기본값과 무관하게 0.0 고정 (이전 단계 재현), STUDY_OVERRIDES < 직접 준 값
        self.assertEqual(replay.study_params("veto").lost_grace_s, 0.0)
        saved = dict(replay.STUDY_OVERRIDES)
        try:
            replay.STUDY_OVERRIDES["lost_grace_s"] = 0.5
            self.assertEqual(replay.study_params("off").lost_grace_s, 0.5)
            self.assertEqual(replay.study_params("veto", ai_suppress_pfall_th=0.3).lost_grace_s, 0.5)
            self.assertEqual(replay.study_params("off", lost_grace_s=0.25).lost_grace_s, 0.25)
            self.assertEqual(replay.study_params("off").abnormal_timeout_s, 10.0)
        finally:
            replay.STUDY_OVERRIDES.clear()
            replay.STUDY_OVERRIDES.update(saved)
        self.assertEqual(replay.study_params().lost_grace_s, 0.0)

    def test_study_params_pins_post_alarm_latch_off(self):
        # 래치(운영 기본 켬)는 실험 재생에서 끔 (progression_v1, p7b_*, p8_full 재현), 직접 주면 켤 수 있음
        self.assertFalse(replay.study_params("veto").post_alarm_latch)
        self.assertTrue(replay.study_params("off", post_alarm_latch=True).post_alarm_latch)

    def test_ema_reset_per_clip(self):
        clip_a, _ = _synth_clip("fast_fall", lie_s=2.0, seed=1)
        clip_b, _ = _synth_clip("stand", seed=2)
        fresh_b = replay.clip_to_obs(clip_b, tail_s=0.0)
        replay.clip_to_obs(clip_a, tail_s=0.0)
        after_a = replay.clip_to_obs(clip_b, tail_s=0.0)
        for x, y in zip(fresh_b, after_a):
            self.assertEqual(x["tracks"][0]["keypoints_smooth"], y["tracks"][0]["keypoints_smooth"])
        # 첫 프레임의 스무딩 값은 원시 값과 같음 (이전 클립 상태를 물려받지 않음)
        t0 = after_a[0]["tracks"][0]
        np.testing.assert_allclose(_kp_array(t0["keypoints_smooth"]), _kp_array(t0["keypoints_raw"]))
        # 다른 사용자에게 EMA 상태를 남기지 않음
        self.assertIsNone(smoothing._prev_kp_ema)

    def test_freeze_tail(self):
        clip, _ = _synth_clip("fast_fall", lie_s=2.0)
        main = replay.clip_to_obs(clip, tail_s=0.0)
        obs = replay.clip_to_obs(clip, hz=15.0, tail_s=20.0)
        n_main = len(main)
        self.assertEqual(len(obs), n_main + 300)
        self.assertFalse(any(o["meta"]["replay_tail"] for o in obs[:n_main]))
        self.assertTrue(all(o["meta"]["replay_tail"] for o in obs[n_main:]))
        last_src = main[-1]["meta"]["replay_src_index"]
        self.assertEqual(last_src, len(clip["ts"]) - 1)
        for o in obs[n_main:]:
            self.assertEqual(o["meta"]["replay_src_index"], last_src)
            self.assertEqual(o["tracks"][0]["keypoints_raw"], main[-1]["tracks"][0]["keypoints_raw"])
        # 시간축은 끊김 없이 1/15초 간격, 프레임 번호 연속
        ts = np.array([o["ts"] for o in obs])
        np.testing.assert_allclose(np.diff(ts), 1.0 / 15.0, atol=1e-9)
        self.assertEqual([o["frame_index"] for o in obs], list(range(len(obs))))
        self.assertAlmostEqual(obs[0]["ts"], 1000.0)
        # 꼬리 obs는 서로 다른 객체 (EMA 제자리 수정이 서로 섞이지 않음)
        self.assertIsNot(obs[-1]["tracks"][0]["keypoints_raw"], obs[-2]["tracks"][0]["keypoints_raw"])
        self.assertEqual(replay.clip_to_obs({"ts": np.zeros(0), "kps": np.zeros((0, 17, 3)),
                                             "bbox": np.zeros((0, 4)), "present": np.zeros(0, bool),
                                             "frame_wh": np.array([640, 480])}), [])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class RunClipTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_tail_lets_short_fall_reach_level1(self):
        clip, _ = _synth_clip("fast_fall", lie_s=3.0)
        row = {"clip_id": "synth/fast_fall", "dataset": "synth", "label": "fall"}
        ev_short, _ = replay.run_clip(replay.clip_to_obs(clip, tail_s=0.0), _params())
        self.assertFalse(replay.clip_outcome(row, ev_short)["alarm"])
        seen = []
        events, engine = replay.run_clip(replay.clip_to_obs(clip), _params(),
                                         on_event=lambda ev, eng: seen.append((ev["type"], eng)))
        out = replay.clip_outcome(row, events)
        self.assertTrue(out["alarm"])
        self.assertTrue(out["reached_ai"])
        self.assertFalse(out["suppressed"])
        self.assertGreaterEqual(out["n_abnormal"], 1)
        self.assertEqual([t for t, _ in seen], [e["type"] for e in events])
        self.assertTrue(all(eng is engine for _, eng in seen))

    def test_standing_adl_no_alarm(self):
        clip, _ = _synth_clip("stand")
        events, _ = replay.run_clip(replay.clip_to_obs(clip), _params())
        self.assertFalse(replay.clip_outcome({"label": "adl"}, events)["alarm"])

    def test_default_verifier_is_fail_open_null(self):
        # verifier=None이면 운영 가중치를 찾지 않고 판단 불가(None) -> 규칙 결과 유지
        clip, _ = _synth_clip("fast_fall", lie_s=3.0)
        events, engine = replay.run_clip(replay.clip_to_obs(clip), _params("veto"))
        self.assertIsInstance(engine._verifier, replay.NullVerifier)
        level1 = [e for e in events if e["type"] == "level1"]
        self.assertEqual(len(level1), 1)
        self.assertEqual(level1[0]["values"]["ai"]["decision"], "unavailable")

    def test_obs_reusable_across_runs(self):
        clip, _ = _synth_clip("fast_fall", lie_s=3.0)
        obs = replay.clip_to_obs(clip)
        e1, _ = replay.run_clip(obs, _params())
        e2, _ = replay.run_clip(obs, _params())
        self.assertEqual([(e["type"], e["ts"]) for e in e1], [(e["type"], e["ts"]) for e in e2])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class OutcomeSummaryTest(unittest.TestCase):
    def test_clip_outcome(self):
        events = [
            {"type": "abnormal_enter", "ts": 1.0},
            {"type": "ai_suppressed", "ts": 12.0, "reason": "still_for_sustain_s",
             "values": {"ai": {"p_fall": 0.1}}},
            {"type": "level1", "ts": 80.0, "reason": "escalated_after_ai_suppression",
             "values": {"ai": {"p_fall": 0.1}}},
        ]
        out = replay.clip_outcome({"clip_id": "d/x", "dataset": "d", "label": "fall"}, events)
        self.assertEqual(out["clip_id"], "d/x")
        self.assertTrue(out["alarm"])
        self.assertEqual(out["reasons"], ["escalated_after_ai_suppression"])
        self.assertTrue(out["suppressed"])
        self.assertTrue(out["reached_ai"])
        self.assertEqual(out["n_abnormal"], 1)
        self.assertEqual(out["p_fall"], [0.1])  # 승격 이벤트는 같은 판정의 반복이라 한 번만
        self.assertEqual(out["alarm_ts"], 80.0)

        none = replay.clip_outcome({"clip_id": "d/y", "dataset": "d", "label": "adl"},
                                   [{"type": "abnormal_enter", "ts": 1.0}, {"type": "abnormal_exit", "ts": 3.0}])
        self.assertFalse(none["alarm"] or none["suppressed"] or none["reached_ai"])
        self.assertEqual(none["reasons"], [])

    def test_summarize_math(self):
        def o(ds, label, alarm, suppressed, reached):
            return {"clip_id": f"{ds}/{label}{alarm}{suppressed}{reached}", "dataset": ds, "label": label,
                    "alarm": alarm, "suppressed": suppressed, "reached_ai": reached}
        outs = [
            o("a", "fall", True, False, True),    # 탐지
            o("a", "fall", False, True, True),    # AI가 지운 낙상
            o("b", "fall", False, False, False),  # 규칙이 못 잡음
            o("b", "fall", True, True, True),     # 억제 후 승격
            o("a", "adl", True, False, True),     # 오탐
            o("a", "adl", False, True, True),     # AI가 막은 오탐
            o("b", "adl", False, False, False),
            o("b", "adl", False, False, False),
        ]
        s = replay.summarize(outs)
        self.assertEqual((s["n_fall"], s["n_adl"], s["n_detected"], s["n_adl_alarm"]), (4, 4, 2, 1))
        self.assertAlmostEqual(s["recall"], 0.5)
        self.assertAlmostEqual(s["adl_fa_rate"], 0.25)
        self.assertAlmostEqual(s["balanced"], 2 * 0.5 * 0.75 / 1.25)
        self.assertEqual(s["suppressed_true_falls"], 1)
        self.assertEqual(s["n_fall_suppressed"], 2)
        self.assertEqual(s["n_adl_suppressed"], 1)
        self.assertAlmostEqual(s["rules_ceiling"], 0.75)
        self.assertAlmostEqual(s["adl_rules_fa_rate"], 0.5)
        self.assertAlmostEqual(s["precision"], 2 / 3)
        self.assertAlmostEqual(s["f1"], 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
        self.assertEqual(sorted(s["by_dataset"]), ["a", "b"])
        a = s["by_dataset"]["a"]
        self.assertEqual((a["n_fall"], a["n_detected"], a["n_adl"], a["n_adl_alarm"]), (2, 1, 2, 1))
        self.assertAlmostEqual(a["recall"], 0.5)
        self.assertAlmostEqual(a["adl_fa_rate"], 0.5)
        b = s["by_dataset"]["b"]
        self.assertAlmostEqual(b["rules_ceiling"], 0.5)
        self.assertEqual(b["adl_fa_rate"], 0.0)
        self.assertNotIn("by_dataset", a)

    def test_summarize_empty_denominators(self):
        s = replay.summarize([{"dataset": "a", "label": "adl", "alarm": False, "suppressed": False,
                               "reached_ai": False}])
        self.assertIsNone(s["recall"])
        self.assertIsNone(s["balanced"])
        self.assertIsNone(s["rules_ceiling"])
        self.assertEqual(s["adl_fa_rate"], 0.0)
        e = replay.summarize([])
        self.assertEqual((e["n_fall"], e["n_adl"], e["by_dataset"]), (0, 0, {}))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TimingTest(unittest.TestCase):
    """시간 기준 탐지: 클립 시각 = ts - t0, 창 [fall_start - 2, fall_end + 3]"""

    ROW = {"clip_id": "d/f", "dataset": "d", "label": "fall", "fall_start_s": 5.0, "fall_end_s": 6.0}

    def _ev(self, enter_s, level1_s, t0=1000.0):
        return [{"type": "abnormal_enter", "ts": t0 + enter_s}, {"type": "level1", "ts": t0 + level1_s}]

    def test_categories(self):
        cases = [
            (4.0, 14.0, True, True, None),                     # 진입이 창 안, LEVEL1이 낙상 이후
            (2.0, 4.5, False, True, "level1_before_fall"),     # LEVEL1이 낙상보다 먼저 (진입도 이름)
            (1.0, 11.0, False, True, "enter_early"),           # 진입이 2초 넘게 먼저, LEVEL1은 창 밖
            (0.5, 8.5, False, True, "enter_early"),            # 진입은 이르지만 LEVEL1이 창 안 -> 느슨한 기준만 통과
            (9.5, 19.5, False, False, "enter_late"),           # 진입이 fall_end + 3초보다 늦음
        ]
        for enter, l1, strict, loose, why in cases:
            t = replay.timing(self.ROW, self._ev(enter, l1))
            self.assertEqual((t["timely_strict"], t["untimely"]), (strict, why), (enter, l1))
            if enter == 1.0:
                self.assertFalse(t["timely_loose"])
            elif strict or enter == 0.5:
                self.assertTrue(t["timely_loose"])
            self.assertAlmostEqual(t["level1_enter_s"], enter)

    def test_episode_is_last_enter_before_level1(self):
        ev = [{"type": "abnormal_enter", "ts": 1001.0}, {"type": "abnormal_exit", "ts": 1003.0},
              {"type": "abnormal_enter", "ts": 1004.5}, {"type": "level1", "ts": 1015.0},
              {"type": "abnormal_enter", "ts": 1030.0}]
        t = replay.timing(self.ROW, ev, end_s=12.0)
        self.assertEqual(t["abnormal_s"], [1.0, 4.5, 30.0])
        self.assertAlmostEqual(t["level1_enter_s"], 4.5)
        self.assertTrue(t["timely_strict"])
        self.assertTrue(t["in_tail"])
        self.assertFalse(replay.timing(self.ROW, ev, end_s=20.0)["in_tail"])
        self.assertIsNone(replay.timing(self.ROW, ev)["in_tail"])

    def test_no_label_or_no_alarm(self):
        adl = replay.timing({"label": "adl"}, self._ev(1.0, 11.0))
        self.assertIsNone(adl["timely_strict"])
        self.assertAlmostEqual(adl["level1_s"], 11.0)
        miss = replay.timing(self.ROW, [{"type": "abnormal_enter", "ts": 1004.0}])
        self.assertIsNone(miss["level1_s"])
        self.assertIsNone(miss["timely_strict"])

    def test_multiple_falls(self):
        """낙상이 두 번인 클립: 두 번째 낙상에 대한 알람도 제때, 기준 낙상은 진입이 창 안에 드는 구간"""
        row = dict(self.ROW, segments=[{"cls": "walk", "start_s": 0.0, "end_s": 5.0},
                                       {"cls": "fall", "start_s": 5.0, "end_s": 6.0},
                                       {"cls": "stand_up", "start_s": 8.0, "end_s": 10.0},
                                       {"cls": "fall", "start_s": 20.0, "end_s": 21.0}])
        self.assertEqual(replay.fall_intervals(row), [(5.0, 6.0), (20.0, 21.0)])
        t = replay.timing(row, self._ev(19.5, 30.0))
        self.assertTrue(t["timely_strict"])
        self.assertEqual(t["ref_fall_s"], [20.0, 21.0])
        late = replay.timing(row, self._ev(26.0, 36.0))
        self.assertEqual((late["timely_strict"], late["untimely"], late["ref_fall_s"]), (False, "enter_late", [20.0, 21.0]))
        # 구간 라벨이 없으면 fall_start_s/fall_end_s
        self.assertEqual(replay.fall_intervals(self.ROW), [(5.0, 6.0)])
        self.assertEqual(replay.fall_intervals({"label": "adl", "segments": row["segments"]}), [])

    def test_level1_without_entry_is_not_timely(self):
        t = replay.timing(self.ROW, [{"type": "level1", "ts": 1010.0}])
        self.assertEqual((t["timely_strict"], t["untimely"]), (False, "no_entry"))

    def test_timed_gate_does_not_depend_on_alarms(self):
        """구간 라벨이 없는 낙상이 하나라도 있으면 알람 여부와 관계없이 시간 기준 지표는 None"""
        unlabeled = {"clip_id": "d/u", "dataset": "d", "label": "fall"}
        for ev in ([], self._ev(1.0, 11.0)):
            outs = [replay.clip_outcome(self.ROW, self._ev(4.0, 14.0)), replay.clip_outcome(unlabeled, ev)]
            self.assertFalse(outs[1]["has_interval"])
            self.assertIsNone(replay.summarize(outs)["timely_recall"])

    def test_summarize_timely(self):
        def o(row, events):
            return replay.clip_outcome(row, events)
        fall2 = dict(self.ROW, clip_id="d/g")
        outs = [o(self.ROW, self._ev(4.0, 14.0)),        # 제때
                o(fall2, self._ev(1.0, 11.0)),           # 탐지했지만 제때 아님
                o(dict(self.ROW, clip_id="d/h"), []),    # 놓침
                o({"clip_id": "d/a", "dataset": "d", "label": "adl"}, [])]
        s = replay.summarize(outs)
        self.assertEqual((s["n_detected"], s["n_timely"], s["n_timely_loose"]), (2, 1, 1))
        self.assertAlmostEqual(s["timely_recall"], 1 / 3)
        self.assertAlmostEqual(s["timely_balanced"], 2 * (1 / 3) * 1.0 / (1 / 3 + 1.0))
        # 옛 결과 파일(시간 필드 없음)은 None
        old = [{k: v for k, v in x.items() if k in ("clip_id", "dataset", "label", "alarm", "suppressed",
                                                     "reached_ai")} for x in outs]
        self.assertIsNone(replay.summarize(old)["timely_recall"])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class SwitchReplayTest(unittest.TestCase):
    def test_switch_flags_cover_skipped_frames(self):
        sw = np.zeros(10, dtype=bool)
        sw[3] = True    # 고르지 않은 프레임의 전환은 다음에 고른 프레임에
        sw[8] = True
        flags = replay._switch_flags(sw, np.array([0, 2, 4, 6, 8]), 10)
        self.assertEqual(flags.tolist(), [False, False, True, False, True])
        self.assertFalse(replay._switch_flags(None, np.array([0, 1]), 2).any())
        self.assertFalse(replay._switch_flags(np.ones(3, bool), np.array([0, 1]), 2).any())  # 길이 불일치

    def test_clip_to_obs_sets_target_switched(self):
        clip, _ = _synth_clip()
        t = len(clip["ts"])
        base = replay.clip_to_obs(clip)
        self.assertFalse(any(o["target_switched"] for o in base))
        sw = np.zeros(t, dtype=bool)
        sw[t // 2] = True
        obs = replay.clip_to_obs(dict(clip, switched=sw))
        hit = [o for o in obs if o["target_switched"]]
        self.assertEqual(len(hit), 1)
        self.assertFalse(hit[0]["meta"]["replay_tail"])
        self.assertGreaterEqual(hit[0]["meta"]["replay_src_index"], t // 2)


@unittest.skipUnless(HAS_TORCH, "numpy/torch not installed")
class VerifierTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @staticmethod
    def _v2_model(bias=(0.0, 0.0)):
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        m = nn.Sequential(nn.Conv1d(92, 4, 1), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(4, 2))
        if bias != (0.0, 0.0):
            with torch.no_grad():
                m[3].weight.zero_()
                m[3].bias.copy_(torch.tensor(bias))
        return m

    def test_v1_matches_runtime_classifier(self):
        import torch
        from app.ai_classifier.inference import FallClassifier
        from app.ai_classifier.model import Fall1DCNN
        from training.experiments.verifiers import V1SeqVerifier

        torch.manual_seed(0)
        model = Fall1DCNN().eval()
        # 운영 분류기 객체를 가중치 파일 없이 같은 모델로 구성
        clf = FallClassifier.__new__(FallClassifier)
        clf.enabled, clf.model, clf._torch, clf.device, clf.seq_len = True, model, torch, torch.device("cpu"), 30
        rng = np.random.default_rng(0)
        seq = rng.random((35, 34)).tolist()
        seq[-3] = [0.0] * 34  # 미검출 프레임 포함
        v = V1SeqVerifier(model, "cpu", model_id="m1")
        p = v(seq)
        self.assertAlmostEqual(p, clf.predict_proba(seq), places=6)
        self.assertEqual(v.model_id, "m1")
        self.assertIsNone(v(seq[:29]))
        self.assertEqual((v.calls, v.n_none), (2, 1))

    def test_event_window_fail_open(self):
        from training.experiments.verifiers import EventWindowVerifier, event_window_to_v2

        model = self._v2_model()
        self.assertIsNone(EventWindowVerifier(model, "cpu", engine_ref=lambda: None)([]))

        class _Eng:
            def __init__(self, win):
                self.win = win

            def get_incident_snapshot(self):
                return {"event_window": self.win}

        self.assertIsNone(EventWindowVerifier(model, "cpu", engine_ref=lambda: _Eng(None))([]))
        self.assertIsNone(EventWindowVerifier(model, "cpu", engine_ref=lambda: _Eng({"ts": []}))([]))
        empty = {"t_start": 0.0, "t_end": 4.0, "ts": [0.0, 1.0, 2.0], "kps": [None] * 3, "bbox": [None] * 3,
                 "frame_wh": [640, 480]}
        self.assertIsNone(EventWindowVerifier(model, "cpu", engine_ref=lambda: _Eng(empty))([]))
        self.assertIsNone(event_window_to_v2(None))
        # 사람이 창 앞 여유(pad) 구간에만 있음: 창 안 관절 마스크가 모두 0 -> 판단 불가
        frame = [[0.5, 0.5, 0.9]] * 17
        ts = [0.75] + [1.0 + i / 15.0 for i in range(61)]
        pad_only = {"t_start": 1.0, "t_end": 5.0, "ts": ts, "kps": [frame] + [None] * 61,
                    "bbox": [[0.4, 0.2, 0.6, 0.9]] + [None] * 61, "frame_wh": [640, 480]}
        self.assertIsNone(event_window_to_v2(pad_only))
        visible = dict(pad_only, kps=[frame] * 62, bbox=[[0.4, 0.2, 0.6, 0.9]] * 62)
        self.assertEqual(event_window_to_v2(visible).shape, (92, 60))

    def test_event_window_verifier_in_replay(self):
        from training.experiments.verifiers import EventWindowVerifier

        clip, _ = _synth_clip("fast_fall", lie_s=3.0)
        obs = replay.clip_to_obs(clip)
        row = {"clip_id": "synth/fast_fall", "dataset": "synth", "label": "fall"}

        # 중립 모델: 사건 구간을 v2 (92, 60)로 받아 p_fall 계산 (run_clip이 엔진을 자동 연결)
        v = EventWindowVerifier(self._v2_model(), "cpu", model_id="gap")
        events, engine = replay.run_clip(obs, _params("veto"), verifier=v)
        self.assertEqual(v.calls, 1)
        self.assertEqual(v.last_x.shape, (92, 60))
        self.assertTrue(np.all(np.isfinite(v.last_x)))
        self.assertTrue(0.0 <= v.last_p <= 1.0)
        judged = [e for e in events if e["type"] in ("level1", "ai_suppressed")]
        self.assertEqual(judged[0]["values"]["ai"]["model_id"], "gap")
        self.assertAlmostEqual(judged[0]["values"]["ai"]["p_fall"], v.last_p)

        # ADL로 치우친 모델: 억제되고, 정지 꼬리(20초)는 승격(60초)까지 가지 않으므로 알람 없음
        v_adl = EventWindowVerifier(self._v2_model(bias=(5.0, -5.0)), "cpu")
        events, _ = replay.run_clip(obs, _params("veto"), verifier=v_adl)
        out = replay.clip_outcome(row, events)
        self.assertTrue(out["suppressed"])
        self.assertFalse(out["alarm"])
        self.assertTrue(out["reached_ai"])
        self.assertLess(out["p_fall"][0], 0.4)
        self.assertEqual(replay.summarize([out])["suppressed_true_falls"], 1)

        # 같은 모델이라도 shadow 모드면 알람 유지
        events, _ = replay.run_clip(obs, _params("shadow"), verifier=v_adl)
        self.assertTrue(replay.clip_outcome(row, events)["alarm"])

        # engine_ref가 주어지면 그것이 우선 (사건 구간 없음 -> None -> fail-open으로 알람)
        v_none = EventWindowVerifier(self._v2_model(bias=(5.0, -5.0)), "cpu", engine_ref=lambda: None)
        events, _ = replay.run_clip(obs, _params("veto"), verifier=v_none)
        out = replay.clip_outcome(row, events)
        self.assertTrue(out["alarm"])
        self.assertFalse(out["suppressed"])
        self.assertEqual(v_none.n_none, 1)


@unittest.skipUnless(HAS_SKLEARN, "numpy/sklearn not installed")
class FoldsTest(unittest.TestCase):
    @staticmethod
    def _rows():
        rows = []
        for ds in ("le2i", "caucafall"):
            for g in range(12):
                label = "fall" if g % 3 else "adl"
                for cam in range(2 if ds == "caucafall" else 1):
                    rows.append({"clip_id": f"{ds}/e{g}/cam{cam}", "dataset": ds, "label": label,
                                 "event_group": f"{ds}/e{g}"})
        for s in range(1, 8):
            for i, label in enumerate(("adl", "fall", "fall")):
                cid = f"figshare_ursul29/{label.upper()}/SBJ_{s:02d}_LOC1/ACT{i}/{s}{i}"
                rows.append({"clip_id": cid, "dataset": "figshare_ursul29", "label": label,
                             "subject": str(s), "event_group": cid})
        return rows

    def test_build_split_save_load(self):
        from training.experiments import folds as F

        rows = self._rows()
        f = F.build_folds(rows, n_folds=3, seed=0)
        self.assertEqual(set(f), {r["clip_id"] for r in rows})
        self.assertEqual(set(f.values()), {0, 1, 2})
        self.assertEqual(f, F.build_folds(list(reversed(rows)), n_folds=3, seed=0))
        by_group = {}
        for r in rows:
            by_group.setdefault(F.fold_group(r), set()).add(f[r["clip_id"]])
        self.assertTrue(all(len(v) == 1 for v in by_group.values()))
        self.assertEqual(F.fold_group(rows[-1]), "figshare/SBJ_07")
        self.assertEqual(sum(1 for g in by_group if g.startswith("figshare/")), 7)

        train, test = F.split_fold(rows, f, 1)
        self.assertTrue(test)
        self.assertTrue(all(r["dataset"] in F.EVAL_DATASETS and f[r["clip_id"]] == 1 for r in test))
        self.assertTrue(all(f[r["clip_id"]] != 1 for r in train))
        self.assertTrue(any(r["dataset"] == "figshare_ursul29" for r in train))

        tr, va = F.split_train_val(train, frac=0.15, seed=0)
        self.assertEqual(len(tr) + len(va), len(train))
        self.assertTrue(va)
        self.assertFalse({F.fold_group(r) for r in tr} & {F.fold_group(r) for r in va})
        self.assertEqual((tr, va), F.split_train_val(train, frac=0.15, seed=0))

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "folds.json"
            F.save_folds(p, f, {"seed": 0})
            self.assertEqual(F.load_folds(p), f)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["n_folds"], 3)
            self.assertEqual(data["meta"], {"seed": 0})


if __name__ == "__main__":
    unittest.main()
