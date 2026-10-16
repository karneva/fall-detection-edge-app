"""
E* 단계(진입 이후 여러 창 판정, 재학습 없음) 테스트

- 단계 등록, paired 비교, 정지 꼬리 분리, 표 렌더링: 표준 라이브러리만 (numpy 없이도 실행)
- 창 시작 시각, SlidingWindowVerifier, 재생 연동: numpy/torch가 있을 때만
"""
import importlib.util
import json
import logging
import unittest

from training.experiments import run_progression as RP

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = HAS_NUMPY and importlib.util.find_spec("torch") is not None


def _fake_summarize(outcomes):
    """replay.summarize의 필요한 키만 표준 라이브러리로 계산 (timing 필드는 없음)"""
    def stats(os_):
        f = [o for o in os_ if o["label"] == "fall"]
        a = [o for o in os_ if o["label"] == "adl"]
        det = sum(o["alarm"] for o in f)
        fa = sum(o["alarm"] for o in a)
        r = det / len(f) if f else None
        far = fa / len(a) if a else None
        bal = None if r is None or far is None else (0.0 if r + 1 - far <= 0 else 2 * r * (1 - far) / (r + 1 - far))
        return {"n_fall": len(f), "n_adl": len(a), "n_detected": det, "n_adl_alarm": fa, "recall": r,
                "adl_fa_rate": far, "balanced": bal,
                "suppressed_true_falls": sum(1 for o in f if o["suppressed"] and not o["alarm"]),
                "rules_ceiling": (sum(o["reached_ai"] for o in f) / len(f)) if f else None}
    out = stats(outcomes)
    out["by_dataset"] = {d: stats([o for o in outcomes if o["dataset"] == d])
                         for d in sorted({o["dataset"] for o in outcomes})}
    return out


def _o(cid, label, alarm, suppressed=False, in_tail=None, ds="le2i"):
    o = {"clip_id": cid, "dataset": ds, "label": label, "alarm": alarm, "suppressed": suppressed,
         "reached_ai": True, "p_fall": [0.5]}
    if in_tail is not None or alarm:
        o["in_tail"] = in_tail
    return o


def _fold(k, runs):
    return {"fold": k, "params": 1, "time_s": 1.0,
            "runs": [{"seed": s, "outcomes": outs, "threshold": 0.4} for s, outs in runs]}


class StageTest(unittest.TestCase):
    def test_registered_and_filesystem_safe(self):
        self.assertIn("Es", RP.STAGES)
        self.assertLess(RP.STAGES.index("E"), RP.STAGES.index("Es"))
        for v in RP.STAGE_VARIANTS["Es"]:
            self.assertIn(v, RP.VARIANTS)
            self.assertIn(v, RP.VARIANT_INFO)
            self.assertTrue(v.isalnum(), v)  # Windows 파일 이름에 쓸 수 있음 ('*' 없음)
            base, mode = RP.ES_BASE[v]
            self.assertIn(base, RP.STAGE_VARIANTS["E"])
            self.assertIn(mode, ("max", "last"))
        self.assertEqual(RP.ES_BASE["E1max"], ("E1", "max"))
        self.assertEqual(RP.ES_BASE["E2last"], ("E2", "last"))

    def test_parse_stages_case(self):
        self.assertEqual(RP.parse_stages("es"), ["Es"])
        self.assertEqual(RP.parse_stages("F,Es,E,a"), ["A", "E", "Es", "F"])
        with self.assertRaises(ValueError):
            RP.parse_stages("E*")


class PairedTest(unittest.TestCase):
    def setUp(self):
        # E: f1 억제로 놓침, f2 탐지, f3 억제로 놓침, a1 알람 없음, a2 알람
        self.base = [_fold(0, [(0, [_o("f1", "fall", False, True), _o("f2", "fall", True, in_tail=False),
                                    _o("a1", "adl", False), _o("a2", "adl", True, in_tail=True)]),
                               (1, [_o("f1", "fall", False, True), _o("f2", "fall", True, in_tail=False),
                                    _o("a1", "adl", False), _o("a2", "adl", True, in_tail=True)])]),
                     _fold(1, [(0, [_o("f3", "fall", False, True), _o("a3", "adl", False)]),
                               (1, [_o("f3", "fall", False, True), _o("a3", "adl", False)])])]
        # E*: seed 0은 f1 회복 + a1 새 알람, seed 1은 f1, f3 회복
        self.new = [_fold(0, [(0, [_o("f1", "fall", True, True, in_tail=True), _o("f2", "fall", True, in_tail=False),
                                   _o("a1", "adl", True, in_tail=True), _o("a2", "adl", True, in_tail=True)]),
                              (1, [dict(_o("f1", "fall", True, True, in_tail=False), timely_strict=True),
                                   _o("f2", "fall", True, in_tail=False),
                                   _o("a1", "adl", False), _o("a2", "adl", True, in_tail=True)])]),
                    _fold(1, [(0, [_o("f3", "fall", False, True), _o("a3", "adl", False)]),
                              (1, [_o("f3", "fall", True, True, in_tail=True), _o("a3", "adl", False)])])]

    def test_paired_compare(self):
        pc = RP.paired_compare(self.base, self.new, summarize_fn=_fake_summarize)
        self.assertEqual(pc["folds"], [0, 1])
        self.assertEqual(pc["n_seeds"], 2)
        c = pc["clips"]
        self.assertAlmostEqual(c["base_supp_true"], 2.0)   # seed마다 f1, f3
        self.assertAlmostEqual(c["recovered"], 1.5)        # seed 0: 1, seed 1: 2
        self.assertAlmostEqual(c["recovered_tail"], 1.0)   # seed 0: f1, seed 1: f3
        self.assertAlmostEqual(c["recovered_timely"], 0.5)  # seed 1의 f1만 제때
        self.assertAlmostEqual(c["fall_gained"], 1.5)
        self.assertAlmostEqual(c["fall_lost"], 0.0)
        self.assertAlmostEqual(c["adl_new_alarm"], 0.5)
        self.assertAlmostEqual(c["adl_removed"], 0.0)
        d = pc["delta"]
        # fold 0 recall: E 0.5 -> E* 1.0, fold 1: 0 -> 0.5
        self.assertAlmostEqual(d["recall"]["values"][0], 0.5)
        self.assertAlmostEqual(d["recall"]["values"][1], 0.5)
        self.assertEqual((d["recall"]["n_better"], d["recall"]["n_worse"]), (2, 0))
        # fold 0 ADL 오탐률 0.5 -> 0.75 (나빠짐), fold 1 0 -> 0
        self.assertEqual((d["adl_fa_rate"]["n_better"], d["adl_fa_rate"]["n_worse"]), (0, 1))
        self.assertIsNone(RP.paired_compare(self.base[:1], self.new[1:], summarize_fn=_fake_summarize))

    def test_tail_split(self):
        outs = [_o("f1", "fall", True, in_tail=True), _o("f2", "fall", True, in_tail=False),
                _o("f3", "fall", False), _o("a1", "adl", True, in_tail=True)]
        legacy = {"clip_id": "x", "dataset": "le2i", "label": "adl", "alarm": True}  # 옛 결과: in_tail 없음
        self.assertEqual(RP._tail_split(outs + [legacy]),
                         {"fall_live": 1, "fall_tail": 1, "adl_live": 0, "adl_tail": 1, "unknown": 1})

    def test_summary_and_markdown(self):
        s = RP.build_summary({"E1": self.base, "E1max": self.new}, summarize_fn=_fake_summarize)
        self.assertEqual(list(s["variants"]), ["E1", "E1max"])
        self.assertEqual(set(s["paired"]), {"E1max"})
        self.assertEqual(s["paired"]["E1max"]["base"], "E1")
        self.assertEqual(s["variants"]["E1max"]["folds"], [0, 1])
        md = RP.render_markdown(s)
        self.assertIn("| E1max | E* 진입 이후 최대 p_fall (E1)", md)
        self.assertIn("## 제때 탐지와 정지 꼬리", md)
        self.assertIn("## E* 대 E", md)
        self.assertIn("| E1max | E1 | +", md)
        self.assertIn("1.5/2.0 | 0.5/1.0 |", md)   # 회복/억제, 회복 중 제때/꼬리
        # timing 필드가 없는 요약 함수 -> 제때 지표는 '-'
        sec = md.split("## 제때 탐지와 정지 꼬리")[1].split("## ")[0]
        row = next(line for line in sec.splitlines() if line.startswith("| E1max | -"))
        self.assertIn("1.5 / 1.0", row)  # 탐지 낙상 실제/꼬리 (seed 평균)
        json.dumps(s)


@unittest.skipUnless(HAS_TORCH, "numpy/torch not installed")
class SlidingWindowTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_window_starts(self):
        from training.experiments.verifiers import entry_window_starts
        self.assertEqual(entry_window_starts(0.0, 6.0), [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(entry_window_starts(0.0, 6.2), [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(entry_window_starts(0.0, 3.0), [0.0])   # 창 하나보다 짧아도 첫 창
        self.assertEqual(entry_window_starts(10.0, 14.0), [10.0])

    @staticmethod
    def _engine(n_s=12.0, anchor=1000.0 + 2.0, horizon=20.0):
        """TimedKeypointBuffer에 15Hz 프레임을 채운 가짜 엔진 (event_window는 첫 창과 같은 규칙으로 캡처)"""
        from collections import deque
        from types import SimpleNamespace

        from app.core.kp_buffer import TimedKeypointBuffer
        from app.core.level1 import Level1Params

        p = Level1Params()
        buf = TimedKeypointBuffer(horizon)
        kp_ts = deque(maxlen=30)
        t0 = 1000.0
        for i in range(int(n_s * 15) + 1):
            ts = t0 + i / 15.0
            frame = [[0.5, 0.3 + 0.02 * j + 0.001 * i, 0.9] for j in range(17)]
            buf.append(ts, frame, [0.4, 0.2, 0.6, 0.9], frame_wh=(640, 480))
            kp_ts.append(ts)
        win = buf.window(anchor - p.event_window_pre_s, anchor + p.event_window_post_s, pad_s=p.event_window_pad_s)
        eng = SimpleNamespace(p=p, kp_buffer=buf, kp_ts=kp_ts, event_anchor_ts=anchor, abnormal_start_ts=anchor,
                              get_incident_snapshot=lambda: {"event_window": win})
        return eng, t0 + int(n_s * 15) / 15.0

    def test_max_and_last(self):
        import numpy as np

        from training.experiments.verifiers import SlidingWindowVerifier, entry_window_starts, event_window_to_v2

        eng, ts = self._engine()
        seen = []

        def score(X):
            seen.append(X.copy())
            return np.linspace(0.1, 0.9, len(X)) if len(X) > 1 else np.array([0.3])

        v = SlidingWindowVerifier(score, mode="max", engine_ref=lambda: eng)
        p = v([])
        n_exp = len(entry_window_starts(eng.event_anchor_ts - eng.p.event_window_pre_s, ts))
        self.assertGreater(n_exp, 5)
        self.assertEqual(seen[-1].shape, (n_exp, 92, 60))
        self.assertAlmostEqual(p, 0.9)
        # 첫 창은 진입 사건 구간 그대로 (E와 같은 입력)
        x0 = event_window_to_v2(eng.get_incident_snapshot()["event_window"])
        np.testing.assert_array_equal(seen[-1][0], x0)
        self.assertEqual(v.stats()["max_later"], 1)
        self.assertEqual(v.stats()["windows"], n_exp)

        v_last = SlidingWindowVerifier(score, mode="last", engine_ref=lambda: eng)
        self.assertAlmostEqual(v_last([]), 0.3)
        self.assertEqual(seen[-1].shape, (1, 92, 60))
        x_last = event_window_to_v2(eng.kp_buffer.window(ts - 4.0, ts, pad_s=eng.p.event_window_pad_s))
        np.testing.assert_array_equal(seen[-1][0], x_last)

        with self.assertRaises(ValueError):
            SlidingWindowVerifier(score, mode="mean")

    def test_truncated_windows_skipped_and_fail_open(self):
        import numpy as np

        from training.experiments.verifiers import SlidingWindowVerifier

        # 버퍼 보관 시간(5초)보다 긴 에피소드: 버퍼 앞이 잘린 창은 건너뜀 (첫 창은 캡처해 둔 사건 구간)
        eng, ts = self._engine(n_s=12.0, horizon=5.0)
        got = []
        v = SlidingWindowVerifier(lambda X: got.append(len(X)) or np.full(len(X), 0.2), engine_ref=lambda: eng)
        self.assertAlmostEqual(v([]), 0.2)
        self.assertGreater(v.stats()["windows_skipped"], 0)
        self.assertEqual(got[-1], v.stats()["windows"])

        self.assertIsNone(SlidingWindowVerifier(lambda X: X, engine_ref=lambda: None)([]))
        eng.get_incident_snapshot = lambda: {"event_window": None}
        eng.kp_buffer.clear()
        v2 = SlidingWindowVerifier(lambda X: np.ones(len(X)), engine_ref=lambda: eng)
        self.assertIsNone(v2([]))
        self.assertEqual(v2.n_none, 1)

    def test_in_replay_first_window_matches_event_verifier(self):
        import numpy as np
        import torch
        import torch.nn as nn

        from app.core.level1 import Level1Params
        from app.eval.synth import make_scenario
        from training.datasets.clipfmt import from_obs_records
        from training.experiments import replay
        from training.experiments.verifiers import EventWindowVerifier, SlidingWindowVerifier, nn_batch_scorer

        torch.manual_seed(0)
        m = nn.Sequential(nn.Conv1d(92, 4, 1), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(4, 2)).eval()
        obs_rec, _ = make_scenario("fast_fall", fps=30.0, seed=0, lie_s=3.0)
        obs = replay.clip_to_obs(from_obs_records(obs_rec, dataset="synth", clip_id="synth/fast_fall"))
        params = replay.study_params("shadow")
        self.assertIsInstance(params, Level1Params)

        ve = EventWindowVerifier(m, "cpu")
        replay.run_clip(obs, params, verifier=ve)
        vm = SlidingWindowVerifier(nn_batch_scorer(m, "cpu"), mode="max")
        events, _ = replay.run_clip(obs, params, verifier=vm)
        self.assertEqual(vm.calls, 1)
        self.assertGreater(len(vm.last_ps), 1)
        self.assertEqual(vm.last_ps[0], ve.last_p)   # 배치 1 경로라 E와 비트 단위로 같음
        self.assertAlmostEqual(vm.last_p, max(vm.last_ps))
        self.assertGreaterEqual(vm.last_p, ve.last_p - 1e-6)
        judged = [e for e in events if e["type"] in ("level1", "ai_suppressed")]
        self.assertAlmostEqual(judged[0]["values"]["ai"]["p_fall"], vm.last_p)

        # batched=True도 같은 값 (CPU에서는 근사적으로)
        X = np.random.default_rng(0).normal(size=(3, 92, 60)).astype(np.float32)
        np.testing.assert_allclose(nn_batch_scorer(m, "cpu", batched=True)(X), nn_batch_scorer(m, "cpu")(X),
                                   rtol=0, atol=1e-6)

        vl = SlidingWindowVerifier(nn_batch_scorer(m, "cpu"), mode="last")
        replay.run_clip(obs, params, verifier=vl)
        self.assertEqual(len(vl.last_ps), 1)


if __name__ == "__main__":
    unittest.main()
