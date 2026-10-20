"""
training/bench/pose_bench.py 순수 함수 (표준 라이브러리만, numpy/torch 없이 실행)

지연 통계, 상자 IoU, OKS 유사 일치도, 프레임 일치도 요약, 회전 폴백 시도 재구성, 실효 지연, 서 있는 시각 고르기, 표.
"""
import math
import random
import unittest

from training.bench import pose_bench as B


class LatencyStatsTest(unittest.TestCase):
    def test_mean_percentiles_fps(self):
        s = B.latency_stats([10, 20, 30, 40])
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["mean_ms"], 25.0)
        self.assertAlmostEqual(s["p50_ms"], 25.0)          # numpy linear 보간과 같음
        self.assertAlmostEqual(s["p95_ms"], 38.5)
        self.assertAlmostEqual(s["fps"], 40.0)

    def test_single_and_empty(self):
        s = B.latency_stats([5.0])
        self.assertEqual((s["p50_ms"], s["p95_ms"]), (5.0, 5.0))
        with self.assertRaises(ValueError):
            B.latency_stats([])


class GeometryTest(unittest.TestCase):
    def test_box_iou(self):
        self.assertAlmostEqual(B.box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(B.box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)
        self.assertEqual(B.box_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)
        self.assertEqual(B.box_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    def test_oks_identical_and_shifted(self):
        ref = [[10.0 * i, 5.0, 0.9] for i in range(17)]
        self.assertAlmostEqual(B.oks(ref, ref, area=1000.0), 1.0)
        # 관절 하나만 d만큼 이동: (16 + exp(-d^2 / (2 area (2 sigma)^2))) / 17
        moved = [list(k) for k in ref]
        moved[0][0] += 3.0
        want = (16 + math.exp(-9.0 / (2 * 1000.0 * (2 * B.COCO_SIGMAS[0]) ** 2))) / 17
        self.assertAlmostEqual(B.oks(ref, moved, area=1000.0), want)

    def test_oks_uses_only_visible_reference_joints(self):
        ref = [[0.0, 0.0, 0.9]] + [[0.0, 0.0, 0.1]] * 16
        far = [[0.0, 0.0, 0.9]] + [[500.0, 500.0, 0.9]] * 16
        self.assertAlmostEqual(B.oks(ref, far, area=100.0), 1.0)
        self.assertIsNone(B.oks([[0.0, 0.0, 0.1]] * 17, far, area=100.0))
        self.assertIsNone(B.oks(ref, far, area=0.0))


def _det(boxes, confs, kps=None):
    kps = kps or [[[b[0], b[1], 0.9]] * 17 for b in boxes]
    return {"boxes": boxes, "conf": confs, "kps": kps}


class AgreementTest(unittest.TestCase):
    def test_reference_top_conf_matched_by_iou(self):
        ref = _det([[0, 0, 10, 10], [100, 100, 120, 120]], [0.6, 0.9])
        var = _det([[0, 0, 10, 10], [101, 100, 121, 120]], [0.95, 0.7])
        a = B.frame_agreement(ref, var)
        self.assertTrue(a["ref_det"] and a["var_det"])
        self.assertAlmostEqual(a["iou"], B.box_iou([100, 100, 120, 120], [101, 100, 121, 120]))
        self.assertLess(a["oks"], 1.0)

    def test_missing_detection(self):
        a = B.frame_agreement(_det([[0, 0, 10, 10]], [0.9]), _det([], []))
        self.assertEqual((a["ref_det"], a["var_det"], a["iou"], a["oks"]), (True, False, None, None))

    def test_summary(self):
        recs = [{"ref_det": True, "var_det": True, "iou": 1.0, "oks": 1.0},
                {"ref_det": True, "var_det": True, "iou": 0.5, "oks": 0.8},
                {"ref_det": True, "var_det": False, "iou": None, "oks": None},
                {"ref_det": False, "var_det": False, "iou": None, "oks": None}]
        s = B.summarize_agreement(recs)
        self.assertEqual((s["n"], s["n_both"]), (4, 2))
        self.assertAlmostEqual(s["ref_det_rate"], 0.75)
        self.assertAlmostEqual(s["var_det_rate"], 0.5)
        self.assertAlmostEqual(s["det_agree_rate"], 0.75)
        self.assertAlmostEqual(s["oks_mean"], 0.9)
        self.assertAlmostEqual(s["iou_mean"], 0.75)
        self.assertAlmostEqual(s["oks_ge_0p9_rate"], 0.5)
        self.assertEqual(B.summarize_agreement([]), {"n": 0})


class FallbackTest(unittest.TestCase):
    def test_attempts_follow_runtime_condition(self):
        # fps 1: 0 보임, 1~2 후보 0명(시도), 2는 폴백 성공(present), 3 후보 있음, 4~ 후보 0명
        present = [True, False, True, True, False, False, False, False, False, False, False]
        n_cand = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        att = B.fallback_attempts(present, n_cand, fps=1.0, recent_s=5.0)
        # 마지막으로 본 프레임 3 -> 4..8 (5초 이내) 시도, 9부터 안 함
        self.assertEqual(att, [False, True, True, False, True, True, True, True, True, False, False])

    def test_no_attempt_before_first_sighting(self):
        self.assertEqual(B.fallback_attempts([False, False], [0, 0], fps=30.0), [False, False])

    def test_effective_ms(self):
        e = B.effective_ms(10.0, 20.0, 0.1)
        self.assertAlmostEqual(e["ms"], 12.0)
        self.assertAlmostEqual(e["fps"], 1000 / 12.0, places=2)
        self.assertAlmostEqual(B.effective_ms(10.0, 20.0, 1.0)["ms"], 30.0)


class UprightPickTest(unittest.TestCase):
    def test_picks_inside_upright_segment_before_fall(self):
        segs = [{"cls": "walk", "start_s": 0.0, "end_s": 2.0}, {"cls": "sitting", "start_s": 2.0, "end_s": 4.0},
                {"cls": "fall", "start_s": 4.0, "end_s": 5.0}]
        rng = random.Random(0)
        for _ in range(50):
            t, src = B.pick_upright_time(segs, 4.0, rng)
            self.assertEqual(src, "segment")
            self.assertTrue(0.2 <= t <= 1.8, t)

    def test_segment_clipped_at_fall_start(self):
        segs = [{"cls": "standing", "start_s": 0.0, "end_s": 10.0}]
        rng = random.Random(1)
        for _ in range(50):
            t, _ = B.pick_upright_time(segs, 3.0, rng)
            self.assertTrue(0.2 <= t <= 2.8, t)

    def test_unlabeled_lead_in(self):
        segs = [{"cls": "fall", "start_s": 2.7, "end_s": 3.7}, {"cls": "lying", "start_s": 3.7, "end_s": 5.3}]
        t, src = B.pick_upright_time(segs, 2.7, random.Random(0))
        self.assertEqual(src, "lead_in")
        self.assertTrue(0.2 <= t <= 2.2, t)
        # 낙상 직전이 너무 짧으면 없음
        self.assertEqual(B.pick_upright_time(segs[:1], 0.6, random.Random(0)), (None, None))

    def test_labeled_non_upright_before_fall_is_not_lead_in(self):
        segs = [{"cls": "sitting", "start_s": 0.0, "end_s": 3.0}, {"cls": "fall", "start_s": 3.0, "end_s": 4.0}]
        self.assertEqual(B.pick_upright_time(segs, 3.0, random.Random(0)), (None, None))


class TableTest(unittest.TestCase):
    def test_md_table(self):
        s = B.md_table(["a", "b"], [[1, None], ["x", 2.5]])
        self.assertEqual(s.splitlines(), ["| a | b |", "|---|---|", "| 1 | - |", "| x | 2.5 |"])

    def test_variants_are_consistent(self):
        self.assertIn(B.REFERENCE, B.VARIANTS)
        for name, v in B.VARIANTS.items():
            self.assertIn(v["device"], ("gpu", "cpu"), name)
            self.assertIn(v["fmt"], ("pt", "onnx", "engine"), name)
        for name in B.FALLBACK_COST_VARIANTS:
            self.assertEqual(B.VARIANTS[name]["device"], "gpu")
        # GPU 블록 순서는 GPU 변형 전부를 한 번씩, n과 s가 번갈아
        gpu = {n for n, v in B.VARIANTS.items() if v["device"] == "gpu"}
        self.assertEqual(set(B.GPU_ORDER), gpu)
        self.assertEqual(len(B.GPU_ORDER), len(gpu))
        self.assertEqual([B.VARIANTS[n]["model"] for n in B.GPU_ORDER[:4]], ["s", "n", "s", "n"])
        for name in B.HIRES_VARIANTS:
            self.assertEqual(B.VARIANTS[name]["device"], "gpu")
        self.assertEqual(B.VARIANTS["s_ort_int8u8pc_cpu"]["device"], "cpu")


class RoundsTest(unittest.TestCase):
    def test_round_spread(self):
        self.assertAlmostEqual(B.round_spread([9.3, 13.5]), round(4.2 / 9.3, 4))
        self.assertAlmostEqual(B.round_spread([10.0, 10.0]), 0.0)
        self.assertIsNone(B.round_spread([10.0]))
        self.assertIsNone(B.round_spread([0.0, 1.0]))
        # 15% 경계
        self.assertLessEqual(B.round_spread([10.0, 11.5]), B.UNSTABLE_REL)
        self.assertGreater(B.round_spread([10.0, 11.6]), B.UNSTABLE_REL)

    def test_parse_wh(self):
        self.assertEqual(B.parse_wh("1920x1080"), (1920, 1080))
        self.assertEqual(B.parse_wh("640X480"), (640, 480))
        for bad in ("1920", "0x10", "ax b"):
            with self.assertRaises(ValueError):
                B.parse_wh(bad)


if __name__ == "__main__":
    unittest.main()
