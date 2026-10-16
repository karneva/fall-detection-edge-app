"""
training/datasets/data_card.py (데이터 카드 통계) 검증. 합성 데이터만 사용합니다.
순수 계산은 numpy 없이, 키포인트 통계는 numpy가 있을 때만 확인합니다.
"""
import importlib.util
import math
import unittest

from training.datasets import data_card as D

HAS_NUMPY = importlib.util.find_spec("numpy") is not None

FRAME_WH = (640, 480)
TORSO_PX = 60.0

# 골반 중점 원점, 몸통 길이 1 단위, y는 아래 방향 (서 있는 자세)
_SKELETON = [
    (0.0, -1.6), (0.08, -1.7), (-0.08, -1.7), (0.16, -1.65), (-0.16, -1.65),
    (0.3, -1.0), (-0.3, -1.0), (0.4, -0.5), (-0.4, -0.5), (0.45, 0.0), (-0.45, 0.0),
    (0.15, 0.0), (-0.15, 0.0), (0.17, 0.9), (-0.17, 0.9), (0.18, 1.8), (-0.18, 1.8),
]


def make_clip(duration_s=6.0, fps=30.0, drop_at=2.0, drop_s=0.5, drop_px=120.0, absent=()):
    """
    합성 클립 (clipfmt 형식): drop_at초부터 drop_s초 동안 몸 전체가 drop_px만큼 일정 속도로 내려감.
    골반 하강 속도 = drop_px / drop_s / TORSO_PX (몸통 길이/초). absent: 사람이 없는 시각 구간 [(a, b)].
    """
    import numpy as np
    n = int(round(duration_s * fps)) + 1
    ts = np.arange(n) / fps
    base = np.array(_SKELETON) * TORSO_PX
    w, h = FRAME_WH
    kps = np.zeros((n, 17, 3), dtype=np.float32)
    bbox = np.zeros((n, 4), dtype=np.float32)
    present = np.ones(n, dtype=bool)
    for i, t in enumerate(ts):
        if any(a <= t <= b for a, b in absent):
            present[i] = False
            continue
        dy = drop_px * min(max((t - drop_at) / drop_s, 0.0), 1.0)
        px = 320.0 + base[:, 0]
        py = 200.0 + dy + base[:, 1]
        kps[i, :, 0] = px / w
        kps[i, :, 1] = py / h
        kps[i, :, 2] = 0.9
        bbox[i] = [px.min() / w, py.min() / h, px.max() / w, py.max() / h]
    meta = {"native_fps": fps, "fps_source": "container", "container_fps": fps,
            "n_target_switch": 0, "n_multi_person_frames": 0, "crop_px": None}
    return {"ts": ts, "kps": kps, "bbox": bbox, "present": present,
            "frame_wh": np.array(FRAME_WH, dtype=np.int32), "meta": meta}


def make_row(clip_id="le2i/x/video_1", dataset="le2i", label="fall", segments=None, event_group=None,
             subject="1", cam="1"):
    segs = segments if segments is not None else (
        [{"cls": "walk", "start_s": 0.0, "end_s": 2.0}, {"cls": "fall", "start_s": 2.0, "end_s": 2.5},
         {"cls": "fallen", "start_s": 2.5, "end_s": 6.0}] if label == "fall" else [])
    falls = [s for s in segs if s["cls"] == "fall"]
    return {"clip_id": clip_id, "dataset": dataset, "label": label, "segments": segs,
            "fall_start_s": falls[0]["start_s"] if falls else None,
            "fall_end_s": falls[0]["end_s"] if falls else None,
            "subject": subject, "cam": cam, "event_group": event_group or clip_id,
            "label_source": "omnifall", "license": "test"}


class TestPureStats(unittest.TestCase):
    def test_quantile_matches_linear_interpolation(self):
        self.assertEqual(D.quantile([3, 1, 2, 4], 0.5), 2.5)
        self.assertAlmostEqual(D.quantile([0, 10], 0.1), 1.0)
        self.assertEqual(D.quantile([5], 0.9), 5)
        self.assertIsNone(D.quantile([], 0.5))
        self.assertEqual(D.quantile([None, float("nan"), 1.0], 0.5), 1.0)   # 결측/nan은 버림

    def test_describe_and_share(self):
        d = D.describe([1, 2, 3, 4, 5, None])
        self.assertEqual((d["n"], d["median"], d["min"], d["max"]), (5, 3, 1, 5))
        self.assertEqual(D.describe([])["n"], 0)
        self.assertEqual(D.share([1, 2, 3, 4], lambda x: x > 2), 0.5)
        self.assertIsNone(D.share([], lambda x: True))

    def test_pixel_threshold(self):
        # 정규화 h/w = 픽셀 h/w * W / H -> 픽셀 임계값 = th * H / W
        self.assertAlmostEqual(D.pixel_threshold(1.6, 1920, 1080), 0.9)
        self.assertAlmostEqual(D.pixel_threshold(1.6, 320, 240), 1.2)
        self.assertAlmostEqual(D.pixel_threshold(1.0, 720, 480), 2.0 / 3.0)
        self.assertAlmostEqual(D.normalized_aspect(D.pixel_threshold(0.7, 1280, 720), 1280, 720), 0.7)
        # 같은 픽셀 자세: 4:3에서 0.75 -> 16:9에서 0.75 * (3/4) * (16/9) = 1.0
        self.assertAlmostEqual(D.to_frame(0.75, (640, 480), (1920, 1080)), 1.0)

    def test_rule_thresholds_and_deploy_frame(self):
        th = D.rule_aspect_thresholds()
        self.assertEqual(th, {"aspect_abs_th": 1.6, "force_abnormal_aspect_th": 0.7, "confirm_aspect_abs": 1.0})
        w, h = D.deploy_frame_wh()
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_segments_and_lying_interval(self):
        row = make_row(segments=[{"cls": "sit_down", "start_s": 0.5, "end_s": 1.0},
                                 {"cls": "fall", "start_s": 2.0, "end_s": 2.5},
                                 {"cls": "fallen", "start_s": 2.5, "end_s": 9.0},
                                 {"cls": "lie_down", "start_s": 0.1, "end_s": 0.4}])
        self.assertEqual(D.segments_of(row, D.HARD_ADL_CLS), [("lie_down", 0.1, 0.4), ("sit_down", 0.5, 1.0)])
        self.assertEqual(D.lying_interval(row, 6.0), (2.5, 6.0, "label"))         # 클립 끝에서 자름
        no_post = make_row(segments=[{"cls": "fall", "start_s": 2.0, "end_s": 2.5}])
        self.assertEqual(D.lying_interval(no_post, 6.0), (5.0, 6.0, "clip_end"))
        # 낙상 뒤 일어나서 걸어 나가면 클립 끝은 누운 자세가 아님 -> 누운 구간 없음
        got_up = make_row(segments=[{"cls": "walk", "start_s": 0.0, "end_s": 2.0},
                                    {"cls": "fall", "start_s": 2.0, "end_s": 2.5},
                                    {"cls": "stand_up", "start_s": 2.5, "end_s": 4.0},
                                    {"cls": "walk", "start_s": 4.0, "end_s": 6.0}])
        self.assertEqual(D.lying_interval(got_up, 6.0), (None, None, "recovered"))
        # 라벨 없는 클립(Figshare)은 클립 끝
        self.assertEqual(D.lying_interval(make_row(segments=[]), 1.8)[2], "clip_end")

    def test_label_overrun_and_overlap(self):
        row = make_row()
        self.assertEqual(D.label_overrun_s(row, 6.0), 0.0)
        self.assertAlmostEqual(D.label_overrun_s(row, 5.5), 0.5)
        self.assertIsNone(D.label_overrun_s(make_row(label="adl"), 5.0))
        o = D.overlap([1, 2, 3, 4, 5], [0.5, 1.5, 3.5])
        self.assertAlmostEqual(o["adl_gt_fall_median"], 1 / 3)
        self.assertIsNone(D.overlap([], [1.0])["adl_gt_fall_median"])

    def test_format_helpers(self):
        self.assertEqual(D._fmt_num(24.0000001), "24")
        self.assertEqual(D._fmt_num(0.456), "0.46")
        self.assertEqual(D._fmt_num(None), "-")
        t = D.md_table(["a", "b"], [["x|y", 1]])
        self.assertIn("x\\|y", t)
        self.assertEqual(len(t.splitlines()), 3)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestArrayStats(unittest.TestCase):
    def test_constants_match_features_v2(self):
        from app.ai_classifier import features as F
        from training.experiments import windows as W
        self.assertEqual((D.HZ, D.MIN_CONF, D.MAX_GAP_S), (F.TARGET_HZ, F.MIN_CONF, F.MAX_GAP_S))
        self.assertEqual(D._MASK0, F.V2_CHANNELS.index("mask_nose"))
        self.assertEqual(D.LOW_VIS, W.MIN_VISIBILITY)

    def test_mid_point_and_torso_length(self):
        import numpy as np
        xy = np.zeros((3, 17, 2))
        xy[:, 5] = [0, 0]
        xy[:, 6] = [10, 0]
        xy[:, 11] = [0, 30]
        xy[:, 12] = [10, 30]
        m = np.ones((3, 17), dtype=bool)
        m[1, 6] = False          # 한쪽 어깨만 -> 그 관절 사용
        m[2, [11, 12]] = False   # 골반 없음 -> 그 프레임 제외
        mid, ok = D.mid_point(xy, m, 5, 6)
        self.assertEqual(mid[1].tolist(), [0.0, 0.0])
        self.assertTrue(ok.all())
        _, hip_ok = D.mid_point(xy, m, 11, 12)
        self.assertEqual(hip_ok.tolist(), [True, True, False])
        # 프레임 0: 30, 프레임 1: sqrt(5^2 + 30^2) -> 중앙값은 두 값의 평균
        self.assertAlmostEqual(D.torso_length(xy, m), (30.0 + math.hypot(5, 30)) / 2)
        self.assertTrue(math.isnan(D.torso_length(xy, np.zeros((3, 17), dtype=bool))))

    def test_down_velocity_constant_and_gaps(self):
        import numpy as np
        t = np.arange(10) / 15.0
        y = 450.0 * t   # 450 px/s
        ok = np.ones(10, dtype=bool)
        v = D.down_velocity(t, y, ok, scale=45.0)
        self.assertTrue(np.isnan(v[0]) and np.isnan(v[-1]))     # 끝점은 중앙 차분 불가
        np.testing.assert_allclose(v[1:-1], 10.0)                # 450 / 45 = 10 몸통 길이/초
        ok[4] = False
        v = D.down_velocity(t, y, ok, scale=45.0)
        self.assertTrue(np.isnan(v[3]) and np.isnan(v[4]) and np.isnan(v[5]))   # 공백 주변은 잇지 않음
        self.assertAlmostEqual(v[2], 10.0)
        self.assertTrue(np.all(np.isnan(D.down_velocity(t, y, ok, scale=float("nan")))))
        self.assertEqual(D.max_in(t, v, 0.0, 0.2), 10.0)
        self.assertIsNone(D.max_in(t, v, 0.25, 0.35))

    def test_visibility_groups_and_aspect(self):
        import numpy as np
        m = np.zeros((4, 17), dtype=bool)
        m[:, list(D.JOINT_GROUPS["legs"])] = True
        self.assertAlmostEqual(D.visibility(m), 4 / 17)
        self.assertEqual(D.visibility(m, D.JOINT_GROUPS["legs"]), 1.0)
        self.assertEqual(D.visibility(m, D.JOINT_GROUPS["head"]), 0.0)
        self.assertIsNone(D.visibility(np.zeros((0, 17))))
        a = D.bbox_aspect_norm([[0.1, 0.1, 0.2, 0.4], [0, 0, 0, 0], [0.1, 0.1, 0.2, 0.4]], [True, True, False])
        self.assertAlmostEqual(a[0], 3.0)
        self.assertTrue(np.isnan(a[1]) and np.isnan(a[2]))

    def test_clip_stats_fall_velocity_and_quality(self):
        clip = make_clip(drop_px=120.0, drop_s=0.5, absent=[(4.0, 6.0)])
        st = D.clip_stats(make_row(), clip)
        self.assertAlmostEqual(st["torso_px"], TORSO_PX, places=3)
        # 120 px / 0.5 s / 60 px = 4 몸통 길이/초 (15Hz 중앙 차분이라 경계에서 조금 낮을 수 있음)
        self.assertEqual(len(st["fall_vel"]), 1)
        self.assertAlmostEqual(st["fall_vel"][0], 4.0, delta=0.05)
        self.assertLess(st["clip_vel"], 4.0 + 0.05)
        # 사람 없음 4~6초: 낙상 구간 [2, 4.5]의 뒤쪽 0.5초가 빠짐
        self.assertLess(st["fall_present"], 1.0)
        self.assertGreater(st["onset_vis"], 0.5)
        self.assertAlmostEqual(st["vis_legs"], st["vis_head"])          # 모든 관절 conf 0.9
        self.assertLess(st["lying_present"], 0.5)                       # 누운 뒤 2.5~6초 중 4~6초 없음
        self.assertEqual(st["lying_src"], "label")
        self.assertAlmostEqual(st["stand_aspect"], (3.5 * TORSO_PX / 480) / (0.9 * TORSO_PX / 640), places=3)

    def test_switched_frames_excluded_from_velocity(self):
        import numpy as np
        clip = make_clip(drop_px=120.0, drop_s=0.5)
        # 낙상 도중 골반이 순간 이동(다른 사람으로 전환)한 것처럼 만듦: 속도 최댓값이 크게 튐
        jump = (clip["ts"] >= 2.2) & (clip["ts"] < 2.3)
        clip["kps"][jump, :, 1] += 0.3
        st = D.clip_stats(make_row(), clip)
        self.assertGreater(st["fall_vel"][0], 10.0)
        sw = np.zeros(len(clip["ts"]), dtype=bool)
        sw[np.flatnonzero(jump)[[0, -1]] + [0, 1]] = True   # 전환 들어감/나옴 프레임
        clip["switched"] = sw
        st = D.clip_stats(make_row(), clip)
        self.assertLess(st["fall_vel"][0], 4.0 + 0.05)

    def test_recovered_fall_has_no_lying_stats(self):
        row = make_row(segments=[{"cls": "fall", "start_s": 2.0, "end_s": 2.5},
                                 {"cls": "stand_up", "start_s": 2.5, "end_s": 6.0}])
        st = D.clip_stats(row, make_clip())
        self.assertEqual(st["lying_src"], "recovered")
        self.assertIsNone(st["lying_present"])
        self.assertIsNone(st["lying_aspect"])
        self.assertIsNotNone(st["stand_aspect"])
        self.assertNotIn("stand_aspect", D.clip_stats(make_row(label="adl"), make_clip()))   # 서 있음은 낙상 클립만

    def test_aggregate_and_render_smoke(self):
        rows = [make_row("le2i/a/video_1"),
                make_row("le2i/a/video_2", label="adl",
                         segments=[{"cls": "sit_down", "start_s": 2.0, "end_s": 2.5}]),
                make_row("le2i/a/video_3", event_group="le2i/a/video_1"),
                make_row("urfd/fall-01", dataset="urfd", subject=None, cam="0")]
        stats = [D.clip_stats(rows[0], make_clip(drop_px=120.0)),
                 D.clip_stats(rows[1], make_clip(drop_px=30.0)),
                 D.clip_stats(rows[2], make_clip(drop_px=120.0))]
        agg = D.aggregate(rows, stats, D.rule_aspect_thresholds(), (1920, 1080))
        le = agg["per"]["le2i"]
        self.assertEqual((le["clips"], le["fall"], le["adl"], le["event_groups"]), (3, 2, 1, 2))
        self.assertEqual(le["hard_adl_vel"]["n"], 1)
        self.assertAlmostEqual(le["hard_adl_vel"]["median"], 1.0, delta=0.05)   # 30 / 0.5 / 60
        self.assertEqual(le["overlap"]["adl_gt_fall_median"], 0.0)
        self.assertEqual(agg["issues"]["kp_missing"], ["urfd/fall-01"])
        self.assertEqual(agg["issues"]["multi_clip_groups"]["le2i"], [["le2i/a/video_1", "le2i/a/video_3"]])
        self.assertEqual(agg["le2i_subjects"]["subjects"], {"1": 3})
        md = D.render(agg, "m.jsonl", "kp", 1.0)
        self.assertIn("## 1. 구성", md)
        self.assertIn("이 데이터로 할 수 없는 것", md)
        self.assertIn("1920x1080 (배포)", md)


if __name__ == "__main__":
    unittest.main()
