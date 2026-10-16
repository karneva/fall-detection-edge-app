"""
features v2 (리샘플링 + 스케일 정규화 + 속도/전역 궤적/몸통 각도/bbox 종횡비/마스크) 검증.
numpy가 없는 시스템 python에서는 건너뜁니다.
"""
import importlib.util
import math
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None

FRAME_WH = (640, 480)

# 골반 중점 원점, 몸통 길이(어깨 중점-골반 중점) 1 단위, y는 아래 방향
_BASE_SKELETON = [
    (0.0, -1.6),                    # nose
    (0.08, -1.7), (-0.08, -1.7),    # l_eye, r_eye
    (0.16, -1.65), (-0.16, -1.65),  # l_ear, r_ear
    (0.3, -1.0), (-0.3, -1.0),      # l_shoulder, r_shoulder
    (0.4, -0.5), (-0.4, -0.5),      # l_elbow, r_elbow
    (0.45, 0.0), (-0.45, 0.0),      # l_wrist, r_wrist
    (0.15, 0.0), (-0.15, 0.0),      # l_hip, r_hip
    (0.17, 0.9), (-0.17, 0.9),      # l_knee, r_knee
    (0.18, 1.8), (-0.18, 1.8),      # l_ankle, r_ankle
]


def _theta(t):
    # 2초 부근에서 부드럽게 90도 넘어지는 동작
    return (math.pi / 2) / (1.0 + math.exp(-(t - 2.0) / 0.5))


def make_clip(ts, unit_px=60.0, hip0=(320.0, 240.0), hip_vel=(30.0, 0.0), frame_wh=FRAME_WH,
              fall=True, conf=0.9):
    """합성 클립: 시각 ts마다 (kps (T,17,3) 정규화, bbox (T,4), present (T,))"""
    import numpy as np
    ts = np.asarray(ts, dtype=np.float64)
    base = np.array(_BASE_SKELETON, dtype=np.float64)
    w, h = frame_wh
    kps = np.zeros((len(ts), 17, 3), dtype=np.float64)
    bbox = np.zeros((len(ts), 4), dtype=np.float64)
    for i, t in enumerate(ts):
        th = _theta(t) if fall else 0.0
        c, s = math.cos(th), math.sin(th)
        rot = base @ np.array([[c, s], [-s, c]])
        px = rot * unit_px + np.array([hip0[0] + hip_vel[0] * t, hip0[1] + hip_vel[1] * t])
        kps[i, :, 0] = px[:, 0] / w
        kps[i, :, 1] = px[:, 1] / h
        kps[i, :, 2] = conf
        x1, y1 = px.min(axis=0) - 5
        x2, y2 = px.max(axis=0) + 5
        bbox[i] = [x1 / w, y1 / h, x2 / w, y2 / h]
    present = np.ones(len(ts), dtype=bool)
    return kps.astype(np.float32), bbox.astype(np.float32), present


def _ch(name):
    from app.ai_classifier.features import V2_CHANNELS
    return V2_CHANNELS.index(name)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class ChannelLayoutTest(unittest.TestCase):
    def test_constants(self):
        from app.ai_classifier import features as f
        self.assertEqual(f.FEATURE_VERSION_V1, "v1")
        self.assertEqual(f.FEATURE_VERSION_V2, "v2")
        self.assertEqual((f.TARGET_HZ, f.WINDOW_S, f.MIN_CONF, f.MAX_GAP_S), (15.0, 4.0, 0.3, 0.2))

    def test_channel_count_and_order(self):
        from app.ai_classifier.features import V2_CHANNELS
        self.assertEqual(len(V2_CHANNELS), 92)
        self.assertEqual(len(set(V2_CHANNELS)), 92)
        self.assertEqual(V2_CHANNELS[0:2], ["rel_nose_x", "rel_nose_y"])
        self.assertEqual(V2_CHANNELS[33], "rel_r_ankle_y")
        self.assertEqual(V2_CHANNELS[34], "vel_nose_x")
        self.assertEqual(V2_CHANNELS[67], "vel_r_ankle_y")
        self.assertEqual(V2_CHANNELS[68:72], ["hip_dx", "hip_dy", "hip_vx", "hip_vy"])
        self.assertEqual(V2_CHANNELS[72:75], ["torso_angle", "torso_angvel", "bbox_aspect"])
        self.assertEqual(V2_CHANNELS[75], "mask_nose")
        self.assertEqual(V2_CHANNELS[91], "mask_r_ankle")

    def test_output_shape_dtype(self):
        import numpy as np
        from app.ai_classifier.features import window_v2
        ts = np.arange(120) / 30.0
        kps, bbox, present = make_clip(ts)
        feats = window_v2(ts, kps, bbox, present, FRAME_WH, 0.0, 4.0)
        self.assertEqual(feats.shape, (92, 60))
        self.assertEqual(feats.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(feats)))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class ResampleTest(unittest.TestCase):
    def linear_clip(self, fps=30.0, dur=4.0):
        import numpy as np
        ts = np.arange(int(round(dur * fps))) / fps
        kps = np.zeros((len(ts), 17, 3), dtype=np.float32)
        kps[:, :, 0] = (0.1 + 0.1 * ts)[:, None] + np.arange(17)[None, :] * 0.01
        kps[:, :, 1] = 0.5
        kps[:, :, 2] = 0.9
        bbox = np.tile(np.array([0.1, 0.1, 0.3, 0.9], dtype=np.float32), (len(ts), 1))
        present = np.ones(len(ts), dtype=bool)
        return ts, kps, bbox, present

    def test_default_grid(self):
        import numpy as np
        from app.ai_classifier.features import resample_clip
        ts, kps, bbox, present = self.linear_clip(fps=30.0, dur=4.0)   # 0 .. 3.9667
        t_grid, kps_r, mask, bbox_r, pres_r = resample_clip(ts, kps, bbox, present)
        self.assertEqual(len(t_grid), 60)                               # 0 .. 3.9333
        self.assertAlmostEqual(t_grid[1] - t_grid[0], 1 / 15.0)
        self.assertEqual(kps_r.shape, (60, 17, 2))
        self.assertEqual(mask.shape, (60, 17))
        self.assertEqual(bbox_r.shape, (60, 4))
        self.assertEqual(pres_r.shape, (60,))
        self.assertTrue(np.all(mask == 1))
        self.assertTrue(np.all(pres_r))
        np.testing.assert_allclose(kps_r[:, 0, 0], 0.1 + 0.1 * t_grid, atol=1e-6)
        np.testing.assert_allclose(bbox_r[:, 2], 0.3, atol=1e-6)

    def test_window_grid(self):
        import numpy as np
        from app.ai_classifier.features import resample_clip
        ts, kps, bbox, present = self.linear_clip()
        t_grid, kps_r, mask, _, _ = resample_clip(ts, kps, bbox, present, t_start=1.0, t_end=3.0)
        self.assertEqual(len(t_grid), 30)
        self.assertAlmostEqual(t_grid[0], 1.0)
        np.testing.assert_allclose(kps_r[:, 3, 0], 0.13 + 0.1 * t_grid, atol=1e-6)
        with self.assertRaises(ValueError):
            resample_clip(ts, kps, bbox, present, t_start=1.0)

    def test_long_gap_masked_short_gap_interpolated(self):
        import numpy as np
        from app.ai_classifier.features import resample_clip
        ts, kps, bbox, present = self.linear_clip()
        frame = np.round(ts * 30).astype(int)
        present[(frame > 30) & (frame < 45)] = False   # 1.0 ~ 1.5초 사이 결측: 간격 0.5초
        present[(frame > 60) & (frame < 66)] = False   # 2.0 ~ 2.2초: 간격 정확히 0.2초 -> 보간
        present[(frame > 90) & (frame < 97)] = False   # 3.0 ~ 3.2333초: 간격 0.2333초 -> 마스크
        t_grid, kps_r, mask, bbox_r, pres_r = resample_clip(ts, kps, bbox, present)
        k = np.round(t_grid * 15).astype(int)

        long_gap = (k > 15) & (k < 23)                 # 1.0667 ~ 1.4667
        self.assertTrue(np.all(mask[long_gap] == 0))
        self.assertTrue(np.all(kps_r[long_gap] == 0))
        self.assertTrue(np.all(bbox_r[long_gap] == 0))
        self.assertFalse(np.any(pres_r[long_gap]))
        self.assertTrue(np.all(mask[k == 15] == 1))     # 1.0초 샘플 자체는 유효
        self.assertTrue(np.all(mask[k == 23] == 1))     # 1.5333초는 1.5~1.5667 사이

        short_gap = (k > 30) & (k < 33)                # 2.0667, 2.1333
        self.assertTrue(np.all(mask[short_gap] == 1))
        self.assertTrue(np.all(pres_r[short_gap]))
        np.testing.assert_allclose(kps_r[short_gap, 0, 0], 0.1 + 0.1 * t_grid[short_gap], atol=1e-6)

        just_over = (k > 45) & (k < 49)                # 3.0667 ~ 3.2
        self.assertTrue(np.all(mask[just_over] == 0))

    def test_low_conf_joint_masked(self):
        import numpy as np
        from app.ai_classifier.features import resample_clip
        ts, kps, bbox, present = self.linear_clip()
        kps[:, 9, 2] = 0.29        # l_wrist: 임계값 미만
        kps[:, 10, 2] = 0.3        # r_wrist: 임계값과 같으면 유효
        kps[:, 0, 2] = 0.0
        kps[:, 0, :2] = 0.0        # 미검출 표기
        _, kps_r, mask, _, _ = resample_clip(ts, kps, bbox, present)
        self.assertTrue(np.all(mask[:, 9] == 0))
        self.assertTrue(np.all(kps_r[:, 9] == 0))
        self.assertTrue(np.all(mask[:, 10] == 1))
        self.assertTrue(np.all(mask[:, 0] == 0))
        self.assertTrue(np.all(mask[:, [1, 2, 3, 11, 12]] == 1))

    def test_low_conf_run_acts_as_gap(self):
        import numpy as np
        from app.ai_classifier.features import resample_clip
        ts, kps, bbox, present = self.linear_clip()
        frame = np.round(ts * 30).astype(int)
        kps[(frame > 30) & (frame < 45), 5, 2] = 0.1   # l_shoulder만 0.5초 동안 저신뢰
        t_grid, _, mask, _, pres_r = resample_clip(ts, kps, bbox, present)
        k = np.round(t_grid * 15).astype(int)
        gap = (k > 15) & (k < 23)
        self.assertTrue(np.all(mask[gap, 5] == 0))
        self.assertTrue(np.all(mask[gap, 6] == 1))
        self.assertTrue(np.all(pres_r))

    def test_outside_valid_range_masked(self):
        import numpy as np
        from app.ai_classifier.features import resample_clip
        ts, kps, bbox, present = self.linear_clip(dur=2.0)            # 0 ~ 1.9667
        t_grid, _, mask, _, pres_r = resample_clip(ts, kps, bbox, present, t_start=-1.0, t_end=3.0)
        self.assertEqual(len(t_grid), 60)
        inside = (t_grid >= -1e-9) & (t_grid <= ts[-1] + 1e-9)
        self.assertTrue(np.all(mask[inside] == 1))
        self.assertTrue(np.all(mask[~inside] == 0))
        self.assertTrue(np.array_equal(pres_r, inside))

    def test_nonfinite_timestamp_does_not_drop_clip(self):
        import numpy as np
        from app.ai_classifier.features import resample_clip
        ts, kps, bbox, present = self.linear_clip()
        ref_grid, ref_kps, ref_mask, _, _ = resample_clip(ts, kps, bbox, present)
        for bad_idx, bad in ((len(ts) - 1, np.nan), (5, np.nan), (30, -np.inf), (60, np.inf)):
            ts2 = ts.copy()
            ts2[bad_idx] = bad
            t_grid, kps_r, mask, _, _ = resample_clip(ts2, kps, bbox, present)
            self.assertGreater(len(t_grid), 50, (bad_idx, bad))
            self.assertTrue(np.all(np.isfinite(t_grid)))
            # 깨진 샘플 하나만 빠지고 나머지 격자는 기준과 같음 (30fps 샘플 1개 결측은 보간됨)
            k = min(len(t_grid), len(ref_grid))
            np.testing.assert_allclose(t_grid[:k], ref_grid[:k])
            np.testing.assert_allclose(kps_r[:k], ref_kps[:k], atol=1e-6)
            self.assertTrue(np.all(mask[:k] == 1))

    def test_unsorted_timestamps(self):
        import numpy as np
        from app.ai_classifier.features import resample_clip
        ts, kps, bbox, present = self.linear_clip()
        perm = np.random.default_rng(0).permutation(len(ts))
        a = resample_clip(ts, kps, bbox, present)
        b = resample_clip(ts[perm], kps[perm], bbox[perm], present[perm])
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class FeatureValuesTest(unittest.TestCase):
    def test_upright_rel_matches_skeleton(self):
        import numpy as np
        from app.ai_classifier.features import window_v2
        ts = np.arange(120) / 30.0
        kps, bbox, present = make_clip(ts, fall=False)
        feats = window_v2(ts, kps, bbox, present, FRAME_WH, 0.0, 4.0)
        base = np.array(_BASE_SKELETON)
        expect_rel = (base - (base[11] + base[12]) / 2).reshape(34)       # 몸통 길이 1 단위
        np.testing.assert_allclose(feats[:34].T, np.tile(expect_rel, (60, 1)), atol=1e-4)
        np.testing.assert_allclose(feats[34:68], 0.0, atol=1e-4)         # 순수 평행이동 -> 상대 속도 0
        np.testing.assert_allclose(feats[_ch("torso_angle")], 0.0, atol=1e-5)
        np.testing.assert_allclose(feats[_ch("torso_angvel")], 0.0, atol=1e-4)
        # 골반이 30 px/s 로 이동, 몸통 60 px -> 0.5 s/초
        np.testing.assert_allclose(feats[_ch("hip_vx")], 0.5, atol=1e-4)
        np.testing.assert_allclose(feats[_ch("hip_vy")], 0.0, atol=1e-4)
        np.testing.assert_allclose(feats[_ch("hip_dx")], 0.5 * np.arange(60) / 15.0, atol=1e-4)
        self.assertTrue(np.all(feats[75:] == 1))
        # bbox: 폭 (0.9 단위*60 + 10) px, 높이 (3.5 단위*60 + 10) px
        np.testing.assert_allclose(feats[_ch("bbox_aspect")], (3.5 * 60 + 10) / (0.9 * 60 + 10), rtol=1e-4)

    def test_fall_angle_goes_to_horizontal(self):
        import numpy as np
        from app.ai_classifier.features import window_v2
        ts = np.arange(150) / 30.0
        kps, bbox, present = make_clip(ts)
        feats = window_v2(ts, kps, bbox, present, FRAME_WH, 0.0, 5.0)
        ang = feats[_ch("torso_angle")]
        self.assertLess(abs(ang[0]), 0.1)
        self.assertAlmostEqual(abs(float(ang[-1])), math.pi / 2, delta=0.1)
        # 각속도 부호/크기가 각도 변화와 일치
        angvel = feats[_ch("torso_angvel")]
        np.testing.assert_allclose(np.sum(angvel[1:-1]) / 15.0, ang[-2] - ang[1], atol=0.05)
        self.assertGreater(float(np.max(np.abs(angvel))), 0.5)

    def test_fps_invariance(self):
        import numpy as np
        from app.ai_classifier.features import window_v2
        feats = {}
        for fps in (10.0, 15.0, 30.0):
            ts = np.arange(int(5 * fps)) / fps
            kps, bbox, present = make_clip(ts)
            feats[fps] = window_v2(ts, kps, bbox, present, FRAME_WH, 0.6, 4.6)   # 0.6초 = 15/30fps 샘플 시각
            self.assertEqual(feats[fps].shape, (92, 60))
        pos = list(range(34)) + [68, 69, 72]
        vel = list(range(34, 68)) + [70, 71, 73]
        # 15/30 fps는 격자가 샘플과 겹쳐 거의 동일
        np.testing.assert_allclose(feats[15.0][pos], feats[30.0][pos], atol=1e-4)
        np.testing.assert_allclose(feats[15.0][vel], feats[30.0][vel], atol=1e-3)
        # 10 fps는 선형 보간 오차만큼만 차이
        np.testing.assert_allclose(feats[10.0][pos], feats[30.0][pos], atol=0.005)
        np.testing.assert_allclose(feats[10.0][vel], feats[30.0][vel], atol=0.05)
        np.testing.assert_allclose(feats[10.0][74], feats[30.0][74], rtol=0.02)
        self.assertTrue(np.all(feats[10.0][75:] == 1))

    def test_scale_and_translation_invariance(self):
        import numpy as np
        from app.ai_classifier.features import window_v2
        ts = np.arange(150) / 30.0
        kps, bbox, present = make_clip(ts, unit_px=40.0, hip0=(250.0, 200.0), hip_vel=(20.0, 5.0))
        big, big_bbox, _ = make_clip(ts, unit_px=80.0, hip0=(350.0, 260.0), hip_vel=(40.0, 10.0))
        a = window_v2(ts, kps, bbox, present, FRAME_WH, 0.0, 5.0)
        b = window_v2(ts, big, big_bbox, present, FRAME_WH, 0.0, 5.0)
        # bbox 패딩(5 px)이 스케일과 무관해 종횡비 채널은 제외
        idx = [i for i in range(92) if i != _ch("bbox_aspect")]
        np.testing.assert_allclose(a[idx], b[idx], atol=1e-4)

    def test_isotropic_pixels(self):
        import numpy as np
        from app.ai_classifier.features import window_v2
        ts = np.arange(120) / 30.0
        a_wh, b_wh = (640, 480), (1280, 720)
        ka, ba, pa = make_clip(ts, frame_wh=a_wh)
        kb, bb, pb = make_clip(ts, frame_wh=b_wh)
        a = window_v2(ts, ka, ba, pa, a_wh, 0.0, 4.0)
        b = window_v2(ts, kb, bb, pb, b_wh, 0.0, 4.0)
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_masked_joint_channels_zero(self):
        import numpy as np
        from app.ai_classifier.features import window_v2
        ts = np.arange(120) / 30.0
        kps, bbox, present = make_clip(ts)
        kps[:, 9, 2] = 0.1                       # l_wrist 저신뢰
        feats = window_v2(ts, kps, bbox, present, FRAME_WH, 0.0, 4.0)
        for name in ("rel_l_wrist_x", "rel_l_wrist_y", "vel_l_wrist_x", "vel_l_wrist_y", "mask_l_wrist"):
            self.assertTrue(np.all(feats[_ch(name)] == 0), name)
        self.assertTrue(np.all(feats[_ch("mask_r_wrist")] == 1))

    def test_scale_fallbacks(self):
        import numpy as np
        from app.ai_classifier.features import build_features_v2
        n = 10
        kps_r = np.zeros((n, 17, 2), dtype=np.float32)
        kps_r[:, 11] = [0.4, 0.5]
        kps_r[:, 12] = [0.5, 0.5]
        kps_r[:, 15] = [0.4, 0.9]
        mask = np.zeros((n, 17), dtype=np.float32)
        mask[:, [11, 12, 15]] = 1                 # 어깨 없음
        bbox = np.tile(np.array([0.3, 0.25, 0.6, 1.0], dtype=np.float32), (n, 1))
        # bbox 높이 0.75*400 = 300 px -> s = 90 px; 발목은 골반 아래 0.4*400 = 160 px
        f = build_features_v2(kps_r, mask, bbox, (400, 400))
        np.testing.assert_allclose(f[_ch("rel_l_ankle_y")], 160 / 90, rtol=1e-5)
        # bbox도 없으면 s = 1 px
        f = build_features_v2(kps_r, mask, np.zeros((n, 4)), (400, 400))
        np.testing.assert_allclose(f[_ch("rel_l_ankle_y")], 160.0, rtol=1e-5)
        self.assertTrue(np.all(f[_ch("torso_angle")] == 0))
        self.assertTrue(np.all(f[_ch("bbox_aspect")] == 0))

    def test_frames_without_hips_zeroed(self):
        import numpy as np
        from app.ai_classifier.features import window_v2
        ts = np.arange(120) / 30.0
        kps, bbox, present = make_clip(ts)
        frame = np.round(ts * 30).astype(int)
        kps[(frame >= 30) & (frame < 60), 11:13, 2] = 0.0   # 1~2초 골반 미검출
        feats = window_v2(ts, kps, bbox, present, FRAME_WH, 0.0, 4.0)
        gap = slice(16, 29)
        self.assertTrue(np.all(feats[:72, gap] == 0))
        self.assertTrue(np.all(feats[_ch("mask_nose"), gap] == 1))
        self.assertTrue(np.all(feats[_ch("mask_l_hip"), gap] == 0))
        self.assertTrue(np.all(np.isfinite(feats)))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class RobustnessTest(unittest.TestCase):
    def test_all_missing(self):
        import numpy as np
        from app.ai_classifier.features import resample_clip, build_features_v2, window_v2
        ts = np.arange(60) / 15.0
        kps = np.zeros((60, 17, 3), dtype=np.float32)
        bbox = np.zeros((60, 4), dtype=np.float32)
        present = np.zeros(60, dtype=bool)
        feats = window_v2(ts, kps, bbox, present, FRAME_WH, 0.0, 4.0)
        self.assertEqual(feats.shape, (92, 60))
        self.assertTrue(np.all(feats == 0))
        _, kps_r, mask, bbox_r, pres_r = resample_clip(ts, kps, bbox, present)
        self.assertFalse(np.any(pres_r))
        f2 = build_features_v2(kps_r, mask, bbox_r, None)
        self.assertTrue(np.all(np.isfinite(f2)))
        self.assertTrue(np.all(f2 == 0))

    def test_empty_input(self):
        import numpy as np
        from app.ai_classifier.features import resample_clip, build_features_v2, window_v2
        empty_k = np.zeros((0, 17, 3), dtype=np.float32)
        t_grid, kps_r, mask, bbox_r, pres_r = resample_clip([], empty_k, np.zeros((0, 4)), [])
        self.assertEqual(len(t_grid), 0)
        self.assertEqual(build_features_v2(kps_r, mask, bbox_r, FRAME_WH).shape, (92, 0))
        feats = window_v2([], empty_k, None, None, FRAME_WH, 0.0, 4.0)
        self.assertEqual(feats.shape, (92, 60))
        self.assertTrue(np.all(feats == 0))

    def test_nan_and_degenerate_inputs_finite(self):
        import numpy as np
        from app.ai_classifier.features import window_v2, build_features_v2
        ts = np.arange(120) / 30.0
        kps, bbox, present = make_clip(ts)
        kps[10:20, 3, 0] = np.nan
        kps[30:40, :, 2] = np.inf
        bbox[50:60] = [0.5, 0.2, 0.5, 0.8]        # 폭 0
        bbox[60:70] = np.nan
        feats = window_v2(ts, kps, bbox, present, FRAME_WH, 0.0, 4.0)
        self.assertTrue(np.all(np.isfinite(feats)))
        self.assertLessEqual(float(feats[_ch("bbox_aspect")].max()), 10.0)
        # 모든 관절이 한 점에 몰림 (몸통 길이 0)
        n = 8
        kps_r = np.full((n, 17, 2), 0.5, dtype=np.float32)
        f = build_features_v2(kps_r, np.ones((n, 17)), np.zeros((n, 4)), FRAME_WH)
        self.assertTrue(np.all(np.isfinite(f)))
        self.assertTrue(np.all(f[:68] == 0))

    def test_single_frame(self):
        import numpy as np
        from app.ai_classifier.features import resample_clip, build_features_v2
        kps, bbox, present = make_clip([0.0])
        t_grid, kps_r, mask, bbox_r, _ = resample_clip([0.0], kps, bbox, present)
        self.assertEqual(len(t_grid), 1)
        f = build_features_v2(kps_r, mask, bbox_r, FRAME_WH)
        self.assertEqual(f.shape, (92, 1))
        self.assertTrue(np.all(f[34:68] == 0))
        self.assertTrue(np.all(np.isfinite(f)))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class FlipTest(unittest.TestCase):
    def test_involution(self):
        import numpy as np
        from app.ai_classifier.features import flip_lr_coco, flip_lr_bbox
        rng = np.random.default_rng(0)
        for shape in ((17, 2), (17, 3), (5, 17, 3)):
            k = rng.random(shape).astype(np.float32)
            np.testing.assert_allclose(flip_lr_coco(flip_lr_coco(k)), k, atol=1e-6)
        b = rng.random((7, 4))
        b[3] = 0
        np.testing.assert_allclose(flip_lr_bbox(flip_lr_bbox(b)), b, atol=1e-12)
        with self.assertRaises(ValueError):
            flip_lr_coco(np.zeros((16, 2)))

    def test_swaps_pairs_and_mirrors_x(self):
        import numpy as np
        from app.ai_classifier.features import flip_lr_coco, flip_lr_bbox
        k = np.random.default_rng(1).random((17, 3))
        k[4] = 0                                 # r_ear 미검출
        f = flip_lr_coco(k)
        self.assertEqual(f.shape, k.shape)
        np.testing.assert_allclose(f[0], [1 - k[0, 0], k[0, 1], k[0, 2]])   # nose는 제자리
        for a in range(1, 17, 2):
            b = a + 1
            if a == 3:
                continue
            np.testing.assert_allclose(f[a], [1 - k[b, 0], k[b, 1], k[b, 2]])
            np.testing.assert_allclose(f[b], [1 - k[a, 0], k[a, 1], k[a, 2]])
        self.assertTrue(np.all(f[3] == 0))       # 미검출 관절은 0 유지 (자리만 이동)
        np.testing.assert_allclose(f[4], [1 - k[3, 0], k[3, 1], k[3, 2]])
        self.assertTrue(np.array_equal(k[4], np.zeros(3)))   # 입력은 변경하지 않음
        np.testing.assert_allclose(flip_lr_bbox([0.1, 0.2, 0.4, 0.9]), [0.6, 0.2, 0.9, 0.9])

    def test_feature_symmetry(self):
        import numpy as np
        from app.ai_classifier.features import window_v2, flip_lr_coco, flip_lr_bbox, COCO_NAMES
        ts = np.arange(150) / 30.0
        kps, bbox, present = make_clip(ts)
        a = window_v2(ts, kps, bbox, present, FRAME_WH, 0.0, 5.0)
        b = window_v2(ts, flip_lr_coco(kps), flip_lr_bbox(bbox), present, FRAME_WH, 0.0, 5.0)

        def mirror(name):
            if name.startswith("l_"):
                return "r_" + name[2:]
            if name.startswith("r_"):
                return "l_" + name[2:]
            return name

        for n in COCO_NAMES:
            m = mirror(n)
            for kind in ("rel", "vel"):
                np.testing.assert_allclose(b[_ch(f"{kind}_{n}_x")], -a[_ch(f"{kind}_{m}_x")], atol=1e-4)
                np.testing.assert_allclose(b[_ch(f"{kind}_{n}_y")], a[_ch(f"{kind}_{m}_y")], atol=1e-4)
            np.testing.assert_array_equal(b[_ch(f"mask_{n}")], a[_ch(f"mask_{m}")])
        for name, sign in (("hip_dx", -1), ("hip_vx", -1), ("hip_dy", 1), ("hip_vy", 1),
                           ("torso_angle", -1), ("torso_angvel", -1), ("bbox_aspect", 1)):
            np.testing.assert_allclose(b[_ch(name)], sign * a[_ch(name)], atol=1e-4, err_msg=name)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class V1UntouchedTest(unittest.TestCase):
    def test_v1_still_hip_centered(self):
        import numpy as np
        from app.ai_classifier.features import normalize_frame_v1, normalize_sequence_v1
        k = np.random.default_rng(3).random((17, 2)).astype(np.float32)
        out = normalize_frame_v1(k)
        self.assertEqual(out.shape, (34,))
        np.testing.assert_array_equal(out, (k - (k[11] + k[12]) / 2).flatten())
        seq = normalize_sequence_v1(np.tile(k.reshape(1, 34), (3, 1)))
        self.assertEqual((seq.shape, seq.dtype), ((3, 34), np.float32))


if __name__ == "__main__":
    unittest.main()
