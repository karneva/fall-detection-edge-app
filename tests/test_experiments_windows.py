"""
training/experiments/windows.py (단계별 학습 창 생성기) 검증. 합성 데이터만 사용하고 빠르게 끝납니다.
numpy가 없는 시스템 python에서는 건너뜁니다.
"""
import importlib.util
import math
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None

FRAME_WH = (640, 480)

# 골반 중점 원점, 몸통 길이 1 단위, y는 아래 방향 (서 있는 자세)
_SKELETON = [
    (0.0, -1.6), (0.08, -1.7), (-0.08, -1.7), (0.16, -1.65), (-0.16, -1.65),
    (0.3, -1.0), (-0.3, -1.0), (0.4, -0.5), (-0.4, -0.5), (0.45, 0.0), (-0.45, 0.0),
    (0.15, 0.0), (-0.15, 0.0), (0.17, 0.9), (-0.17, 0.9), (0.18, 1.8), (-0.18, 1.8),
]


def make_arrays(duration_s=10.0, hz=15.0, fall_at=None, absent=()):
    """합성 클립 배열 (clipfmt 형식): fall_at초부터 1초 동안 몸이 90도 넘어지며 골반이 내려감."""
    import numpy as np
    n = int(round(duration_s * hz)) + 1
    ts = np.arange(n) / hz
    base = np.array(_SKELETON)
    kps = np.zeros((n, 17, 3), dtype=np.float32)
    bbox = np.zeros((n, 4), dtype=np.float32)
    present = np.ones(n, dtype=bool)
    w, h = FRAME_WH
    for i, t in enumerate(ts):
        th, drop = 0.0, 0.0
        if fall_at is not None and t >= fall_at:
            a = min(1.0, (t - fall_at) / 1.0)
            th, drop = a * math.pi / 2, a * 80.0
        c, s = math.cos(th), math.sin(th)
        rot = base @ np.array([[c, s], [-s, c]])
        px = 320.0 + rot[:, 0] * 60.0
        py = 300.0 + drop + rot[:, 1] * 60.0
        kps[i, :, 0] = px / w
        kps[i, :, 1] = py / h
        kps[i, :, 2] = 0.9
        bbox[i] = [px.min() / w, py.min() / h, px.max() / w, py.max() / h]
    for i in absent:
        kps[i] = 0
        bbox[i] = 0
        present[i] = False
    return {"ts": ts, "kps": kps, "bbox": bbox, "present": present,
            "frame_wh": np.array(FRAME_WH, dtype=np.int32), "meta": {}}


def make_obs(ts, kps17, bbox=(0.4, 0.2, 0.6, 0.9), quality=0.9, has_person=True, t0=1000.0):
    kl = [{"id": j, "x": float(kps17[j][0]), "y": float(kps17[j][1]), "conf": quality} for j in range(17)]
    return {"ts": t0 + ts, "tracks": [{"has_person": has_person, "bbox": None if bbox is None else [float(v) for v in bbox],
                                       "keypoints_raw": kl, "keypoints": kl, "keypoints_smooth": kl,
                                       "quality_score": quality, "frame_shape": (480, 640)}]}


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class OriginalWindowsTest(unittest.TestCase):
    def setUp(self):
        from training.experiments import windows
        self.W = windows

    def test_original_sequence_repeats_last_and_zero_before_first(self):
        import numpy as np
        from app.ai_classifier.features import normalize_frame_v1
        clip = make_arrays(duration_s=2.0, hz=30.0, absent=(0, 1, 10, 11))
        seq = self.W.original_sequence(clip)
        self.assertEqual(seq.shape, (len(clip["ts"]), 34))
        np.testing.assert_array_equal(seq[0], np.zeros(34))
        np.testing.assert_array_equal(seq[1], np.zeros(34))
        np.testing.assert_allclose(seq[2], normalize_frame_v1(clip["kps"][2, :, :2]), atol=1e-6)
        np.testing.assert_array_equal(seq[10], seq[9])
        np.testing.assert_array_equal(seq[11], seq[9])

    def test_window_count_step_and_whole_video_label(self):
        clip = make_arrays(duration_s=3.0, hz=30.0)          # 91 프레임
        rows = [{"clip_id": "a", "dataset": "urfd", "label": "fall"},
                {"clip_id": "b", "dataset": "urfd", "label": "adl"},
                {"clip_id": "missing", "dataset": "urfd", "label": "adl"}]
        X, y, meta = self.W.windows_original_urfd(rows, {"a": clip, "b": clip}, return_meta=True)
        per_clip = (91 - 30) // 5 + 1
        self.assertEqual(X.shape, (2 * per_clip, 34, 30))
        self.assertEqual(X.dtype.name, "float32")
        self.assertEqual(list(y), [1] * per_clip + [0] * per_clip)
        self.assertEqual(self.W.count_windows(meta, by=("dataset", "y")), {"urfd/0": per_clip, "urfd/1": per_clip})

    def test_short_clip_padded_to_one_window(self):
        import numpy as np
        clip = make_arrays(duration_s=0.5, hz=30.0)          # 16 프레임
        X, y = self.W.windows_original_urfd([{"clip_id": "s", "label": "fall"}], {"s": clip})
        self.assertEqual(X.shape, (1, 34, 30))
        np.testing.assert_array_equal(X[0, :, -1], X[0, :, 15])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class RuntimeStreamTest(unittest.TestCase):
    def setUp(self):
        from training.experiments import windows
        self.W = windows
        self.k = [(0.5, 0.5)] * 17

    def test_lost_frames_skipped_and_bbox_reset_splits(self):
        obs = [make_obs(0.0, self.k), make_obs(0.1, self.k, has_person=False),
               make_obs(0.2, self.k, quality=0.05), make_obs(0.3, self.k, bbox=None), make_obs(0.4, self.k)]
        ts, seq, seg = self.W.runtime_v1_stream(obs, min_quality=0.15)
        self.assertEqual([round(t, 3) for t in ts], [0.0, 0.4])
        self.assertEqual(seg[0] != seg[1], True)
        self.assertEqual(seq.shape, (2, 34))

    def test_event_arrays_mark_lost_absent(self):
        obs = [make_obs(0.0, self.k), make_obs(0.1, self.k, quality=0.05), make_obs(0.2, self.k, bbox=None)]
        arr = self.W.obs_event_arrays(obs, min_quality=0.15)
        self.assertEqual(list(arr["present"]), [True, False])
        self.assertEqual(arr["frame_wh"], (640, 480))
        self.assertEqual(float(arr["kps"][1].sum()), 0.0)

    def test_model_window_bounds(self):
        self.assertEqual(self.W.model_window_bounds(3.0, 10.0), (3.0, 7.0))


def _fake_rep(W, clip, entries=(), level1=()):
    """재생 결과 대신 쓰는 가짜 (클립 배열을 그대로 15Hz obs로 본 것)."""
    import numpy as np
    obs = []
    for i, t in enumerate(clip["ts"]):
        obs.append(make_obs(float(t), clip["kps"][i, :, :2], bbox=clip["bbox"][i],
                            has_person=bool(clip["present"][i])))
    return {
        "clip_end": float(clip["ts"][-1]),
        "v1": W.runtime_v1_stream(obs, 0.15),
        "arrays": W.obs_event_arrays(obs, 0.15),
        "level1_seqs": [(t, np.zeros((30, 34), dtype=np.float32)) for t in level1],
        "entries": [{"anchor_s": a, "start_s": a, "reason": "x"} for a in entries],
    }


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class LabelledWindowsTest(unittest.TestCase):
    def setUp(self):
        from training.experiments import windows
        from app.core.level1 import Level1Params
        self.W = windows
        self.P = Level1Params()
        self.fall = make_arrays(duration_s=10.0, fall_at=5.0)
        self.adl = make_arrays(duration_s=10.0)
        self.rows = [
            {"clip_id": "f", "dataset": "d1", "label": "fall", "label_source": "omnifall",
             "segments": [{"cls": "walk", "start_s": 0.0, "end_s": 5.0}, {"cls": "fall", "start_s": 5.0, "end_s": 6.0},
                          {"cls": "fallen", "start_s": 6.0, "end_s": 10.0}]},
            {"clip_id": "a", "dataset": "d1", "label": "adl", "label_source": "urfd_csv", "segments": []},
            {"clip_id": "g", "dataset": "fig", "label": "fall", "label_source": "folder", "segments": []},
        ]
        self.clips = {"f": self.fall, "a": self.adl, "g": make_arrays(duration_s=1.9, fall_at=0.5)}
        reps = {"f": _fake_rep(windows, self.fall, entries=(5.4, 2.0), level1=(17.0,)),
                "a": _fake_rep(windows, self.adl, entries=(4.0,)),
                "g": _fake_rep(windows, self.clips["g"], entries=(1.0,))}
        self._orig = windows._replay
        windows._replay = lambda row, clip, params, use_cache=True: reps[row["clip_id"]]

    def tearDown(self):
        self.W._replay = self._orig

    def test_post_fall_interval_extends_into_tail(self):
        iv = self.W._post_fall_intervals(self.rows[0], clip_end=10.0, tail_use_s=2.0)
        self.assertEqual(iv, [(6.0, 12.0)])
        row = {"segments": [{"cls": "fall", "start_s": 1.0, "end_s": 2.0}, {"cls": "fallen", "start_s": 2.0, "end_s": 3.0},
                            {"cls": "stand_up", "start_s": 3.0, "end_s": 4.0}, {"cls": "lying", "start_s": 4.0, "end_s": 9.0}]}
        self.assertEqual(self.W._post_fall_intervals(row, clip_end=9.0, tail_use_s=2.0), [(2.0, 3.0)])

    def test_runtime_v1_labels(self):
        X, y, meta = self.W.windows_runtime_v1(self.rows, self.clips, self.P, return_meta=True)
        self.assertEqual(X.shape[1:], (34, 30))
        for m in meta:
            if m["kind"] == "post_fall":
                self.assertTrue(6.0 <= m["t_end"] <= 12.0 and m["y"] == 1)
            elif m["kind"] == "pre_fall":
                self.assertTrue(m["t_end"] <= 5.0 and m["y"] == 0)
            elif m["kind"] == "adl":
                self.assertEqual(m["clip_id"], "a")
            elif m["kind"] == "clip_end":
                self.assertEqual((m["clip_id"], m["y"]), ("g", 1))
        kinds = self.W.count_windows(meta, by=("clip_id", "kind", "y"))
        self.assertIn("f/level1/1", kinds)
        self.assertIn("f/pre_fall/0", kinds)
        self.assertIn("a/adl/0", kinds)
        # 낙상 동작 중(5~6초)에 끝나는 창은 쓰지 않음
        self.assertFalse(any(m["clip_id"] == "f" and 5.0 < m["t_end"] < 6.0 for m in meta if m["kind"] != "level1"))

    def test_event_v2_kinds_and_shapes(self):
        import numpy as np
        X, y, meta = self.W.windows_event_v2(self.rows, self.clips, self.P, return_meta=True)
        self.assertEqual(X.shape[1:], (92, 60))
        self.assertTrue(np.all(np.isfinite(X)))
        c = self.W.count_windows(meta, by=("clip_id", "kind", "y"))
        self.assertEqual(c.get("f/onset/1"), self.W.N_ONSET)
        self.assertEqual(c.get("f/entry_fall/1"), 1)        # 5.4초 진입
        self.assertIsNone(c.get("f/entry_pre_fall/0"))     # 2.0 + 2.5 > 5.0 - 0.5
        self.assertEqual(c.get("a/entry_adl/0"), 1)
        self.assertEqual(c.get("g/figshare/1"), self.W.N_ONSET)
        self.assertEqual(c.get("g/entry_fall/1"), 1)
        self.assertGreater(c.get("a/adl/0", 0), 5)
        onset = sorted(m["anchor_s"] for m in meta if m["kind"] == "onset")
        self.assertEqual(onset, [4.5, 4.75, 5.0, 5.25, 5.5])

    def test_onset_offset_moves_positive_anchors_only(self):
        _, _, base = self.W.windows_event_v2(self.rows, self.clips, self.P, return_meta=True)
        _, _, moved = self.W.windows_event_v2(self.rows, self.clips, self.P, onset_offset_s=1.0, return_meta=True)
        onset = sorted(m["anchor_s"] for m in moved if m["kind"] == "onset")
        self.assertEqual(onset, [5.5, 5.75, 6.0, 6.25, 6.5])
        # 음성 창(pre_fall/adl/entry)은 그대로
        strip = lambda ms: sorted((m["clip_id"], m["kind"], m["anchor_s"]) for m in ms if m["kind"] != "onset")
        self.assertEqual(strip(base), strip(moved))

    def test_tail_use_longer_than_cached_tail_rejected(self):
        with self.assertRaises(ValueError):
            self.W.windows_event_v2(self.rows, self.clips, self.P, tail_use_s=self.W.TAIL_USE_S + 1.0)

    def test_visibility_filter_drops_invisible_windows(self):
        absent = make_arrays(duration_s=10.0, absent=range(0, 151))
        rep = _fake_rep(self.W, absent)
        self.W._replay = lambda row, clip, params, use_cache=True: rep
        X, y = self.W.windows_event_v2([self.rows[1]], {"a": absent}, self.P)
        self.assertEqual(len(y), 0)
        self.assertEqual(X.shape, (0, 92, 60))

    def test_augment_is_seeded_and_finite(self):
        import numpy as np
        a = self.W.windows_event_v2(self.rows, self.clips, self.P, rng=np.random.default_rng(3), augment=True)
        b = self.W.windows_event_v2(self.rows, self.clips, self.P, rng=np.random.default_rng(3), augment=True)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertTrue(np.all(np.isfinite(a[0])))
        plain = self.W.windows_event_v2(self.rows, self.clips, self.P)
        self.assertFalse(a[0].shape == plain[0].shape and np.allclose(a[0], plain[0]))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class HandcraftedTest(unittest.TestCase):
    def test_shape_finite_and_fall_signal(self):
        import numpy as np
        from app.ai_classifier.features import window_v2
        from training.experiments import windows as W
        fall = make_arrays(duration_s=8.0, fall_at=3.0)
        stand = make_arrays(duration_s=8.0)
        xf = window_v2(fall["ts"], fall["kps"], fall["bbox"], fall["present"], FRAME_WH, 1.5, 5.5)
        xs = window_v2(stand["ts"], stand["kps"], stand["bbox"], stand["present"], FRAME_WH, 1.5, 5.5)
        H = W.handcrafted(np.stack([xf, xs, np.zeros_like(xs)]))
        self.assertEqual(H.shape, (3, len(W.HANDCRAFTED_NAMES)))
        self.assertTrue(np.all(np.isfinite(H)))
        i = W.HANDCRAFTED_NAMES.index
        self.assertGreater(H[0, i("hip_vy_max")], H[1, i("hip_vy_max")] + 0.5)
        self.assertGreater(H[0, i("torso_abs_end")], 1.2)
        self.assertLess(H[1, i("torso_abs_end")], 0.1)
        self.assertEqual(H[2].tolist(), [0.0] * H.shape[1])
        self.assertEqual(W.handcrafted(np.zeros((0, 92, 60))).shape, (0, len(W.HANDCRAFTED_NAMES)))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class ReplayParityTest(unittest.TestCase):
    """실제 replay + Level1Engine: LEVEL1 시점의 kp_seq와 사건 구간이 학습 창 규칙과 같아야 함."""

    def test_level1_seq_and_event_window_match_training_windows(self):
        import dataclasses
        import numpy as np
        from app.ai_classifier import features as F
        from app.core.level1 import Level1Params
        from training.datasets import clipfmt
        from training.experiments import replay as R
        from training.experiments import windows as W
        clip = make_arrays(duration_s=8.0, hz=30.0, fall_at=3.0)
        p = dataclasses.replace(Level1Params(), ai_mode="off")
        obs = R.clip_to_obs(clip, t0=1000.0)
        ts, seq, _ = W.runtime_v1_stream(obs, p.min_quality)
        arr = W.obs_event_arrays(obs, p.min_quality)
        hits = []

        def on_event(ev, eng):
            if ev["type"] != "level1":
                return
            i = int(np.searchsorted(ts, ev["ts"] - 1000.0 + 1e-6))
            np.testing.assert_allclose(np.asarray(list(eng.kp_seq), dtype=np.float32), seq[i - 30:i], atol=1e-6)
            snap = eng.get_incident_snapshot()
            c = clipfmt.from_sidecar({"snapshot": snap})
            ws, we = W.model_window_bounds(c["meta"]["window_start_s"], c["meta"]["window_end_s"])
            x_rt = F.window_v2(c["ts"], c["kps"], c["bbox"], c["present"], c["frame_wh"], ws, we)
            x_tr = W._v2_from_arrays(arr, snap["event_anchor_ts"] - 1000.0)
            np.testing.assert_allclose(x_rt, x_tr, atol=1e-4)
            hits.append(ev)

        R.run_clip(obs, p, on_event=on_event)
        if not hits:
            self.skipTest("합성 클립에서 LEVEL1이 나지 않음 (규칙 파라미터 변경)")


if __name__ == "__main__":
    unittest.main()
