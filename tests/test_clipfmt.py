"""
클립 키포인트 파일 형식(training/datasets/clipfmt.py)과 추출기(extract_keypoints.py) 테스트

추출기는 cv2/ultralytics 없이 가짜 영상/가짜 예측기로 검증합니다. numpy가 없으면 건너뜁니다.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

HAS_NUMPY = importlib.util.find_spec("numpy") is not None

if HAS_NUMPY:
    import numpy as np

    from app.core.kp_buffer import TimedKeypointBuffer
    from app.eval.synth import make_scenario
    from training.datasets import extract_keypoints as ek
    from training.datasets.clipfmt import (CLIP_FORMAT_VERSION, from_obs_records, from_sidecar, load_clip,
                                           save_clip, save_clip_dict)


def _clip(t=5):
    rng = np.random.default_rng(0)
    ts = np.arange(t) / 10.0
    kps = rng.random((t, 17, 3)).astype(np.float32)
    bbox = np.tile(np.array([0.1, 0.2, 0.5, 0.9], np.float32), (t, 1))
    present = np.ones(t, bool)
    present[1] = False
    kps[1] = 0
    bbox[1] = 0
    return ts, kps, bbox, present, (640, 480)


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class ClipFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_keys_dtypes_meta(self):
        ts, kps, bbox, present, wh = _clip()
        path = self.dir / "a" / "b" / "clip.npz"
        save_clip(path, ts.tolist(), kps.astype(np.float64), bbox, present.astype(int), list(wh),
                  {"dataset": "test", "clip_id": "test/clip", "extractor": {"model": "m"}})
        self.assertTrue(path.is_file())
        self.assertEqual(sorted(os.listdir(path.parent)), ["clip.npz"])  # 임시 파일이 남지 않음
        c = load_clip(path)
        self.assertEqual(set(c), {"ts", "kps", "bbox", "present", "frame_wh", "meta"})
        self.assertEqual(c["ts"].dtype, np.float64)
        self.assertEqual(c["kps"].dtype, np.float32)
        self.assertEqual(c["bbox"].dtype, np.float32)
        self.assertEqual(c["present"].dtype, np.bool_)
        self.assertEqual(c["frame_wh"].dtype, np.int32)
        np.testing.assert_array_equal(c["kps"], kps)
        np.testing.assert_array_equal(c["present"], present)
        self.assertEqual(c["frame_wh"].tolist(), [640, 480])
        self.assertEqual(c["meta"]["clip_format_version"], CLIP_FORMAT_VERSION)
        self.assertEqual(c["meta"]["n_frames"], 5)
        self.assertEqual(c["meta"]["extractor"], {"model": "m"})
        self.assertIn("created_at", c["meta"])

    def test_meta_is_plain_string_array(self):
        ts, kps, bbox, present, wh = _clip()
        path = self.dir / "c.npz"
        save_clip(path, ts, kps, bbox, present, wh, {"clip_id": "한글/id"})
        with np.load(path, allow_pickle=False) as z:
            self.assertEqual(z["meta"].shape, ())
            self.assertEqual(json.loads(str(z["meta"][()]))["clip_id"], "한글/id")

    def test_empty_clip(self):
        path = self.dir / "e.npz"
        save_clip(path, [], [], [], [], (1, 1), {})
        c = load_clip(path)
        self.assertEqual(c["kps"].shape, (0, 17, 3))
        self.assertEqual(c["bbox"].shape, (0, 4))

    def test_rejects_bad_shapes_and_values(self):
        ts, kps, bbox, present, wh = _clip()
        bad = [
            (ts, kps[:, :16], bbox, present, wh),
            (ts, kps, bbox[:4], present, wh),
            (ts, kps, bbox, present[:3], wh),
            (ts, kps, bbox, present, (1, 2, 3)),
            (ts[::-1], kps, bbox, present, wh),  # 시간 역행
        ]
        nan_kps = kps.copy()
        nan_kps[0, 0, 0] = np.nan
        bad.append((ts, nan_kps, bbox, present, wh))
        for i, args in enumerate(bad):
            with self.subTest(i=i), self.assertRaises(ValueError):
                save_clip(self.dir / f"bad{i}.npz", *args, {})
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_rejects_newer_version(self):
        ts, kps, bbox, present, wh = _clip()
        path = self.dir / "v.npz"
        # 미래 버전 파일은 save_clip으로 만들 수 없으므로 직접 작성
        meta = json.dumps({"clip_format_version": CLIP_FORMAT_VERSION + 1})
        np.savez_compressed(path, ts=ts, kps=kps, bbox=bbox, present=present,
                            frame_wh=np.array(wh, np.int32), meta=np.array(meta))
        with self.assertRaises(ValueError):
            load_clip(path)

    def test_save_overwrites_stale_derived_meta(self):
        # 잘라낸 클립을 원래 meta 그대로 다시 저장해도 n_frames/버전은 실제 배열 기준
        ts, kps, bbox, present, wh = _clip(5)
        path = self.dir / "s.npz"
        save_clip(path, ts[:3], kps[:3], bbox[:3], present[:3], wh,
                  {"n_frames": 5, "clip_format_version": CLIP_FORMAT_VERSION + 7, "clip_id": "x"})
        c = load_clip(path)
        self.assertEqual(c["meta"]["n_frames"], 3)
        self.assertEqual(c["meta"]["clip_format_version"], CLIP_FORMAT_VERSION)
        self.assertEqual(c["meta"]["clip_id"], "x")


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class FromRuntimeDataTest(unittest.TestCase):
    def test_from_obs_records(self):
        obs, _ = make_scenario("fast_fall", fps=15.0, lie_s=5.0, t0=1000.0)
        obs = json.loads(json.dumps(obs))  # JSONL로 기록된 것처럼 (frame_shape 튜플 -> 리스트)
        obs[3]["tracks"] = []  # 트랙 없는 obs
        # keypoints_raw를 우선 사용
        obs[5]["tracks"][0]["keypoints"] = [{"id": i, "x": 0.0, "y": 0.0, "conf": 0.0} for i in range(17)]
        c = from_obs_records(obs, dataset="runs", clip_id="runs/obs_x")
        t = len(obs)
        self.assertEqual(c["kps"].shape, (t, 17, 3))
        self.assertAlmostEqual(float(c["ts"][0]), 0.0)
        self.assertAlmostEqual(float(c["ts"][-1]), obs[-1]["ts"] - obs[0]["ts"], places=6)
        self.assertFalse(c["present"][3])
        self.assertTrue(np.all(c["kps"][3] == 0) and np.all(c["bbox"][3] == 0))
        raw5 = obs[5]["tracks"][0]["keypoints_raw"]
        self.assertAlmostEqual(float(c["kps"][5, raw5[0]["id"], 0]), raw5[0]["x"], places=6)
        h, w = obs[0]["tracks"][0]["frame_shape"]
        self.assertEqual(c["frame_wh"].tolist(), [w, h])
        self.assertEqual(c["meta"]["frame_wh_source"], "recorded")
        self.assertEqual(c["meta"]["clip_id"], "runs/obs_x")
        self.assertAlmostEqual(c["meta"]["native_fps"], 15.0, places=1)
        with tempfile.TemporaryDirectory() as d:
            save_clip_dict(Path(d) / "o.npz", c)
            np.testing.assert_array_equal(load_clip(Path(d) / "o.npz")["kps"], c["kps"])

    def _window(self, with_frame_wh):
        b = TimedKeypointBuffer(horizon_s=10.0)
        for i in range(20):
            frame = None if i in (4, 5) else [[0.5, 0.25 + i * 0.01, 0.9]] * 17
            b.append(500.0 + i / 15.0, frame, [0.1, 0.1, 0.6, 0.9] if frame else None)
        win = b.window(500.0, 502.0)
        win.pop("frame_wh", None)
        if with_frame_wh:
            win["frame_wh"] = [1280, 720]
        return win

    def test_from_sidecar(self):
        sidecar = {"schema_version": 1, "event": {"type": "level1", "ts": 501.0},
                   "review": {"human_label": "fall"},
                   "snapshot": {"abnormal_start_ts": 500.5, "event_window": self._window(True)}}
        c = from_sidecar(json.loads(json.dumps(sidecar)))
        self.assertEqual(c["kps"].shape, (20, 17, 3))
        self.assertEqual(c["present"].tolist(), [i not in (4, 5) for i in range(20)])
        self.assertTrue(np.all(c["kps"][4] == 0) and np.all(c["bbox"][5] == 0))
        self.assertAlmostEqual(float(c["kps"][6, 0, 1]), 0.31, places=5)
        self.assertEqual(c["frame_wh"].tolist(), [1280, 720])
        self.assertAlmostEqual(c["meta"]["t0_abs"], 500.0)
        self.assertEqual(c["meta"]["human_label"], "fall")
        self.assertEqual(c["meta"]["event_type"], "level1")
        self.assertIsNone(c["meta"]["review_fall_start_s"])

    def test_from_sidecar_keeps_review_fall_times_relative(self):
        sidecar = {"review": {"human_label": "fall", "fall_start_ts": 500.8, "fall_end_ts": 501.5},
                   "snapshot": {"event_window": self._window(True)}}
        c = from_sidecar(sidecar)
        self.assertAlmostEqual(c["meta"]["review_fall_start_s"], 0.8, places=6)
        self.assertAlmostEqual(c["meta"]["review_fall_end_s"], 1.5, places=6)

    def test_from_sidecar_without_frame_wh_uses_config_default(self):
        from app.config import FRAME_H, FRAME_W
        c = from_sidecar({"snapshot": {"event_window": self._window(False)}})
        self.assertEqual(c["frame_wh"].tolist(), [FRAME_W, FRAME_H])
        self.assertEqual(c["meta"]["frame_wh_source"], "config_default")

    def test_from_sidecar_without_window_raises(self):
        with self.assertRaises(ValueError):
            from_sidecar({"snapshot": {"event_window": None}})


# ---------------------------------------------------------------------------
# 추출기 (가짜 영상 + 가짜 예측기)
# ---------------------------------------------------------------------------

def _person(x, y, w=100.0, h=200.0):
    """픽셀 박스와 그 안의 키포인트 17개."""
    box = np.array([x, y, x + w, y + h], np.float32)
    kxy = np.stack([np.linspace(x, x + w, 17), np.linspace(y, y + h, 17)], axis=1).astype(np.float32)
    return box, kxy


class FakePredictor:
    """프레임 순서대로 미리 정한 사람 목록 [(x, y, conf), ...]을 돌려줍니다."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.frames_seen = 0

    def __call__(self, frames):
        self.calls += 1
        out = []
        for _ in frames:
            people = self.script[self.frames_seen % len(self.script)]
            self.frames_seen += 1
            if not people:
                out.append(ek.empty_detections())
                continue
            boxes, kxy = zip(*[_person(x, y) for x, y, _ in people])
            confs = np.array([c for _, _, c in people], np.float32)
            kcf = np.full((len(people), 17), 0.8, np.float32)
            out.append((np.stack(boxes), confs, np.stack(kxy), kcf))
        return out


def _frames(n, w=640, h=480):
    return (np.zeros((h, w, 3), np.uint8) for _ in range(n))


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class ExtractFramesTest(unittest.TestCase):
    def test_tracks_person_like_runtime(self):
        # 0: 한 명(A) / 1: A(낮은 conf) + 더 높은 conf B -> A 유지 / 2: 없음 -> 리셋 / 3: A + B -> 최고 conf B
        script = [
            [(100, 100, 0.7)],
            [(105, 100, 0.6), (400, 200, 0.95)],
            [],
            [(105, 100, 0.6), (400, 200, 0.95)],
        ]
        c = ek.extract_frames(_frames(4), 20.0, FakePredictor(script), batch=3)
        np.testing.assert_allclose(c["ts"], [0, 0.05, 0.1, 0.15])
        self.assertEqual(c["present"].tolist(), [True, True, False, True])
        self.assertAlmostEqual(float(c["bbox"][1, 0]), 105 / 640, places=6)   # A 유지
        self.assertAlmostEqual(float(c["bbox"][3, 0]), 400 / 640, places=6)   # 리셋 후 최고 conf
        self.assertTrue(np.all(c["kps"][2] == 0) and np.all(c["bbox"][2] == 0))
        self.assertEqual(c["frame_wh"], [640, 480])
        self.assertEqual(c["n_switch"], 0)  # 리셋 후 첫 선택은 교체로 세지 않음 (런타임과 같음)
        self.assertEqual(c["n_multi"], 2)

    def test_switch_counted_when_tracked_person_disappears(self):
        script = [[(100, 100, 0.9)], [(400, 200, 0.8)]]
        c = ek.extract_frames(_frames(2), 30.0, FakePredictor(script), batch=1)
        self.assertEqual(c["n_switch"], 1)

    def test_normalized_and_clipped(self):
        script = [[(600, 400, 0.9)]]  # 박스가 프레임 밖으로 나감 (600+100 > 640)
        c = ek.extract_frames(_frames(1), 30.0, FakePredictor(script))
        self.assertTrue(np.all(c["kps"][..., :2] >= 0) and np.all(c["kps"][..., :2] <= 1))
        self.assertAlmostEqual(float(c["kps"][0, 0, 0]), 600 / 640, places=6)
        self.assertEqual(float(c["kps"][0, 16, 0]), 1.0)
        self.assertAlmostEqual(float(c["kps"][0, 0, 2]), 0.8, places=6)
        self.assertEqual(float(c["bbox"][0, 2]), 1.0)

    def test_batch_size_does_not_change_result(self):
        script = [[(100, 100, 0.7)], [(105, 100, 0.6), (400, 200, 0.95)], [], [(300, 100, 0.9)]] * 3
        a = ek.extract_frames(_frames(12), 25.0, FakePredictor(script), batch=1)
        b = ek.extract_frames(_frames(12), 25.0, FakePredictor(script), batch=5)
        for k in ("ts", "kps", "bbox", "present"):
            np.testing.assert_array_equal(a[k], b[k])

    def test_normalize_fps(self):
        self.assertEqual(ek.normalize_fps(25.0), (25.0, "container"))
        for bad in (None, 0.0, -1, float("nan"), 1000.0, "x"):
            self.assertEqual(ek.normalize_fps(bad), (30.0, "fallback"))

    def test_clip_out_path(self):
        out = Path("/tmp/out")
        self.assertEqual(ek.clip_out_path(out, "le2i/Coffee_room_01/video_14"),
                         out / "le2i" / "Coffee_room_01" / "video_14.npz")
        self.assertEqual(ek.clip_out_path(out, "a/b:c").name, "b_c.npz")
        for bad in ("../x", "a/../../x", "", "/", None, "  "):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ek.clip_out_path(out, bad)


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class RunExtractionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "videos").mkdir()
        for name in ("a.mp4", "b.avi", "broken.mp4"):
            (self.root / "videos" / name).write_bytes(b"x")
        self.rows = [
            {"clip_id": "ds/a", "dataset": "ds", "video": "videos/a.mp4"},
            {"clip_id": "ds/missing", "dataset": "ds", "video": "videos/none.mp4"},
            {"clip_id": "ds/broken", "dataset": "ds", "video": "videos/broken.mp4"},
            {"clip_id": "ds/b", "dataset": "ds", "video": "videos/b.avi"},
        ]
        self.out = self.root / "out"
        self.made = 0

    def tearDown(self):
        self._tmp.cleanup()

    def open_video(self, path):
        if path.name == "broken.mp4":
            raise RuntimeError("cannot open video")
        return (0.0 if path.suffix == ".avi" else 25.0), _frames(6, 320, 240)

    def get_predictor(self):
        self.made += 1
        return FakePredictor([[(10, 10, 0.9)], []])

    def run_once(self, **kw):
        return ek.run_extraction(self.rows, self.out, self.get_predictor, self.open_video,
                                 {"model": "fake.pt", "conf": 0.5}, repo_root=self.root, log=lambda *_: None, **kw)

    def read_log(self):
        with open(self.out / ek.LOG_NAME, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def test_extracts_logs_and_survives_bad_videos(self):
        stats = self.run_once()
        self.assertEqual(stats, {"total": 4, "done": 2, "failed": 2, "skipped": 0})
        self.assertEqual(self.made, 1)  # 모델은 한 번만 로드
        c = load_clip(self.out / "ds" / "a.npz")
        self.assertEqual(c["kps"].shape, (6, 17, 3))
        self.assertEqual(c["present"].tolist(), [True, False] * 3)
        self.assertEqual(c["frame_wh"].tolist(), [320, 240])
        self.assertEqual(c["meta"]["clip_id"], "ds/a")
        self.assertEqual(c["meta"]["native_fps"], 25.0)
        self.assertEqual(c["meta"]["fps_source"], "container")
        self.assertEqual(c["meta"]["extractor"]["model"], "fake.pt")
        self.assertAlmostEqual(c["meta"]["present_ratio"], 0.5)
        b = load_clip(self.out / "ds" / "b.npz")
        self.assertEqual(b["meta"]["fps_source"], "fallback")
        np.testing.assert_allclose(b["ts"][1], 1 / 30.0)

        log = {r["clip_id"]: r for r in self.read_log()}
        self.assertTrue(log["ds/a"]["ok"] and log["ds/b"]["ok"])
        self.assertEqual(log["ds/a"]["frames"], 6)
        self.assertIn("FileNotFoundError", log["ds/missing"]["error"])
        self.assertIn("cannot open video", log["ds/broken"]["error"])
        for r in log.values():
            self.assertTrue({"clip_id", "ok", "frames", "secs", "error"} <= set(r))

    def test_resume_skips_existing_and_limit_counts_new_attempts(self):
        stats = self.run_once(limit=1)
        self.assertEqual((stats["done"], stats["failed"]), (1, 0))
        self.assertTrue((self.out / "ds" / "a.npz").is_file())
        stats = self.run_once(limit=1)  # a는 건너뛰고 missing 1개만 시도
        self.assertEqual((stats["skipped"], stats["done"], stats["failed"]), (1, 0, 1))
        stats = self.run_once()
        self.assertEqual((stats["skipped"], stats["done"], stats["failed"]), (1, 1, 2))
        stats = self.run_once()
        self.assertEqual((stats["skipped"], stats["done"]), (2, 0))

    def test_records_container_frame_count(self):
        class Frames:
            frame_count = 9.0  # cv2 CAP_PROP_FRAME_COUNT는 float

            def __iter__(self):
                return iter(list(_frames(6, 320, 240)))

        self.open_video = lambda path: (25.0, Frames())
        self.rows = [self.rows[0]]
        self.run_once()
        c = load_clip(self.out / "ds" / "a.npz")
        self.assertEqual((c["meta"]["n_frames"], c["meta"]["container_frames"]), (6, 9))  # 디코딩이 짧게 끝남
        self.assertEqual(self.read_log()[0]["container_frames"], 9)
        # frame_count가 없는 이터러블(제너레이터)이면 None
        self.open_video = lambda path: (25.0, _frames(6, 320, 240))
        self.rows = [dict(self.rows[0], clip_id="ds/a2")]
        self.run_once()
        self.assertIsNone(load_clip(self.out / "ds" / "a2.npz")["meta"]["container_frames"])

    def test_crop_px_applied_before_detection(self):
        seen = []

        class Recorder(FakePredictor):
            def __call__(self, frames):
                seen.extend(f.shape for f in frames)
                return super().__call__(frames)

        self.get_predictor = lambda: Recorder([[(10, 10, 0.9)]])
        self.rows = [dict(self.rows[0], crop_px=[160, 0, 320, 240])]
        self.run_once()
        c = load_clip(self.out / "ds" / "a.npz")
        self.assertEqual(c["frame_wh"].tolist(), [160, 240])
        self.assertEqual(c["meta"]["crop_px"], [160, 0, 320, 240])
        self.assertTrue(seen and all(s[:2] == (240, 160) for s in seen))

    def test_crop_outside_frame_fails_that_clip_only(self):
        self.rows = [dict(self.rows[0], crop_px=[0, 0, 999, 240]), self.rows[3]]
        stats = self.run_once()
        self.assertEqual((stats["done"], stats["failed"]), (1, 1))
        log = {r["clip_id"]: r for r in self.read_log()}
        self.assertIn("crop_px", log["ds/a"]["error"])

    def test_manifest_fps_overrides_container_header(self):
        # MCFD처럼 헤더 fps(25)가 실제(30)와 다르면 manifest fps를 사용
        self.rows = [dict(self.rows[0], fps=30.0, segments=[{"cls": "fall", "start_s": 0.05, "end_s": 0.15}])]
        self.run_once()
        c = load_clip(self.out / "ds" / "a.npz")
        self.assertEqual((c["meta"]["native_fps"], c["meta"]["fps_source"], c["meta"]["container_fps"]),
                         (30.0, "manifest", 25.0))
        np.testing.assert_allclose(c["ts"][1], 1 / 30.0)

    def test_label_timebase_mismatch_fails_clip(self):
        # 6프레임 @25fps = 0.24초인데 라벨이 3초에 끝나면 시간축 오류 -> 이 클립만 실패
        self.rows = [dict(self.rows[0], segments=[{"cls": "fall", "start_s": 2.0, "end_s": 3.0}]), self.rows[3]]
        stats = self.run_once()
        self.assertEqual((stats["done"], stats["failed"]), (1, 1))
        log = {r["clip_id"]: r for r in self.read_log()}
        self.assertIn("timebase", log["ds/a"]["error"])
        # 허용 범위 안(길이 x 1.05 + 0.5초)이면 통과
        ek.check_label_timebase({"segments": [{"cls": "fall", "start_s": 0.0, "end_s": 0.7}]}, 6, 25.0)

    def test_cv2_frames_releases_capture(self):
        class Cap:
            def __init__(self):
                self.left, self.released = 3, False

            def read(self):
                if self.left == 0:
                    return False, None
                self.left -= 1
                return True, np.zeros((2, 2, 3), np.uint8)

            def release(self):
                self.released = True

        cap = Cap()
        frames = ek._Cv2Frames(cap, 3.0)
        self.assertEqual(len(list(frames)), 3)
        self.assertTrue(cap.released)
        self.assertEqual(ek._container_frame_count(frames), 3)

    def test_no_model_load_when_everything_done_or_missing(self):
        self.rows = [self.rows[1]]
        self.run_once()
        self.assertEqual(self.made, 0)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class SidecarExtractorParityTest(unittest.TestCase):
    """같은 동작을 추출기 클립과 운영 sidecar(kp_buffer 사건 구간)로 만들었을 때 v2 특징이 같아야 함."""

    def test_window_v2_matches(self):
        from app.ai_classifier.features import MAX_GAP_S, window_v2

        rng = np.random.default_rng(3)
        fps, n = 15.0, 90
        t0 = 1000.0 + 0.013  # 운영 ts는 프레임 격자와 어긋나 있음
        ts_abs = t0 + np.arange(n) / fps
        kps = np.zeros((n, 17, 3), np.float32)
        base = rng.random((17, 2)) * 0.2 + 0.4
        for i in range(n):
            kps[i, :, :2] = base + 0.002 * i
            kps[i, :, 2] = 0.9
        bbox = np.tile(np.array([0.3, 0.2, 0.6, 0.9], np.float32), (n, 1))
        wh = (1920, 1080)

        # 운영: kp_buffer에 쌓고 [ts_s, ts_e] 구간을 여유와 함께 잘라 sidecar로 저장
        buf = TimedKeypointBuffer(horizon_s=30.0)
        for i in range(n):
            buf.append(ts_abs[i], kps[i].tolist(), bbox[i].tolist(), wh)
        ts_s, ts_e = ts_abs[30] + 0.031, ts_abs[30] + 0.031 + 4.0
        win = buf.window(ts_s, ts_e, pad_s=0.25)
        side = from_sidecar(json.loads(json.dumps({"snapshot": {"event_window": win}})))
        m = side["meta"]
        f_side = window_v2(side["ts"], side["kps"], side["bbox"], side["present"], side["frame_wh"],
                           m["window_start_s"], m["window_end_s"])

        # 학습: 영상 전체 클립에서 같은 구간
        f_clip = window_v2(ts_abs - t0, kps, bbox, np.ones(n, bool), np.array(wh), ts_s - t0, ts_e - t0)
        self.assertEqual(f_side.shape, (92, 60))
        self.assertGreater(f_side[-17:, 0].sum(), 0)  # 첫 열도 마스크되지 않음
        np.testing.assert_allclose(f_side, f_clip, atol=1e-4)
        # 여유가 없으면 첫 열이 마스크됨 (여유가 필요한 이유)
        bare = from_sidecar(json.loads(json.dumps({"snapshot": {"event_window": buf.window(ts_s, ts_e)}})))
        f_bare = window_v2(bare["ts"], bare["kps"], bare["bbox"], bare["present"], bare["frame_wh"],
                           bare["meta"]["window_start_s"], bare["meta"]["window_end_s"])
        self.assertEqual(f_bare[-17:, 0].sum(), 0)
        self.assertGreaterEqual(0.25, MAX_GAP_S)
