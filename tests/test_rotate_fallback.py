"""
회전 폴백(app/engine/rotate_fallback) 테스트.

- fallback_detect: 회전 영상의 검출을 원래 좌표로 되돌리고, 직전 대상 근처의 가장 확실한 것만 고름
- 추출기(extract_frames, fallback=True): 놓친 프레임에서만 폴백, 최근에 본 대상이 없으면 폴백 안 함,
  fallback / switched_base 기록, 재생에서 폴백을 끄면 폴백 없는 추출과 같아짐
- 운영 LivePipeline(ROTATE_FALLBACK)과 추출기가 같은 프레임에서 폴백을 쓰는지 (torch가 있을 때, 별도 프로세스)
numpy와 cv2가 없으면 건너뜁니다.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_CV2 = HAS_NUMPY and importlib.util.find_spec("cv2") is not None
HAS_TORCH = HAS_CV2 and importlib.util.find_spec("torch") is not None
REPO_ROOT = Path(__file__).resolve().parents[1]

if HAS_NUMPY:
    import numpy as np

    from app.engine import rotate_fallback as RF

W, H, FPS = 64, 48, 15.0
STAND = (10.0, 5.0, 30.0, 45.0)      # 원래 방향에서 잡히는 사람 (픽셀)
LYING = (8.0, 30.0, 48.0, 44.0)      # 누운 사람 (원래 방향에서는 안 잡힘, 회전하면 잡힘)


def _script():
    """프레임별 상태: stand / lying / empty. 0~19 서 있음, 20~39 누움(폴백 대상), 40~129 아무도 없음(6초),
    130~139 누움 (마지막으로 본 뒤 5초가 넘어 폴백 안 함), 140~149 서 있음"""
    return ["stand"] * 20 + ["lying"] * 20 + ["empty"] * 90 + ["lying"] * 10 + ["stand"] * 10


def _frame(i):
    # 프레임 번호를 화소 값에 넣음 (회전해도 그대로라 가짜 모델이 어떤 프레임인지 알 수 있음)
    return np.full((H, W, 3), i % 250, np.uint8)


def _det(box, conf=0.9):
    b = np.array([box], np.float32)
    kxy = np.stack([np.linspace(box[0], box[2], 17), np.linspace(box[1], box[3], 17)], 1)[None].astype(np.float32)
    return b, np.array([conf], np.float32), kxy, np.full((1, 17), 0.8, np.float32)


def _empty():
    return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 17, 2), np.float32), np.zeros((0, 17),
                                                                                                          np.float32)


def fake_predict_np(imgs, script):
    """영상 목록 -> [(boxes, confs, kxy, kcf)]. 원래 방향(H x W)은 서 있을 때만, 90도 회전 영상은 누웠을 때만 검출"""
    out = []
    for im in imgs:
        state = script[int(im[0, 0, 0])]
        if im.shape[:2] == (H, W):
            out.append(_det(STAND) if state == "stand" else _empty())
        elif state == "lying" and im.shape[:2] == (W, H):
            # 90도 회전 영상의 좌표로 줌 (270도 영상에는 일부러 아무것도 없음)
            c = RF.to_rotated(np.array([[LYING[0], LYING[1]], [LYING[2], LYING[3]]]), W, H, 90)
            box = (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())
            probe = RF.rotate_image(np.zeros((H, W), np.uint8), 90)
            out.append(_det(box, 0.8) if probe.shape == im.shape[:2] and _is_90(im) else _empty())
        else:
            out.append(_empty())
    return out


def _is_90(im):
    # 가짜 영상은 값이 고르므로 회전 방향을 구분할 수 없음: 90/270 모두 (W, H) 크기라 둘 다 검출이 나오면
    # 같은 사람이 두 번 잡히는 셈이라 첫 번째(90도)만 검출하도록 호출 순서로 구분 (fallback_detect는 90 -> 270 순서)
    _is_90.calls = getattr(_is_90, "calls", 0) + 1
    return _is_90.calls % 2 == 1


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class FallbackDetectTest(unittest.TestCase):
    def test_maps_back_and_filters_by_distance(self):
        img = np.zeros((H, W, 3), np.uint8)
        true_box = np.array(LYING)

        def predict(imgs):
            out = []
            for a, im in zip(RF.FALLBACK_ANGLES, imgs):
                self.assertEqual(im.shape[:2][::-1], RF.rotated_size(W, H, a))
                c = RF.to_rotated(np.array([[LYING[0], LYING[1]], [LYING[2], LYING[3]]]), W, H, a)
                b = np.array([[c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()]])
                k = np.tile(np.r_[RF.to_rotated(np.array([20.0, 35.0]), W, H, a), 0.7], (17, 1))[None]
                out.append((b, np.array([0.6 if a == 90 else 0.9]), k))
            return out
        fb = RF.fallback_detect(predict, img, np.array(STAND))
        np.testing.assert_allclose(fb[0], true_box)
        np.testing.assert_allclose(fb[2][0, :2], [20.0, 35.0])
        self.assertEqual(fb[3], 270)                      # 신뢰도가 더 높은 방향
        far = np.array([200.0, 200.0, 220.0, 240.0])
        self.assertIsNone(RF.fallback_detect(predict, img, far))   # 직전 대상과 멀면 버림

    def test_near(self):
        self.assertTrue(RF.near(np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15])))
        self.assertFalse(RF.near(np.array([0, 0, 10, 10]), np.array([50, 50, 60, 60])))


@unittest.skipUnless(HAS_CV2, "numpy/cv2 없음")
class ExtractorFallbackTest(unittest.TestCase):
    def _extract(self, fallback):
        from training.datasets import extract_keypoints as ek
        script = _script()
        _is_90.calls = 0
        frames = [_frame(i) for i in range(len(script))]
        return ek.extract_frames(iter(frames), FPS, lambda fr: fake_predict_np(fr, script), batch=4,
                                 fallback=fallback), script

    def test_fallback_only_on_recent_lost_frames(self):
        clip, script = self._extract(True)
        fb = clip["fallback"].tolist()
        self.assertEqual([i for i, v in enumerate(fb) if v], list(range(20, 40)))   # 130~139는 5초가 넘어 안 씀
        self.assertTrue(all(clip["present"][20:40]))
        self.assertFalse(any(clip["present"][130:140]))
        self.assertEqual(clip["n_fallback"], 20)
        np.testing.assert_allclose(clip["bbox"][25] * [W, H, W, H], LYING, atol=1e-4)
        base, _ = self._extract(False)
        self.assertIsNone(base["fallback"])
        self.assertEqual(base["switched"].tolist(), clip["switched_base"].tolist())
        self.assertEqual(base["present"].tolist(), (clip["present"] & ~clip["fallback"]).tolist())

    def test_replay_without_fallback_matches_plain_extraction(self):
        from training.experiments import replay
        clip, _ = self._extract(True)
        base, _ = self._extract(False)
        clip.update(meta={}, frame_wh=[W, H])
        base.update(meta={}, frame_wh=[W, H])
        old = replay.USE_FALLBACK
        try:
            replay.USE_FALLBACK = False
            a = replay.clip_to_obs(clip)
            b = replay.clip_to_obs(base)
        finally:
            replay.USE_FALLBACK = old
        self.assertEqual([o["tracks"][0]["has_person"] for o in a], [o["tracks"][0]["has_person"] for o in b])
        self.assertEqual([o["target_switched"] for o in a], [o["target_switched"] for o in b])
        with_fb = replay.clip_to_obs(clip)
        self.assertGreater(sum(o["tracks"][0]["has_person"] for o in with_fb),
                           sum(o["tracks"][0]["has_person"] for o in a))


_LIVE_SCRIPT = textwrap.dedent(
    """
    import sys, json
    from unittest import mock
    import numpy as np
    import torch
    sys.path.insert(0, "TESTS_DIR")
    import test_rotate_fallback as T
    from app.engine import live
    from training.datasets import extract_keypoints as ek

    script = T._script()

    class Boxes:
        def __init__(self, xyxy, conf):
            self.xyxy, self.conf = xyxy, conf
        def __len__(self):
            return int(self.xyxy.shape[0])
        def __getitem__(self, s):
            return Boxes(self.xyxy[s], self.conf[s])

    class Kps:
        def __init__(self, xy, conf):
            self.xy, self.conf = xy, conf
        def __len__(self):
            return int(self.xy.shape[0])
        def __getitem__(self, s):
            return Kps(self.xy[s], self.conf[s])

    class Res:
        def __init__(self, det):
            b, c, kxy, kcf = det
            self.boxes = Boxes(torch.from_numpy(b), torch.from_numpy(c))
            self.keypoints = Kps(torch.from_numpy(kxy), torch.from_numpy(kcf))

    def predict(src, **kw):
        imgs = src if isinstance(src, list) else [src]
        return [Res(d) for d in T.fake_predict_np(imgs, script)]

    frames = [T._frame(i) for i in range(len(script))]
    cap = mock.MagicMock()
    cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
    model = mock.MagicMock()
    model.predict.side_effect = predict
    clock = iter(1000.0 + i / T.FPS for i in range(10 ** 6))
    T._is_90.calls = 0
    with mock.patch.object(live, "time", mock.MagicMock(time=lambda: next(clock))):
        pipe = live.LivePipeline(model, cap, jpeg_quality=80)
        pipe.rotate_fallback = True
        runtime_fb, runtime_present = [], []
        while True:
            obs, _, _ = pipe.step()
            if obs is None:
                break
            runtime_fb.append(bool(obs["meta"].get("rotate_fallback")))
            runtime_present.append(bool(obs["tracks"][0]["has_person"]))
    T._is_90.calls = 0
    clip = ek.extract_frames(iter(frames), T.FPS, lambda fr: T.fake_predict_np(fr, script), batch=4, fallback=True)
    ok = runtime_fb == clip["fallback"].tolist() and runtime_present == clip["present"].tolist() and sum(runtime_fb) == 20
    print(json.dumps({"runtime_fallback": sum(runtime_fb), "extractor_fallback": int(clip["fallback"].sum()),
                      "present_mismatch": sum(a != b for a, b in zip(runtime_present, clip["present"].tolist()))}))
    sys.exit(0 if ok else 1)
    """
)


@unittest.skipUnless(HAS_TORCH and importlib.util.find_spec("ultralytics") is not None, "torch/ultralytics 없음")
class LiveFallbackParityTest(unittest.TestCase):
    def test_live_pipeline_matches_extractor(self):
        with tempfile.TemporaryDirectory() as cwd:
            env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), RUNS_DIR=os.path.join(cwd, "runs"))
            code = _LIVE_SCRIPT.replace("TESTS_DIR", Path(__file__).resolve().parent.as_posix())
            r = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True,
                               timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr[-2000:])
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1])["runtime_fallback"], 20)


if __name__ == "__main__":
    unittest.main()
