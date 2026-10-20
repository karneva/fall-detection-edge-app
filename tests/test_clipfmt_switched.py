"""
클립의 프레임별 대상 전환 기록(clipfmt "switched")과 추출기/리플레이 연결 테스트

- 저장/읽기, 검증, from_obs_records/from_sidecar 매핑
- 추출기(extract_frames)의 프레임별 전환이 런타임(LivePipeline.step의 obs.target_switched)과 같은지
  (torch가 있을 때 별도 프로세스에서 cv2/ultralytics를 스텁으로 바꿔 확인. 회전 폴백은 끄고, 추출기가 기록하는
  "all" 정책은 그대로, 운영 기본 "multi" 정책은 switched & n_cand >= 2와 같은지)
- 저장한 클립을 replay.clip_to_obs로 바꾸면 해당 프레임 obs에 target_switched가 실리는지
numpy가 없으면 건너뜁니다.
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
HAS_TORCH = HAS_NUMPY and importlib.util.find_spec("torch") is not None
REPO_ROOT = Path(__file__).resolve().parents[1]

if HAS_NUMPY:
    import numpy as np

    from app.eval.synth import make_scenario
    from training.datasets import extract_keypoints as ek
    from training.datasets.clipfmt import (from_obs_records, from_sidecar, load_clip, save_clip,
                                           save_clip_dict)
    from training.experiments import replay


def _clip(t=6):
    ts = np.arange(t) / 15.0
    kps = np.full((t, 17, 3), 0.5, np.float32)
    bbox = np.tile(np.array([0.1, 0.2, 0.5, 0.9], np.float32), (t, 1))
    return ts, kps, bbox, np.ones(t, bool), (640, 480)


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class SwitchedFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_with_switched(self):
        ts, kps, bbox, present, wh = _clip()
        for i, sw in enumerate(([False, False, True, False, False, True], [0, 0, 1, 0, 0, 1],
                                np.array([0, 0, 1, 0, 0, 1], np.uint8))):
            with self.subTest(i=i):
                path = self.dir / f"s{i}.npz"
                save_clip(path, ts, kps, bbox, present, wh, {}, switched=sw)
                c = load_clip(path)
                self.assertEqual(c["switched"].dtype, np.bool_)
                self.assertEqual(c["switched"].tolist(), [False, False, True, False, False, True])

    def test_without_switched_key_is_omitted(self):
        ts, kps, bbox, present, wh = _clip()
        path = self.dir / "n.npz"
        save_clip(path, ts, kps, bbox, present, wh, {})
        with np.load(path, allow_pickle=False) as z:
            self.assertNotIn("switched", z.files)
        c = load_clip(path)
        self.assertNotIn("switched", c)
        self.assertIsNone(c.get("switched"))

    def test_old_file_without_switched_loads(self):
        # 이 키가 생기기 전에 저장된 v1 파일 (save_clip을 거치지 않고 직접 작성)
        ts, kps, bbox, present, wh = _clip()
        path = self.dir / "old.npz"
        np.savez_compressed(path, ts=ts, kps=kps, bbox=bbox, present=present, frame_wh=np.array(wh, np.int32),
                            meta=np.array(json.dumps({"clip_format_version": 1})))
        c = load_clip(path)
        self.assertEqual(set(c), {"ts", "kps", "bbox", "present", "frame_wh", "meta"})

    def test_empty_clip_with_switched(self):
        path = self.dir / "e.npz"
        save_clip(path, [], [], [], [], (1, 1), {}, switched=[])
        self.assertEqual(load_clip(path)["switched"].shape, (0,))

    def test_rejects_bad_switched(self):
        ts, kps, bbox, present, wh = _clip()
        bad = [np.zeros(5, bool), np.zeros(7, bool), np.zeros((6, 1), bool), [0, 0, 2, 0, 0, 0],
               [0, 0, 0.5, 0, 0, 0], ["a"] * 6]
        for i, sw in enumerate(bad):
            with self.subTest(i=i), self.assertRaises(ValueError):
                save_clip(self.dir / f"bad{i}.npz", ts, kps, bbox, present, wh, {}, switched=sw)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_load_rejects_wrong_length_in_file(self):
        ts, kps, bbox, present, wh = _clip()
        path = self.dir / "w.npz"
        np.savez_compressed(path, ts=ts, kps=kps, bbox=bbox, present=present, frame_wh=np.array(wh, np.int32),
                            switched=np.zeros(3, bool), meta=np.array(json.dumps({"clip_format_version": 1})))
        with self.assertRaises(ValueError):
            load_clip(path)

    def test_save_clip_dict_keeps_switched(self):
        ts, kps, bbox, present, wh = _clip()
        path = self.dir / "d.npz"
        save_clip(path, ts, kps, bbox, present, wh, {}, switched=[0, 1, 0, 0, 0, 0])
        save_clip_dict(self.dir / "d2.npz", load_clip(path))
        self.assertEqual(load_clip(self.dir / "d2.npz")["switched"].tolist(), [False, True] + [False] * 4)


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class SwitchedFromRuntimeDataTest(unittest.TestCase):
    def test_from_obs_records_maps_target_switched(self):
        obs, _ = make_scenario("fast_fall", fps=15.0, lie_s=2.0, t0=1000.0)
        obs = json.loads(json.dumps(obs))
        obs[4]["target_switched"] = True
        del obs[6]["target_switched"]  # 키가 없으면 False
        c = from_obs_records(obs)
        self.assertEqual(c["switched"].dtype, np.bool_)
        self.assertEqual(np.flatnonzero(c["switched"]).tolist(), [4])
        with tempfile.TemporaryDirectory() as d:
            save_clip_dict(Path(d) / "o.npz", c)
            self.assertEqual(np.flatnonzero(load_clip(Path(d) / "o.npz")["switched"]).tolist(), [4])

    def test_from_obs_records_sorted_with_frames(self):
        obs, _ = make_scenario("fast_fall", fps=15.0, lie_s=2.0, t0=1000.0)
        obs = json.loads(json.dumps(obs))[:10]
        obs[7]["target_switched"] = True
        obs = obs[5:] + obs[:5]  # 시간이 섞인 녹화
        c = from_obs_records(obs)
        self.assertEqual(np.flatnonzero(c["switched"]).tolist(), [7])

    def test_from_sidecar_has_no_switched(self):
        win = {"ts": [500.0, 500.1], "kps": [[[0.5, 0.5, 0.9]] * 17] * 2, "bbox": [[0.1, 0.1, 0.5, 0.9]] * 2}
        self.assertNotIn("switched", from_sidecar({"snapshot": {"event_window": win}}))


class _Fake:
    """프레임 순서대로 미리 정한 사람 목록 [(x, y, conf), ...]을 돌려주는 예측기."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def __call__(self, frames):
        out = []
        for _ in frames:
            people = self.script[self.i % len(self.script)]
            self.i += 1
            if not people:
                out.append(ek.empty_detections())
                continue
            boxes = np.array([[x, y, x + 100, y + 200] for x, y, _ in people], np.float32)
            confs = np.array([c for _, _, c in people], np.float32)
            kxy = np.stack([np.stack([np.linspace(b[0], b[2], 17), np.linspace(b[1], b[3], 17)], 1)
                            for b in boxes]).astype(np.float32)
            out.append((boxes, confs, kxy, np.full((len(people), 17), 0.8, np.float32)))
        return out


def _frames(n, w=640, h=480):
    return (np.zeros((h, w, 3), np.uint8) for _ in range(n))


# 0 A / 1 A / 2 B만 (전환) / 3 없음 / 4 B (놓친 뒤 다시 잡음: 전환 아님) / 5 B + 먼 곳의 높은 conf C (B 유지)
# 6 C만 (전환) / 7 C
SCRIPT = [
    [(100, 100, 0.9)],
    [(102, 100, 0.9)],
    [(400, 200, 0.8)],
    [],
    [(400, 200, 0.8)],
    [(402, 200, 0.6), (50, 50, 0.95)],
    [(50, 50, 0.95)],
    [(52, 50, 0.95)],
]
SCRIPT_SWITCHED = [False, False, True, False, False, False, True, False]


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class ExtractorSwitchedTest(unittest.TestCase):
    def test_per_frame_flags(self):
        for batch in (1, 3, 8):
            with self.subTest(batch=batch):
                c = ek.extract_frames(_frames(len(SCRIPT)), 15.0, _Fake(SCRIPT), batch=batch)
                self.assertEqual(c["switched"].dtype, np.bool_)
                self.assertEqual(c["switched"].tolist(), SCRIPT_SWITCHED)
                self.assertEqual(c["n_switch"], int(c["switched"].sum()))
                self.assertFalse(np.any(c["switched"] & ~c["present"]))  # 전환은 대상이 있는 프레임에만

    def test_empty_video(self):
        c = ek.extract_frames(_frames(0), 15.0, _Fake(SCRIPT))
        self.assertEqual((c["switched"].shape, c["n_switch"]), ((0,), 0))


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class SwitchedReplayIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "v.mp4").write_bytes(b"x")

    def tearDown(self):
        self._tmp.cleanup()

    def _extract(self, fps, script, n):
        stats = ek.run_extraction([{"clip_id": "ds/v", "dataset": "ds", "video": "v.mp4"}], self.root / "out",
                                  lambda: _Fake(script), lambda p: (fps, _frames(n)), {"model": "fake.pt"},
                                  repo_root=self.root, log=lambda *_: None)
        self.assertEqual(stats["done"], 1)
        return load_clip(self.root / "out" / "ds" / "v.npz")

    def test_extracted_clip_carries_switch_into_obs(self):
        c = self._extract(15.0, SCRIPT, len(SCRIPT))
        self.assertEqual(c["switched"].tolist(), SCRIPT_SWITCHED)
        self.assertEqual(c["meta"]["n_target_switch"], 2)
        obs = replay.clip_to_obs(c, tail_s=1.0)
        hit = [o["meta"]["replay_src_index"] for o in obs if o["target_switched"]]
        self.assertEqual(hit, [2, 6])  # 15fps = 운영 주기 -> 원본 프레임과 1:1
        self.assertFalse(any(o["target_switched"] for o in obs if o["meta"]["replay_tail"]))
        # 키가 없는 예전 클립은 전환 없음
        old = {k: v for k, v in c.items() if k != "switched"}
        self.assertFalse(any(o["target_switched"] for o in replay.clip_to_obs(old, tail_s=0)))

    def test_switch_on_skipped_frame_moves_to_next_obs(self):
        # 30fps 원본을 15Hz로 고르면 건너뛴 프레임의 전환은 다음 obs에 실림
        script = [[(100, 100, 0.9)]] * 7 + [[(400, 200, 0.8)]] * 13
        c = self._extract(30.0, script, len(script))
        self.assertEqual(np.flatnonzero(c["switched"]).tolist(), [7])
        obs = replay.clip_to_obs(c, tail_s=0)
        src = [o["meta"]["replay_src_index"] for o in obs]
        hit = [k for k, o in enumerate(obs) if o["target_switched"]]
        self.assertEqual(len(hit), 1)
        self.assertTrue(src[hit[0]] >= 7 and (hit[0] == 0 or src[hit[0] - 1] < 7))


LIVE_SWITCH_PARITY_SCRIPT = textwrap.dedent(
    """
    import json, random, sys
    from unittest import mock
    import numpy as np
    import torch
    for name in ("cv2", "ultralytics"):
        sys.modules[name] = mock.MagicMock()
    from app.engine import live
    from training.datasets import extract_keypoints as ek

    live.encode_jpeg = lambda img, q: b"x"

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

    class Results:
        def __init__(self, det):
            boxes, confs, kxy, kcf = det
            self.boxes = Boxes(torch.from_numpy(boxes), torch.from_numpy(confs))
            self.keypoints = Kps(torch.from_numpy(kxy), torch.from_numpy(kcf))

    # 사람 0~3명이 걷다가 사라지고, 다시 나타나고, 순간이동하는 무작위 장면
    rng = random.Random(7)
    W, H = 640, 480
    people = []
    dets = []
    for f in range(3000):
        if rng.random() < 0.04 and len(people) < 3:
            people.append([rng.uniform(0, 540), rng.uniform(0, 280), rng.uniform(0.3, 1.0)])
        if people and rng.random() < 0.03:
            people.pop(rng.randrange(len(people)))
        for p in people:
            if rng.random() < 0.03:
                p[0], p[1] = rng.uniform(0, 540), rng.uniform(0, 280)
            else:
                p[0] += rng.uniform(-15, 15); p[1] += rng.uniform(-10, 10)
            p[2] = min(1.0, max(0.25, p[2] + rng.uniform(-0.05, 0.05)))
        shown = [p for p in people if rng.random() > 0.08]  # 잠깐 놓침
        if not shown:
            dets.append(ek.empty_detections())
            continue
        boxes = np.array([[x, y, x + 100, y + 200] for x, y, _ in shown], np.float32)
        confs = np.array([c for _, _, c in shown], np.float32)
        kxy = np.stack([np.stack([np.linspace(b[0], b[2], 17), np.linspace(b[1], b[3], 17)], 1)
                        for b in boxes]).astype(np.float32)
        dets.append((boxes, confs, kxy, np.full((len(shown), 17), 0.8, np.float32)))

    frame = np.zeros((H, W, 3), np.uint8)

    def run_live(policy):
        cap = mock.MagicMock()
        cap.read.side_effect = [(True, frame)] * len(dets) + [(False, None)]
        model = mock.MagicMock()
        model.predict.side_effect = [[Results(d)] for d in dets]
        pipe = live.LivePipeline(model, cap, jpeg_quality=80)
        pipe.rotate_fallback = False
        pipe.switch_policy = policy
        sw, present = [], []
        while True:
            obs, _, _ = pipe.step()
            if obs is None:
                break
            sw.append(bool(obs["target_switched"]))
            present.append(bool(obs["tracks"] and obs["tracks"][0].get("has_person")))
        return sw, present

    runtime_sw, runtime_present = run_live("all")
    multi_sw, _ = run_live("multi")

    it = iter(dets)
    clip = ek.extract_frames((frame for _ in dets), 15.0, lambda fr: [next(it) for _ in fr], batch=4)
    ext_sw = clip["switched"].tolist()
    ext_multi = [bool(a and n >= 2) for a, n in zip(ext_sw, clip["n_cand"].tolist())]
    print(json.dumps({"n": len(runtime_sw), "runtime_switches": sum(runtime_sw), "extractor_switches": sum(ext_sw),
                      "reacquire": sum(1 for a, b in zip(runtime_present, runtime_present[1:]) if b and not a),
                      "flag_mismatch": sum(a != b for a, b in zip(runtime_sw, ext_sw)),
                      "present_mismatch": sum(a != b for a, b in zip(runtime_present, clip["present"].tolist())),
                      "multi_switches": sum(multi_sw), "multi_mismatch": sum(a != b for a, b in zip(multi_sw, ext_multi))}))
    ok = (len(runtime_sw) == len(ext_sw) == len(dets) and runtime_sw == ext_sw and sum(ext_sw) > 10
          and runtime_present == clip["present"].tolist() and multi_sw == ext_multi and 0 < sum(multi_sw) < sum(ext_sw))
    sys.exit(0 if ok else 1)
    """
)


@unittest.skipUnless(HAS_TORCH, "numpy/torch 없음")
class LiveSwitchParityTest(unittest.TestCase):
    def test_extractor_switch_flags_match_live_pipeline(self):
        # import 부작용(runs/ 생성 등)이 저장소에 남지 않도록 임시 폴더에서 실행
        with tempfile.TemporaryDirectory() as cwd:
            env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), RUNS_DIR=os.path.join(cwd, "runs"))
            proc = subprocess.run([sys.executable, "-c", LIVE_SWITCH_PARITY_SCRIPT], cwd=cwd, env=env,
                                  capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "numpy 없음")
class NCandTest(unittest.TestCase):
    """프레임별 사람 후보 수(n_cand) 저장과 대상 전환 정책 (replay.effective_switched)"""

    def _clip(self, switched, n_cand):
        import numpy as np
        t = len(switched)
        return {"ts": np.arange(t) / 15.0, "kps": np.zeros((t, 17, 3), np.float32),
                "bbox": np.zeros((t, 4), np.float32), "present": np.ones(t, bool), "frame_wh": [640, 480],
                "switched": np.asarray(switched, bool), "n_cand": n_cand}

    def test_roundtrip_and_validation(self):
        import tempfile

        import numpy as np

        from training.datasets.clipfmt import load_clip, save_clip
        c = self._clip([False, True, False], [1, 2, 0])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.npz"
            save_clip(p, c["ts"], c["kps"], c["bbox"], c["present"], c["frame_wh"], {}, switched=c["switched"],
                      n_cand=c["n_cand"])
            got = load_clip(p)
            self.assertEqual(got["n_cand"].tolist(), [1, 2, 0])
            save_clip(p, c["ts"], c["kps"], c["bbox"], c["present"], c["frame_wh"], {})
            self.assertNotIn("n_cand", load_clip(p))
            for bad in ([1, 2], [1, -1, 0], [0.5, 1, 1]):
                with self.assertRaises(ValueError):
                    save_clip(p, c["ts"], c["kps"], c["bbox"], c["present"], c["frame_wh"], {}, n_cand=np.array(bad))

    def test_switch_policy(self):
        from training.experiments import replay
        c = self._clip([False, True, True, False], [1, 1, 2, 1])
        self.assertEqual(replay.effective_switched(c, "all").tolist(), [False, True, True, False])
        self.assertEqual(replay.effective_switched(c, "multi").tolist(), [False, False, True, False])
        no_nc = dict(c)
        no_nc.pop("n_cand")
        self.assertEqual(replay.effective_switched(no_nc, "multi").tolist(), [False, True, True, False])
        self.assertIsNone(replay.effective_switched({"ts": []}, "multi"))
        with self.assertRaises(ValueError):
            replay.effective_switched(c, "bogus")
