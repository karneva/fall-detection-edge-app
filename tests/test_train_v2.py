"""
배포용 v2 분류기 학습(training/train_v2.py) 테스트: 합성 클립 12개, 1 epoch, CPU.
numpy/torch/sklearn이 없으면 skip (모듈 import 자체는 표준 라이브러리만으로 가능해야 함).
"""
import importlib.util
import json
import logging
import tempfile
import unittest
from pathlib import Path

from training import train_v2 as TV

HAS_STACK = (importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("torch") is not None
             and importlib.util.find_spec("sklearn") is not None)


class ConstTest(unittest.TestCase):
    def test_frontend_defaults(self):
        self.assertEqual(TV.FRONTEND, {"lost_grace_s": 0.5, "switch_policy": "multi", "rotate_fallback": True,
                                      "post_alarm_latch": True})
        self.assertEqual((TV.FEATURE_VERSION, TV.META_ARCH, TV.MODE, TV.STEP_S, TV.THRESHOLD),
                         ("v2", "fall_cnn_gap", "max", 0.5, 0.4))


@unittest.skipUnless(HAS_STACK, "numpy/torch/sklearn not installed")
class TrainV2SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        from app.eval.synth import make_scenario
        from training.datasets.clipfmt import from_obs_records

        cls.rows, cls.clips = [], {}
        n = 0
        for ds in ("urfd", "le2i", "figshare"):
            for label, scen in (("fall", "fast_fall"), ("adl", "lie_on_bed_slow")):
                for _ in range(2):
                    cid = f"{ds}/{label}/c{n}"
                    n += 1
                    obs, _ = make_scenario(scen, fps=30.0, seed=n, lie_s=3.0)
                    cls.clips[cid] = from_obs_records(obs, dataset=ds, clip_id=cid)
                    segs = []
                    if label == "fall" and ds != "figshare":
                        segs = [{"cls": "walk", "start_s": 0.0, "end_s": 3.0},
                                {"cls": "fall", "start_s": 3.0, "end_s": 3.5},
                                {"cls": "fallen", "start_s": 3.5, "end_s": 6.5}]
                    cls.rows.append({"clip_id": cid, "dataset": ds, "label": label, "segments": segs,
                                     "event_group": cid, "label_source": "synthetic"})
        # 키포인트가 없는 행은 빠져야 함
        cls.rows.append({"clip_id": "urfd/fall/missing", "dataset": "urfd", "label": "fall", "segments": [],
                         "event_group": "missing", "label_source": "synthetic"})

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_train_and_save(self):
        import contextlib
        import io

        import torch

        from app.ai_classifier import model_meta as MM
        from training.experiments import models as M
        from training.experiments import replay as R
        from training.experiments import windows as W

        before = (R.SWITCH_POLICY, R.USE_FALLBACK, dict(R.STUDY_OVERRIDES))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m" / "fall_cnn_gap_v2.pt"
            with contextlib.redirect_stdout(io.StringIO()):
                meta = TV.train_v2(self.rows, self.clips, out, seed=0, device="cpu", epochs=1, keypoints="synth",
                                   git_sha="abc")
            # 재생 전역 설정은 원래대로
            self.assertEqual((R.SWITCH_POLICY, R.USE_FALLBACK, dict(R.STUDY_OVERRIDES)), before)
            ck = torch.load(out, map_location="cpu", weights_only=False)
            self.assertEqual(ck["arch"], "E1")
            self.assertEqual(tuple(ck["mu"].shape), (W.V2_CHANNELS,))
            self.assertEqual(tuple(ck["sd"].shape), (W.V2_CHANNELS,))
            model = M.Fall1DCNNGAP(W.V2_CHANNELS)
            model.load_state_dict(ck["state_dict"])
            self.assertTrue(all(v.device.type == "cpu" for v in ck["state_dict"].values()))

            on_disk = MM.load_meta(out)
            self.assertEqual(on_disk, meta)
            self.assertFalse(b"\r\n" in MM.meta_path_for(out).read_bytes())
            for key, val in (("feature_version", "v2"), ("arch", "fall_cnn_gap"), ("window_s", 4.0), ("hz", 15),
                             ("mode", "max"), ("step_s", 0.5), ("threshold", 0.4), ("keypoints", "synth"),
                             ("in_channels", W.V2_CHANNELS), ("seq_len", W.V2_STEPS), ("git_sha", "abc")):
                self.assertEqual(meta[key], val, key)
            self.assertEqual(meta["frontend"], TV.FRONTEND)
            self.assertEqual(meta["trained_on"], ["figshare", "le2i", "urfd"])
            self.assertGreater(meta["n_windows"]["train"], 0)
            self.assertIn("val", meta)
            self.assertEqual(meta["train"]["seed"], 0)
            json.dumps(meta)


if __name__ == "__main__":
    unittest.main()
