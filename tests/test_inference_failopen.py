"""
분류기 fail-open 테스트. numpy가 없으면 전체 skip, torch가 없으면 로드 테스트 skip.
"""
import importlib
import importlib.util
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

SEQ = [[0.01 * (i % 7)] * 34 for i in range(30)]


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class NoTorchTest(unittest.TestCase):
    def test_disabled_without_torch(self):
        # patch.dict는 종료 시 패치 중 새로 import된 모듈까지 지우므로, 필요한 모듈은 미리 import
        import numpy  # noqa: F401
        from app.ai_classifier import inference

        logging.disable(logging.CRITICAL)
        try:
            with mock.patch.dict(sys.modules, {"torch": None}):
                inference.reset_classifier()
                clf = inference.FallClassifier()
                self.assertFalse(clf.enabled)
                self.assertIn("import failed", clf.load_error)
                self.assertIsNone(clf.predict_proba(SEQ))
                self.assertEqual(clf.predict(SEQ), (None, 0.0))
                self.assertIsNone(clf.warmup())
                self.assertFalse(inference.get_classifier().enabled)
                inference.reset_classifier()
        finally:
            logging.disable(logging.NOTSET)

    def test_get_classifier_never_raises(self):
        from app.ai_classifier import inference
        inference.reset_classifier()
        with mock.patch.object(inference, "FallClassifier", side_effect=RuntimeError("boom")):
            logging.disable(logging.CRITICAL)
            try:
                clf = inference.get_classifier()
            finally:
                logging.disable(logging.NOTSET)
        self.assertFalse(clf.enabled)
        self.assertIsNone(clf.predict_proba(SEQ))
        self.assertIn("boom", clf.status()["load_error"])
        self.assertIsNone(clf.status()["meta"])
        self.assertIsNone(clf.status()["feature_version"])
        inference.reset_classifier()

    def test_singleton_built_once_under_concurrency(self):
        import threading
        import time as _time
        from app.ai_classifier import inference

        built = []

        def slow_ctor(*args, **kwargs):
            built.append(1)
            _time.sleep(0.05)
            return inference._DisabledClassifier("fake")

        inference.reset_classifier()
        self.assertIsNone(inference.peek_classifier())
        with mock.patch.object(inference, "FallClassifier", side_effect=slow_ctor):
            threads = [threading.Thread(target=inference.get_classifier) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(len(built), 1)
        self.assertIsNotNone(inference.peek_classifier())
        inference.reset_classifier()


@unittest.skipUnless(HAS_NUMPY and HAS_TORCH, "numpy/torch not installed")
class TorchLoadTest(unittest.TestCase):
    def setUp(self):
        import torch
        from app.ai_classifier.model import Fall1DCNN

        self.torch = torch
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        torch.manual_seed(0)
        self.ref = Fall1DCNN().eval()
        self.path = self.dir / "w.pth"
        torch.save(self.ref.state_dict(), self.path)
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)
        self._tmp.cleanup()

    def clf(self, path):
        from app.ai_classifier.inference import FallClassifier
        return FallClassifier(model_path=str(path), device=self.torch.device("cpu"))

    def test_loads_and_matches_manual_forward(self):
        import numpy as np

        c = self.clf(self.path)
        self.assertTrue(c.enabled)
        self.assertEqual(len(c.weights_hash), 8)

        x = c.normalize_sequence(np.asarray(SEQ, dtype=np.float32)).T[None]
        with self.torch.no_grad():
            probs = self.torch.softmax(self.ref(self.torch.from_numpy(np.ascontiguousarray(x))), dim=1)[0]
        p_fall = float(probs[1])
        self.assertAlmostEqual(c.predict_proba(SEQ), p_fall, places=5)

        # predict()는 기존 의미 유지: argmax + max-softmax 임계값
        label, conf = c.predict(SEQ)
        exp_conf = float(probs.max())
        self.assertAlmostEqual(conf, exp_conf, places=5)
        if exp_conf < c.threshold:
            self.assertIsNone(label)
        else:
            self.assertEqual(label, int(probs.argmax()))
        self.assertIsNotNone(c.warmup())

    def test_short_sequence_returns_none(self):
        self.assertIsNone(self.clf(self.path).predict_proba(SEQ[:29]))

    def test_uses_last_30_frames(self):
        c = self.clf(self.path)
        longer = [[0.5] * 34] * 5 + SEQ
        self.assertAlmostEqual(c.predict_proba(longer), c.predict_proba(SEQ), places=6)

    def test_missing_path_disabled(self):
        self.assertFalse(self.clf(self.dir / "nope.pth").enabled)

    def test_corrupt_file_disabled(self):
        bad = self.dir / "bad.pth"
        bad.write_text("not a checkpoint")
        c = self.clf(bad)
        self.assertFalse(c.enabled)
        self.assertIn("load failed", c.load_error)

    def test_shape_mismatch_disabled(self):
        from app.ai_classifier.model import Fall1DCNN

        other = self.dir / "other.pth"
        self.torch.save(Fall1DCNN(seq_len=20).state_dict(), other)
        self.assertFalse(self.clf(other).enabled)

    def test_missing_meta_is_legacy_v1(self):
        from app.ai_classifier.model_meta import legacy_v1_meta

        c = self.clf(self.path)
        self.assertTrue(c.enabled)
        st = c.status()
        self.assertEqual(st["meta"], legacy_v1_meta())
        self.assertEqual(st["feature_version"], "v1")

    def test_compatible_meta_loads_and_status_exposes_it(self):
        from app.ai_classifier.model_meta import make_meta, write_meta

        meta = make_meta(trained_on=["urfd"], git_sha="abc")
        write_meta(self.path, meta)
        c = self.clf(self.path)
        self.assertTrue(c.enabled)
        self.assertEqual(c.status()["meta"], meta)
        self.assertEqual(c.status()["feature_version"], "v1")
        self.assertIsNotNone(c.predict_proba(SEQ))

    def test_incompatible_meta_disabled(self):
        from app.ai_classifier.model_meta import legacy_v1_meta, write_meta

        variants = {"arch": "Other", "feature_version": "v2", "in_channels": 92, "seq_len": 60, "meta_version": 99}
        for key, value in variants.items():
            meta = legacy_v1_meta()
            meta[key] = value
            write_meta(self.path, meta)
            c = self.clf(self.path)
            self.assertFalse(c.enabled, key)
            self.assertIn("meta", c.load_error)
            self.assertIn(key, c.load_error)
            self.assertIsNone(c.predict_proba(SEQ))
            self.assertEqual(c.predict(SEQ), (None, 0.0))
            self.assertIsNone(c.model_path)
            st = c.status()
            self.assertEqual(st["meta"], meta)
            self.assertEqual(st["feature_version"], meta["feature_version"])

    def test_corrupt_meta_disabled(self):
        from app.ai_classifier.model_meta import meta_path_for

        meta_path_for(self.path).write_text("{oops", encoding="utf-8")
        c = self.clf(self.path)
        self.assertFalse(c.enabled)
        self.assertIn("meta", c.load_error)
        self.assertIsNone(c.status()["meta"])


if __name__ == "__main__":
    unittest.main()
