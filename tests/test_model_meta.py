"""
가중치 메타(<weights>.meta.json) 저장/읽기/호환성 검사 테스트 (표준 라이브러리만 사용)
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from app.ai_classifier.model_meta import (
    check_compatible,
    current_git_sha,
    legacy_v1_meta,
    load_meta,
    make_meta,
    meta_path_for,
    write_meta,
)

CONTRACT_KEYS = {"meta_version", "arch", "feature_version", "in_channels", "seq_len", "hz", "temperature",
                 "suppress_pfall_th", "trained_on", "git_sha", "created_at"}


class ModelMetaTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.weights = Path(self._tmp.name) / "fall_model_best.pth"
        self.weights.write_bytes(b"dummy")

    def tearDown(self):
        self._tmp.cleanup()

    def test_meta_path(self):
        self.assertEqual(meta_path_for(self.weights).name, "fall_model_best.pth.meta.json")

    def test_legacy_v1_is_compatible_and_complete(self):
        meta = legacy_v1_meta()
        self.assertEqual(set(meta), CONTRACT_KEYS)
        self.assertIsNone(check_compatible(meta))
        self.assertEqual(meta["feature_version"], "v1")
        # 매번 새 객체 (호출 측이 수정해도 기본값이 오염되지 않음)
        meta["trained_on"].append("x")
        self.assertEqual(legacy_v1_meta()["trained_on"], [])

    def test_round_trip(self):
        meta = make_meta(trained_on=["app/ai_classifier/data"], git_sha="abc123")
        self.assertEqual(set(meta), CONTRACT_KEYS)
        self.assertIsNotNone(meta["created_at"])
        path = write_meta(self.weights, meta)
        self.assertTrue(Path(path).is_file())
        self.assertEqual(load_meta(self.weights), meta)
        self.assertEqual(load_meta(str(self.weights)), meta)
        # 임시 파일이 남지 않음
        self.assertEqual(sorted(p.name for p in Path(self._tmp.name).iterdir()),
                         ["fall_model_best.pth", "fall_model_best.pth.meta.json"])

    def test_missing_meta_returns_none(self):
        self.assertIsNone(load_meta(self.weights))

    def test_corrupt_meta_raises(self):
        meta_path_for(self.weights).write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_meta(self.weights)
        meta_path_for(self.weights).write_text("[1, 2]", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_meta(self.weights)

    def test_write_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            write_meta(self.weights, ["v1"])

    def test_incompatible_variants(self):
        variants = {
            "arch": "TransformerNet",
            "feature_version": "v2",
            "in_channels": 92,
            "seq_len": 60,
            "meta_version": 2,
        }
        for key, value in variants.items():
            meta = legacy_v1_meta()
            meta[key] = value
            reason = check_compatible(meta)
            self.assertIsNotNone(reason, key)
            self.assertIn(key, reason)
        missing = legacy_v1_meta()
        del missing["arch"]
        self.assertIsNotNone(check_compatible(missing))
        self.assertIsNotNone(check_compatible(None))
        self.assertIsNotNone(check_compatible("v1"))

    def test_failed_write_keeps_old_meta_and_no_temp(self):
        old = make_meta(git_sha="old")
        write_meta(self.weights, old)
        bad = make_meta()
        bad["trained_on"] = [object()]  # JSON 직렬화 불가
        with self.assertRaises(TypeError):
            write_meta(self.weights, bad)
        self.assertEqual(load_meta(self.weights), old)
        self.assertEqual(sorted(p.name for p in Path(self._tmp.name).iterdir()),
                         ["fall_model_best.pth", "fall_model_best.pth.meta.json"])

    def test_extra_fields_still_compatible(self):
        meta = make_meta(hz=15.0, temperature=1.3, suppress_pfall_th=0.4, note="x")
        self.assertIsNone(check_compatible(meta))

    def test_meta_is_json(self):
        write_meta(self.weights, make_meta())
        with open(meta_path_for(self.weights), encoding="utf-8") as f:
            self.assertIsInstance(json.load(f), dict)

    def test_git_sha_never_raises(self):
        self.assertIsNone(current_git_sha(self._tmp.name))
        sha = current_git_sha()
        self.assertTrue(sha is None or isinstance(sha, str))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"), "torch/numpy 필요")
class ActiveTrainedOnTest(unittest.TestCase):
    def test_trained_on_lists_only_merged_files(self):
        import numpy as np

        from training.train_active import load_active_learning_arrays

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            np.save(root / "X_ok.npy", np.zeros((2, 34, 30), dtype=np.float32))
            np.save(root / "y_ok.npy", np.zeros(2, dtype=np.int64))
            np.save(root / "X_noy.npy", np.zeros((1, 34, 30), dtype=np.float32))  # y 없음 -> 건너뜀
            np.save(root / "X_bad.npy", np.zeros((3, 34, 30), dtype=np.float32))
            np.save(root / "y_bad.npy", np.zeros(1, dtype=np.int64))  # 길이 불일치 -> 건너뜀
            loaded = []
            X, y = load_active_learning_arrays(str(root), loaded)
            self.assertEqual(len(X), 2)
            self.assertEqual([Path(p).name for p in loaded], ["X_ok.npy"])
            # 기존 호출 방식도 그대로 동작
            X2, _ = load_active_learning_arrays(str(root))
            self.assertEqual(len(X2), 2)


if __name__ == "__main__":
    unittest.main()
