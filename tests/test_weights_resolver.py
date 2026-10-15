import os
import tempfile
import unittest
from pathlib import Path

from app.ai_classifier.weights_resolver import (
    DEFAULT_WEIGHT_NAMES,
    DEFAULT_WEIGHTS_DIR,
    REPO_ROOT,
    resolve_weights_path,
    sha256_8,
)

ACTIVE, BEST = DEFAULT_WEIGHT_NAMES


class WeightsResolverTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def touch(self, name: str) -> Path:
        p = self.dir / name
        p.write_bytes(b"weights")
        return p

    def test_explicit_existing(self):
        p = self.touch("custom.pth")
        self.assertEqual(resolve_weights_path(str(p), env={}, search_dir=self.dir), str(p.resolve()))

    def test_explicit_missing_does_not_fall_back(self):
        self.touch(ACTIVE)
        self.assertIsNone(resolve_weights_path(str(self.dir / "nope.pth"), env={}, search_dir=self.dir))

    def test_env_path(self):
        p = self.touch("env.pth")
        env = {"AI_MODEL_PATH": str(p)}
        self.assertEqual(resolve_weights_path(env=env, search_dir=self.dir), str(p.resolve()))

    def test_env_missing_does_not_fall_back(self):
        self.touch(ACTIVE)
        env = {"AI_MODEL_PATH": str(self.dir / "nope.pth")}
        self.assertIsNone(resolve_weights_path(env=env, search_dir=self.dir))

    def test_relative_path_resolves_against_repo_root(self):
        rel = os.path.relpath(Path(__file__).resolve(), REPO_ROOT)
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            found = resolve_weights_path(rel, env={})
        finally:
            os.chdir(cwd)
        self.assertEqual(found, str(Path(__file__).resolve()))

    def test_default_prefers_active_best(self):
        self.touch(BEST)
        a = self.touch(ACTIVE)
        self.assertEqual(resolve_weights_path(env={}, search_dir=self.dir), str(a.resolve()))

    def test_default_falls_back_to_best(self):
        b = self.touch(BEST)
        self.assertEqual(resolve_weights_path(env={}, search_dir=self.dir), str(b.resolve()))

    def test_nothing_found(self):
        self.assertIsNone(resolve_weights_path(env={}, search_dir=self.dir))

    def test_default_weights_dir_location(self):
        # 장치에 배포된 가중치 경로가 바뀌면 분류기가 조용히 꺼지므로 위치를 고정
        self.assertEqual(DEFAULT_WEIGHTS_DIR, REPO_ROOT / "app" / "ai_classifier" / "weights")

    def test_sha256_8(self):
        h = sha256_8(str(self.touch(ACTIVE)))
        self.assertEqual(len(h), 8)
        int(h, 16)


if __name__ == "__main__":
    unittest.main()
