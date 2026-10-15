"""
training 패키지: 기본 경로가 이전 스크립트의 위치와 같은지, 모듈이 import되는지 확인합니다.
"""
import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


class TrainingPathsTest(unittest.TestCase):
    def test_defaults_match_previous_locations(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRAIN_DATA_DIR", None)
            import training.paths as paths
            paths = importlib.reload(paths)
        # 이전 스크립트는 저장소 루트에서 실행하는 상대경로를 사용했음
        self.assertEqual(paths.DATA_DIR, REPO_ROOT / "app/ai_classifier/data")
        self.assertEqual(paths.ACTIVE_DATA_DIR, REPO_ROOT / "app/ai_classifier/data/active_learning")
        self.assertEqual(paths.WEIGHTS_DIR, REPO_ROOT / "app/ai_classifier/weights")
        self.assertEqual(paths.URFD_ROOT, REPO_ROOT / "dataset/URFD")
        self.assertEqual(paths.RUNS_NOT_FALL_CLIPS, REPO_ROOT / "runs/clips/not_fall")

    def test_data_dir_override(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"TRAIN_DATA_DIR": d}):
                import training.paths as paths
                paths = importlib.reload(paths)
                self.assertEqual(paths.DATA_DIR, Path(d))
                self.assertEqual(paths.ACTIVE_DATA_DIR, Path(d) / "active_learning")
        importlib.reload(paths)


@unittest.skipUnless(HAS_TORCH, "torch/numpy not installed")
class TrainingImportTest(unittest.TestCase):
    def test_modules_import_without_side_effects(self):
        """import만으로 폴더를 만들거나 학습을 시작하지 않아야 함 (YOLO/tqdm/cv2는 스텁)."""
        script = textwrap.dedent(
            """
            import importlib, sys, os
            from unittest import mock
            for name in ("ultralytics", "tqdm", "cv2"):
                sys.modules[name] = mock.MagicMock()
            for m in ("training.dataset_builder", "training.video_to_data", "training.train",
                      "training.train_active", "training.retrain_from_runs"):
                importlib.import_module(m)
            print(sorted(os.listdir(".")))
            """
        )
        with tempfile.TemporaryDirectory() as cwd:
            env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), TRAIN_DATA_DIR=os.path.join(cwd, "data"))
            proc = subprocess.run([sys.executable, "-c", script], cwd=cwd, env=env,
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertEqual(proc.stdout.strip(), "[]")
            self.assertFalse(os.path.exists(os.path.join(cwd, "data")))


if __name__ == "__main__":
    unittest.main()
