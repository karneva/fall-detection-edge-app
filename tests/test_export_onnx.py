"""
training/export_onnx.py: PyTorch vs ONNX Runtime 출력 동일성 (최대 절대 오차 < 1e-4)

torch/onnx/onnxruntime이 없으면 skip. 실제 가중치(progression_v1 fold0 E1_s0.pt)와 실제 창(v3_fb 클립)은 있을 때만.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_DEPS = all(importlib.util.find_spec(m) is not None for m in ("numpy", "torch", "onnx", "onnxruntime"))

REPO_ROOT = Path(__file__).resolve().parents[1]
E1_CKPT = REPO_ROOT / "dataset" / "_experiments" / "progression_v1" / "fold0" / "models" / "E1_s0.pt"
REAL_CLIP = REPO_ROOT / "dataset" / "_keypoints" / "v3_fb" / "le2i" / "Coffee_room_01" / "video_6.npz"


def _random_ckpt(path):
    import numpy as np
    import torch

    from app.ai_classifier.model import Fall1DCNNGAP
    torch.manual_seed(0)
    m = Fall1DCNNGAP(92)
    # BN running 통계도 기본값이 아니게 (학습된 모델과 비슷한 조건)
    with torch.no_grad():
        for bn in (m.bn1, m.bn2, m.bn3):
            bn.running_mean.normal_(0, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
    rng = np.random.default_rng(0)
    torch.save({"arch": "E1", "state_dict": m.eval().state_dict(),
                "mu": (rng.normal(size=92) * 0.2).astype(np.float32),
                "sd": rng.uniform(0.3, 3.0, size=92).astype(np.float32)}, path)


def _real_windows(n=8):
    """v3_fb 클립에서 0.5초 간격 4초 창 n개 (window_v2, 표준화 전)."""
    import numpy as np

    from app.ai_classifier.features import window_v2
    from training.datasets.clipfmt import load_clip
    c = load_clip(REAL_CLIP)
    t_end = float(c["ts"][-1]) - 4.0
    xs = []
    for t in np.linspace(0.0, max(t_end, 0.0), n):
        xs.append(window_v2(c["ts"], c["kps"], c["bbox"], c["present"], c["frame_wh"], float(t), float(t) + 4.0))
    return np.stack(xs).astype(np.float32)


@unittest.skipUnless(HAS_DEPS, "numpy/torch/onnx/onnxruntime not installed")
class ExportOnnxTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _check(self, weights, X, standardize):
        import onnx

        from app.ai_classifier import model_meta as MM
        from training import export_onnx as EX
        out = self.dir / f"m_{int(standardize)}.onnx"
        meta = EX.export_onnx(weights, out, standardize=standardize)
        onnx.checker.check_model(str(out))
        model = onnx.load(str(out))
        self.assertEqual(model.opset_import[0].version, 17)
        dim0 = model.graph.input[0].type.tensor_type.shape.dim[0]
        self.assertEqual(dim0.dim_param, "batch")          # 배치 크기 동적
        self.assertEqual(MM.load_meta(out)["format"], "onnx")
        self.assertEqual(meta["feature_version"], "v2")
        d = EX.onnx_parity(weights, out, X, standardize=standardize)
        self.assertLess(d["logits"], EX.PARITY_ATOL, d)
        self.assertLess(d["p_fall"], EX.PARITY_ATOL, d)
        return d

    def test_random_model_random_windows(self):
        import numpy as np
        w = self.dir / "rand.pt"
        _random_ckpt(w)
        X = np.random.default_rng(1).normal(size=(7, 92, 60)).astype(np.float32)
        for std in (True, False):
            self._check(w, X, std)
        self._check(w, X[:1], True)   # 배치 1 (운영 경로)

    @unittest.skipUnless(E1_CKPT.is_file(), "progression_v1 E1_s0.pt not available")
    def test_saved_e1_random_and_real_windows(self):
        import numpy as np
        X = np.random.default_rng(2).normal(size=(5, 92, 60)).astype(np.float32)
        if REAL_CLIP.is_file():
            X = np.concatenate([X, _real_windows()], axis=0)
        self._check(E1_CKPT, X, True)
        self._check(E1_CKPT, X, False)

    def test_wrapper_matches_runtime_classifier(self):
        """그래프 안 표준화 + softmax = 운영 FallClassifier.score_windows (같은 가중치)."""
        import numpy as np
        import torch

        from app.ai_classifier import model_meta as MM
        from app.ai_classifier.inference import FallClassifier
        from training import export_onnx as EX
        w = self.dir / "rand.pt"
        _random_ckpt(w)
        MM.write_meta(w, MM.make_meta_v2())
        clf = FallClassifier(model_path=str(w), device=torch.device("cpu"))
        self.assertTrue(clf.is_v2, clf.load_error)
        X = np.random.default_rng(3).normal(size=(4, 92, 60)).astype(np.float32)
        model, mu, sd = EX.load_checkpoint(w)
        with torch.no_grad():
            _, p = EX.ExportWrapper(model, mu, sd).eval()(torch.from_numpy(X))
        np.testing.assert_allclose(p.numpy(), clf.score_windows(X), rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
