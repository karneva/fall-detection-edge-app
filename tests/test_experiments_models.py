"""
training/experiments models/trainer 검증 (합성 데이터, 빠름)
numpy/torch가 없는 시스템 python에서는 건너뜁니다.
"""
import importlib.util
import unittest

HAS_TORCH = (importlib.util.find_spec("numpy") is not None
             and importlib.util.find_spec("torch") is not None)


def _has_cuda():
    if not HAS_TORCH:
        return False
    import torch
    return torch.cuda.is_available()


def _separable(n=200, ch=8, t=20, seed=0):
    """클래스 1은 앞쪽 채널 평균이 +1만큼 이동한 쉬운 합성 과제 (불균형 3:1)"""
    import numpy as np
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(np.int64)
    X = rng.normal(0.0, 0.5, size=(n, ch, t)).astype(np.float32)
    X[y == 1, : ch // 2, :] += 1.0
    return X, y


@unittest.skipUnless(HAS_TORCH, "numpy/torch 없음")
class ModelShapeTest(unittest.TestCase):
    def test_fall1dcnn_reexport_is_runtime_class(self):
        from app.ai_classifier.model import Fall1DCNN as Runtime
        from training.experiments.models import Fall1DCNN
        self.assertIs(Fall1DCNN, Runtime)

    def test_fall1dcnn_shape(self):
        import torch
        from training.experiments.models import Fall1DCNN, count_params
        m = Fall1DCNN().eval()
        self.assertEqual(tuple(m(torch.randn(4, 34, 30)).shape), (4, 2))
        self.assertGreater(count_params(m), 100_000)

    def test_gap_length_agnostic(self):
        import torch
        from training.experiments.models import Fall1DCNNGAP
        m = Fall1DCNNGAP(92).eval()
        for t in (1, 2, 30, 60, 97):
            self.assertEqual(tuple(m(torch.randn(3, 92, t)).shape), (3, 2), t)

    def test_tcn_length_agnostic(self):
        import torch
        from training.experiments.models import FallTCN
        m = FallTCN(92).eval()
        for t in (1, 15, 60, 120):
            self.assertEqual(tuple(m(torch.randn(3, 92, t)).shape), (3, 2), t)
        w = m.attention_weights(torch.randn(2, 92, 60))
        self.assertEqual(tuple(w.shape), (2, 60))
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(2), atol=1e-5))

    def test_tcn_param_count(self):
        from training.experiments.models import FallTCN, count_params
        n = count_params(FallTCN(92, width=40))
        self.assertTrue(40_000 <= n <= 46_000, n)

    def test_param_counts_ordering(self):
        from training.experiments.models import Fall1DCNNGAP, FallTCN, build_model, count_params
        self.assertLess(count_params(FallTCN(92)), count_params(Fall1DCNNGAP(92)))
        self.assertEqual(count_params(build_model("tcn", 92)), count_params(FallTCN(92)))
        with self.assertRaises(ValueError):
            build_model("nope", 92)


@unittest.skipUnless(HAS_TORCH, "numpy/torch 없음")
class TrainerTest(unittest.TestCase):
    def test_recipes(self):
        from training.experiments.trainer import get_recipe
        o = get_recipe("original")
        self.assertEqual((o["optimizer"], o["lr"], o["epochs"], o["batch_size"], o["select"]),
                         ("adam", 1e-3, 50, 32, "val_acc"))
        self.assertFalse(o["class_weighted"])
        i = get_recipe("improved")
        self.assertEqual((i["optimizer"], i["weight_decay"], i["epochs"], i["batch_size"], i["patience"]),
                         ("adamw", 1e-4, 60, 64, 8))
        self.assertTrue(i["class_weighted"])
        self.assertEqual(get_recipe({"name": "improved", "epochs": 3})["epochs"], 3)
        with self.assertRaises(ValueError):
            get_recipe("nope")

    def test_class_weights(self):
        import numpy as np
        from training.experiments.trainer import class_weights
        w = class_weights(np.array([0, 0, 0, 1]))
        self.assertAlmostEqual(float(w[0]), 4 / 6, places=5)
        self.assertAlmostEqual(float(w[1]), 2.0, places=5)
        self.assertEqual(float(class_weights(np.array([0, 0]))[1]), 0.0)

    def _fit(self, model_name, recipe, seed=0, device="cpu"):
        from training.experiments.models import build_model
        from training.experiments.trainer import set_seed, train_classifier
        X, y = _separable(seed=1)
        Xv, yv = _separable(n=80, seed=2)
        set_seed(seed)
        m = build_model(model_name, X.shape[1], seq_len=X.shape[2])
        return train_classifier(m, X, y, Xv, yv, recipe, seed=seed, device=device)

    def test_learns_separable_improved(self):
        import numpy as np
        from sklearn.metrics import roc_auc_score
        from training.experiments.trainer import predict_proba
        Xt, yt = _separable(n=120, seed=3)
        for name in ("tcn", "cnn_gap", "cnn"):
            m, hist = self._fit(name, {"name": "improved", "epochs": 12})
            p = predict_proba(m, Xt, "cpu")
            self.assertEqual(p.shape, (120,))
            self.assertTrue(np.all((p >= 0) & (p <= 1)))
            self.assertGreater(roc_auc_score(yt, p), 0.95, name)
            self.assertIsNotNone(hist["best_epoch"])
            self.assertLessEqual(hist["stopped_epoch"], 12)
            self.assertEqual(len(hist["class_weights"]), 2)

    def test_learns_separable_original(self):
        import numpy as np
        from training.experiments.trainer import predict_proba
        Xt, yt = _separable(n=120, seed=3)
        m, hist = self._fit("cnn", {"name": "original", "epochs": 6})
        acc = float(np.mean((predict_proba(m, Xt, "cpu") > 0.5) == yt))
        self.assertGreater(acc, 0.9)
        self.assertIsNone(hist["class_weights"])
        best = max(e["val_acc"] for e in hist["epochs"])
        self.assertEqual(hist["best_val_acc"], best)
        # 동률이면 먼저 나온 epoch (원본 `>` 비교)
        first = next(e["epoch"] for e in hist["epochs"] if e["val_acc"] == best)
        self.assertEqual(hist["best_epoch"], first)

    def test_early_stopping(self):
        import numpy as np
        from training.experiments.models import FallTCN
        from training.experiments.trainer import train_classifier
        rng = np.random.default_rng(0)
        X = rng.normal(size=(64, 4, 10)).astype(np.float32)
        y = rng.integers(0, 2, 64)  # 학습 불가능한 랜덤 라벨 -> 검증 손실이 개선되지 않음
        Xv = rng.normal(size=(32, 4, 10)).astype(np.float32)
        yv = rng.integers(0, 2, 32)
        _, hist = train_classifier(FallTCN(4, width=8), X, y, Xv, yv,
                                   {"name": "improved", "epochs": 60, "patience": 3}, seed=0, device="cpu")
        self.assertLess(hist["stopped_epoch"], 60)
        self.assertEqual(hist["stopped_epoch"] - hist["best_epoch"], 3)

    def test_no_validation_uses_last_epoch(self):
        from training.experiments.models import FallTCN
        from training.experiments.trainer import train_classifier
        X, y = _separable(n=40)
        _, hist = train_classifier(FallTCN(8, width=8), X, y, None, None,
                                   {"name": "improved", "epochs": 2}, seed=0, device="cpu")
        self.assertEqual(hist["best_epoch"], 2)

    def test_deterministic_cpu(self):
        import torch
        m1, h1 = self._fit("tcn", {"name": "improved", "epochs": 3}, seed=7)
        m2, h2 = self._fit("tcn", {"name": "improved", "epochs": 3}, seed=7)
        for (k, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
            self.assertTrue(torch.equal(a, b), k)
        self.assertEqual([e["train_loss"] for e in h1["epochs"]], [e["train_loss"] for e in h2["epochs"]])
        m3, _ = self._fit("tcn", {"name": "improved", "epochs": 3}, seed=8)
        self.assertFalse(all(torch.equal(a, b) for a, b in
                             zip(m1.state_dict().values(), m3.state_dict().values())))

    def test_predict_proba_empty(self):
        import numpy as np
        from training.experiments.models import FallTCN
        from training.experiments.trainer import predict_proba
        self.assertEqual(predict_proba(FallTCN(8, width=8), np.zeros((0, 8, 10)), "cpu").shape, (0,))

    def test_val_loss_ignores_zero_weight_batches(self):
        """학습에 없는 클래스(가중치 0)만 든 검증 배치가 검증 손실을 nan으로 오염시키지 않음"""
        import math
        import numpy as np
        from training.experiments import trainer
        from training.experiments.models import FallTCN
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 4, 10)).astype(np.float32)
        y = np.zeros(40, dtype=np.int64)  # 클래스 1 없음 -> 가중치 [1, 0]
        Xv = rng.normal(size=(6, 4, 10)).astype(np.float32)
        yv = np.array([0, 0, 0, 1, 1, 1])
        old = trainer.EVAL_BATCH
        trainer.EVAL_BATCH = 3  # 두 번째 배치는 클래스 1만
        try:
            _, hist = trainer.train_classifier(FallTCN(4, width=8), X, y, Xv, yv,
                                               {"name": "improved", "epochs": 2}, seed=0, device="cpu")
        finally:
            trainer.EVAL_BATCH = old
        for e in hist["epochs"]:
            self.assertFalse(math.isnan(e["val_loss"]), e)

    def test_predict_proba_keeps_model_device_by_default(self):
        import numpy as np
        from training.experiments.models import FallTCN
        from training.experiments.trainer import predict_proba
        m = FallTCN(8, width=8).to("cpu")
        predict_proba(m, np.zeros((2, 8, 10), dtype=np.float32))
        self.assertEqual(next(m.parameters()).device.type, "cpu")

    @unittest.skipUnless(_has_cuda(), "CUDA 없음")
    def test_gpu_training(self):
        import numpy as np
        from training.experiments.trainer import predict_proba
        Xt, yt = _separable(n=120, seed=3)
        m, hist = self._fit("tcn", {"name": "improved", "epochs": 8}, device="cuda")
        self.assertTrue(hist["device"].startswith("cuda"))
        p = predict_proba(m, Xt, "cuda")
        self.assertGreater(float(np.mean((p > 0.5) == yt)), 0.9)


if __name__ == "__main__":
    unittest.main()
