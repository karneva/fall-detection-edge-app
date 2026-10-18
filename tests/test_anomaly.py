"""
training/anomaly (단계 G 이상 탐지) 검증. 합성 데이터만 쓰고 빠르게 끝납니다.
- 패키지 import, 잘못된 종류: 표준 라이브러리만
- 세그먼트, 보정기: numpy
- md: numpy + sklearn, ae/nf: numpy + torch (+ sklearn: windows.handcrafted 경로 없음)
"""
import importlib.util
import math
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = HAS_NUMPY and importlib.util.find_spec("torch") is not None
HAS_SKLEARN = HAS_NUMPY and importlib.util.find_spec("sklearn") is not None

# 골반 중점 원점, 몸통 길이 1 단위, y는 아래 방향 (서 있는 자세)
_SKELETON = [
    (0.0, -1.6), (0.08, -1.7), (-0.08, -1.7), (0.16, -1.65), (-0.16, -1.65),
    (0.3, -1.0), (-0.3, -1.0), (0.4, -0.5), (-0.4, -0.5), (0.45, 0.0), (-0.45, 0.0),
    (0.15, 0.0), (-0.15, 0.0), (0.17, 0.9), (-0.17, 0.9), (0.18, 1.8), (-0.18, 1.8),
]


def make_window(t=60, seed=0, drift=(0.0, 0.0), noise=0.02, fall=False):
    """
    합성 v2 창 (92, t): rel = 골격 + 잡음, hip_dx/dy = drift * 스텝, 모든 관절 유효.
    fall=True면 후반에 골반이 아래로 1.5 몸통 길이 내려가고 몸이 옆으로 눕는 이상 동작.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    base = np.array(_SKELETON)                                # (17, 2)
    X = np.zeros((92, t), dtype=np.float32)
    steps = np.arange(t)
    for k in range(t):
        pose = base + rng.normal(0, noise, base.shape)
        hip = np.array(drift) * k
        if fall and k >= t // 2:
            a = min(1.0, (k - t // 2) / 8.0)
            rot = np.array([[math.cos(a * 1.5), -math.sin(a * 1.5)], [math.sin(a * 1.5), math.cos(a * 1.5)]])
            pose = pose @ rot.T
            hip = hip + np.array([0.0, 1.5 * a])
        X[0:34, k] = pose.reshape(-1)
        X[68, k], X[69, k] = hip
    X[34:68] = np.gradient(X[0:34], axis=1) * 15.0
    X[70:72] = np.gradient(X[68:70], axis=1) * 15.0
    X[72] = 0.0
    X[74] = 2.5
    X[75:92] = 1.0
    del steps
    return X


def make_set(n, seed=0, fall=False):
    import numpy as np
    rng = np.random.default_rng(seed)
    return np.stack([make_window(seed=int(rng.integers(1 << 30)), fall=fall,
                                 drift=tuple(rng.normal(0, 0.01, 2))) for _ in range(n)])


class PackageImportTest(unittest.TestCase):
    def test_import_without_numpy_and_bad_kind(self):
        import training.anomaly as A
        self.assertEqual(A.KINDS, ("md", "ae", "nf"))
        with self.assertRaises(ValueError):
            A.fit_detector("zz", None, None, device="cpu")
        with self.assertRaises(ValueError):
            A.load_detector("zz", {}, device="cpu")


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class SegmentTest(unittest.TestCase):
    def test_fill_linear(self):
        import numpy as np
        from training.anomaly.segments import fill_linear
        v = np.array([[0.0, 9, 9, 3, 9], [9, 9, 9, 9, 9], [9, 2, 9, 9, 9]])
        ok = np.array([[1, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=bool)
        f, anyv = fill_linear(v, ok)
        np.testing.assert_allclose(f[0], [0, 1, 2, 3, 3])
        np.testing.assert_allclose(f[2], [2, 2, 2, 2, 2])
        self.assertEqual(anyv.tolist(), [True, False, True])

    def test_starts_and_shapes(self):
        import numpy as np
        from training.anomaly.segments import segment_starts, window_segments
        self.assertEqual(segment_starts(60).tolist(), [0, 6, 12, 18, 24, 30, 36])
        self.assertEqual(len(segment_starts(23)), 0)
        seg, keep, frac = window_segments(make_set(3))
        self.assertEqual(seg.shape, (3, 7, 2, 24, 17))
        self.assertTrue(keep.all())
        np.testing.assert_allclose(frac, 1.0)
        seg, keep, _ = window_segments(np.zeros((2, 92, 20), dtype=np.float32))
        self.assertEqual(seg.shape[:2], (2, 0))

    def test_coords_centering_and_fill(self):
        import numpy as np
        from training.anomaly.segments import window_segments
        X = make_window(drift=(0.05, 0.02), noise=0.0)
        # 관절 9(l_wrist): 스텝 2~4 무효 -> 보간, 관절 16(r_ankle): 전체 무효 -> 골반 좌표
        X[75 + 9, 2:5] = 0.0
        X[0 + 18:20, 2:5] = 0.0
        X[75 + 16, :] = 0.0
        X[32:34, :] = 0.0
        seg, keep, frac = window_segments(X[None])
        s0 = seg[0, 0]                                         # 첫 세그먼트 (스텝 0~23)
        hip = np.stack([0.05 * np.arange(24), 0.02 * np.arange(24)])   # (2, 24)
        center = hip.mean(axis=1, keepdims=True)
        exp9 = np.array(_SKELETON[9])[:, None] + hip - center
        np.testing.assert_allclose(s0[:, :, 9], exp9, atol=1e-5)   # 정지 골격이라 보간 = 원래 값
        np.testing.assert_allclose(s0[:, :, 16], hip - center, atol=1e-5)
        # 골반 궤적의 세그먼트 평균이 0 (l_hip, r_hip 평균 = 골반 중점)
        mid = (s0[:, :, 11] + s0[:, :, 12]) / 2
        np.testing.assert_allclose(mid.mean(axis=1), 0.0, atol=1e-5)
        self.assertLess(frac[0, 0], 1.0)
        self.assertTrue(keep[0].all())

    def test_low_visibility_segments_dropped(self):
        from training.anomaly.segments import window_segments
        X = make_window()
        X[75:92, 30:] = 0.0                                    # 후반 30스텝은 사람 없음
        _, keep, frac = window_segments(X[None])
        # 시작 0, 6, 12 (무효 비율 <= 0.25), 18 (0.5 유효)까지 사용, 24 이후 제외
        self.assertEqual(keep[0].tolist(), [True, True, True, True, False, False, False])
        self.assertAlmostEqual(float(frac[0, 3]), 0.5)

    def test_adjacency(self):
        import numpy as np
        from training.anomaly.segments import coco_adjacency
        A = coco_adjacency()
        np.testing.assert_allclose(A, A.T)
        self.assertTrue((np.diag(A) > 0).all())
        self.assertEqual(int((A > 0).sum()), 17 + 2 * 19)
        # 그래프가 연결되어 있어야 함 (머리와 몸통이 끊기면 결합층이 두 부분을 섞지 못함)
        reach = np.linalg.matrix_power((A > 0).astype(int), 16)
        self.assertTrue((reach > 0).all())
        np.testing.assert_allclose(coco_adjacency("full").sum(axis=1), 1.0)


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class CalibratorTest(unittest.TestCase):
    def test_quantile_veto_and_monotone(self):
        import numpy as np
        from training.anomaly.calibrate import Calibrator
        rng = np.random.default_rng(0)
        val = np.concatenate([rng.gamma(2.0, 1.0, 500), np.full(20, 1.5)])   # 동률 포함
        cal = Calibrator.fit(val, q=0.8, p_at_q=0.4)
        xq = np.quantile(val, 0.8)
        self.assertEqual(cal.threshold, xq)
        self.assertEqual(float(cal([xq])[0]), 0.4)
        test = np.concatenate([rng.gamma(2.0, 1.5, 2000), [xq, np.nextafter(xq, np.inf), np.nextafter(xq, -np.inf)]])
        p = cal(test)
        np.testing.assert_array_equal(p <= 0.4, test <= xq)       # 억제 <=> 점수 <= 검증 80% 분위수
        order = np.argsort(test)
        self.assertTrue((np.diff(p[order]) >= 0).all())
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        # 검증 집합 자체: 약 80%가 억제
        self.assertAlmostEqual(float((cal(val) <= 0.4).mean()), 0.8, delta=0.01)
        self.assertEqual(float(cal([val.max() * 10])[0]), 1.0)
        self.assertEqual(float(cal([val.min() - 100])[0]), 0.0)

    def test_nan_roundtrip_and_degenerate(self):
        import json

        import numpy as np
        from training.anomaly.calibrate import Calibrator
        cal = Calibrator.fit([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
        self.assertEqual(cal.n_val, 5)
        p = cal([np.nan, 0.0, 10.0])
        self.assertTrue(math.isnan(p[0]))
        cal2 = Calibrator.from_params(json.loads(json.dumps(cal.params())))
        x = np.linspace(-2, 8, 101)
        np.testing.assert_array_equal(cal(x), cal2(x))
        const = Calibrator.fit([2.0] * 10)
        self.assertEqual(const([2.0, 1.0, 3.0]).tolist()[0], 0.4)
        self.assertLess(float(const([1.0])[0]), 0.4)
        self.assertGreater(float(const([3.0])[0]), 0.4)
        with self.assertRaises(ValueError):
            Calibrator.fit([np.nan])

    def test_calibrated_scorer(self):
        import numpy as np
        from training.anomaly.calibrate import Calibrator, calibrated_scorer

        class _Det:
            def score(self, X):
                return np.asarray(X, dtype=np.float64)

        cal = Calibrator.fit(np.arange(10.0))
        fn = calibrated_scorer(_Det(), cal)
        self.assertTrue(math.isnan(fn([np.nan, 1.0])[0]))
        self.assertEqual(calibrated_scorer(_Det(), cal, nan_to=0.0)([np.nan])[0], 0.0)


@unittest.skipUnless(HAS_SKLEARN, "numpy/sklearn 없음")
class MahalanobisTest(unittest.TestCase):
    def test_fit_score_roundtrip(self):
        import numpy as np

        import training.anomaly as A
        X_tr, X_va = make_set(60, seed=1), make_set(20, seed=2)
        det = A.fit_detector("md", X_tr, X_va, seed=0)
        self.assertEqual(det.kind, "md")
        normal = det.score(make_set(10, seed=3))
        falls = det.score(make_set(10, seed=4, fall=True))
        self.assertGreater(np.median(falls), np.max(normal))
        empty = np.zeros((1, 92, 60), dtype=np.float32)
        self.assertTrue(np.isnan(det.score(empty)[0]))
        det2 = A.load_detector("md", det.state())
        np.testing.assert_allclose(det2.score(X_va), det.score(X_va))
        self.assertIn("val_loss", det.info)
        self.assertEqual(det.score(X_va[0]).shape, (1,))


@unittest.skipUnless(HAS_TORCH, "numpy/torch 없음")
class AutoencoderTest(unittest.TestCase):
    def test_joint_weights(self):
        from training.anomaly.autoencoder import joint_weights
        X = make_set(2)
        X[0, 75 + 9, 5] = 0.0                     # 관절 9 스텝 5 무효
        X[1, 75 + 11, 7] = X[1, 75 + 12, 7] = 0.0  # 골반 무효 -> 그 스텝 모든 관절 무효
        w = joint_weights(X)
        self.assertEqual(w[0, 18, 5], 0.0)
        self.assertEqual(w[0, 19, 5], 0.0)
        self.assertEqual(w[0, 34 + 18, 5], 0.0)
        self.assertEqual(w[0, 18, 6], 1.0)
        self.assertTrue((w[1, 0:68, 7] == 0).all())
        self.assertTrue((w[:, 68:] == 1).all())
        self.assertEqual(float(w.sum()), w.size - 4 - 68)   # rel x,y + vel x,y / 골반 무효 스텝 68채널

    def test_fit_deterministic_roundtrip(self):
        import numpy as np

        import training.anomaly as A
        from training.anomaly.autoencoder import ConvAE
        X_tr, X_va = make_set(48, seed=1), make_set(16, seed=2)
        d1 = A.fit_detector("ae", X_tr, X_va, seed=0, device="cpu", max_epochs=2)
        d2 = A.fit_detector("ae", X_tr, X_va, seed=0, device="cpu", max_epochs=2)
        s1 = d1.score(X_va)
        np.testing.assert_allclose(s1, d2.score(X_va), rtol=1e-6)
        self.assertEqual(s1.dtype, np.float64)
        self.assertEqual(len(d1.info["epochs"]), 2)
        d3 = A.load_detector("ae", d1.state(), device="cpu")
        np.testing.assert_allclose(d3.score(X_va), s1, rtol=1e-6)
        self.assertTrue(np.isnan(d1.score(np.zeros((1, 92, 60), dtype=np.float32))[0]))
        import torch
        m = ConvAE().eval()
        for t in (60, 61, 58):
            self.assertEqual(tuple(m(torch.zeros(2, 92, t)).shape), (2, 92, t))


@unittest.skipUnless(HAS_TORCH, "numpy/torch 없음")
class STGNFTest(unittest.TestCase):
    def _flow(self, seed=0, **kw):
        import torch

        from training.anomaly.stgnf import STGNF
        torch.manual_seed(seed)
        flow = STGNF(**kw).double()
        flow.data_init(torch.randn(16, 2, 6, 17, dtype=torch.float64) * 0.7 + 0.3)
        with torch.no_grad():                    # 0 초기화된 coupling 출력층을 흔들어 항등이 아니게
            for st in flow.steps:
                st.coupling.nn.out.weight.normal_(0, 0.3)
                st.coupling.nn.out.bias.normal_(0, 0.3)
        return flow.eval().requires_grad_(False)   # 입력에 대한 야코비안만 필요

    def test_identity_at_init(self):
        import torch

        from training.anomaly.stgnf import STGNF
        torch.manual_seed(0)
        flow = STGNF().double().eval()           # ActNorm 미초기화(항등), coupling 항등 -> z = W_K..W_1 x (직교)
        x = torch.randn(3, 2, 24, 17, dtype=torch.float64)
        z, ld = flow(x)
        torch.testing.assert_close(ld, torch.zeros(3, dtype=torch.float64), atol=1e-4, rtol=0)  # float32 QR 직교 오차 x T*V*K
        nll = flow.nll(x)
        exp = 0.5 * (x ** 2).flatten(1).mean(1) + 0.5 * math.log(2 * math.pi)
        torch.testing.assert_close(nll, exp, atol=1e-6, rtol=0)

    def test_invertible(self):
        import torch
        for adj in ("skeleton", "full"):
            flow = self._flow(adjacency=adj)
            x = torch.randn(4, 2, 24, 17, dtype=torch.float64)
            z, _ = flow(x)
            self.assertGreater(float((z - x).abs().max()), 1e-2)
            torch.testing.assert_close(flow.inverse(z), x, atol=1e-8, rtol=1e-8)

    def test_logdet_matches_bruteforce_jacobian(self):
        import torch
        from torch.autograd.functional import jacobian
        flow = self._flow(K=3)
        x = torch.randn(1, 2, 4, 17, dtype=torch.float64)
        _, ld = flow(x)

        def f(v):
            return flow(v.reshape(1, 2, 4, 17))[0].reshape(-1)

        J = jacobian(f, x.reshape(-1))
        sign, logabs = torch.linalg.slogdet(J)
        self.assertNotEqual(float(sign), 0.0)
        self.assertAlmostEqual(float(ld[0]), float(logabs), places=8)
        # 층별로도 확인
        for layer in (flow.steps[0].actnorm, flow.steps[0].invconv, flow.steps[0].coupling):
            _, l1 = layer(x)
            Jl = jacobian(lambda v: layer(v.reshape(1, 2, 4, 17))[0].reshape(-1), x.reshape(-1))
            self.assertAlmostEqual(float(l1[0]), float(torch.linalg.slogdet(Jl)[1]), places=8)

    def test_param_count(self):
        from training.anomaly.stgnf import STGNF
        n = sum(p.numel() for p in STGNF().parameters())
        self.assertTrue(1_000 <= n <= 10_000, n)

    def test_detector_scores(self):
        import numpy as np

        import training.anomaly as A
        X_tr, X_va = make_set(40, seed=1), make_set(12, seed=2)
        det = A.fit_detector("nf", X_tr, X_va, seed=0, device="cpu", max_epochs=2)
        self.assertEqual(det.kind, "nf")
        self.assertEqual(det.info["n_train_segments"], 40 * 7)
        X = make_set(3, seed=5)
        X[1, 75:92] = 0.0                                     # 보이는 관절 없음 -> 세그먼트 없음 -> NaN
        X[2, 75:92, 24:] = 0.0                                # 앞쪽 세그먼트만 사용
        s = det.score(X)
        self.assertTrue(np.isfinite(s[0]) and np.isnan(s[1]) and np.isfinite(s[2]))
        seg, keep = det.segments(X[2:3])
        nll = det.segment_nll(seg[0][keep[0]])
        self.assertAlmostEqual(float(s[2]), float(nll.max()), places=10)
        det2 = A.load_detector("nf", det.state(), device="cpu")
        np.testing.assert_allclose(det2.score(X), s, rtol=1e-6)
        self.assertTrue(np.isnan(det.score(np.zeros((1, 92, 20), dtype=np.float32))[0]))
        d2 = A.fit_detector("nf", X_tr, X_va, seed=0, device="cpu", max_epochs=2)
        np.testing.assert_allclose(d2.score(X_va), det.score(X_va), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
