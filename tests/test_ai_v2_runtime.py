"""
운영 v2 분류기 (사건 구간 4초 창 E1 + E*max) 테스트

- 메타 검사, _ClassifierVerifier v1/v2 분기: 표준 라이브러리만 (numpy 없이도 실행)
- event_window_to_v2 / 창 생성: 예전 오프라인 구현(clipfmt.from_sidecar 기반, 아래 _legacy_*)과 비트 단위 동일
- FallClassifier v2 로드/상태/fail-open, 재생에서 운영 경로 = 오프라인 E*max 경로 (p_fall 열이 비트 단위 동일)
- 실제 클립: dataset/_experiments/progression_v1/fold0/models/E1_s0.pt + dataset/_keypoints/v3_fb 3개 (없으면 skip)
"""
import importlib.util
import logging
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.ai_classifier import model_meta as MM

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = HAS_NUMPY and importlib.util.find_spec("torch") is not None

REPO_ROOT = Path(__file__).resolve().parents[1]
E1_CKPT = REPO_ROOT / "dataset" / "_experiments" / "progression_v1" / "fold0" / "models" / "E1_s0.pt"
V3_FB = REPO_ROOT / "dataset" / "_keypoints" / "v3_fb"
REAL_CLIPS = ("le2i/Coffee_room_01/video_6", "caucafall/adl/KneelS8", "le2i/Home_02/video_40")


# ---------------------------------------------------------------------------
# 메타
# ---------------------------------------------------------------------------
class MetaV2Test(unittest.TestCase):
    def test_v2_default_compatible(self):
        meta = MM.make_meta_v2(trained_on=["x"], git_sha="abc")
        self.assertIsNone(MM.check_compatible(meta))
        self.assertEqual((meta["feature_version"], meta["arch"], meta["in_channels"], meta["seq_len"]),
                         ("v2", "fall_cnn_gap", 92, 60))
        self.assertEqual((meta["window_s"], meta["hz"], meta["mode"], meta["step_s"], meta["threshold"]),
                         (4.0, 15.0, "max", 0.5, 0.4))

    def test_train_v2_style_meta_compatible(self):
        # training/train_v2.build_meta와 같은 구성 (v1 make_meta 위에 v2 필드 덮어쓰기, hz 정수)
        meta = MM.make_meta(arch="fall_cnn_gap", feature_version="v2", in_channels=92, seq_len=60, hz=15,
                            window_s=4.0, mode="max", step_s=0.5, threshold=0.4, suppress_pfall_th=0.4)
        self.assertIsNone(MM.check_compatible(meta))

    def test_v2_variants_rejected(self):
        bad = {"arch": "Fall1DCNN", "in_channels": 34, "hz": 30, "window_s": 3.0, "seq_len": 30, "mode": "last",
               "step_s": 0, "threshold": 1.5}
        for key, value in bad.items():
            meta = MM.make_meta_v2()
            meta[key] = value
            reason = MM.check_compatible(meta)
            self.assertIsNotNone(reason, key)
            self.assertIn("feature_version", reason)
            self.assertIn(key, reason)
        for key in ("hz", "window_s", "mode", "step_s", "threshold"):
            meta = MM.make_meta_v2()
            del meta[key]
            self.assertIsNotNone(MM.check_compatible(meta), key)
        meta = MM.make_meta_v2()
        del meta["seq_len"]  # 없어도 됨 (window_s x hz)
        self.assertIsNone(MM.check_compatible(meta))
        for fv in ("v3", None):
            meta = MM.make_meta_v2(feature_version=fv)
            self.assertIsNotNone(MM.check_compatible(meta))
        self.assertIsNotNone(MM.check_compatible(MM.make_meta_v2(meta_version=2)))


# ---------------------------------------------------------------------------
# _ClassifierVerifier 분기 (표준 라이브러리)
# ---------------------------------------------------------------------------
class _StubV1:
    enabled, feature_version, weights_hash = True, "v1", "v1hash"

    def __init__(self):
        self.seqs = []

    def predict_proba(self, seq):
        self.seqs.append(seq)
        return 0.7


class _StubSliding:
    def __init__(self):
        self.engine = None
        self.calls = 0
        self.last_starts = None

    def bind_engine(self, engine):
        self.engine = engine

    def __call__(self, seq):
        self.calls += 1
        if self.engine is None:
            return None
        self.last_starts = [self.engine.get_incident_snapshot()["event_window"]["t_start"]]
        return 0.9


class _StubV2:
    enabled, feature_version, weights_hash = True, "v2", "v2hash"

    def __init__(self):
        self.made = []

    def predict_proba(self, seq):
        raise AssertionError("v2 must not use the 30-frame path")

    def make_engine_verifier(self, model_id=None):
        v = _StubSliding()
        self.made.append(v)
        return v


class ClassifierVerifierRoutingTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_v1_uses_seq(self):
        from app.core.level1 import _ClassifierVerifier
        clf = _StubV1()
        v = _ClassifierVerifier(classifier=clf)
        self.assertEqual(v([[0.0] * 34] * 30), 0.7)
        self.assertEqual(len(clf.seqs), 1)
        self.assertEqual((v.version, v.model_id, v.last_window_starts), ("v1", "v1hash", None))

    def test_v2_needs_engine_and_binds_once(self):
        from app.core.level1 import _ClassifierVerifier
        clf = _StubV2()
        v = _ClassifierVerifier(classifier=clf)
        self.assertIsNone(v([]))   # 엔진 없음 -> fail-open
        self.assertEqual(clf.made, [])
        eng = SimpleNamespace(get_incident_snapshot=lambda: {"event_window": {"t_start": 12.5}})
        v.bind_engine(eng)
        self.assertEqual(v([]), 0.9)
        self.assertEqual(v([]), 0.9)
        self.assertEqual(len(clf.made), 1)        # 엔진당 검증기 하나
        self.assertIs(clf.made[0].engine, eng)
        self.assertEqual(v.last_window_starts, [12.5])
        self.assertEqual(v.version, "v2")
        v.bind_engine(eng)                         # 다시 묶으면 새 검증기
        v([])
        self.assertEqual(len(clf.made), 2)

    def test_disabled_v2_falls_back_to_predict_proba_none(self):
        from app.core.level1 import _ClassifierVerifier
        clf = SimpleNamespace(enabled=False, feature_version=None, weights_hash=None,
                              predict_proba=lambda seq: None)
        v = _ClassifierVerifier(classifier=clf)
        v.bind_engine(object())
        self.assertIsNone(v([]))
        self.assertIsNone(v.version)

    def test_default_engine_binds_default_verifier(self):
        from app.core import level1
        clf = _StubV2()
        with mock.patch.object(level1._ClassifierVerifier, "_get", lambda self: clf):
            eng = level1.Level1Engine(level1.Level1Params())
            self.assertIs(eng._verifier._engine, eng)
        injected = _StubSliding()
        eng2 = level1.Level1Engine(level1.Level1Params(), verifier=injected)
        self.assertIs(injected.engine, eng2)  # 주입한 검증기도 bind_engine이 있으면 엔진을 받음
        self.assertIs(eng2._verifier, injected)


# ---------------------------------------------------------------------------
# 예전 오프라인 구현 (Day 7까지 training/experiments/verifiers.py, 비트 단위 비교 기준)
# ---------------------------------------------------------------------------
def _legacy_event_window_to_v2(win, fixed_len_s=4.0):
    import numpy as np

    from app.ai_classifier.features import NUM_KPS, V2_CHANNELS, window_v2
    from training.datasets.clipfmt import from_sidecar

    mask0 = V2_CHANNELS.index("mask_nose")
    if not isinstance(win, dict) or not win.get("ts"):
        return None
    try:
        clip = from_sidecar({"snapshot": {"event_window": win}})
    except ValueError:
        return None
    if not np.any(clip["present"]):
        return None
    meta = clip["meta"]
    t_start = meta.get("window_start_s")
    t_end = meta.get("window_end_s")
    if t_start is None:
        return None
    if fixed_len_s is not None:
        t_end = float(t_start) + float(fixed_len_s)
    if t_end is None or float(t_end) <= float(t_start):
        return None
    x = window_v2(clip["ts"], clip["kps"], clip["bbox"], clip["present"], clip["frame_wh"],
                  float(t_start), float(t_end))
    if x.shape[1] == 0 or not np.any(x[mask0:mask0 + NUM_KPS] > 0):
        return None
    return x


def _legacy_entry_window_starts(start_s, ts, win_s=4.0, step_s=0.5):
    n = int(math.floor((float(ts) - float(win_s) - float(start_s)) / float(step_s) + 1e-6))
    return [float(start_s) + i * float(step_s) for i in range(max(n, 0) + 1)]


class _LegacySlidingMax:
    """예전 SlidingWindowVerifier(mode max) + nn_batch_scorer(batched=False)와 같은 계산."""

    def __init__(self, model, std, device="cpu", win_s=4.0, step_s=0.5):
        self.model, self.std, self.device = model.to(device).eval(), std, device
        self.win_s, self.step_s = win_s, step_s
        self.engine = None
        self.model_id = "legacy"

    def bind_engine(self, engine):
        self.engine = engine

    def _p(self, x):
        import numpy as np
        import torch
        x = np.asarray(self.std(x), dtype=np.float32)
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1)
        return float(probs[0, 1].item())

    def windows(self, engine):
        kp_ts = getattr(engine, "kp_ts", None)
        ts = float(kp_ts[-1]) if kp_ts else None
        pad = float(engine.p.event_window_pad_s)
        snap = engine.get_incident_snapshot() or {}
        xs = [_legacy_event_window_to_v2(snap.get("event_window"), self.win_s)]
        start = engine.event_anchor_ts if engine.event_anchor_ts is not None else engine.abnormal_start_ts
        if ts is None or start is None:
            return xs
        for t in _legacy_entry_window_starts(float(start) - float(engine.p.event_window_pre_s), ts,
                                             self.win_s, self.step_s)[1:]:
            win = engine.kp_buffer.window(t, t + self.win_s, pad_s=pad)
            xs.append(None if win.get("truncated") else _legacy_event_window_to_v2(win, self.win_s))
        return xs

    def __call__(self, seq):
        import numpy as np
        if self.engine is None:
            return None
        xs = [x for x in self.windows(self.engine) if x is not None]
        if not xs:
            return None
        ps = np.asarray([self._p(x) for x in xs], dtype=np.float64)
        return float(ps[int(np.argmax(ps))])


class _Rec:
    """검증기 출력(p_fall) 기록용 래퍼 (bind_engine은 그대로 전달)."""

    def __init__(self, inner):
        self.inner = inner
        self.ps = []
        self.model_id = "rec"

    def bind_engine(self, engine):
        self.inner.bind_engine(engine)

    def __call__(self, seq):
        p = self.inner(seq)
        self.ps.append(p)
        return p


def _buffer(n_s=10.0, hz=15.0, missing_every=0, bbox_none_every=0, frame_wh=(640, 480), t0=1000.0, seed=0):
    import numpy as np

    from app.core.kp_buffer import TimedKeypointBuffer
    rng = np.random.default_rng(seed)
    buf = TimedKeypointBuffer(20.0)
    for i in range(int(n_s * hz) + 1):
        ts = t0 + i / hz + float(rng.uniform(-0.01, 0.01))
        if missing_every and i % missing_every == 0:
            buf.append(ts, None, frame_wh=frame_wh)
            continue
        frame = [[float(rng.uniform(0.3, 0.7)), float(0.2 + 0.03 * j + 0.002 * i), float(rng.uniform(0.0, 1.0))]
                 for j in range(17)]
        bbox = None if (bbox_none_every and i % bbox_none_every == 0) else [0.3, 0.1, 0.7, 0.9]
        buf.append(ts, frame, bbox, frame_wh=frame_wh)
    return buf


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class EventWindowParityTest(unittest.TestCase):
    def test_bit_identical_to_legacy(self):
        import numpy as np

        from app.ai_classifier.event_window import entry_window_starts, event_window_to_v2
        n_cmp = n_none = 0
        cases = [dict(), dict(missing_every=3), dict(bbox_none_every=4), dict(frame_wh=None),
                 dict(missing_every=2, bbox_none_every=3, seed=1)]
        for kw in cases:
            buf = _buffer(**kw)
            for t in np.arange(997.0, 1010.0, 0.37):
                for fixed in (4.0, None):
                    for pad in (0.0, 0.25):
                        win = buf.window(float(t), float(t) + 4.0, pad_s=pad)
                        a = _legacy_event_window_to_v2(win, fixed)
                        b = event_window_to_v2(win, fixed)
                        self.assertEqual(a is None, b is None, (kw, t, fixed, pad))
                        if a is None:
                            n_none += 1
                            continue
                        self.assertEqual(a.dtype, b.dtype)
                        self.assertTrue(np.array_equal(a, b), (kw, t, fixed, pad))
                        n_cmp += 1
        self.assertGreater(n_cmp, 100)
        self.assertGreater(n_none, 0)   # 사람이 pad에만 있거나 버퍼 밖인 창
        # 형식 오류 -> None (예전과 같음)
        self.assertIsNone(event_window_to_v2({"ts": [1.0, 2.0], "kps": [None], "bbox": None, "t_start": 1.0}))
        self.assertIsNone(event_window_to_v2({"ts": [2.0, 1.0], "kps": None, "bbox": None, "t_start": 1.0}))
        self.assertIsNone(event_window_to_v2({"ts": []}))
        self.assertIsNone(event_window_to_v2(None))
        for s, ts in ((0.0, 6.0), (0.0, 6.2), (0.0, 3.0), (10.0, 14.0), (3.1, 19.7)):
            self.assertEqual(entry_window_starts(s, ts), _legacy_entry_window_starts(s, ts))

    def test_standardizer_matches_training(self):
        import numpy as np

        from app.ai_classifier.event_window import Standardizer
        from training.experiments.run_progression import _Standardizer
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5, 92, 60)).astype(np.float32)
        ref = _Standardizer.fit(X)
        std = Standardizer(ref.mu, ref.sd)
        self.assertTrue(np.array_equal(std(X[0]), ref(X[0])))
        self.assertTrue(np.array_equal(std(X), ref(X)))


# ---------------------------------------------------------------------------
# FallClassifier v2 (torch)
# ---------------------------------------------------------------------------
def _save_v2(path, seed=0, mu=None, sd=None, drop=()):
    import numpy as np
    import torch

    from app.ai_classifier.model import Fall1DCNNGAP
    torch.manual_seed(seed)
    m = Fall1DCNNGAP(92).eval()
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=92).astype(np.float32) * 0.1 if mu is None else mu
    sd = (rng.uniform(0.5, 2.0, size=92)).astype(np.float32) if sd is None else sd
    ck = {"arch": "E1", "state_dict": m.state_dict(), "mu": mu, "sd": sd}
    for k in drop:
        ck.pop(k)
    torch.save(ck, path)
    MM.write_meta(path, MM.make_meta_v2())
    return m, mu, sd


def _replay_three(case, obs, params, clf, offline_model, std):
    """같은 obs를 운영 경로 / 오프라인(re-export) 경로 / 예전 오프라인 구현으로 재생해 p 열 3개를 반환 (이벤트 열도 비교)."""
    from app.core.level1 import _ClassifierVerifier
    from training.experiments import replay
    from training.experiments.verifiers import SlidingWindowVerifier, nn_batch_scorer

    app_v = _Rec(_ClassifierVerifier(classifier=clf))
    ev_app, _ = replay.run_clip(obs, params, verifier=app_v)
    off_v = _Rec(SlidingWindowVerifier(nn_batch_scorer(offline_model, "cpu", preprocess=std), mode="max"))
    ev_off, _ = replay.run_clip(obs, params, verifier=off_v)
    leg_v = _Rec(_LegacySlidingMax(offline_model, std))
    ev_leg, _ = replay.run_clip(obs, params, verifier=leg_v)
    strip = [[(e["type"], e["ts"]) for e in evs] for evs in (ev_app, ev_off, ev_leg)]
    case.assertEqual(strip[0], strip[1])
    case.assertEqual(strip[0], strip[2])
    return app_v.ps, off_v.ps, leg_v.ps


@unittest.skipUnless(HAS_TORCH, "numpy/torch not installed")
class FallClassifierV2Test(unittest.TestCase):
    def setUp(self):
        import torch
        logging.disable(logging.CRITICAL)
        self.torch = torch
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "fall_cnn_gap_v2.pt"
        self.model, self.mu, self.sd = _save_v2(self.path)

    def tearDown(self):
        logging.disable(logging.NOTSET)
        self._tmp.cleanup()

    def clf(self, path=None):
        from app.ai_classifier.inference import FallClassifier
        return FallClassifier(model_path=str(path or self.path), device=self.torch.device("cpu"))

    def test_loads_v2_and_status(self):
        import numpy as np
        c = self.clf()
        self.assertTrue(c.enabled, c.load_error)
        self.assertTrue(c.is_v2)
        st = c.status()
        self.assertEqual(st["version"], "v2")
        self.assertEqual(st["feature_version"], "v2")
        self.assertEqual(st["protocol"], {"window_s": 4.0, "step_s": 0.5, "mode": "max", "threshold": 0.4,
                                          "batch": 1})
        self.assertEqual(st["decision_threshold"], 0.4)
        self.assertIsNone(c.predict_proba([[0.0] * 34] * 30))   # v2는 30프레임 경로를 쓰지 않음
        self.assertEqual(c.predict([[0.0] * 34] * 30), (None, 0.0))
        self.assertIsNotNone(c.warmup())
        # score_windows = 창마다 표준화 후 배치 1 softmax[1]
        X = np.random.default_rng(1).normal(size=(3, 92, 60)).astype(np.float32)
        got = c.score_windows(X)
        for i in range(3):
            x = (X[i] - self.mu[:, None]) / self.sd[:, None]
            with self.torch.no_grad():
                p = self.torch.softmax(self.model(self.torch.from_numpy(x[None])), dim=1)[0, 1].item()
            self.assertEqual(got[i], p)
        v = c.make_engine_verifier()
        self.assertEqual((v.mode, v.step_s, v.win_s, v.model_id), ("max", 0.5, 4.0, c.weights_hash))

    def test_v1_still_loads(self):
        from app.ai_classifier.model import Fall1DCNN
        p = self.dir / "v1.pth"
        self.torch.save(Fall1DCNN().state_dict(), p)
        c = self.clf(p)
        self.assertTrue(c.enabled)
        self.assertFalse(c.is_v2)
        self.assertEqual(c.status()["version"], "v1")
        self.assertIsNone(c.status()["protocol"])
        self.assertIsNone(c.make_engine_verifier())
        self.assertIsNotNone(c.predict_proba([[0.0] * 34] * 30))
        with self.assertRaises(RuntimeError):
            c.score_windows([[[0.0] * 60] * 92])

    def test_fail_open_variants(self):
        import numpy as np
        p = self.dir / "nomu.pt"
        _save_v2(p, drop=("mu",))
        c = self.clf(p)
        self.assertFalse(c.enabled)
        self.assertIn("mu", c.load_error)
        self.assertIsNone(c.make_engine_verifier())
        self.assertIsNone(c.status()["version"])
        p = self.dir / "badsd.pt"
        _save_v2(p, sd=np.zeros(92, np.float32))
        self.assertFalse(self.clf(p).enabled)
        p = self.dir / "short.pt"
        _save_v2(p, mu=np.zeros(10, np.float32))
        self.assertFalse(self.clf(p).enabled)
        # v2 메타인데 v1 가중치 -> 로드 실패 (fail-open)
        from app.ai_classifier.model import Fall1DCNN
        p = self.dir / "mismatch.pt"
        self.torch.save(Fall1DCNN().state_dict(), p)
        MM.write_meta(p, MM.make_meta_v2())
        c = self.clf(p)
        self.assertFalse(c.enabled)
        self.assertIn("load failed", c.load_error)
        # 지원하지 않는 모드 -> 메타 단계에서 거부
        MM.write_meta(self.path, MM.make_meta_v2(mode="last"))
        c = self.clf()
        self.assertFalse(c.enabled)
        self.assertIn("incompatible meta", c.load_error)

    def test_resolver_prefers_v2_with_meta(self):
        from app.ai_classifier.weights_resolver import DEFAULT_WEIGHT_NAMES, resolve_weights_path
        legacy = self.dir / DEFAULT_WEIGHT_NAMES[1]
        legacy.write_bytes(b"x")
        self.assertEqual(resolve_weights_path(None, env={}, search_dir=self.dir), str(self.path.resolve()))
        MM.meta_path_for(self.path).unlink()   # 메타 없는 v2 파일은 기본값으로 쓰지 않음
        self.assertEqual(resolve_weights_path(None, env={}, search_dir=self.dir), str(legacy.resolve()))

    def test_load_classifier_swaps_singleton(self):
        from app.ai_classifier import inference
        try:
            c = inference.load_classifier(str(self.path))
            self.assertIs(inference.peek_classifier(), c)
            self.assertTrue(c.is_v2)
            c2 = inference.load_classifier(str(self.dir / "nope.pt"))
            self.assertFalse(c2.enabled)
        finally:
            inference.reset_classifier()

    def test_replay_app_equals_offline_synthetic(self):
        from app.eval.synth import make_scenario
        from training.datasets.clipfmt import from_obs_records
        from training.experiments import replay
        from training.experiments.run_progression import _Standardizer

        c = self.clf()
        std = _Standardizer(self.mu, self.sd)
        n_calls = 0
        for name in ("fast_fall", "fall_writhe", "fall_lie_then_get_up", "lie_on_bed_slow", "fall_then_recover"):
            for fps in (15.0, 30.0):
                rec, _ = make_scenario(name, fps=fps, seed=0, lie_s=30.0)
                for obs in (rec, replay.clip_to_obs(from_obs_records(rec, dataset="synth", clip_id=name))):
                    for mode in ("veto", "shadow"):
                        a, b, l = _replay_three(self, obs, replay.study_params(mode), c, self.model, std)
                        self.assertEqual(a, b, (name, fps, mode))
                        self.assertEqual(a, l, (name, fps, mode))
                        n_calls += sum(p is not None for p in a)
        self.assertGreater(n_calls, 5)


@unittest.skipUnless(HAS_TORCH and E1_CKPT.is_file() and V3_FB.is_dir(),
                     "torch / progression_v1 E1_s0.pt / v3_fb keypoints not available")
class RealClipParityTest(unittest.TestCase):
    """저장된 E1_s0.pt를 운영 FallClassifier(v2 메타)와 오프라인 로더(run_progression)로 각각 읽어 실제 클립 재생."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "E1_s0.pt"
        shutil.copyfile(E1_CKPT, self.path)   # 원본 폴더에는 메타를 쓰지 않음
        MM.write_meta(self.path, MM.make_meta_v2())

    def tearDown(self):
        logging.disable(logging.NOTSET)
        self._tmp.cleanup()

    def test_three_real_clips_bit_identical(self):
        import torch

        from app.ai_classifier.inference import FallClassifier
        from training.experiments import replay
        from training.experiments.run_progression import _load_e_model

        c = FallClassifier(model_path=str(self.path), device=torch.device("cpu"))
        self.assertTrue(c.is_v2, c.load_error)
        kind, model, std = _load_e_model(E1_CKPT.parent, "E1", 0)
        self.assertEqual(kind, "nn")
        n_p = 0
        for cid in REAL_CLIPS:
            p = replay.clip_path(cid, V3_FB)
            if not p.is_file():
                self.skipTest(f"missing {p}")
            clip = replay.load_row_clip({"clip_id": cid}, V3_FB)
            obs = replay.clip_to_obs(clip)
            a, b, l = _replay_three(self, obs, replay.study_params("veto"), c, model, std)
            self.assertEqual(a, b, cid)
            self.assertEqual(a, l, cid)
            n_p += sum(x is not None for x in a)
        self.assertGreater(n_p, 0)


if __name__ == "__main__":
    unittest.main()
