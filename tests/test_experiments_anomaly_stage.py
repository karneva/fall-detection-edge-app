"""
G 단계(정상 창만으로 학습한 이상 탐지) 실행기 통합 테스트 (training/experiments/run_progression.py)

- 단계 등록, 라벨 효율/같은 오탐률 표의 G 행: 표준 라이브러리만 (numpy 없이도 실행)
- NaN 창을 건너뛰는 검증기, fail-open(모든 세그먼트 사용 불가 -> None), 결합 p: numpy/torch 필요
- G fold 실행 스모크 테스트(합성 클립, E 1 epoch + G 1 epoch, CPU): numpy/torch/sklearn이 있을 때만
"""
import importlib.util
import json
import logging
import math
import tempfile
import unittest
from pathlib import Path

from training.experiments import run_progression as RP

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = HAS_NUMPY and importlib.util.find_spec("torch") is not None
HAS_STACK = HAS_TORCH and importlib.util.find_spec("sklearn") is not None


def _fake_summarize(outcomes):
    """replay.summarize의 필요한 키만 표준 라이브러리로 계산"""
    def stats(os_):
        f = [o for o in os_ if o["label"] == "fall"]
        a = [o for o in os_ if o["label"] == "adl"]
        det = sum(o["alarm"] for o in f)
        fa = sum(o["alarm"] for o in a)
        r = det / len(f) if f else None
        far = fa / len(a) if a else None
        bal = None if r is None or far is None else (0.0 if r + 1 - far <= 0 else 2 * r * (1 - far) / (r + 1 - far))
        return {"n_fall": len(f), "n_adl": len(a), "n_detected": det, "n_adl_alarm": fa, "recall": r,
                "adl_fa_rate": far, "balanced": bal,
                "suppressed_true_falls": sum(1 for o in f if o["suppressed"] and not o["alarm"]),
                "rules_ceiling": (sum(o["reached_ai"] for o in f) / len(f)) if f else None}
    out = stats(outcomes)
    out["by_dataset"] = {d: stats([o for o in outcomes if o["dataset"] == d])
                         for d in sorted({o["dataset"] for o in outcomes})}
    return out


def _o(cid, label, p, th=0.4, reached=True):
    """veto 재생 결과 흉내: 첫 p_fall <= th면 억제, p가 None이면 fail-open"""
    alarm = reached and (p is None or p > th)
    return {"clip_id": cid, "dataset": "le2i", "label": label, "alarm": alarm,
            "suppressed": reached and not alarm, "reached_ai": reached, "p_fall": [] if p is None else [p]}


def _fold(k, outs, n_pos_clips=None):
    return {"fold": k, "params": 1, "time_s": 1.0,
            "runs": [{"seed": 0, "outcomes": outs, "threshold": 0.4, "train": {"n_pos_clips": n_pos_clips}}]}


class StageTest(unittest.TestCase):
    def test_registered(self):
        self.assertIn("G", RP.STAGES)
        self.assertLess(RP.STAGES.index("L"), RP.STAGES.index("G"))
        self.assertEqual(RP.parse_stages("g,es"), ["Es", "G"])
        self.assertEqual(RP.STAGE_VARIANTS["G"], ("Gmd", "Gae", "Gnf", "GnfS", "GnfE0"))
        for v in RP.STAGE_VARIANTS["G"]:
            self.assertIn(v, RP.VARIANTS)
            self.assertIn(v, RP.VARIANT_INFO)
            self.assertTrue(v.isalnum(), v)
            kind = RP.G_KIND.get(v) or RP.G_STRICT.get(v) or RP.G_COMBO[v][0]
            self.assertIn(kind, ("md", "ae", "nf"))
            self.assertIn(kind, RP.G_PARAMS_UNIT)
        self.assertEqual(RP.G_COMBO["GnfE0"], ("nf", "E0"))
        # 라벨 효율 표: Gmd/Gae/Gnf는 0%, GnfE0는 100% 라벨 E0를 섞으므로 100%
        rows = dict((v, f) for f, vs in RP.LABEL_EFF_ROWS for v in vs)
        self.assertEqual([rows[v] for v in ("Gmd", "Gae", "Gnf", "GnfS", "GnfE0")], [0.0, 0.0, 0.0, 0.0, 1.0])


class SummaryWithGTest(unittest.TestCase):
    def _results(self):
        # 기준 E0max: 낙상 p [0.9, 0.5, 0.3], ADL p [0.6, 0.45, 0.1] -> th 0.4 근사 오탐률 2/3
        ref = [_fold(0, [_o("f1", "fall", 0.9), _o("f2", "fall", 0.5), _o("a1", "adl", 0.6)]),
               _fold(1, [_o("f3", "fall", 0.3), _o("a2", "adl", 0.45), _o("a3", "adl", 0.1)])]
        # Gnf: f2는 모든 창이 NaN이라 p 없음 (fail-open 알람), 나머지는 보정 p
        gnf = [_fold(0, [_o("f1", "fall", 0.8), _o("f2", "fall", None), _o("a1", "adl", 0.7)], n_pos_clips=0),
               _fold(1, [_o("f3", "fall", 0.6), _o("a2", "adl", 0.2), _o("a3", "adl", 0.65)], n_pos_clips=0)]
        gmd = [_fold(k, [_o(f"f{k}", "fall", 0.5), _o(f"a{k}", "adl", 0.5)], n_pos_clips=0) for k in (0, 1)]
        comb = [_fold(k, [_o(f"f{k}", "fall", 0.9), _o(f"a{k}", "adl", 0.1)]) for k in (0, 1)]
        return {"E0max": ref, "Gnf": gnf, "Gmd": gmd, "GnfE0": comb}

    def test_matched_fa_and_label_efficiency(self):
        s = RP.build_summary(self._results(), summarize_fn=_fake_summarize)
        V = s["matched_fa"]["variants"]
        self.assertEqual(set(V), {"E0max", "Gmd", "Gnf", "GnfE0"})
        # Gnf: 낙상 [0.8, None, 0.6], ADL [0.7, 0.2, 0.65] -> 오탐률 <= 2/3 최소 임계값 0.2, recall 1 (None은 통과)
        self.assertAlmostEqual(V["Gnf"]["threshold"]["mean"], 0.2)
        self.assertAlmostEqual(V["Gnf"]["recall_at_fa"], 1.0)
        rows = s["label_efficiency"]
        self.assertEqual([(r["frac"], r["variant"]) for r in rows],
                         [(0.0, "Gmd"), (0.0, "Gnf"), (1.0, "E0max"), (1.0, "GnfE0")])
        self.assertEqual(rows[1]["train_pos_clips"], 0.0)
        self.assertIsNone(rows[3]["train_pos_clips"])
        self.assertAlmostEqual(rows[1]["matched_recall"], 1.0)
        md = RP.render_markdown(s)
        self.assertIn("| 0% | Gnf | STG-NF | 0.0 |", md)
        self.assertIn("| 100% | GnfE0 |", md)
        self.assertIn("| Gnf | 0.667 / 0.667 | 0.200 | 1.000 |", md)
        json.dumps(s)


@unittest.skipUnless(HAS_TORCH, "numpy/torch not installed")
class NanSafeVerifierTest(unittest.TestCase):
    def _pair(self, xs, score_fn):
        from training.experiments.verifiers import SlidingWindowVerifier
        ref = SlidingWindowVerifier(score_fn, mode="max")
        ver = RP.NanSafeSlidingVerifier(score_fn, model_id="t")
        for v in (ref, ver._inner):
            v._windows = lambda engine, xs=xs: list(xs)
        ref.bind_engine(object())
        ver.bind_engine(object())
        return ref, ver

    def test_same_as_sliding_when_finite(self):
        import numpy as np
        xs = [np.full((92, 60), 0.1, np.float32), None, np.full((92, 60), 0.7, np.float32),
              np.full((92, 60), 0.3, np.float32)]

        def score(X):
            return X[:, 0, 0].astype(np.float64)

        ref, ver = self._pair(xs, score)
        p = ver([])
        self.assertAlmostEqual(p, ref([]))
        self.assertAlmostEqual(p, 0.7, places=6)
        st, rs = ver.stats(), ref.stats()
        for key in ("calls", "none", "windows", "windows_skipped", "max_later"):
            self.assertEqual(st[key], rs[key], key)
        self.assertEqual((st["nan_windows"], st["all_nan_calls"]), (0, 0))

    def test_nan_windows_skipped_and_all_nan_fail_open(self):
        import numpy as np
        xs = [np.zeros((92, 60), np.float32), np.ones((92, 60), np.float32), np.full((92, 60), 2, np.float32)]

        def score(X):  # 첫 창(진입 구간)과 마지막 창은 NaN, 가운데 창만 점수
            return np.where(X[:, 0, 0] == 1, 0.3, np.nan)

        ref, ver = self._pair(xs, score)
        self.assertTrue(math.isnan(ref([])))  # 원래 검증기는 NaN을 최댓값으로 고름
        self.assertAlmostEqual(ver([]), 0.3)
        st = ver.stats()
        self.assertEqual((st["windows"], st["windows_skipped"], st["nan_windows"], st["max_later"]), (1, 2, 2, 1))

        _, ver = self._pair(xs, lambda X: np.full(len(X), np.nan))
        self.assertIsNone(ver([]))
        self.assertIsNone(ver([]))
        st = ver.stats()
        self.assertEqual((st["calls"], st["none"], st["all_nan_calls"], st["windows"]), (2, 2, 2, 0))
        # 창 자체가 없을 때도 None (all-NaN 호출로 세지 않음), 엔진 없으면 None
        _, ver = self._pair([None, None], lambda X: np.zeros(len(X)))
        self.assertIsNone(ver([]))
        self.assertEqual(ver.stats()["all_nan_calls"], 0)
        self.assertIsNone(RP.NanSafeSlidingVerifier(lambda X: X)([]))

    def test_stgnf_unusable_segments_fail_open(self):
        """관절 유효 50% 이상 세그먼트가 없는 창 -> nf 점수 NaN -> 보정 p NaN -> 검증기 None (fail-open)"""
        import numpy as np

        from training.anomaly import Calibrator, calibrated_scorer
        from training.anomaly.stgnf import STGNF, STGNFDetector

        det = STGNFDetector(STGNF(K=2, hidden=4))
        good = np.zeros((92, 60), np.float32)
        good[75:92] = 1.0                                    # 모든 관절 유효 (좌표 0이어도 사용 가능)
        good[0:34] = np.random.default_rng(0).normal(0, 0.3, (34, 60))
        bad = good.copy()
        bad[75:92] = 0.0
        bad[75:92, :10] = 1.0                                # 앞 10스텝만 유효: 어느 세그먼트도 50% 미만
        s = det.score(np.stack([good, bad]))
        self.assertTrue(np.isfinite(s[0]))
        self.assertTrue(np.isnan(s[1]))
        cal = Calibrator.fit(np.linspace(0.0, 1.0, 50))
        fn = calibrated_scorer(det, cal)
        _, ver = self._pair([bad, bad.copy()], fn)
        self.assertIsNone(ver([]))
        _, ver = self._pair([bad, good], fn)
        p = ver([])
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p, float(fn(good[None])[0]))

    def test_combine_and_window_metrics(self):
        import numpy as np
        p = RP.combine_mean_p([0.2, np.nan, 0.8, np.nan], [0.6, 0.3, np.nan, np.nan])
        np.testing.assert_allclose(p[:3], [0.4, 0.3, 0.8])
        self.assertTrue(np.isnan(p[3]))
        w = RP._window_metrics_nan([0, 1, 0, 1, 1], [0.1, 0.9, np.nan, 0.8, np.nan],
                                   [{"dataset": "a"}] * 5)
        self.assertEqual((w["n"], w["n_pos"], w["n_nan"], w["roc_auc"]), (3, 2, 2, 1.0))
        self.assertEqual(w["by_dataset"]["a"]["n"], 3)
        v = RP._veto_rates([0.1, 0.5, np.nan, 0.3], [0, 0, 1, 1])
        self.assertEqual((v["neg_vetoed"], v["pos_passed"], v["n_nan"]), (0.5, 0.5, 1))


@unittest.skipUnless(HAS_STACK, "numpy/torch/sklearn not installed")
class RunFoldGTest(unittest.TestCase):
    """합성 클립으로 E(1 epoch, E0 모델용) + G(1 epoch) 한 fold (CPU): 결과/모델 파일, 보정기, 요약 표."""

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        from app.eval.synth import make_scenario
        from training.datasets.clipfmt import from_obs_records

        cls.rows, cls.clips, cls.folds = [], {}, {}
        n = 0
        for ds in ("urfd", "le2i"):
            for fold, n_each in ((1, 3), (0, 1)):
                for label, scen in (("fall", "fast_fall"), ("adl", "lie_on_bed_slow")):
                    for _ in range(n_each):
                        cid = f"{ds}/{label}/c{n}"
                        n += 1
                        obs, _ = make_scenario(scen, fps=30.0, seed=n, lie_s=3.0)
                        cls.clips[cid] = from_obs_records(obs, dataset=ds, clip_id=cid)
                        segs = []
                        if label == "fall":
                            segs = [{"cls": "walk", "start_s": 0.0, "end_s": 3.0},
                                    {"cls": "fall", "start_s": 3.0, "end_s": 3.5},
                                    {"cls": "fallen", "start_s": 3.5, "end_s": 6.5}]
                        cls.rows.append({"clip_id": cid, "dataset": ds, "label": label, "segments": segs,
                                         "event_group": cid, "label_source": "synthetic"})
                        cls.folds[cid] = fold

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_stage_g_one_fold(self):
        import contextlib
        import io

        from training.anomaly import Detector

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            with contextlib.redirect_stdout(io.StringIO()):
                res = RP.run_fold(0, self.rows, self.folds, self.clips, ["E", "G"], seeds=1, out_dir=out,
                                  device="cpu", epochs=1, hgb_iter=5)
            self.assertTrue(set(RP.STAGE_VARIANTS["G"]) <= set(res))
            n_test = sum(1 for r in self.rows if self.folds[r["clip_id"]] == 0)
            mdir = out / "fold0" / "models"
            for v in RP.STAGE_VARIANTS["G"]:
                d = json.loads((out / "fold0" / f"{v}.json").read_text(encoding="utf-8"))
                self.assertEqual((d["variant"], d["mode"]), (v, "max"))
                self.assertEqual(d["label_frac"], 1.0 if v == "GnfE0" else 0.0)
                run = d["runs"][0]
                self.assertEqual(len(run["outcomes"]), n_test)
                self.assertEqual(run["threshold"], RP.DEFAULT_TH)
                self.assertEqual((run["calibrator"]["q"], run["calibrator"]["p_at_q"]), (0.8, 0.4))
                for key in ("calls", "nan_windows", "all_nan_calls"):
                    self.assertIn(key, run["verifier"])
                self.assertIn("n_nan", run["window"])
                self.assertIn("neg_vetoed", run["window_veto"])
                if v != "GnfE0":
                    self.assertEqual(run["train"]["n_pos_clips"], 0)
                    self.assertGreater(run["train"]["n_train"], 0)
                    self.assertGreater(d["params"], 0)
            # 학습 정상 창에는 양성 창이 없음
            d = json.loads((out / "fold0" / "Gae.json").read_text(encoding="utf-8"))
            self.assertTrue(d["n_windows"]["train_kinds"])
            self.assertTrue(all(k.endswith("/0") for k in d["n_windows"]["train_kinds"]))
            for kind in ("md", "ae", "nf"):
                self.assertTrue((mdir / f"G{kind}_s0.pt").exists())
                det, cal = RP._load_g_model(mdir, kind, 0)
                self.assertIsInstance(det, Detector)
                self.assertEqual(det.kind, kind)
                self.assertEqual(cal.q, 0.8)
            s = RP.summarize_out(out, fa_ref="E0")
            got = [(r["frac"], r["variant"]) for r in s["label_efficiency"]]
            self.assertEqual(got, [(0.0, "Gmd"), (0.0, "Gae"), (0.0, "Gnf"), (0.0, "GnfS"), (1.0, "GnfE0")])
            # GnfS: ADL 클립 창만으로 학습 (Gnf보다 학습 창이 적음)
            gs = json.loads((out / "fold0" / "GnfS.json").read_text(encoding="utf-8"))
            gn = json.loads((out / "fold0" / "Gnf.json").read_text(encoding="utf-8"))
            self.assertEqual(gs["normal_set"], "adl_clips_only")
            self.assertLessEqual(gs["runs"][0]["train"]["n_train"], gn["runs"][0]["train"]["n_train"])
            self.assertIn("ADL clips", gs["calibration"]["fit_on"])
            self.assertTrue({"Gmd", "Gae", "Gnf", "GnfE0"} <= set(s["matched_fa"]["variants"]))
            md = (out / "summary.md").read_text(encoding="utf-8")
            self.assertIn("## 라벨 효율", md)
            self.assertIn("| Gnf |", md)


if __name__ == "__main__":
    unittest.main()
