"""
L 단계(라벨 효율)와 같은 오탐률(matched FA) 비교 테스트 (training/experiments/run_progression.py)

- 단계 등록, 양성 클립 중첩 표본, matched FA, 라벨 효율 표: 표준 라이브러리만 (numpy 없이도 실행)
- L fold 실행 스모크 테스트(합성 클립, 1 epoch, CPU): numpy/torch/sklearn이 있을 때만
"""
import importlib.util
import json
import logging
import tempfile
import unittest
from pathlib import Path

from training.experiments import run_progression as RP

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_STACK = (HAS_NUMPY and importlib.util.find_spec("torch") is not None
             and importlib.util.find_spec("sklearn") is not None)


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
    """veto 재생 결과 흉내: 첫 p_fall <= th면 억제 (승격 없음), p가 None이면 fail-open"""
    alarm = reached and (p is None or p > th)
    return {"clip_id": cid, "dataset": "le2i", "label": label, "alarm": alarm,
            "suppressed": reached and not alarm, "reached_ai": reached, "p_fall": [] if p is None else [p]}


def _fold(k, runs, **extra):
    d = {"fold": k, "params": 1, "time_s": 1.0,
         "runs": [{"seed": s, "outcomes": outs, "threshold": 0.4} for s, outs in runs]}
    d.update(extra)
    return d


class StageTest(unittest.TestCase):
    def test_registered(self):
        self.assertIn("L", RP.STAGES)
        self.assertLess(RP.STAGES.index("F"), RP.STAGES.index("L"))
        self.assertEqual(RP.parse_stages("l,e"), ["E", "L"])
        self.assertEqual(RP.L_FRACTIONS, (0.1, 0.25, 0.5))
        for v in RP.STAGE_VARIANTS["L"]:
            self.assertIn(v, RP.VARIANTS)
            self.assertIn(v, RP.VARIANT_INFO)
            self.assertTrue(v.isalnum(), v)
            base, frac = RP.L_BASE[v]
            self.assertIn(base, ("E0", "E1"))
            self.assertIn(frac, RP.L_FRACTIONS)
        self.assertEqual(RP.L_BASE["E1p25"], ("E1", 0.25))
        self.assertIn("E0max", RP.VARIANTS)
        self.assertEqual(RP.DEFAULT_FA_REF, "E0max")


class SubsampleTest(unittest.TestCase):
    def setUp(self):
        self.clips = [f"le2i/fall/c{i:03d}" for i in range(37)] + [f"urfd/fall/c{i}" for i in range(23)]

    def test_nested_sizes_and_deterministic(self):
        a = RP.nested_clip_subsets(self.clips, [0.1, 0.25, 0.5], fold=2, seed=1)
        b = RP.nested_clip_subsets(list(reversed(self.clips)), [0.5, 0.1, 0.25], fold=2, seed=1)
        self.assertEqual(a, b)  # 입력 순서/비율 순서와 무관
        self.assertEqual([len(a[f]) for f in (0.1, 0.25, 0.5)], [6, 15, 30])
        self.assertTrue(set(a[0.1]) <= set(a[0.25]) <= set(a[0.5]) <= set(self.clips))
        self.assertEqual(a[0.1], sorted(a[0.1]))

    def test_request_subset_gives_same_sample(self):
        full = RP.nested_clip_subsets(self.clips, RP.L_FRACTIONS, fold=0, seed=0)
        only = RP.nested_clip_subsets(self.clips, [0.1], fold=0, seed=0)
        self.assertEqual(list(only), [0.1])
        self.assertEqual(only[0.1], full[0.1])

    def test_depends_on_fold_and_seed(self):
        s00 = RP.nested_clip_subsets(self.clips, [0.25], fold=0, seed=0)[0.25]
        self.assertNotEqual(s00, RP.nested_clip_subsets(self.clips, [0.25], fold=0, seed=1)[0.25])
        self.assertNotEqual(s00, RP.nested_clip_subsets(self.clips, [0.25], fold=1, seed=0)[0.25])

    def test_edges(self):
        self.assertEqual(RP.nested_clip_subsets([], [0.1], fold=0, seed=0), {0.1: []})
        self.assertEqual(len(RP.nested_clip_subsets(["a", "b"], [0.1], fold=0, seed=0)[0.1]), 1)  # 최소 1개
        self.assertEqual(RP.nested_clip_subsets(["b", "a"], [1.0], fold=0, seed=0), {1.0: ["a", "b"]})

    def test_positive_ids_and_mask(self):
        meta = [{"clip_id": "f1", "label": "fall"}, {"clip_id": "f1", "label": "fall"},
                {"clip_id": "f1", "label": "fall"}, {"clip_id": "f2", "label": "fall"},
                {"clip_id": "f2", "label": "fall"}, {"clip_id": "a1", "label": "adl"}]
        y = [1, 1, 0, 1, 0, 0]
        self.assertEqual(RP.positive_clip_ids(meta, y), ["f1", "f2"])
        # f1만 남김: f1의 창 모두 (pre_fall 음성 포함), f2는 양성과 pre_fall 음성 모두 빠짐, ADL은 남음
        self.assertEqual(RP.keep_positive_mask(meta, y, ["f1"]), [True, True, True, False, False, True])
        self.assertEqual(RP.keep_positive_mask(meta, y, []), [False, False, False, False, False, True])
        # label 없는 옛 meta: y == 1 창만 낙상 클립 창으로 봄
        old = [{"clip_id": "f1"}, {"clip_id": "a1"}]
        self.assertEqual(RP.keep_positive_mask(old, [1, 0], []), [False, True])

    def test_hgb_recipe_early_stopping(self):
        import importlib.util
        if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("sklearn") is None:
            self.skipTest("numpy/sklearn 없음")
        import numpy as np
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(2000, 5)), rng.integers(0, 2, 2000)
        clf = RP._make_hgb(0, 20)
        clf.fit(X, y)
        # 표본이 1만 개 미만이어도 early stopping이 켜짐 (E0와 L 변형의 학습 방식이 같음)
        self.assertTrue(clf.do_early_stopping_)


class MatchedFATest(unittest.TestCase):
    def test_sim_and_match(self):
        pts = [("fall", True, 0.9), ("fall", True, 0.5), ("fall", True, 0.2), ("fall", False, None),
               ("adl", True, 0.6), ("adl", True, 0.3), ("adl", True, None), ("adl", False, None)]
        self.assertEqual(RP.sim_rates(pts, RP.NO_VETO), (0.75, 0.75))
        self.assertEqual(RP.sim_rates(pts, 0.4), (0.5, 0.5))
        # 오탐률 0.5 이하가 되는 가장 작은 임계값 = 0.3 (ADL 0.3 억제), recall 0.5 (0.2 낙상도 억제)
        m = RP.match_fa_threshold(pts, 0.5)
        self.assertEqual((m["threshold"], m["recall"], m["adl_fa_rate"], m["matched"]), (0.3, 0.5, 0.5, True))
        m = RP.match_fa_threshold(pts, 0.9)
        self.assertEqual((m["threshold"], m["recall"]), (RP.NO_VETO, 0.75))
        # 판단 불가(fail-open) ADL 알람 때문에 0.25 아래로는 못 내려감
        m = RP.match_fa_threshold(pts, 0.1)
        self.assertFalse(m["matched"])
        self.assertEqual((m["threshold"], m["adl_fa_rate"], m["recall"]), (0.9, 0.25, 0.0))

    def test_ai_points_use_reached_not_post_veto_alarm(self):
        o = _o("a", "fall", 0.3)  # veto로 억제됨
        self.assertFalse(o["alarm"])
        self.assertEqual(RP.ai_points([o]), [("fall", True, 0.3)])

    def _results(self):
        # 기준 E0max: 낙상 p [0.9, 0.5, 0.3], ADL p [0.6, 0.45, 0.1] -> th 0.4 근사 오탐률 2/3
        ref = [_fold(0, [(0, [_o("f1", "fall", 0.9), _o("f2", "fall", 0.5), _o("a1", "adl", 0.6)]),
                         (1, [_o("f1", "fall", 0.9), _o("f2", "fall", 0.5), _o("a1", "adl", 0.6)])]),
               _fold(1, [(0, [_o("f3", "fall", 0.3), _o("a2", "adl", 0.45), _o("a3", "adl", 0.1)]),
                         (1, [_o("f3", "fall", 0.3), _o("a2", "adl", 0.45), _o("a3", "adl", 0.1)])])]
        # 새 변형: 낙상 p 높음, ADL p 낮음 (같은 오탐률이면 recall이 더 높아야)
        new = [_fold(0, [(0, [_o("f1", "fall", 0.95), _o("f2", "fall", 0.8), _o("a1", "adl", 0.7)]),
                         (1, [_o("f1", "fall", 0.95), _o("f2", "fall", 0.3), _o("a1", "adl", 0.7)])]),
               _fold(1, [(0, [_o("f3", "fall", 0.6), _o("a2", "adl", 0.2), _o("a3", "adl", 0.65)]),
                         (1, [_o("f3", "fall", 0.6), _o("a2", "adl", 0.2), _o("a3", "adl", 0.65)])])]
        a = [_fold(k, [(None, [dict(_o("f1", "fall", None), p_fall=[])])]) for k in (0, 1)]
        return {"A": a, "E0max": ref, "E0p10": new}

    def test_matched_fa(self):
        mfa = RP.matched_fa(self._results())
        self.assertEqual(mfa["ref"], "E0max")
        V = mfa["variants"]
        self.assertNotIn("A", V)  # p_fall 없음
        self.assertAlmostEqual(V["E0max"]["target_fa"], 2 / 3)
        # 자기 자신과 맞추면 임계값 0.1에서 같은 오탐률, 억제된 낙상(0.3)도 살아남 (oracle)
        self.assertAlmostEqual(V["E0max"]["recall_at_fa"], 1.0)
        self.assertAlmostEqual(V["E0max"]["threshold"]["mean"], 0.1)
        self.assertAlmostEqual(V["E0max"]["own_recall"], 2 / 3)
        r = V["E0p10"]
        # seed 0: 낙상 [0.95, 0.8, 0.6], ADL [0.7, 0.2, 0.65] -> 오탐률 <= 2/3 최소 임계값 0.2, recall 1
        # seed 1: 낙상 [0.95, 0.3, 0.6] -> 임계값 0.2, recall 1
        self.assertEqual(r["recall_at_fa_seeds"], [1.0, 1.0])
        self.assertAlmostEqual(r["threshold"]["mean"], 0.2)
        self.assertAlmostEqual(r["fa_at_th"], 2 / 3)
        self.assertAlmostEqual(r["own_recall"], (1.0 + 2 / 3) / 2)
        self.assertEqual(r["folds"], [0, 1])
        self.assertEqual(r["n_seeds"], 2)
        # 기준을 바꿀 수 있음, 기준 결과가 없으면 빔
        self.assertIn("E0max", RP.matched_fa(self._results(), ref="E0p10")["variants"])
        self.assertEqual(RP.matched_fa(self._results(), ref="E1max")["variants"], {})

    def test_summary_with_l(self):
        s = RP.build_summary(self._results(), summarize_fn=_fake_summarize, meta={"folds": [0, 1]})
        self.assertEqual([r["variant"] for r in s["label_efficiency"]], ["E0p10", "E0max"])
        self.assertEqual([r["frac"] for r in s["label_efficiency"]], [0.1, 1.0])
        self.assertEqual(s["label_efficiency"][0]["matched_recall"], 1.0)
        md = RP.render_markdown(s)
        self.assertIn("## 라벨 효율", md)
        self.assertIn("| 10% | E0p10 |", md)
        self.assertIn("| 100% | E0max |", md)
        self.assertIn("## 같은 오탐률에서의 recall (기준 E0max", md)
        self.assertIn("| E0p10 | 0.667 / 0.667 | 0.200 | 1.000 |", md)
        json.dumps(s)

    def test_old_results_without_l(self):
        res = self._results()
        del res["E0p10"]
        s = RP.build_summary(res, summarize_fn=_fake_summarize)
        self.assertEqual(s["label_efficiency"], [])
        md = RP.render_markdown(s)
        self.assertNotIn("## 라벨 효율", md)
        self.assertIn("## 같은 오탐률", md)
        # 기준 변형도 없으면 matched 표도 없음
        s = RP.build_summary({"A": res["A"]}, summarize_fn=_fake_summarize)
        self.assertEqual(s["matched_fa"]["variants"], {})
        self.assertNotIn("## 같은 오탐률", RP.render_markdown(s))

    def test_label_efficiency_rows_with_g(self):
        # G 변형(다른 단계)이 summary에 있으면 0% 행, GnfE0는 100% 행
        V = {"E0max": {"model": "HGB", "fold": {"recall": {"mean": 0.7, "std": 0.1}}, "ai_call_auc": 0.8},
             "Gnf": {"model": "STG-NF", "fold": {"recall": {"mean": 0.6, "std": 0.1}}, "ai_call_auc": 0.7},
             "GnfE0": {"model": "STG-NF + HGB", "fold": {}, "ai_call_auc": None}}
        rows = RP.label_efficiency_rows(V, {"variants": {"Gnf": {"recall_at_fa": 0.5}}})
        self.assertEqual([(r["frac"], r["variant"]) for r in rows], [(0.0, "Gnf"), (1.0, "E0max"), (1.0, "GnfE0")])
        self.assertEqual(rows[0]["matched_recall"], 0.5)
        md = "\n".join(RP._render_label_efficiency(rows, "E0max"))
        self.assertIn("| 0% | Gnf | STG-NF |", md)
        self.assertEqual(RP.label_efficiency_rows({"E0max": V["E0max"]}), [])


@unittest.skipUnless(HAS_STACK, "numpy/torch/sklearn not installed")
class RunFoldLTest(unittest.TestCase):
    """합성 클립으로 L 단계만 1 epoch (CPU): 결과/모델 파일, 중첩 표본 기록, 저장 모델 다시 읽기."""

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

    def test_stage_l_one_fold(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            with contextlib.redirect_stdout(io.StringIO()):
                res = RP.run_fold(0, self.rows, self.folds, self.clips, ["L"], seeds=1, out_dir=out,
                                  device="cpu", epochs=1, hgb_iter=5)
            self.assertEqual(set(res), set(RP.STAGE_VARIANTS["L"]))
            n_test = sum(1 for r in self.rows if self.folds[r["clip_id"]] == 0)
            kept = {}
            for v in RP.STAGE_VARIANTS["L"]:
                d = json.loads((out / "fold0" / f"{v}.json").read_text(encoding="utf-8"))
                base, frac = RP.L_BASE[v]
                self.assertEqual((d["variant"], d["base"], d["label_frac"], d["mode"]), (v, base, frac, "max"))
                run = d["runs"][0]
                self.assertEqual(len(run["outcomes"]), n_test)
                self.assertEqual(run["threshold"], RP.DEFAULT_TH)
                tr = run["train"]
                self.assertEqual(tr["label_frac"], frac)
                self.assertEqual(tr["n_pos_clips"], RP._n_keep(frac, tr["n_pos_clips_total"]))
                self.assertIn("calls", run["verifier"])
                kept[v] = tr["n_pos_clips"]
                ext = ".pkl" if base == "E0" else ".pt"
                self.assertTrue((out / "fold0" / "models" / f"{v}_s0{ext}").exists())
                entry = RP._load_e_model(out / "fold0" / "models", v, 0)
                self.assertEqual(entry[0], "hgb" if base == "E0" else "nn")
            self.assertLessEqual(kept["E0p10"], kept["E0p25"])
            self.assertLessEqual(kept["E0p25"], kept["E0p50"])
            self.assertEqual(kept["E0p50"], kept["E1p50"])
            s = RP.summarize_out(out)
            self.assertEqual([r["variant"] for r in s["label_efficiency"]],
                             ["E0p10", "E1p10", "E0p25", "E1p25", "E0p50", "E1p50"])
            self.assertIn("## 라벨 효율", (out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
