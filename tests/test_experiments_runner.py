"""
progression study 실행기(training/experiments/run_progression.py) 테스트

- 표준 라이브러리만 쓰는 부분(인자 해석, 임계값 선택, AUC, 집계/표)은 numpy 없이도 실행
- fold 실행 스모크 테스트(합성 클립, 1 epoch, CPU)는 numpy/torch/sklearn이 있을 때만
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
HAS_STACK = (HAS_NUMPY and importlib.util.find_spec("torch") is not None
             and importlib.util.find_spec("sklearn") is not None)


def _fake_summarize(outcomes):
    """replay.summarize의 필요한 키만 표준 라이브러리로 계산 (집계 테스트용)"""
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


def _o(ds, label, alarm, p=None, suppressed=False, reached=True):
    return {"clip_id": f"{ds}/{label}/{alarm}/{p}", "dataset": ds, "label": label, "alarm": alarm,
            "suppressed": suppressed, "reached_ai": reached, "p_fall": [] if p is None else [p]}


class ArgsTest(unittest.TestCase):
    def test_parse_folds(self):
        self.assertEqual(RP.parse_folds("0-4"), [0, 1, 2, 3, 4])
        self.assertEqual(RP.parse_folds("3,1, 1"), [1, 3])
        self.assertEqual(RP.parse_folds("0-1,4"), [0, 1, 4])
        with self.assertRaises(ValueError):
            RP.parse_folds("3-1")
        with self.assertRaises(ValueError):
            RP.parse_folds("")

    def test_parse_stages_orders_and_validates(self):
        self.assertEqual(RP.parse_stages("f,a,E"), ["A", "E", "F"])
        self.assertEqual(RP.parse_stages("F,es,E"), ["E", "Es", "F"])
        with self.assertRaises(ValueError):
            RP.parse_stages("A,Z")


class ThresholdTest(unittest.TestCase):
    def test_recall_one_threshold(self):
        # 가장 낮은 검증 낙상 p_fall - margin, [0, 0.95]
        self.assertAlmostEqual(RP.select_threshold([0.9, 0.5, 0.7], margin=0.05), 0.45)
        self.assertEqual(RP.select_threshold([0.02], margin=0.05), 0.0)
        self.assertEqual(RP.select_threshold([0.999], margin=0.01), RP.F_TH_MAX)
        self.assertEqual(RP.select_threshold([], default=0.4), 0.4)

    def test_threshold_keeps_all_val_falls(self):
        ps = [0.31, 0.8, 0.55]
        th = RP.select_threshold(ps)
        self.assertTrue(all(p >= th for p in ps))  # 억제 조건은 p_fall < th (엔진과 같은 엄격 경계)

    def test_sim_balanced_and_balanced_threshold(self):
        pts = [("fall", True, 0.9), ("fall", True, 0.6), ("fall", False, None),
               ("adl", True, 0.2), ("adl", True, 0.5), ("adl", False, None)]
        # th 0: 알람 그대로 -> recall 2/3, 오탐 2/3
        self.assertAlmostEqual(RP.sim_balanced(pts, 0.0), 2 * (2 / 3) * (1 / 3) / 1.0)
        # th 0.5 바로 위: ADL 두 개 억제, 낙상 유지 -> recall 2/3, 특이도 1
        up = math.nextafter(0.5, math.inf)
        self.assertAlmostEqual(RP.sim_balanced(pts, up), 2 * (2 / 3) / (2 / 3 + 1))
        # th == 0.5: 엄격 경계라 p 0.5 ADL은 억제 안 됨 -> 오탐 1/3
        self.assertAlmostEqual(RP.sim_balanced(pts, 0.5), 2 * (2 / 3) * (2 / 3) / (4 / 3))
        # 고른 임계값은 그 p를 낸 클립까지 억제하도록 1 ulp 위
        self.assertEqual(RP.select_threshold_balanced(pts), up)
        # p_fall 없는 클립(판단 불가)은 규칙 결과 유지
        self.assertIsNone(RP.sim_balanced([("fall", True, 0.9)], 0.5))
        self.assertEqual(RP.select_threshold_balanced([], default=0.4), 0.4)

    def test_sim_no_veto_at_or_below_zero(self):
        # 엔진과 같이 th <= 0이면 veto 꺼짐 (p_fall 0.0도 억제 안 됨)
        pts = [("fall", True, 0.0), ("adl", True, 0.0)]
        for th in (0.0, -1.0, RP.NO_VETO):
            self.assertEqual(RP.sim_rates(pts, th), (1.0, 1.0))
        self.assertEqual(RP.sim_rates(pts, math.nextafter(0.0, math.inf)), (0.0, 0.0))

    def test_sim_matches_engine_boundary(self):
        # th == p, 그 바로 아래/위에서 근사 알람 여부가 Level1Engine._verify_with_ai의 억제 여부와 같음
        from app.core.level1 import Level1Engine, Level1Params
        for p in (0.0, 0.3, 0.4, 0.5, 1.0):
            for th in (p, math.nextafter(p, math.inf), math.nextafter(p, -math.inf), 0.0, -1.0):
                eng = Level1Engine(Level1Params(ai_mode="veto", ai_suppress_pfall_th=th), verifier=lambda s, p=p: p)
                for i in range(30):
                    eng.kp_seq.append([0.0] * 51)
                    eng.kp_ts.append(i / 10.0)
                ai = eng._verify_with_ai(3.0)
                self.assertEqual(ai["p_fall"], p)
                engine_alarm = not ai["suppress"]
                r, fa = RP.sim_rates([("fall", True, p), ("adl", True, p)], th)
                self.assertEqual((r == 1.0, fa == 1.0), (engine_alarm, engine_alarm), (p, th))
                pts = [("fall", True, p), ("adl", False, None)]
                self.assertEqual(RP.sim_balanced(pts, th) == 1.0, engine_alarm, (p, th))

    def test_auc(self):
        self.assertEqual(RP.auc_mann_whitney([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(RP.auc_mann_whitney([0.1], [0.9]), 0.0)
        self.assertEqual(RP.auc_mann_whitney([0.5], [0.5]), 0.5)
        self.assertAlmostEqual(RP.auc_mann_whitney([0.9, 0.4], [0.5, 0.1]), 0.75)
        self.assertIsNone(RP.auc_mann_whitney([], [0.1]))


class AggregateTest(unittest.TestCase):
    def _fold(self, k, runs):
        return {"fold": k, "params": 123, "time_s": 2.0,
                "runs": [{"seed": i, "outcomes": outs, "threshold": 0.4,
                          "window": {"roc_auc": 0.8 + 0.1 * i, "pr_auc": 0.5}, "train": {"time_s": 1.0}}
                         for i, outs in enumerate(runs)]}

    def test_pooled_and_fold_stats(self):
        f0 = self._fold(0, [[_o("le2i", "fall", True, 0.9), _o("le2i", "adl", False, 0.1)],
                            [_o("le2i", "fall", False, 0.2, suppressed=True), _o("le2i", "adl", False, 0.3)]])
        f1 = self._fold(1, [[_o("urfd", "fall", True, 0.7), _o("urfd", "adl", True, 0.6)],
                            [_o("urfd", "fall", True, 0.8), _o("urfd", "adl", True, 0.6)]])
        agg = RP.aggregate_variant([f1, f0], summarize_fn=_fake_summarize)
        self.assertEqual(agg["n_folds"], 2)
        self.assertEqual(agg["n_runs_per_fold"], 2)
        # pooled: seed 0 -> 2/2 탐지, seed 1 -> 1/2 탐지 => 평균 1.5
        self.assertAlmostEqual(agg["pooled"]["n_detected"], 1.5)
        self.assertAlmostEqual(agg["pooled"]["recall"], 0.75)
        self.assertAlmostEqual(agg["pooled"]["suppressed_true_falls"], 0.5)
        self.assertAlmostEqual(agg["pooled"]["by_dataset"]["urfd"]["recall"], 1.0)
        # fold: fold0 recall 평균 0.5, fold1 1.0 -> mean 0.75, 표본 std
        self.assertAlmostEqual(agg["fold"]["recall"]["mean"], 0.75)
        self.assertAlmostEqual(agg["fold"]["recall"]["std"], (2 * 0.25 ** 2) ** 0.5)
        self.assertAlmostEqual(agg["window"]["roc_auc"]["mean"], 0.85)
        # AI 호출 AUC: seed 0 낙상 [0.9, 0.7] vs ADL [0.1, 0.6] -> 1.0, seed 1 [0.2, 0.8] vs [0.3, 0.6] -> 0.5
        self.assertAlmostEqual(agg["ai_call_auc"], 0.75)
        self.assertAlmostEqual(agg["threshold"]["mean"], 0.4)

    def test_summary_and_markdown(self):
        fa = {"fold": 0, "params": None, "time_s": 1.0,
              "runs": [{"seed": None, "outcomes": [_o("le2i", "fall", True), _o("le2i", "adl", True)]}]}
        fb = self._fold(0, [[_o("le2i", "fall", True, 0.9), _o("le2i", "adl", False, 0.1)]])
        ff = dict(self._fold(0, [[_o("le2i", "fall", True, 0.9), _o("le2i", "adl", False, 0.1)]]))
        ff["runs"][0]["val"] = {"balanced_at_th": 0.9}
        s = RP.build_summary({"A": [fa], "D": [fb], "F1b": [ff]}, summarize_fn=_fake_summarize, meta={"folds": [0]})
        self.assertEqual(list(s["variants"]), ["A", "D", "F1b"])
        self.assertEqual(s["best_f"], "F1b")
        md = RP.render_markdown(s)
        self.assertIn("| A | 규칙만", md)
        self.assertIn("| D |", md)
        self.assertIn("1.000 (1/1)", md)
        self.assertIn("le2i recall", md)
        json.dumps(s)  # 직렬화 가능


class SpawnArgsTest(unittest.TestCase):
    """fold 하위 프로세스에 앞단 설정(--switch-policy, --no-fallback, --lost-grace)이 전달되는지 (프로세스는 가짜)"""

    def _cmds(self, **kw):
        import argparse
        from unittest import mock

        cmds = []

        class _P:
            def __init__(self, cmd, **_):
                cmds.append(cmd)

            def poll(self):
                return 0

        base = dict(stages_list=["A"], seeds=1, device="cpu", f_margin=0.05, hgb_iter=5, manifest="m.jsonl",
                    folds_file="f.json", epochs=None, kp_dir=None, switch_policy="all", no_fallback=False,
                    lost_grace=None, skip_existing=False, jobs=2)
        base.update(kw)
        with tempfile.TemporaryDirectory() as td,                 mock.patch.object(RP.subprocess, "Popen", _P), mock.patch.object(RP.time, "sleep", lambda s: None),                 mock.patch.object(RP, "_log", lambda m: None):
            failed = RP._spawn_folds(argparse.Namespace(**base), [0, 1], Path(td))
        self.assertEqual(failed, 0)
        self.assertEqual(len(cmds), 2)
        return cmds

    def test_default_has_no_frontend_flags(self):
        for cmd in self._cmds():
            self.assertNotIn("--lost-grace", cmd)
            self.assertNotIn("--switch-policy", cmd)
            self.assertNotIn("--no-fallback", cmd)

    def test_frontend_flags_passed(self):
        for cmd in self._cmds(switch_policy="multi", no_fallback=True, lost_grace=0.5):
            i = cmd.index("--lost-grace")
            self.assertEqual(float(cmd[i + 1]), 0.5)
            self.assertEqual(cmd[cmd.index("--switch-policy") + 1], "multi")
            self.assertIn("--no-fallback", cmd)

    def test_markdown_shows_frontend(self):
        md = RP.render_markdown({"meta": {"protocol": "p", "frontend": {"kp_dir": "kp", "switch_policy": "multi",
                                                                         "use_fallback": True, "lost_grace_s": 0.5,
                                                                         "post_alarm_latch": False}},
                                 "variants": {}})
        self.assertIn("lost_grace_s 0.5", md)
        self.assertIn("post_alarm_latch False", md)
        # 래치 기록이 없는 옛 meta는 앞단 줄이 그대로
        md = RP.render_markdown({"meta": {"protocol": "p", "frontend": {"kp_dir": "kp", "switch_policy": "multi",
                                                                         "use_fallback": True, "lost_grace_s": 0.5}},
                                 "variants": {}})
        self.assertNotIn("post_alarm_latch", md)


@unittest.skipUnless(HAS_STACK, "numpy/torch/sklearn not installed")
class RunFoldSmokeTest(unittest.TestCase):
    """합성 클립 12개로 모든 단계를 1 epoch 돌려 결과 파일/형식을 확인 (CPU)."""

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        from app.eval.synth import make_scenario
        from training.datasets.clipfmt import from_obs_records

        cls.rows, cls.clips, cls.folds = [], {}, {}
        n = 0
        for ds in ("urfd", "le2i"):
            for fold, n_each in ((1, 2), (0, 1)):
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

    def test_all_stages_one_fold(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            with contextlib.redirect_stdout(io.StringIO()):
                res = RP.run_fold(0, self.rows, self.folds, self.clips, list(RP.STAGES), seeds=1, out_dir=out,
                                  device="cpu", epochs=1, hgb_iter=5)
            self.assertEqual(set(res), set(RP.VARIANTS))
            n_test = sum(1 for r in self.rows if self.folds[r["clip_id"]] == 0)
            for v in RP.VARIANTS:
                d = json.loads((out / "fold0" / f"{v}.json").read_text(encoding="utf-8"))
                self.assertEqual(d["variant"], v)
                self.assertEqual(len(d["runs"]), 1)
                self.assertEqual(len(d["runs"][0]["outcomes"]), n_test)
                # 앞단 설정 기록: 기본 lost_grace 0.0 고정, R 변형만 0.5
                self.assertEqual(d["frontend"]["lost_grace_s"], RP.LOST_GRACE_R1 if v in RP.STAGE_VARIANTS["R"]
                                 else 0.0, v)
                self.assertIs(d["frontend"]["post_alarm_latch"], False, v)  # study_params 고정값
            for v in ("E1", "E2"):
                self.assertTrue((out / "fold0" / "models" / f"{v}_s0.pt").exists())
            self.assertTrue((out / "fold0" / "models" / "E0_s0.pkl").exists())
            # F만 다시: 저장된 E 모델을 읽어 같은 임계값
            th_before = res["F2"]["runs"][0]["threshold"]
            with contextlib.redirect_stdout(io.StringIO()):
                res2 = RP.run_fold(0, self.rows, self.folds, self.clips, ["F"], seeds=1, out_dir=out,
                                   device="cpu", epochs=1)
            self.assertAlmostEqual(res2["F2"]["runs"][0]["threshold"], th_before, places=5)
            # Es만 다시: 저장된 E 모델로 같은 클립 결과, E 결과 파일은 그대로
            e_before = {v: (out / "fold0" / f"{v}.json").read_bytes() for v in RP.STAGE_VARIANTS["E"]}
            with contextlib.redirect_stdout(io.StringIO()):
                res3 = RP.run_fold(0, self.rows, self.folds, self.clips, ["Es"], seeds=1, out_dir=out,
                                   device="cpu", epochs=1)
            self.assertEqual(set(res3), set(RP.STAGE_VARIANTS["Es"]))
            for v in RP.STAGE_VARIANTS["Es"]:
                a = [(o["clip_id"], o["alarm"], o["p_fall"]) for o in res[v]["runs"][0]["outcomes"]]
                b = [(o["clip_id"], o["alarm"], o["p_fall"]) for o in res3[v]["runs"][0]["outcomes"]]
                self.assertEqual([x[:2] for x in a], [x[:2] for x in b], v)
                for (_, _, pa), (_, _, pb) in zip(a, b):
                    self.assertEqual(len(pa), len(pb))
                    for x, y in zip(pa, pb):
                        self.assertAlmostEqual(x, y, places=5)
                self.assertEqual(res3[v]["base"], RP.ES_BASE[v][0])
                self.assertIn("in_tail", res3[v]["runs"][0]["outcomes"][0])
            for v, raw in e_before.items():
                self.assertEqual((out / "fold0" / f"{v}.json").read_bytes(), raw)
            s = RP.summarize_out(out)
            self.assertIn("A", s["variants"])
            md = (out / "summary.md").read_text(encoding="utf-8")
            self.assertIn("| F2b |", md)
            self.assertIn("| E1max | E1 |", md)
            self.assertIn("E1max", s["paired"])


if __name__ == "__main__":
    unittest.main()
