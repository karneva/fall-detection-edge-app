"""
데이터셋 단위 일반화(LODO) fold 배정 테스트 (training/experiments/folds.build_lodo_folds, run_progression --lodo)

표준 라이브러리만 씁니다 (folds 모듈의 LODO 함수와 분할 함수는 numpy가 필요 없음).
"""
import json
import random
import tempfile
import unittest
from pathlib import Path

from training.experiments import folds as F
from training.experiments import run_progression as RP


def _rows():
    """평가 데이터셋마다 사건 4개(다중 시점 포함) + figshare 피험자 5명"""
    rows = []
    for ds in F.EVAL_DATASETS:
        for g in range(4):
            label = "fall" if g % 2 else "adl"
            for cam in range(2 if ds == "caucafall" else 1):
                rows.append({"clip_id": f"{ds}/e{g}/cam{cam}", "dataset": ds, "label": label,
                             "event_group": f"{ds}/e{g}"})
    for s in range(1, 6):
        for i, label in enumerate(("adl", "fall")):
            cid = f"figshare_ursul29/{label.upper()}/SBJ_{s:02d}_LOC1/ACT{i}/{s}{i}"
            rows.append({"clip_id": cid, "dataset": "figshare_ursul29", "label": label,
                         "subject": str(s), "event_group": cid})
    return rows


class LodoFoldsTest(unittest.TestCase):
    def test_each_eval_dataset_is_one_fold(self):
        rows = _rows()
        f = F.build_lodo_folds(rows)
        self.assertEqual(set(f), {r["clip_id"] for r in rows})
        held = F.lodo_held_out()
        self.assertEqual(held, {0: "le2i", 1: "caucafall", 2: "gmdcsa24", 3: "mcfd", 4: "urfd"})
        by_ds = {}
        for r in rows:
            by_ds.setdefault(r["dataset"], set()).add(f[r["clip_id"]])
        for k, ds in held.items():
            self.assertEqual(by_ds[ds], {k}, ds)
        self.assertEqual(by_ds["figshare_ursul29"], {F.TRAIN_ONLY_FOLD})
        self.assertNotIn(F.TRAIN_ONLY_FOLD, held)

    def test_deterministic_and_order_free(self):
        rows = _rows()
        f = F.build_lodo_folds(rows)
        shuffled = list(rows)
        random.Random(3).shuffle(shuffled)
        self.assertEqual(F.build_lodo_folds(shuffled), f)
        self.assertEqual(list(f), sorted(f))

    def test_unknown_dataset_raises(self):
        rows = _rows() + [{"clip_id": "newset/x", "dataset": "newset", "label": "fall", "event_group": "newset/x"}]
        with self.assertRaises(ValueError):
            F.build_lodo_folds(rows)

    def test_split_holds_out_one_dataset_and_keeps_figshare_in_train(self):
        rows = _rows()
        f = F.build_lodo_folds(rows)
        n_fig = sum(r["dataset"] == "figshare_ursul29" for r in rows)
        tested = []
        for k, ds in F.lodo_held_out().items():
            train, test = F.split_fold(rows, f, k)
            self.assertEqual({r["dataset"] for r in test}, {ds})
            self.assertEqual(len(test), sum(r["dataset"] == ds for r in rows))
            self.assertNotIn(ds, {r["dataset"] for r in train})
            self.assertEqual(sum(r["dataset"] == "figshare_ursul29" for r in train), n_fig)
            self.assertEqual(len(train) + len(test), len(rows))
            # 조기 종료용 검증 분리도 학습 데이터셋에서만
            tr, va = F.split_train_val(train, frac=0.25, seed=0)
            self.assertTrue(va)
            self.assertNotIn(ds, {r["dataset"] for r in tr + va})
            tested.extend(r["clip_id"] for r in test)
        # 평가 데이터셋 클립은 정확히 한 번 test, figshare는 한 번도 test가 아님
        self.assertEqual(sorted(tested), sorted(r["clip_id"] for r in rows if r["dataset"] in F.EVAL_DATASETS))

    def test_save_load_roundtrip(self):
        f = F.build_lodo_folds(_rows())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "folds_lodo.json"
            F.save_folds(p, f, {"mode": "lodo"})
            self.assertEqual(F.load_folds(p), f)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["n_folds"], len(F.EVAL_DATASETS))

    def test_runner_ensure_lodo_folds(self):
        rows = _rows()
        with tempfile.TemporaryDirectory() as d:
            man = Path(d) / "m.jsonl"
            man.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            p = Path(d) / "folds_lodo.json"
            f = RP._ensure_lodo_folds(p, man)
            self.assertTrue(p.exists())
            self.assertEqual(F.load_folds(p), f)
            self.assertEqual(RP._ensure_lodo_folds(p, man), f)  # 있으면 확인만
            # 다른 배정이 들어 있으면 멈춤 (5-fold 파일을 LODO로 잘못 쓰지 않도록)
            F.save_folds(p, {cid: 0 for cid in f})
            with self.assertRaises(SystemExit):
                RP._ensure_lodo_folds(p, man)

    def test_markdown_labels_held_out_datasets(self):
        meta = {"protocol": "p", "folds": [0, 1, 2, 3, 4], "seeds": 3,
                "lodo": {"held_out": {str(k): d for k, d in F.lodo_held_out().items()},
                         "train_only": list(F.TRAIN_ONLY_DATASETS)}}
        md = RP.render_markdown({"meta": meta, "variants": {}})
        self.assertIn("LODO", md)
        self.assertIn("0 le2i, 1 caucafall, 2 gmdcsa24, 3 mcfd, 4 urfd", md)
        self.assertIn("figshare_ursul29", md)
        self.assertNotIn("LODO", RP.render_markdown({"meta": {"protocol": "p", "folds": [0]}, "variants": {}}))


if __name__ == "__main__":
    unittest.main()
