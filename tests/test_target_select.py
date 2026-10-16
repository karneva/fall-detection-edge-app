"""
대상 선택(app.engine.target_select) 테스트

- 규칙 단위 테스트는 표준 라이브러리만 사용
- live.select_best_person이 리팩토링 전 구현과 완전히 같은 결과를 내는지는 torch가 있을 때
  별도 프로세스에서 확인 (cv2/ultralytics는 스텁, 다른 테스트의 sys.modules를 건드리지 않도록)
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from app.engine import target_select as _ts

calculate_iou, select_target_index = _ts.calculate_iou, _ts.select_target_index

REPO_ROOT = Path(__file__).resolve().parents[1]


class IouTest(unittest.TestCase):
    def test_identical_and_disjoint(self):
        self.assertAlmostEqual(calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_partial_overlap(self):
        # 교집합 5x10=50, 합집합 100+100-50=150
        self.assertAlmostEqual(calculate_iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)

    def test_degenerate_boxes(self):
        self.assertEqual(calculate_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)


class SelectTargetTest(unittest.TestCase):
    def test_no_boxes(self):
        self.assertEqual(select_target_index([], [], last_bbox=None), (None, False))
        self.assertEqual(select_target_index([], [], last_bbox=[0, 0, 1, 1]), (None, False))

    def test_no_last_bbox_picks_top_conf_without_switch(self):
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60], [100, 0, 110, 10]]
        self.assertEqual(select_target_index(boxes, [0.6, 0.9, 0.7], last_bbox=None), (1, False))

    def test_tie_takes_first_index(self):
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        self.assertEqual(select_target_index(boxes, [0.8, 0.8], last_bbox=None), (0, False))

    def test_keeps_tracked_person_even_if_lower_conf(self):
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        idx, switched = select_target_index(boxes, [0.9, 0.55], last_bbox=[51, 51, 61, 61])
        self.assertEqual((idx, switched), (1, False))

    def test_low_iou_switches_to_top_conf(self):
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        idx, switched = select_target_index(boxes, [0.6, 0.9], last_bbox=[200, 200, 210, 210])
        self.assertEqual((idx, switched), (1, True))

    def test_threshold_is_inclusive(self):
        # IoU = 50/150 = 0.333.. -> 기준 1/3 이상이면 유지
        boxes = [[5, 0, 15, 10], [100, 100, 110, 110]]
        self.assertEqual(select_target_index(boxes, [0.5, 0.9], [0, 0, 10, 10], iou_thresh=50 / 150), (0, False))
        self.assertEqual(select_target_index(boxes, [0.5, 0.9], [0, 0, 10, 10], iou_thresh=0.34), (1, True))

    def test_switch_can_return_same_index(self):
        # IoU가 낮아도 최고 신뢰도 후보가 같은 사람이면 인덱스는 같고 switched만 True (런타임과 같음)
        boxes = [[0, 0, 10, 10]]
        self.assertEqual(select_target_index(boxes, [0.9], last_bbox=[8, 8, 30, 30]), (0, True))


LIVE_PARITY_SCRIPT = textwrap.dedent(
    """
    import random, sys
    from unittest import mock
    import numpy as np
    import torch
    for name in ("cv2", "ultralytics"):
        sys.modules[name] = mock.MagicMock()
    from app.engine import live

    class Boxes:
        def __init__(self, xyxy, conf):
            self.xyxy, self.conf = xyxy, conf
        def __len__(self):
            return int(self.xyxy.shape[0])
        def __getitem__(self, s):
            return Boxes(self.xyxy[s], self.conf[s])

    class Results:
        def __init__(self, xyxy, conf):
            self.boxes = Boxes(xyxy, conf)
            self.keypoints = torch.arange(xyxy.shape[0], dtype=torch.float32)

    def original(results, last_bbox=None, iou_thresh=0.3):
        # 리팩토링 전 live.select_best_person (복사본)
        if results.boxes is None or len(results.boxes) == 0:
            return results, False
        best_idx = 0
        is_switched = False
        if last_bbox is not None:
            best_iou = -1.0
            boxes = results.boxes.xyxy.cpu().numpy()
            match_idx = -1
            for i, box in enumerate(boxes):
                iou = live.calculate_iou(last_bbox, box)
                if iou > best_iou:
                    best_iou = iou
                    match_idx = i
            top_conf_idx = int(torch.argmax(results.boxes.conf).item())
            if match_idx != -1 and best_iou >= iou_thresh:
                best_idx = match_idx
            else:
                best_idx = top_conf_idx
                is_switched = True
        else:
            best_idx = int(torch.argmax(results.boxes.conf).item())
        results.boxes = results.boxes[best_idx:best_idx + 1]
        results.keypoints = results.keypoints[best_idx:best_idx + 1]
        return results, is_switched

    rng = random.Random(1234)
    bad = 0
    for case in range(3000):
        n = rng.randint(0, 5)
        boxes = []
        for _ in range(n):
            x, y = rng.uniform(0, 600), rng.uniform(0, 400)
            boxes.append([x, y, x + rng.uniform(1, 200), y + rng.uniform(1, 300)])
        conf = [rng.choice([0.5, 0.7, 0.9, rng.random()]) for _ in range(n)]  # 동점 포함
        xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(n, 4)
        cf = torch.tensor(conf, dtype=torch.float32)
        last = None
        if n and rng.random() < 0.8:
            b = xyxy[rng.randrange(n)].numpy().copy()
            b += np.float32(rng.uniform(-40, 40))  # IoU 임계 주변까지 이동
            last = b
        elif rng.random() < 0.5:
            last = np.array([0, 0, 10, 10], dtype=np.float32)
        thr = rng.choice([0.3, 0.0, 0.5, 0.9])
        r1, s1 = original(Results(xyxy, cf), last, thr)
        r2, s2 = live.select_best_person(Results(xyxy, cf), last_bbox=last, iou_thresh=thr)
        same = (s1 == s2 and len(r1.boxes) == len(r2.boxes)
                and torch.equal(r1.boxes.xyxy, r2.boxes.xyxy) and torch.equal(r1.keypoints, r2.keypoints))
        if not same:
            bad += 1
            if bad < 5:
                print("mismatch", case, boxes, conf, last, thr, s1, s2)
    print("mismatches", bad)
    sys.exit(1 if bad else 0)
    """
)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"), "torch/numpy 없음")
class LiveParityTest(unittest.TestCase):
    def test_select_best_person_matches_original(self):
        # import 부작용(runs/ 생성 등)이 저장소에 남지 않도록 임시 폴더에서 실행
        with tempfile.TemporaryDirectory() as cwd:
            env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), RUNS_DIR=os.path.join(cwd, "runs"))
            proc = subprocess.run([sys.executable, "-c", LIVE_PARITY_SCRIPT], cwd=cwd, env=env,
                                  capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
