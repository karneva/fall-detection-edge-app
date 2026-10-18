"""
포즈 의사 라벨 도구(training/pose) 테스트: 회전 좌표 변환, 회전 검출 합치기, YOLO 라벨 줄, 평가셋 COCO 변환.
numpy가 없으면 skip (cv2/ultralytics는 쓰지 않음: 예측 함수는 가짜로 대신).
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None

if HAS_NUMPY:
    import numpy as np

    from training.pose import rotation as RT
    from training.pose.dataset import split_by_video, yolo_label_lines
    from training.pose.evalset import coco_to_yolo, to_coco
    from training.pose.pseudo import detect_rotations, iou, keep_person, merge_rotations


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class RotationTest(unittest.TestCase):
    def test_roundtrip_and_sizes(self):
        w, h = 7, 4
        pts = np.array([[0, 0], [6, 0], [0, 3], [6, 3], [2.5, 1.0]])
        for a in RT.ANGLES:
            r = RT.to_rotated(pts, w, h, a)
            np.testing.assert_allclose(RT.from_rotated(r, w, h, a), pts)
            rw, rh = RT.rotated_size(w, h, a)
            self.assertTrue((r[:, 0] >= 0).all() and (r[:, 0] <= rw - 1).all())
            self.assertTrue((r[:, 1] >= 0).all() and (r[:, 1] <= rh - 1).all())

    @unittest.skipUnless(HAS_CV2, "cv2 없음")
    def test_matches_cv2_rotate(self):
        import cv2
        img = np.zeros((4, 7), dtype=np.uint8)
        img[1, 5] = 255      # (x=5, y=1)
        codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        for a, code in codes.items():
            r = cv2.rotate(img, code)
            yx = np.argwhere(r == 255)[0]
            np.testing.assert_allclose(RT.to_rotated(np.array([5.0, 1.0]), 7, 4, a), [yx[1], yx[0]])

    def test_box_from_rotated(self):
        b = RT.box_from_rotated(np.array([1, 2, 3, 6]), 10, 8, 90)
        corners = RT.from_rotated(np.array([[1, 2], [3, 6]]), 10, 8, 90)
        np.testing.assert_allclose(b, [corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(),
                                       corners[:, 1].max()])


def _person(box, conf, kc=0.9, angle=0):
    k = np.zeros((17, 3))
    k[:, 0] = np.linspace(box[0], box[2], 17)
    k[:, 1] = np.linspace(box[1], box[3], 17)
    k[:, 2] = kc
    return {"box": np.asarray(box, float), "conf": conf, "kps": k, "angle": angle}


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class PseudoTest(unittest.TestCase):
    def test_merge_keeps_best_per_person(self):
        dets = [_person([0, 0, 10, 20], 0.6, angle=0), _person([0, 0, 10, 21], 0.9, angle=90),
                _person([50, 50, 60, 70], 0.7)]
        kept = merge_rotations(dets)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["angle"], 90)
        self.assertAlmostEqual(iou(np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])), 50 / 150)

    @unittest.skipUnless(HAS_CV2, "cv2 없음")
    def test_detect_rotations_maps_back(self):
        img = np.zeros((40, 60, 3), dtype=np.uint8)
        true_box = np.array([10.0, 5.0, 20.0, 30.0])

        def predict(imgs):
            out = []
            for a, im in zip(RT.ANGLES, imgs):
                h, w = 40, 60
                c = RT.to_rotated(np.array([[10, 5], [20, 30]]), w, h, a)
                b = np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()])
                self.assertEqual(im.shape[:2][::-1], RT.rotated_size(w, h, a))
                k = np.tile(np.array([[*RT.to_rotated(np.array([15.0, 10.0]), w, h, a), 0.8]]), (17, 1))
                out.append((b[None], np.array([0.5 + 0.1 * (a == 90)]), k[None]))
            return out
        dets = detect_rotations(predict, img)
        self.assertEqual(len(dets), 4)
        for d in dets:
            np.testing.assert_allclose(d["box"], true_box)
            np.testing.assert_allclose(d["kps"][0, :2], [15.0, 10.0])
        self.assertEqual(merge_rotations(dets)[0]["angle"], 90)

    def test_keep_person(self):
        self.assertTrue(keep_person(_person([0, 0, 10, 20], 0.6)))
        self.assertFalse(keep_person(_person([0, 0, 10, 20], 0.4)))
        self.assertFalse(keep_person(_person([0, 0, 10, 20], 0.9, kc=0.3)))


@unittest.skipUnless(HAS_NUMPY, "numpy 없음")
class LabelTest(unittest.TestCase):
    def test_yolo_lines(self):
        p = _person([10, 20, 30, 60], 0.9)
        p["kps"][3, 2] = 0.1          # 안 보이는 관절
        line = yolo_label_lines([p], 100, 100)[0].split()
        self.assertEqual(len(line), 1 + 4 + 17 * 3)
        self.assertEqual(line[0], "0")
        np.testing.assert_allclose([float(v) for v in line[1:5]], [0.2, 0.4, 0.2, 0.4])
        self.assertEqual(line[5 + 3 * 3: 5 + 3 * 3 + 3], ["0.000000", "0.000000", "0"])
        self.assertEqual(line[5 + 2], "2")

    def test_split_by_video(self):
        items = [{"group": f"g{i}"} for i in range(20)] + [{"group": "g0"}]
        s = split_by_video(items, 0.1, seed=0)
        self.assertEqual(sum(v == "val" for v in s.values()), 2)
        self.assertEqual(s, split_by_video(items, 0.1, seed=0))

    def test_coco_roundtrip(self):
        images = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 50}]
        coco = to_coco(images, [[_person([10, 10, 30, 40], 0.8), _person([0, 0, 5, 5], 0.1)]])
        self.assertEqual(len(coco["annotations"]), 1)       # 신뢰도 0.3 미만은 초안에서 뺌
        self.assertEqual(len(coco["categories"][0]["keypoints"]), 17)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(coco_to_yolo(json.loads(json.dumps(coco)), Path(d), Path(d)), 1)
            line = (Path(d) / "labels" / "val" / "a.txt").read_text().split()
            np.testing.assert_allclose([float(v) for v in line[1:5]], [0.2, 0.5, 0.2, 0.6])
            self.assertIn("kpt_shape: [17, 3]", (Path(d) / "data.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
