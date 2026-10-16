import json
import tempfile
import unittest
from pathlib import Path

from training.datasets import build_manifest as bm
from training.datasets.manifest import load_manifest, make_row, validate_rows, write_manifest

LABEL2ID = "label,id\n" + "".join(f"{n},{i}\n" for i, n in enumerate(
    ("walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up", "standing", "other",
     "kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump")))
HEADER = "path,label,start,end,subject,cam,dataset\n"


def _touch(path: Path, data=None):
    # 내용이 같으면 중복 영상으로 묶이므로 기본값은 경로마다 다른 내용
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.as_posix().encode("utf-8") if data is None else data)


def _seg(cls, a, b):
    return {"cls": cls, "start_s": a, "end_s": b}


class ManifestRowsTest(unittest.TestCase):
    def test_make_row_fall_fields(self):
        r = make_row("le2i", "Home_01/video_1", "dataset/x.avi",
                     [_seg("fallen", 4.0, 6.0), _seg("walk", 0.0, 3.0), _seg("fall", 3.0, 4.0)],
                     "omnifall", "lic", subject=1, cam=2)
        self.assertEqual(r["clip_id"], "le2i/Home_01/video_1")
        self.assertEqual(r["label"], "fall")
        self.assertEqual([s["cls"] for s in r["segments"]], ["walk", "fall", "fallen"])
        self.assertEqual((r["fall_start_s"], r["fall_end_s"]), (3.0, 4.0))
        self.assertEqual((r["subject"], r["cam"], r["event_group"]), ("1", "2", r["clip_id"]))
        self.assertEqual(validate_rows([r]), [])

    def test_make_row_adl_and_folder(self):
        adl = make_row("x", "a", "v.mp4", [_seg("lying", 0, 1)], "omnifall", "lic")
        self.assertEqual((adl["label"], adl["fall_start_s"]), ("adl", None))
        folder = make_row("y", "b", "v.mp4", [], "folder", "lic", label="fall", event_group="y/g")
        self.assertEqual(folder["event_group"], "y/g")
        self.assertEqual(validate_rows([adl, folder]), [])

    def test_validate_catches_errors(self):
        good = make_row("d", "a", "v.mp4", [_seg("fall", 1, 2)], "omnifall", "lic")
        cases = {
            "duplicate": [good, dict(good)],
            "label": [dict(good, label="adl")],
            "segment": [dict(good, segments=[_seg("fall", 2, 2)], fall_start_s=2, fall_end_s=2)],
            "class": [dict(good, segments=[_seg("fall", 1, 2), _seg("dance", 2, 3)])],
            "event_group": [dict(good, event_group="")],
            "fall_start": [dict(good, fall_start_s=0.5)],
            "no fall segment": [make_row("d", "b", "v.mp4", [], "omnifall", "lic", label="fall")],
            "label_source": [dict(good, label_source="guess")],
            "missing": [{k: v for k, v in good.items() if k != "license"}],
        }
        for name, rows in cases.items():
            with self.subTest(name):
                self.assertTrue(validate_rows(rows), name)

    def test_validate_same_start_fall_segments(self):
        # 시작이 같은 fall 구간이 섞여 있어도 make_row과 같은 기준(시작, 끝)으로 첫 구간을 고름
        r = make_row("d", "a", "v.mp4", [_seg("fall", 1, 3), _seg("fall", 1, 2)], "omnifall", "lic")
        self.assertEqual((r["fall_start_s"], r["fall_end_s"]), (1.0, 2.0))
        self.assertEqual(validate_rows([dict(r, segments=list(reversed(r["segments"])))]), [])

    def test_crop_px_validation(self):
        good = make_row("urfd", "fall-01", "v.mp4", [], "urfd_csv", "x", crop_px=[320, 0, 640, 240])
        self.assertEqual(validate_rows([good]), [])
        for bad in ([320, 0, 320, 240], [-1, 0, 10, 10], [0, 0, 10], [0.5, 0, 10, 10], "0,0,10,10", [True, 0, 10, 10]):
            row = dict(good, crop_px=bad)
            self.assertTrue(any("crop_px" in e for e in validate_rows([row])), bad)

    def test_fps_validation(self):
        good = make_row("mcfd", "chute01/cam1", "v.avi", [], "omnifall", "x", fps=30)
        self.assertEqual((good["fps"], validate_rows([good])), (30.0, []))
        for bad in (0, 500, "30", True, None):
            self.assertTrue(any("fps" in e for e in validate_rows([dict(good, fps=bad)])), bad)

    def test_write_load_roundtrip(self):
        rows = [make_row("d", f"c{i}", f"v{i}.mp4", [_seg("fall", i, i + 1)], "omnifall", "라이선스")
                for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sub" / "m.jsonl"
            write_manifest(p, rows)
            self.assertEqual(load_manifest(p), rows)
            self.assertFalse(p.with_name("m.jsonl.tmp").exists())


class BuildManifestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        self.ds = self.repo / "dataset"
        labels = self.ds / "omnifall_labels" / "raw"
        labels.mkdir(parents=True)
        (labels / "labels__label2id.csv").write_text(LABEL2ID, encoding="utf-8")
        # le2i: 장면 폴더가 두 번 반복되고 파일 이름은 "video (N).avi"
        (labels / "labels__le2i.csv").write_text(HEADER + (
            "Coffee_room_01/video_14,0,0.0,1.5,0,1,le2i\n"
            "Coffee_room_01/video_14,1,1.5,2.5,0,1,le2i\n"
            "Lecture_room/video_2,4,0.0,3.0,3,5,le2i\n"
            "Office/video_99,1,0.0,1.0,0,1,le2i\n"), encoding="utf-8")
        le2i = self.ds / "le2i" / "extracted" / "FallDataset"
        _touch(le2i / "Coffee_room_01" / "Coffee_room_01" / "Videos" / "video (14).avi")
        _touch(le2i / "Coffee_room_01" / "Coffee_room_01" / "Videos" / "video (1).avi")
        _touch(le2i / "Lecture_room" / "Lecture room" / "video (2).avi")
        # caucafall: OmniFall 경로의 첫 폴더(adl/side)는 원본 폴더와 다름
        (labels / "labels__caucafall.csv").write_text(HEADER + (
            "adl/HopS1,9,0.0,9.1,1,1,caucafall\n"
            "side/FallLeftS2,1,2.0,3.0,2,1,caucafall\n"), encoding="utf-8")
        cauca = self.ds / "caucafall" / "extracted" / "CAUCAFall-v4" / "Dataset CAUCAFall" / "CAUCAFall"
        _touch(cauca / "Subject.1" / "Hop" / "HopS1.avi")
        _touch(cauca / "Subject.2" / "Fall left" / "FallLeftS2.avi")
        # gmdcsa24: 피험자 폴더 이름에 공백, 파일 이름은 피험자마다 반복
        (labels / "labels__GMDCSA24.csv").write_text(HEADER + (
            "Subject_1/ADL/01,4,0.0,0.5,1,1,GMDCSA24\n"
            "Subject_2/ADL/01,6,0.0,0.5,2,1,GMDCSA24\n"), encoding="utf-8")
        gm = self.ds / "gmdcsa24" / "extracted" / "GMDCSA24-v2.1" / "repo"
        _touch(gm / "Subject 1" / "ADL" / "01.mp4")
        _touch(gm / "Subject 2" / "ADL" / "01.mp4")
        # mcfd: 같은 시나리오(chute01)를 카메라 여러 대로
        (labels / "labels__mcfd.csv").write_text(HEADER + "".join(
            f"chute01/cam{c},0,0.0,1.0,1,{c},mcfd\nchute01/cam{c},1,1.0,2.0,1,{c},mcfd\n" for c in (1, 2, 3)
        ) + "chute02/cam1,4,0.0,1.0,1,1,mcfd\n", encoding="utf-8")
        for c in (1, 2, 3):
            _touch(self.ds / "mcfd" / "extracted" / "dataset" / "dataset" / "chute01" / f"cam{c}.avi")
        _touch(self.ds / "mcfd" / "extracted" / "dataset" / "dataset" / "chute02" / "cam1.avi")
        # urfd: CSV 2개 + cam0 mp4 (adl-02 영상은 없음)
        raw = self.ds / "urfd" / "raw"
        raw.mkdir(parents=True)
        (raw / "urfall-cam0-falls.csv").write_text("fall-01,1,-1,0\nfall-01,2,0,0\nfall-01,3,1,0\n",
                                                   encoding="utf-8")
        (raw / "urfall-cam0-adls.csv").write_text("adl-01,1,-1,0\nadl-01,2,0,0\nadl-01,4,1,0\nadl-02,1,1,0\n",
                                                  encoding="utf-8")
        _touch(raw / "fall-01-cam0.mp4")
        _touch(raw / "adl-01-cam0.mp4")
        # figshare: Fall/ADL 폴더 라벨, 구조가 다른 영상은 건너뜀
        fig = self.ds / "figshare_ursul29" / "extracted" / "VideoDataset" / "VideoDataset"
        _touch(fig / "Fall" / "SBJ_01_LOC1" / "ACT1_L_1" / "20240901.mp4")
        _touch(fig / "ADL" / "SBJ_12_LOC2" / "ACT13_B_1" / "20240902.mp4")
        _touch(fig / "Fall" / "timelapse_20x.mp4")

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, datasets=bm.DATASETS):
        return bm.build(datasets, self.ds, self.repo)

    def test_all_datasets(self):
        rows, summary = self.build()
        self.assertEqual(summary["validation_errors"], 0, summary["validation_error_examples"])
        by_id = {r["clip_id"]: r for r in rows}
        self.assertEqual(set(by_id), {
            "le2i/Coffee_room_01/video_14", "le2i/Lecture_room/video_2",
            "caucafall/adl/HopS1", "caucafall/side/FallLeftS2",
            "gmdcsa24/Subject_1/ADL/01", "gmdcsa24/Subject_2/ADL/01",
            "mcfd/chute01/cam1", "mcfd/chute01/cam2", "mcfd/chute01/cam3", "mcfd/chute02/cam1",
            "urfd/fall-01", "urfd/adl-01",
            "figshare_ursul29/Fall/SBJ_01_LOC1/ACT1_L_1/20240901",
            "figshare_ursul29/ADL/SBJ_12_LOC2/ACT13_B_1/20240902",
        })
        r = by_id["le2i/Coffee_room_01/video_14"]
        self.assertEqual(r["video"], "dataset/le2i/extracted/FallDataset/Coffee_room_01/Coffee_room_01/Videos/"
                                     "video (14).avi")
        self.assertEqual((r["label"], r["fall_start_s"], r["fall_end_s"]), ("fall", 1.5, 2.5))
        self.assertEqual([s["cls"] for s in r["segments"]], ["walk", "fall"])
        self.assertEqual(by_id["le2i/Lecture_room/video_2"]["label"], "adl")
        self.assertTrue(by_id["caucafall/side/FallLeftS2"]["video"].endswith("Subject.2/Fall left/FallLeftS2.avi"))
        self.assertTrue(by_id["gmdcsa24/Subject_2/ADL/01"]["video"].endswith("Subject 2/ADL/01.mp4"))
        self.assertEqual((by_id["gmdcsa24/Subject_2/ADL/01"]["subject"],
                          by_id["gmdcsa24/Subject_2/ADL/01"]["label"]), ("2", "adl"))
        # MCFD는 시나리오(chuteNN) 단위로 묶임
        self.assertEqual({by_id[f"mcfd/chute01/cam{c}"]["event_group"] for c in (1, 2, 3)}, {"mcfd/chute01"})
        self.assertEqual(by_id["mcfd/chute02/cam1"]["event_group"], "mcfd/chute02")
        self.assertEqual(by_id["mcfd/chute01/cam2"]["cam"], "2")
        # MCFD avi 헤더는 120fps지만 실제/라벨 기준은 30fps
        self.assertEqual(by_id["mcfd/chute01/cam1"]["fps"], 30.0)
        self.assertNotIn("fps", by_id["le2i/Coffee_room_01/video_14"])
        # URFD
        fall = by_id["urfd/fall-01"]
        self.assertEqual((fall["label"], fall["label_source"], fall["video"]),
                         ("fall", "urfd_csv", "dataset/urfd/raw/fall-01-cam0.mp4"))
        self.assertEqual([s["cls"] for s in by_id["urfd/adl-01"]["segments"]], ["lie_down", "lying"])
        # URFD cam0는 깊이|RGB 합성 영상이라 RGB 절반만 사용
        self.assertEqual(fall["crop_px"], [320, 0, 640, 240])
        self.assertNotIn("crop_px", by_id["le2i/Coffee_room_01/video_14"])
        self.assertEqual(by_id["urfd/adl-01"]["label"], "adl")
        # figshare
        f = by_id["figshare_ursul29/Fall/SBJ_01_LOC1/ACT1_L_1/20240901"]
        self.assertEqual((f["label"], f["segments"], f["label_source"], f["subject"]), ("fall", [], "folder", "1"))
        self.assertEqual(by_id["figshare_ursul29/ADL/SBJ_12_LOC2/ACT13_B_1/20240902"]["label"], "adl")

        ds = summary["datasets"]
        self.assertEqual((ds["le2i"]["unmatched"], ds["le2i"]["unmatched_examples"]), (1, ["Office/video_99"]))
        self.assertEqual(ds["le2i"]["unlabeled_videos"], 1)
        self.assertEqual((ds["mcfd"]["rows"], ds["mcfd"]["event_groups"], ds["mcfd"]["fall"]), (4, 2, 3))
        self.assertEqual((ds["urfd"]["unmatched"], ds["urfd"]["unmatched_examples"]), (1, ["adl-02"]))
        self.assertEqual(ds["figshare_ursul29"]["skipped"], 1)
        for name in ("caucafall", "gmdcsa24"):
            self.assertEqual((ds[name]["unmatched"], ds[name]["ambiguous"]), (0, 0), name)
        self.assertEqual(summary["rows"], len(rows))

    def test_ambiguous_paths_are_reported(self):
        # 같은 키에 맞는 파일이 둘이면 추측하지 않고 ambiguous로 남김
        _touch(self.ds / "caucafall" / "extracted" / "copy" / "HopS1.avi")
        rows, summary = self.build(["caucafall"])
        self.assertEqual([r["clip_id"] for r in rows], ["caucafall/side/FallLeftS2"])
        self.assertEqual(summary["datasets"]["caucafall"]["ambiguous_examples"], ["adl/HopS1"])

    def test_duplicate_videos_share_event_group(self):
        # Le2i처럼 번호만 다른 같은 영상, MCFD 그룹과 겹치는 사본도 하나의 event_group으로 합침
        (self.ds / "omnifall_labels" / "raw" / "labels__le2i.csv").write_text(HEADER + (
            "Coffee_room_01/video_14,1,1.5,2.5,0,1,le2i\n"
            "Coffee_room_01/video_1,1,1.6,2.4,0,1,le2i\n"
            "Lecture_room/video_2,4,0.0,3.0,3,5,le2i\n"), encoding="utf-8")
        le2i = self.ds / "le2i" / "extracted" / "FallDataset"
        _touch(le2i / "Coffee_room_01" / "Coffee_room_01" / "Videos" / "video (14).avi", b"same")
        _touch(le2i / "Coffee_room_01" / "Coffee_room_01" / "Videos" / "video (1).avi", b"same")
        _touch(le2i / "Lecture_room" / "Lecture room" / "video (2).avi", b"mcfd-copy")
        _touch(self.ds / "mcfd" / "extracted" / "dataset" / "dataset" / "chute02" / "cam1.avi", b"mcfd-copy")
        rows, summary = self.build(["le2i", "mcfd"])
        self.assertEqual(summary["validation_errors"], 0, summary["validation_error_examples"])
        by_id = {r["clip_id"]: r for r in rows}
        self.assertEqual(len(rows), 7)  # 행은 지우지 않음
        self.assertEqual(by_id["le2i/Coffee_room_01/video_1"]["event_group"], "le2i/Coffee_room_01/video_1")
        self.assertEqual(by_id["le2i/Coffee_room_01/video_14"]["event_group"], "le2i/Coffee_room_01/video_1")
        self.assertEqual(by_id["le2i/Lecture_room/video_2"]["event_group"], "le2i/Lecture_room/video_2")
        self.assertEqual(by_id["mcfd/chute02/cam1"]["event_group"], "le2i/Lecture_room/video_2")
        self.assertEqual(by_id["mcfd/chute01/cam1"]["event_group"], "mcfd/chute01")
        self.assertEqual(summary["duplicate_videos"], 2)
        self.assertEqual(summary["duplicate_video_examples"], [
            ["le2i/Coffee_room_01/video_1", "le2i/Coffee_room_01/video_14"],
            ["le2i/Lecture_room/video_2", "mcfd/chute02/cam1"]])
        self.assertEqual((summary["datasets"]["le2i"]["event_groups"], summary["datasets"]["mcfd"]["event_groups"]),
                         (2, 2))

    def test_no_duplicates_in_fixture_and_repeated_dataset_names(self):
        rows, summary = self.build(["mcfd", "mcfd"])
        self.assertEqual(summary["duplicate_videos"], 0)
        self.assertEqual(len(rows), 4)
        self.assertEqual(summary["validation_errors"], 0)

    def test_missing_dataset_is_reported(self):
        rows, summary = bm.build(["le2i", "urfd", "figshare_ursul29"], self.repo / "nothing", self.repo)
        self.assertEqual(rows, [])
        for name in ("le2i", "urfd", "figshare_ursul29"):
            self.assertTrue(summary["datasets"][name]["missing"], name)

    def test_cli_writes_manifest_and_summary(self):
        out = self.repo / "dataset" / "_manifest" / "m.jsonl"
        code = bm.main(["--datasets", "mcfd", "urfd", "--dataset-root", str(self.ds), "--out", str(out)])
        self.assertEqual(code, 0)
        rows = load_manifest(out)
        self.assertEqual(len(rows), 6)
        self.assertEqual(validate_rows(rows), [])
        summary = json.loads((out.parent / "m.summary.json").read_text(encoding="utf-8"))
        self.assertEqual(set(summary["datasets"]), {"mcfd", "urfd"})
        self.assertEqual(summary["rows"], 6)


class NormTest(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(bm.norm("video (14)"), "video_14")
        self.assertEqual(bm.norm("Lecture room"), bm.norm("Lecture_room"))
        self.assertEqual(bm.omnifall_key("caucafall", "side/FallLeftS2"), "falllefts2")
        self.assertEqual(bm.omnifall_key("gmdcsa24", "Subject_1/ADL/01"), "subject_1/adl/01")


if __name__ == "__main__":
    unittest.main()
