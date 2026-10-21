"""
현장 클립 검수 큐(training/review_queue.py)와 검수 라벨만 쓰는 재학습(training/retrain_from_runs.py) 테스트.
표준 라이브러리만 사용 (가짜 sidecar/클립을 임시 폴더에 만들고 subprocess는 mock).
"""
import contextlib
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.core.event_sidecar import write_sidecar
from training import retrain_from_runs as RR
from training import review_queue as RQ
from training import train_v2 as TV
from training.datasets.manifest import load_manifest, validate_rows
from training.paths import REPO_ROOT, WEIGHTS_DIR, is_runtime_weights_path


def _event(typ, ts, p_fall, reason="still_for_sustain_s", **values):
    return {"type": typ, "ts": ts, "frame_index": 1, "reason": reason,
            "values": {"ai": {"p_fall": p_fall, "decision": "adl" if p_fall < 0.4 else "fall"}, **values}}


class _Runs:
    """임시 runs/ 폴더 (런타임처럼 클립 경로를 'runs/clips/...' 상대 경로로 기록)."""

    def __init__(self, root: Path):
        self.root = root
        self.runs = root / "runs"
        self.events = self.runs / "events"

    def clip(self, sub, name):
        p = self.runs / "clips" / sub / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode("utf-8"))
        return f"runs/clips/{sub}/{name}"

    def sidecar(self, ev, clip=None, abn=None, span=(0.0, 12.0)):
        snap = {"abnormal_start_ts": abn} if abn is not None else None
        t0 = abn - 6.0 if abn is not None else None
        path = write_sidecar(str(self.events), ev, device_id="dev1", location_id="loc", snapshot=snap,
                             clip={"path": clip, "t_start": t0, "t_end": t0 + span[1] if t0 is not None else None})
        return Path(path).stem


class ReviewQueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.r = _Runs(Path(self.tmp.name))
        r = self.r
        # 사건 A: 억제(1000) -> 40초 뒤 승격(1040). 같은 abnormal_start_ts
        self.sA = r.sidecar(_event("ai_suppressed", 1000.0, 0.10, reason="still_for_sustain_s"),
                            r.clip("not_fall", "suppressed_A.mp4"), abn=990.0, span=(0, 30.0))
        self.eA = r.sidecar(_event("level1", 1040.0, 0.10, reason=RQ.ESCALATED_REASON, suppressed_for_s=40.0),
                            r.clip("fall", "incident_A.mp4"), abn=990.0)
        # 사건 B: 억제만 (승격 없음)
        self.sB = r.sidecar(_event("ai_suppressed", 2000.0, 0.20), r.clip("not_fall", "suppressed_B.mp4"), abn=1995.0)
        # 사건 C: 일반 알람
        self.lC = r.sidecar(_event("level1", 3000.0, 0.90), r.clip("fall", "incident_C.mp4"), abn=2990.0)
        # sidecar 없는 클립, 클립 없는 sidecar
        r.clip("not_fall", "suppressed_20261021_101500.mp4")
        self.nD = r.sidecar(_event("level1", 4000.0, 0.95))
        self.labels = r.runs / "review" / "labels.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _queue(self):
        entries, problems = RQ.load_queue(self.r.runs, self.labels)
        return {e["id"]: e for e in entries}, entries, problems

    def _label(self, *args, **kw):
        RQ.append_label(self.labels, RQ.make_label_record(*args, **kw))

    def test_ids_are_stable_and_derived_from_sidecar(self):
        by_id, entries, problems = self._queue()
        self.assertEqual(problems, [])
        self.assertEqual(set(by_id), {self.sA, self.eA, self.sB, self.lC, self.nD,
                                      "clip_not_fall_suppressed_20261021_101500"})
        self.assertTrue(self.sA.endswith("_ai_suppressed"))
        # 더 이른 사건이 추가되어 목록 순서가 바뀌어도 기존 id는 그대로
        new = self.r.sidecar(_event("ai_suppressed", 500.0, 0.05), self.r.clip("not_fall", "s0.mp4"), abn=495.0)
        by_id2, entries2, _ = self._queue()
        self.assertEqual(entries2[0]["id"], new)
        for k, e in by_id.items():
            self.assertEqual((by_id2[k]["clip"], by_id2[k]["outcome"]), (e["clip"], e["outcome"]))

    def test_outcomes_links_and_priority(self):
        by_id, entries, _ = self._queue()
        self.assertEqual(by_id[self.sA]["outcome"], "suppressed_escalated")
        self.assertEqual(by_id[self.eA]["outcome"], "escalated")
        self.assertEqual((by_id[self.sA]["linked"], by_id[self.eA]["linked"]), (self.eA, self.sA))
        self.assertEqual(by_id[self.sA]["incident"], by_id[self.eA]["incident"])
        self.assertEqual(by_id[self.sB]["outcome"], "ai_suppressed")
        self.assertEqual(by_id[self.lC]["outcome"], "level1")
        self.assertEqual(by_id["clip_not_fall_suppressed_20261021_101500"]["priority"], 1)
        self.assertEqual(by_id[self.lC]["priority"], 2)
        self.assertIsNone(by_id[self.nD]["clip"])
        self.assertAlmostEqual(by_id[self.sA]["entry_in_clip_s"], 6.0)
        self.assertAlmostEqual(by_id[self.sA]["p_fall"], 0.10)
        # AI 억제가 걸린 사건이 먼저
        prios = [e["priority"] for e in entries]
        self.assertEqual(prios, sorted(prios))
        self.assertEqual({e["id"] for e in entries if e["priority"] == 1},
                         {self.sA, self.eA, self.sB, "clip_not_fall_suppressed_20261021_101500"})

    def test_link_without_snapshot_uses_suppressed_for(self):
        r = self.r
        s = r.sidecar(_event("ai_suppressed", 5000.25, 0.1), r.clip("not_fall", "sE.mp4"))
        e = r.sidecar(_event("level1", 5030.75, 0.1, reason=RQ.ESCALATED_REASON, suppressed_for_s=30.5),
                      r.clip("fall", "iE.mp4"))
        by_id, _, _ = self._queue()
        self.assertEqual(by_id[e]["linked"], s)
        self.assertEqual(by_id[s]["outcome"], "suppressed_escalated")
        self.assertEqual(by_id[e]["incident"], s)

    def test_latest_label_wins_and_bad_lines_skipped(self):
        self._label(self.sB, "fall", fall_start_s=5.0, fall_end_s=6.0)
        with open(self.labels, "a", encoding="utf-8") as f:
            f.write("{not json\n")
            f.write(json.dumps({"id": self.sB, "label": "maybe"}) + "\n")
        self._label(self.sB, "not_fall", note="침대에 누움")
        by_id, _, problems = self._queue()
        self.assertEqual(by_id[self.sB]["status"], "not_fall")
        self.assertEqual(by_id[self.sB]["review"]["note"], "침대에 누움")
        self.assertEqual(len([p for p in problems if "labels.jsonl" in p]), 2)
        self.assertEqual(by_id[self.sA]["status"], "unreviewed")
        # append-only: 앞의 기록도 남아 있음
        self.assertEqual(self.labels.read_text(encoding="utf-8").count(self.sB), 3)

    def test_label_record_validation(self):
        with self.assertRaises(ValueError):
            RQ.make_label_record("x", "adl")
        with self.assertRaises(ValueError):
            RQ.make_label_record("x", "fall", fall_start_s=1.0)
        with self.assertRaises(ValueError):
            RQ.make_label_record("x", "fall", fall_start_s=2.0, fall_end_s=1.0)
        with self.assertRaises(ValueError):
            RQ.make_label_record("x", "not_fall", fall_start_s=1.0, fall_end_s=2.0)
        with self.assertRaises(ValueError):
            RQ.make_label_record("x", "fall", fall_start_s=1.0, fall_end_s=float("inf"))
        rec = RQ.make_label_record("x", "unsure", reviewer="me", now=0)
        self.assertEqual((rec["label"], rec["reviewer"], rec["fall_start_s"]), ("unsure", "me", None))

    def test_append_after_line_without_newline(self):
        # 편집기가 끝 줄바꿈을 지운 경우: 새 기록은 다음 줄에 써져 그대로 읽힘
        self._label(self.sB, "fall", fall_start_s=4.0, fall_end_s=5.0)
        self._label(self.sB, "not_fall")
        self.labels.write_bytes(self.labels.read_bytes().rstrip(b"\n"))
        self._label(self.sB, "unsure")
        by_id, _, problems = self._queue()
        self.assertEqual(by_id[self.sB]["status"], "unsure")
        self.assertEqual([p for p in problems if "labels.jsonl" in p], [])
        # 이전 기록이 쓰다 끊긴 경우: 새 기록은 읽히고, 끊긴 줄은 문제로 남고, strict 읽기(export)는 멈춤
        with open(self.labels, "ab") as f:
            f.write(b'{"id": "' + self.sA.encode() + b'", "label": "fa')
        self._label(self.sB, "not_fall")
        by_id, _, problems = self._queue()
        self.assertEqual(by_id[self.sB]["status"], "not_fall")
        self.assertEqual(len([p for p in problems if "invalid JSON" in p]), 1)
        with self.assertRaises(RQ.LabelLogError):
            RQ.load_queue(self.r.runs, self.labels, strict=True)

    def test_bad_records_are_skipped_not_crashing(self):
        lines = [
            {"id": self.sB, "label": "fall", "fall_start_s": 5, "fall_end_s": 3},  # start >= end
            {"id": self.sB, "label": "fall", "fall_start_s": 5},  # end 없음
            {"id": 5, "label": "fall"},  # id가 문자열이 아님
            {"id": "zz", "label": "fall"},
            {"id": self.lC, "label": "not_fall", "fall_start_s": 1, "fall_end_s": 2},  # fall이 아닌데 구간
            {"id": self.lC, "label": "fall", "fall_start_s": "1", "fall_end_s": 2},  # 숫자가 아님
        ]
        self.labels.parent.mkdir(parents=True, exist_ok=True)
        with open(self.labels, "w", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec) + "\n")
            f.write('{"id": "x", "label": "fall", "fall_start_s": 1, "fall_end_s": Infinity}\n')
        by_id, entries, problems = self._queue()
        self.assertEqual(len([p for p in problems if "bad record" in p]), 6)
        self.assertIn("label for unknown id zz", problems)
        self.assertEqual(by_id[self.sB]["status"], "unreviewed")
        self.assertEqual(RQ.export_rows(entries), ([], []))
        out = Path(self.tmp.name) / "field.jsonl"
        base = ["--runs-dir", str(self.r.runs)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(RQ.main(base + ["list", "--status", "all"]), 0)
            self.assertEqual(RQ.main(base + ["stats"]), 0)
            self.assertEqual(RQ.main(base + ["export", "--out", str(out)]), 1)  # 읽지 못한 줄이 있으면 멈춤
        self.assertIn("labels.jsonl:1", err.getvalue())
        self.assertFalse(out.exists())

    def test_label_cli_warns_on_bad_log_and_refuses_segment_past_clip(self):
        base = ["--runs-dir", str(self.r.runs)]
        self.labels.parent.mkdir(parents=True, exist_ok=True)
        self.labels.write_text('{"id": "' + self.sB + '", "label": "fa', encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(RQ.main(base + ["label", self.sB, "not_fall"]), 0)
        self.assertIn("invalid JSON", err.getvalue())
        by_id, _, _ = self._queue()
        self.assertEqual(by_id[self.sB]["status"], "not_fall")
        # sB 클립은 12초: 구간 끝이 클립 끝을 넘으면 기록하지 않음
        before = self.labels.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(RQ.main(base + ["label", self.sB, "fall", "--fall-start", "5", "--fall-end", "20"]), 2)
        self.assertIn("past the clip end", err.getvalue())
        self.assertEqual(self.labels.read_bytes(), before)

    def test_cli_output_survives_console_without_utf8(self):
        base = ["--runs-dir", str(self.r.runs)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(RQ.main(base + ["label", self.sB, "not_fall", "--note", "의자 — 앉음 ✓"]), 0)
        for cmd in (["show", self.sB], ["list", "--status", "all", "--json"]):
            out = io.TextIOWrapper(io.BytesIO(), encoding="cp949")
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(RQ.main(base + cmd), 0)
            out.flush()
            text = out.buffer.getvalue().decode("cp949")
            self.assertIn(self.sB, text)
        rows = [json.loads(line) for line in text.splitlines()]
        self.assertEqual({r["id"]: r for r in rows}[self.sB]["review"]["note"], "의자 — 앉음 ✓")

    def test_export_only_reviewed_fall_and_not_fall(self):
        by_id, entries, _ = self._queue()
        self.assertEqual(RQ.export_rows(entries), ([], []))  # 아무것도 검수 안 함 -> 없음
        self._label(self.sA, "fall", fall_start_s=4.8, fall_end_s=5.6)
        self._label(self.eA, "fall", fall_start_s=4.8, fall_end_s=5.6)
        self._label(self.sB, "unsure")
        self._label(self.lC, "not_fall")
        self._label("clip_not_fall_suppressed_20261021_101500", "fall")  # 구간 없음
        self._label(self.nD, "fall", fall_start_s=1.0, fall_end_s=2.0)  # 클립 없음
        _, entries, _ = self._queue()
        rows, skipped = RQ.export_rows(entries)
        self.assertEqual(validate_rows(rows), [])
        got = {r["review"]["id"]: r for r in rows}
        # 사건 A는 더 긴 억제 클립 하나만, unsure/구간 없는 fall/클립 없는 항목은 제외
        self.assertEqual(set(got), {self.sA, self.lC})
        reasons = {s["id"]: s["reason"] for s in skipped}
        self.assertEqual(reasons, {self.eA: f"duplicate_of:{self.sA}",
                                   "clip_not_fall_suppressed_20261021_101500": "fall_no_segment",
                                   self.nD: "no_clip"})
        a, c = got[self.sA], got[self.lC]
        self.assertEqual((a["label"], a["fall_start_s"], a["fall_end_s"]), ("fall", 4.8, 5.6))
        self.assertEqual((a["dataset"], a["label_source"], a["clip_id"]), ("field_runs", "review", f"field_runs/{self.sA}"))
        self.assertEqual(a["event_group"], f"field_runs/{self.sA}")
        self.assertEqual((c["label"], c["segments"]), ("adl", []))
        self.assertTrue(Path(a["video"]).is_file() or (REPO_ROOT / a["video"]).is_file())
        self.assertNotIn("fps", a)

    def test_conflicting_labels_in_one_incident_are_not_exported(self):
        self._label(self.sA, "not_fall")
        self._label(self.eA, "fall", fall_start_s=4.8, fall_end_s=5.6)
        _, entries, _ = self._queue()
        rows, skipped = RQ.export_rows(entries)
        self.assertEqual(rows, [])
        self.assertEqual({s["reason"] for s in skipped}, {"conflict"})

    def test_unsure_in_incident_blocks_other_clips(self):
        # 승격 클립을 unsure로 두면, 그 구간을 모두 담는 억제 클립도 내보내지 않음
        self._label(self.eA, "unsure")
        self._label(self.sA, "not_fall")
        _, entries, _ = self._queue()
        rows, skipped = RQ.export_rows(entries)
        self.assertEqual(rows, [])
        self.assertEqual(skipped, [{"id": self.sA, "reason": "incident_unsure"}])

    def test_clip_only_clips_of_one_event_are_grouped(self):
        r = self.r
        r.clip("not_fall", "suppressed_20261021_101010.mp4")
        r.clip("fall", "incident_20261021_101115.mp4")  # 억제 클립 65초 뒤: 승격일 수 있음
        r.clip("fall", "incident_20261021_101530.mp4")  # 10:15:00 억제 클립 30초 뒤: 승격 불가
        r.clip("fall", "incident_20261021_120000.mp4")  # 마지막 억제 클립보다 한참 뒤
        by_id, _, _ = self._queue()
        s, e = "clip_not_fall_suppressed_20261021_101010", "clip_fall_incident_20261021_101115"
        self.assertEqual((by_id[s]["linked"], by_id[e]["linked"]), (e, s))
        self.assertEqual(by_id[e]["incident"], s)
        self.assertEqual(by_id[e]["priority"], 1)
        for cid in ("clip_fall_incident_20261021_101530", "clip_fall_incident_20261021_120000"):
            self.assertIsNone(by_id[cid]["linked"])
            self.assertEqual((by_id[cid]["incident"], by_id[cid]["priority"]), (cid, 2))
        # 라벨이 다르면 둘 다 conflict, 같으면 한 행
        self._label(s, "not_fall")
        self._label(e, "fall", fall_start_s=5.0, fall_end_s=6.0)
        _, entries, _ = self._queue()
        rows, skipped = RQ.export_rows(entries)
        self.assertEqual(rows, [])
        self.assertEqual({x["id"]: x["reason"] for x in skipped}, {s: "conflict", e: "conflict"})
        self._label(s, "fall", fall_start_s=5.0, fall_end_s=6.0)
        _, entries, _ = self._queue()
        rows, skipped = RQ.export_rows(entries)
        self.assertEqual([(x["event_group"], x["label"]) for x in rows], [(f"field_runs/{s}", "fall")])
        self.assertEqual(len(skipped), 1)

    def _snapshot_files(self):
        return {p.relative_to(self.r.runs).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.r.runs.rglob("*") if p.is_file() and p.parent.name != "review"}

    def test_cli_label_and_export_never_touch_clips(self):
        before = self._snapshot_files()
        out = Path(self.tmp.name) / "field.jsonl"
        base = ["--runs-dir", str(self.r.runs)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(RQ.main(base + ["label", self.sB, "fall", "--fall-start", "5", "--fall-end", "6",
                                             "--reviewer", "t"]), 0)
            self.assertEqual(RQ.main(base + ["label", "nope", "fall"]), 2)
            self.assertEqual(RQ.main(base + ["list", "--status", "unreviewed", "--priority", "1"]), 0)
            self.assertEqual(RQ.main(base + ["stats"]), 0)
            self.assertEqual(RQ.main(base + ["export", "--out", str(out)]), 0)
        self.assertIn(f"labeled {self.sB} = fall", buf.getvalue())
        rows = load_manifest(out)
        self.assertEqual([r["review"]["id"] for r in rows], [self.sB])
        self.assertEqual(rows[0]["review"]["reviewer"], "t")
        self.assertEqual(self._snapshot_files(), before)
        self.assertTrue(self.labels.is_file())


class RetrainFromRunsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.r = _Runs(self.root)
        self.s = self.r.sidecar(_event("ai_suppressed", 1000.0, 0.1), self.r.clip("not_fall", "s.mp4"), abn=990.0)
        self.l = self.r.sidecar(_event("level1", 2000.0, 0.9), self.r.clip("fall", "i.mp4"), abn=1990.0)
        self.base = self.root / "manifest.jsonl"
        self.base.write_text("", encoding="utf-8")
        self.field = self.root / "field.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *extra, side_effect=None):
        argv = ["--runs-dir", str(self.r.runs), "--manifest", str(self.base), "--kp-dir", str(self.root / "kp"),
                "--field-manifest", str(self.field), "--field-kp-dir", str(self.root / "fkp"), *extra]
        with mock.patch.object(RR.subprocess, "run", side_effect=side_effect) as run, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            code = RR.main(argv)
        return code, run, err.getvalue()

    def _fake_extract(self, fail_ids=()):
        """추출 단계를 흉내: fail_ids 밖의 행마다 키포인트 파일을 만들고, 모두 실패하면 종료 코드 1."""
        def run(cmd, check):
            if "training.datasets.extract_keypoints" in cmd:
                done = 0
                for row in load_manifest(self.field):
                    if row["review"]["id"] not in fail_ids:
                        p = RR.kp_path(self.root / "fkp", row["clip_id"])
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(b"")
                        done += 1
                if not done:
                    raise subprocess.CalledProcessError(1, cmd)
        return run

    def test_refuses_without_reviewed_labels(self):
        code, run, _ = self._run()
        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertFalse(self.field.exists())
        # unsure만 있어도 마찬가지
        RQ.append_label(RQ.labels_path_for(self.r.runs), RQ.make_label_record(self.s, "unsure"))
        code, run, _ = self._run()
        self.assertEqual(code, 1)
        run.assert_not_called()

    def test_uses_only_reviewed_clips_and_candidate_output(self):
        RQ.append_label(RQ.labels_path_for(self.r.runs), RQ.make_label_record(self.l, "not_fall"))
        out = self.root / "cand" / "m.pt"
        code, run, _ = self._run("--out", str(out), "--device", "cpu", side_effect=self._fake_extract())
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 2)
        rows = load_manifest(self.field)
        self.assertEqual([(r["review"]["id"], r["label"]) for r in rows], [(self.l, "adl")])
        extract, train = (c.args[0] for c in run.call_args_list)
        self.assertIn("training.datasets.extract_keypoints", extract)
        self.assertIn("--rotate-fallback", extract)
        self.assertIn("training.train_v2", train)
        self.assertEqual(train[train.index("--extra-manifest") + 1], str(self.field))
        self.assertEqual(train[train.index("--out") + 1], str(out))
        self.assertNotIn("training.video_to_data", extract + train)

    def test_refuses_deployed_weights_and_removed_v1_options(self):
        RQ.append_label(RQ.labels_path_for(self.r.runs), RQ.make_label_record(self.l, "not_fall"))
        for out in (WEIGHTS_DIR / "fall_cnn_gap_v2.pt", REPO_ROOT / "dataset" / "_models" / "fall_cnn_gap_v2.pt",
                    WEIGHTS_DIR / "anything.pt"):
            code, run, err = self._run("--out", str(out))
            self.assertEqual(code, 2, out)
            run.assert_not_called()
        code, run, err = self._run("--input-dir", str(self.r.runs / "clips" / "not_fall"))
        self.assertEqual(code, 2)
        run.assert_not_called()
        self.assertIn("removed", err)

    def test_stops_when_reviewed_clips_have_no_keypoints(self):
        lp = RQ.labels_path_for(self.r.runs)
        RQ.append_label(lp, RQ.make_label_record(self.l, "not_fall"))
        # 추출이 모두 실패(종료 코드 1): traceback 없이 1, 학습 안 함
        code, run, err = self._run("--out", str(self.root / "a.pt"), side_effect=self._fake_extract({self.l}))
        self.assertEqual((code, run.call_count), (1, 1))
        self.assertIn(f"field_runs/{self.l}", err)
        # 일부만 실패: 기본은 멈춤, --allow-missing이면 남은 클립으로 학습
        RQ.append_label(lp, RQ.make_label_record(self.s, "not_fall"))
        code, run, err = self._run("--out", str(self.root / "b.pt"), side_effect=self._fake_extract({self.l}))
        self.assertEqual((code, run.call_count), (1, 1))
        self.assertIn("--allow-missing", err)
        code, run, _ = self._run("--out", str(self.root / "c.pt"), "--allow-missing",
                                 side_effect=self._fake_extract({self.l}))
        self.assertEqual((code, run.call_count), (0, 2))

    def test_refuses_unreadable_label_lines(self):
        lp = RQ.labels_path_for(self.r.runs)
        RQ.append_label(lp, RQ.make_label_record(self.l, "fall", fall_start_s=4.0, fall_end_s=5.0))
        with open(lp, "ab") as f:
            f.write(b'{"id": "' + self.l.encode() + b'", "label": "not_f')  # 정정이 쓰다 끊김
        code, run, err = self._run("--dry-run")
        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertIn("labels.jsonl:2", err)

    def test_kp_path_matches_extractor(self):
        try:
            from training.datasets.extract_keypoints import clip_out_path
        except ImportError:
            self.skipTest("numpy not installed")
        for cid in ("field_runs/20261021_101010_000_level1", "field_runs/clip_fall_a:b?.mp4", "x\\y/z"):
            self.assertEqual(RR.kp_path(self.root, cid), clip_out_path(self.root, cid))

    def test_dry_run_writes_nothing(self):
        RQ.append_label(RQ.labels_path_for(self.r.runs), RQ.make_label_record(self.l, "not_fall"))
        code, run, _ = self._run("--dry-run")
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertFalse(self.field.exists())


class GuardsTest(unittest.TestCase):
    def test_runtime_weights_paths(self):
        self.assertTrue(is_runtime_weights_path(WEIGHTS_DIR / "fall_cnn_gap_v2.pt"))
        self.assertTrue(is_runtime_weights_path(WEIGHTS_DIR / "fall_model_active_best.pth"))
        self.assertTrue(is_runtime_weights_path(REPO_ROOT / "dataset" / "_models" / "fall_cnn_gap_v2.pt"))
        self.assertFalse(is_runtime_weights_path(REPO_ROOT / "dataset" / "_models" / "candidates" / "x.pt"))
        self.assertFalse(is_runtime_weights_path(REPO_ROOT / "dataset" / "_models" / "v1_active" / "fall_model_active_best.pth"))

    def test_train_v2_extra_manifest_refuses_deployed_out(self):
        with self.assertRaises(SystemExit) as cm:
            TV.main(["--extra-manifest", "x.jsonl", "--out", str(TV.DEFAULT_OUT)])
        self.assertIn("fall_cnn_gap_v2.pt", str(cm.exception))

    def test_merge_extra_rows(self):
        from training.datasets.manifest import make_row
        base = [make_row("le2i", "a", "v.avi", [], "omnifall", "l")]
        extra = [make_row("field_runs", "x", "x.mp4", [], "review", "l")]
        self.assertEqual([r["clip_id"] for r in TV.merge_extra_rows(base, extra)], ["le2i/a", "field_runs/x"])
        with self.assertRaises(ValueError):
            TV.merge_extra_rows(base, [dict(base[0])])
        with self.assertRaises(ValueError):
            TV.merge_extra_rows(base, [dict(extra[0], label="fall")])  # fall인데 구간 없음

    def test_train_active_does_not_save_into_runtime_weights(self):
        src = (REPO_ROOT / "training" / "train_active.py").read_text(encoding="utf-8")
        self.assertIn('MODEL_SAVE_PATH = str(REPO_ROOT / "dataset" / "_models" / "v1_active")', src)
        self.assertFalse(is_runtime_weights_path(REPO_ROOT / "dataset" / "_models" / "v1_active" / "m.pth"))


if __name__ == "__main__":
    unittest.main()
