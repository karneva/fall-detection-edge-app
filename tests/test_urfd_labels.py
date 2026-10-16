import tempfile
import unittest
from pathlib import Path

from training.datasets.urfd_labels import load_urfd_segments, read_states, seq_kind, states_to_segments


def _write(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class UrfdLabelsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_fall_runs_to_segments(self):
        # 프레임 1~3 서 있음, 4~6 넘어짐, 7~9 누움 (30fps)
        states = [(f, -1) for f in (1, 2, 3)] + [(f, 0) for f in (4, 5, 6)] + [(f, 1) for f in (7, 8, 9)]
        segs = states_to_segments(states, "fall")
        self.assertEqual(segs, [
            {"cls": "fall", "start_s": 0.1, "end_s": 0.2},
            {"cls": "lying", "start_s": 0.2, "end_s": 0.3},
        ])

    def test_adl_falling_state_is_lie_down(self):
        segs = states_to_segments([(1, -1), (2, 0), (3, 1)], "adl")
        self.assertEqual([s["cls"] for s in segs], ["lie_down", "lying"])
        self.assertNotIn("fall", [s["cls"] for s in segs])

    def test_gaps_and_order(self):
        # CSV에 빠진 프레임이 있어도 같은 상태면 한 구간, 정렬되지 않은 입력도 처리
        segs = states_to_segments([(10, 0), (6, 0), (20, 1), (8, 0)], "fall")
        self.assertEqual(segs[0], {"cls": "fall", "start_s": round(5 / 30, 6), "end_s": round(10 / 30, 6)})
        self.assertEqual(segs[1]["cls"], "lying")
        # 한 프레임짜리 구간도 start < end
        one = states_to_segments([(5, 0)], "fall")[0]
        self.assertLess(one["start_s"], one["end_s"])

    def test_repeated_state_runs(self):
        segs = states_to_segments([(1, 0), (2, -1), (3, 0)], "fall")
        self.assertEqual([s["cls"] for s in segs], ["fall", "fall"])

    def test_transition_after_lying_is_stand_up(self):
        # -1 -> 0 -> 1 -> 0: 누운 뒤의 전이 구간은 일어나는 동작
        states = [(1, -1), (2, 0), (3, 1), (4, 1), (5, 0), (6, -1)]
        self.assertEqual([s["cls"] for s in states_to_segments(states, "adl")], ["lie_down", "lying", "stand_up"])
        self.assertEqual([s["cls"] for s in states_to_segments(states, "fall")], ["fall", "lying", "stand_up"])

    def test_read_csv_and_load(self):
        falls, adls = self.dir / "falls.csv", self.dir / "adls.csv"
        _write(falls, ["fall-01,1,-1,3.1,2.9", "fall-01,2,0,3.3,2.9", "fall-01,3,1,3.1,3.0", "",
                       "fall-02,1,0,1,1"])
        _write(adls, ["adl-01,6,-1,6.3", "adl-01,8,1,3.6"])
        self.assertEqual(read_states(falls)["fall-01"], [(1, -1), (2, 0), (3, 1)])
        out = load_urfd_segments([falls, adls])
        self.assertEqual(sorted(out), ["adl-01", "fall-01", "fall-02"])
        self.assertEqual(out["fall-01"][0], {"cls": "fall", "start_s": round(1 / 30, 6), "end_s": round(2 / 30, 6)})
        self.assertEqual(out["adl-01"], [{"cls": "lying", "start_s": round(7 / 30, 6), "end_s": round(8 / 30, 6)}])

    def test_rejects_bad_rows(self):
        bad = self.dir / "bad.csv"
        _write(bad, ["fall-01,1,5"])
        with self.assertRaises(ValueError):
            read_states(bad)
        _write(bad, ["fall-01,x,0"])
        with self.assertRaises(ValueError):
            read_states(bad)
        with self.assertRaises(ValueError):
            seq_kind("walk-01")

    def test_rejects_duplicate_frame(self):
        # 같은 프레임이 두 번 있으면 구간이 겹치므로 거부
        bad = self.dir / "dup.csv"
        _write(bad, ["fall-01,1,-1", "fall-01,2,0", "fall-01,2,1", "fall-02,2,0"])
        with self.assertRaises(ValueError):
            read_states(bad)

    def test_duplicate_sequence_across_csvs(self):
        a, b = self.dir / "a.csv", self.dir / "b.csv"
        _write(a, ["fall-01,1,0"])
        _write(b, ["fall-01,1,0"])
        with self.assertRaises(ValueError):
            load_urfd_segments([a, b])


if __name__ == "__main__":
    unittest.main()
