import random
import unittest

from app.core.presence import PresenceEngine, PresenceParams
from app.eval.synth import make_obs, skeleton


def obs(ts, present):
    kps = skeleton(0.5, 0.55, 0.7, 0.0, random.Random(0)) if present else None
    return make_obs(ts, int(ts * 10), kps)


class PresenceEngineTest(unittest.TestCase):
    def test_enter_and_exit_events(self):
        eng = PresenceEngine(PresenceParams(enter_hits=5, exit_hits=10, min_quality=0.1, cool_down_s=0.5))
        events = []
        for i in range(20):
            e = eng.step(obs(i * 0.1, True))
            if e:
                events.append((round(i * 0.1, 1), e["data"]["event"]))
        for i in range(20, 40):
            e = eng.step(obs(i * 0.1, False))
            if e:
                events.append((round(i * 0.1, 1), e["data"]["event"]))
        self.assertEqual(events, [(0.5, "enter"), (2.9, "exit")])

    def test_old_import_path_still_works(self):
        from app.core import PresenceEngine as A
        from app.core.level1 import PresenceEngine as B
        self.assertIs(A, PresenceEngine)
        self.assertIs(B, PresenceEngine)


if __name__ == "__main__":
    unittest.main()
