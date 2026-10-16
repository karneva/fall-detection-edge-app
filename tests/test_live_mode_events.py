"""
LiveMode 이벤트 처리(sidecar 저장, 클립 구간, 사건 시점 유지) 테스트.

cv2/torch/ultralytics/requests 등 무거운 의존성은 스텁으로 대체하고,
LiveMode는 __init__을 거치지 않고 필요한 속성만 채워서 만듭니다.
"""
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.core.event_sidecar import build_sidecar, sidecar_filename, write_sidecar
from app.core.level1 import Level1Engine, Level1Params
from app.eval.synth import CountingVerifier, make_scenario

STUB_MODULES = ("cv2", "torch", "ultralytics", "requests", "websocket", "pyzbar", "pyzbar.pyzbar")


def _load_live_mode(clip_dir: str):
    """스텁 환경에서 live_mode 모듈을 import합니다. (sys.modules는 테스트 후 복원)"""
    import app.config

    names = list(STUB_MODULES)
    if importlib.util.find_spec("numpy") is not None:
        import numpy  # noqa: F401  patch.dict 복원 시 numpy가 지워지지 않도록 미리 import
    else:
        names.append("numpy")
    stubs = {name: mock.MagicMock() for name in names if name not in sys.modules}
    patcher = mock.patch.dict(sys.modules, stubs)
    patcher.start()
    try:
        # ClipRecorder 모듈(app/media/clip_recorder.py)이 import 시점에 만드는 폴더를 임시 경로로
        with mock.patch.object(app.config, "CLIP_DIR", clip_dir):
            mod = importlib.import_module("app.modes.live_mode")
    except Exception:
        patcher.stop()
        raise
    return mod, patcher


class SidecarModuleTest(unittest.TestCase):
    def test_filename_and_write(self):
        ev = {"type": "level1", "ts": 1760000000.25}
        self.assertTrue(sidecar_filename(ev).endswith("_250_level1.json"))
        with tempfile.TemporaryDirectory() as d:
            path = write_sidecar(d, ev, extra_field={"a": 1}, obj=object())
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(doc["event"], ev)
        self.assertEqual(doc["review"]["human_label"], "pending")
        self.assertEqual(doc["extra_field"], {"a": 1})

    def test_write_failure_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            blocker = Path(d) / "file"
            blocker.write_text("x")
            self.assertIsNone(write_sidecar(str(blocker / "sub"), {"type": "level1", "ts": 1.0}))

    def test_build_sidecar_schema(self):
        doc = build_sidecar({"type": "x"})
        self.assertEqual(doc["schema_version"], 1)


class LiveModeEventTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._clip_tmp = tempfile.TemporaryDirectory()
        cls.mod, cls.patcher = _load_live_mode(cls._clip_tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        cls._clip_tmp.cleanup()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.events_dir = self._tmp.name
        self.mode = object.__new__(self.mod.LiveMode)
        self.mode.level1_engine = Level1Engine(
            Level1Params(sustain_s=10.0, abnormal_timeout_s=10.0, ai_mode="veto",
                         ai_suppress_pfall_th=0.4, ai_escalate_s=60.0),
            verifier=CountingVerifier(0.05),
        )
        self.mode.clip_recorder = mock.MagicMock()
        self.mode.clip_recorder.save_range.return_value = "runs/clips/x.mp4"
        self.mode.server_client = mock.MagicMock()
        self.mode.server_client.send_event_server.return_value = None
        self.mode.device_state = mock.MagicMock()
        self.mode.device_state.ensure_valid_token.return_value = ""
        self.mode.last_level1_event = None
        self.mode.local_incident_ts = None
        self.mode._last_obs_ts = None
        self.mode._fps_ema = None
        self.patches = [
            mock.patch.object(self.mod, "EVENTS_DIR", self.events_dir),
            mock.patch.object(self.mod, "SAVE_EVENT_SIDECAR", True),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self._tmp.cleanup()

    def drive(self, obs):
        types = []
        for o in obs:
            self.mode._update_fps(float(o["ts"]))
            ev = self.mode.level1_engine.step(o)
            if ev is not None:
                self.mode._handle_fall_event(ev, o, float(o["ts"]))
                types.append(ev["type"])
        return types

    def sidecars(self):
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(self.events_dir).glob("*.json"))]

    def test_suppressed_then_escalated_saves_sidecars_and_clips(self):
        obs, truth = make_scenario("fast_fall", lie_s=90.0)
        types = self.drive(obs)
        self.assertEqual(types, ["abnormal_enter", "ai_suppressed", "level1"])

        calls = self.mode.clip_recorder.save_range.call_args_list
        self.assertEqual(len(calls), 2)
        supp_call, fall_call = calls[0].kwargs, calls[1].kwargs
        self.assertEqual(supp_call["subdir"], "not_fall")
        self.assertEqual(fall_call["subdir"], "fall")
        # 억제 클립은 AI 판정 시점까지 포함
        sup_doc, l1_doc = self.sidecars()
        self.assertGreaterEqual(supp_call["t_end"], sup_doc["event"]["ts"])
        # 승격된 알람도 같은 사건 시점 기준 클립
        self.assertAlmostEqual(fall_call["t_start"], supp_call["t_start"])

        self.assertEqual(sup_doc["event"]["type"], "ai_suppressed")
        self.assertEqual(l1_doc["event"]["reason"], "escalated_after_ai_suppression")
        for doc in (sup_doc, l1_doc):
            win = doc["snapshot"]["event_window"]
            self.assertLessEqual(win["t_start"], truth["fall_start_ts"])
            self.assertEqual(len(win["kps"]), len(win["ts"]))
            self.assertGreaterEqual(len(win["kps"]), win["n_frames"])  # 앞뒤 여유 프레임 포함
            self.assertEqual(len(doc["snapshot"]["ai_input"]["kps"]), 30)
            self.assertEqual(doc["params"]["ai_mode"], "veto")
            self.assertAlmostEqual(doc["processing_fps"], 15.0, delta=0.5)
            self.assertEqual(doc["clip"]["path"], "runs/clips/x.mp4")
        self.assertEqual(self.mode.local_incident_ts, None)

    def test_sidecar_disabled(self):
        with mock.patch.object(self.mod, "SAVE_EVENT_SIDECAR", False):
            self.drive(make_scenario("fast_fall", lie_s=20.0)[0])
        self.assertEqual(self.sidecars(), [])

    def test_snapshot_failure_still_saves_sidecar(self):
        self.mode.level1_engine = mock.MagicMock()
        self.mode.level1_engine.get_incident_snapshot.side_effect = RuntimeError("boom")
        self.mode.level1_engine.p = Level1Params()
        path = self.mode._save_sidecar({"type": "level1", "ts": 1.0}, None, None)
        self.assertIsNotNone(path)
        self.assertIsNone(self.sidecars()[0]["snapshot"])


if __name__ == "__main__":
    unittest.main()
