"""
모듈 import 스모크 테스트

cv2/torch/ultralytics/fastapi 등 장치 전용 의존성은 스텁으로 바꾸고, 앱의 모든 런타임 모듈이
경로 변경(리팩토링) 후에도 import되는지 확인합니다. 부작용(runs/ 생성 등)이 저장소에 남지 않도록
임시 폴더를 작업 디렉터리로 하는 별도 프로세스에서 실행합니다.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RUNTIME_MODULES = (
    "app.main",
    "app.controller",
    "app.config",
    "app.api.server_client",
    "app.api.streamer",
    "app.core.level1",
    "app.core.kp_buffer",
    "app.core.obs_schema",
    "app.core.event_sidecar",
    "app.engine.live",
    "app.engine.smoothing",
    "app.media.camera",
    "app.media.clip_recorder",
    "app.media.jpeg",
    "app.media.replay",
    "app.modes.live_mode",
    "app.modes.qr_mode",
    "app.state.device_state",
    "app.ai_classifier.weights_resolver",
)

SCRIPT = textwrap.dedent(
    """
    import importlib, importlib.util, sys
    from unittest import mock
    stubs = ["cv2", "torch", "ultralytics", "fastapi", "fastapi.responses", "requests", "websocket",
             "pyzbar", "pyzbar.pyzbar", "scipy", "scipy.ndimage"]
    if importlib.util.find_spec("numpy") is None:
        stubs.append("numpy")
    for name in stubs:
        sys.modules[name] = mock.MagicMock()
    failed = []
    for m in sys.argv[1:]:
        try:
            importlib.import_module(m)
        except Exception as e:
            failed.append(f"{m}: {e!r}")
    print("\\n".join(failed))
    sys.exit(1 if failed else 0)
    """
)


class ImportSmokeTest(unittest.TestCase):
    def test_runtime_modules_import(self):
        with tempfile.TemporaryDirectory() as cwd:
            env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), RUNS_DIR=os.path.join(cwd, "runs"))
            proc = subprocess.run(
                [sys.executable, "-c", SCRIPT, *RUNTIME_MODULES],
                cwd=cwd, env=env, capture_output=True, text=True, timeout=120,
            )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
