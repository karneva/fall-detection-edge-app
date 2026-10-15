"""
장치 상태(페어링/토큰) 저장소 테스트 (기존 루트 test_token_refresh.py 대체)
"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from app.state.device_state import DeviceState


class DeviceStateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "device_state.json"

    def tearDown(self):
        self._tmp.cleanup()

    def write_state(self, **overrides):
        state = {
            "device_id": "TEST_ID",
            "access_token": "OLD_ACCESS_TOKEN",
            "refresh_token": "OLD_REFRESH_TOKEN",
            "group_id": 1,
            "serial_number": "TEST-001",
            "registered_at": time.time() - 7200,
            "token_expiry": time.time() - 3600,
        }
        state.update(overrides)
        self.path.write_text(json.dumps(state), encoding="utf-8")

    def client(self, new_data=None):
        c = MagicMock()
        c.get_access_token.return_value = new_data if new_data is not None else {
            "access_token": "NEW_ACCESS_TOKEN", "refresh_token": "NEW_REFRESH_TOKEN",
        }
        c.send_access_token_to_rpi.return_value = True
        return c

    def test_expired_token_is_refreshed_and_synced(self):
        self.write_state()
        state = DeviceState(state_file=str(self.path))
        self.assertTrue(state.is_token_expired())
        c = self.client()

        self.assertEqual(state.ensure_valid_token(c), "NEW_ACCESS_TOKEN")
        c.get_access_token.assert_called_once_with("OLD_REFRESH_TOKEN")
        c.send_access_token_to_rpi.assert_called_once_with("NEW_ACCESS_TOKEN")
        self.assertFalse(state.is_token_expired())
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["access_token"], "NEW_ACCESS_TOKEN")
        self.assertEqual(saved["refresh_token"], "NEW_REFRESH_TOKEN")

    def test_valid_token_is_returned_without_refresh(self):
        self.write_state(token_expiry=time.time() + 3600)
        c = self.client()
        self.assertEqual(DeviceState(state_file=str(self.path)).ensure_valid_token(c), "OLD_ACCESS_TOKEN")
        c.get_access_token.assert_not_called()

    def test_refresh_failure_returns_none(self):
        self.write_state()
        c = self.client(new_data={})
        self.assertIsNone(DeviceState(state_file=str(self.path)).ensure_valid_token(c))
        c.send_access_token_to_rpi.assert_not_called()

    def test_unregistered_returns_none(self):
        state = DeviceState(state_file=str(self.path))
        self.assertFalse(state.is_registered())
        self.assertIsNone(state.ensure_valid_token(self.client()))

    def test_register_persists(self):
        state = DeviceState(state_file=str(self.path))
        self.assertTrue(state.register("D1", "A", "R", 7, "SN"))
        reloaded = DeviceState(state_file=str(self.path))
        self.assertTrue(reloaded.is_registered())
        self.assertEqual(reloaded.get_state()["group_id"], 7)


if __name__ == "__main__":
    unittest.main()
