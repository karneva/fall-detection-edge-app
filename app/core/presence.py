"""재실(Presence) 감지: 사람이 화면에 들어오고 나가는 이벤트를 판단합니다."""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..config import DEVICE_ID, LOCATION_ID


@dataclass
class PresenceParams:
    """재실 감지 파라미터"""
    enter_hits: int = 5
    exit_hits: int = 10
    min_quality: float = 0.10
    cool_down_s: float = 0.5

class PresenceEngine:
    """재실(Presence) 감지 엔진: 사람의 출입을 판단합니다."""
    def __init__(self, p: PresenceParams):
        self.p = p
        self.state = "ABSENT"
        self.present_cnt = 0
        self.absent_cnt = 0
        self.last_emit_ts: float = 0.0

    def reset(self):
        """내부 카운터를 초기화합니다."""
        self.state = "ABSENT"
        self.present_cnt = 0
        self.absent_cnt = 0
        self.last_emit_ts = 0.0

    def _is_present(self, obs: Dict[str, Any]) -> bool:
        """현재 프레임에 사람이 유효하게 존재하는지 확인합니다."""
        tracks = obs.get("tracks") or []
        if not tracks:
            return False
        t0 = tracks[0]
        if not t0.get("has_person", False):
            return False
        q = float(t0.get("quality_score") or 0.0)
        if q < self.p.min_quality:
            return False
        return True

    def step(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """재실 상태 전이를 판단하고 이벤트를 반환합니다."""
        ts = float(obs.get("ts", 0.0))
        present = self._is_present(obs)

        if present:
            self.present_cnt += 1
            self.absent_cnt = 0
        else:
            self.absent_cnt += 1
            self.present_cnt = 0

        if self.p.cool_down_s > 0 and (ts - self.last_emit_ts) < self.p.cool_down_s:
            return None

        if self.state == "ABSENT" and self.present_cnt >= self.p.enter_hits:
            self.state = "PRESENT"
            self.last_emit_ts = ts
            return {
                "kind": "vision",
                "device_id": DEVICE_ID,
                "data": {"location_id": LOCATION_ID, "event": "enter"},
                "ts": ts,
            }

        if self.state == "PRESENT" and self.absent_cnt >= self.p.exit_hits:
            self.state = "ABSENT"
            self.last_emit_ts = ts
            return {
                "kind": "vision",
                "device_id": DEVICE_ID,
                "data": {"location_id": LOCATION_ID, "event": "exit"},
                "ts": ts,
            }

        return None
