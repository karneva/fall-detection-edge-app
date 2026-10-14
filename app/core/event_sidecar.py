"""
낙상/억제 이벤트의 분석용 sidecar JSON 저장 (표준 라이브러리만 사용)

클립(mp4)과 별도로, 판정 근거를 재현할 수 있는 데이터를 남깁니다.
- 이벤트 원본 (values.ai 포함)
- ABNORMAL 진입 기준 사건 구간의 원시 키포인트 (낙상 동작 포함)
- AI가 실제로 본 30프레임
- 엔진 파라미터, 분류기 상태(가중치 해시), 실측 처리 FPS, 클립 경로
- review.human_label: 사람이 검수할 자리 (pending -> fall / adl / unsure)

저장 실패가 알람 처리 흐름을 막지 않도록 예외를 던지지 않습니다.
"""
import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SIDECAR_SCHEMA_VERSION = 1


def sidecar_filename(event: Dict[str, Any]) -> str:
    ts = float(event.get("ts", time.time()))
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    ms = int(round((ts - int(ts)) * 1000)) % 1000
    return f"{stamp}_{ms:03d}_{event.get('type', 'event')}.json"


def build_sidecar(event: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    doc = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event": event,
        "review": {"human_label": "pending", "fall_start_ts": None, "fall_end_ts": None, "notes": ""},
    }
    doc.update(extra)
    return doc


def write_sidecar(out_dir: str, event: Dict[str, Any], **extra: Any) -> Optional[str]:
    """sidecar를 저장하고 경로를 반환합니다. 실패하면 None."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, sidecar_filename(event))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(build_sidecar(event, **extra), f, ensure_ascii=False, default=str)
        os.replace(tmp, path)
        return path
    except Exception:
        logger.exception("[SIDECAR] failed to write event sidecar")
        return None
