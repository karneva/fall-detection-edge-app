"""
네트워크 IP 보고 유틸리티
에지 장치의 로컬 IP 주소를 주기적으로 감지하여 통합 서버에 보고합니다.
이를 통해 관리자나 앱에서 유동 IP 환경의 장치에 원활하게 접속할 수 있도록 돕습니다.
"""

import os
import time
import socket
import requests
import json
import logging
from pathlib import Path

# 로깅 설정
logger = logging.getLogger("ip_reporter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# 기기 상태 파일 경로 (등록된 토큰 및 그룹 정보를 읽어오기 위함)
state_file = Path("runs/device_state.json")

# 로컬 상태 정보 로드
device_state = {}
if state_file.exists():
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            device_state = json.load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load device state: {e}")
        device_state = {}

# 설정값 구성
SERVER_URL = os.getenv("SERVER_URL", "https://i14a105.p.ssafy.io").rstrip("/")
DEVICE_ID = device_state.get("device_id") or os.getenv("IP_REPORTER_DEVICE_ID", "")
TOKEN = device_state.get("access_token") or os.getenv("IP_REPORTER_TOKEN", "")
FAMILIES_ID = device_state.get("group_id")

# 그룹 ID 보정
if FAMILIES_ID is None:
    FAMILIES_ID = int(os.getenv("IP_REPORTER_FAMILIES_ID", "0"))

# IP 확인 주기 (초)
POLL_SEC = float(os.getenv("IP_POLL_SEC", "5"))


def get_local_ip():
    """현재 장치의 유효한 로컬 IP 주소를 추출합니다."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 외부 사이트 접속을 시도하여 실제 네트워크 인터페이스 주소를 확보
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def report_ip(ipAddress: str):
    """추출된 IP 주소를 서버의 스트리밍 IP 정보에 업데이트(PATCH) 합니다."""
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    payload = {"ipAddress": ipAddress}
    url = f"{SERVER_URL}/api/iot/device/{FAMILIES_ID}/streaming-ip"

    try:
        # 서버 데이터 부분 갱신을 위해 PATCH 메서드 사용
        r = requests.patch(url, json=payload, headers=headers, timeout=5)
        r.raise_for_status()
        logger.info(f"IP updated successfully: {ipAddress} (status={r.status_code})")
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to update IP on server: Status={e.response.status_code}, Msg={e.response.text}")
    except Exception as e:
        logger.error(f"Network error during IP reporting: {e}")


def main():
    """IP 변경 사항을 지속적으로 모니터링하고 보고하는 메인 루프입니다."""
    prev_ip = None
    logger.info("IP Reporter service started.")
    
    while True:
        try:
            current_ip = get_local_ip()
            # IP 주소가 변경된 경우에만 서버에 보고하여 네트워크 부하 최소화
            if current_ip != prev_ip:
                report_ip(current_ip)
                prev_ip = current_ip
        except Exception as e:
            logger.error(f"Unexpected error in IP reporter loop: {e}")
        
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
