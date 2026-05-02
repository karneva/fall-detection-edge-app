"""
인증 토큰 자동 갱신 검증 테스트 스크립트
토큰 만료 상황을 모의하여 리프레시 토큰을 이용한 자동 갱신 및 하위 장치 동기화 로직을 테스트합니다.
"""

import sys
import os
import time
import json
from unittest.mock import MagicMock

# 애플리케이션 모듈 임포트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from app.state.device_state import DeviceState

def test_token_refresh():
    """토큰 만료 감지 및 갱신 프로세스 전반을 테스트합니다."""
    
    # 1. 서버 클라이언트 모킹 (Mocking)
    # 토큰 갱신 API 호출 시 리턴될 정보를 미리 설정합니다.
    mock_client = MagicMock()
    mock_client.get_access_token.return_value = {
        "access_token": "NEW_ACCESS_TOKEN",
        "refresh_token": "NEW_REFRESH_TOKEN",
        "serial_number": "TEST-001",
        "group_id": 1
    }
    mock_client.send_access_token_to_rpi.return_value = True

    # 2. 만료된 토큰 정보가 담긴 가상의 상태 파일 생성
    state_file = "runs/test_device_state.json"
    with open(state_file, "w") as f:
        json.dump({
            "device_id": "TEST_ID",
            "access_token": "OLD_ACCESS_TOKEN",
            "refresh_token": "OLD_REFRESH_TOKEN",
            "group_id": 1,
            "serial_number": "TEST-001",
            "registered_at": time.time() - 7200,
            "token_expiry": time.time() - 3600 # 1시간 전에 만료됨
        }, f)

    # 3. 테스트용 장치 상태 인스턴스 생성
    state = DeviceState(state_file=state_file)
    print(f"[검증] 초기 상태 - 토큰 만료 여부: {state.is_token_expired()}")

    # 4. 토큰 유효성 보장(ensure_valid_token) 메서드 호출
    # 내부적으로 토큰 만료를 감지하고 mock_client를 통해 갱신을 시도해야 합니다.
    new_token = state.ensure_valid_token(mock_client)
    
    # 5. 결과 검증
    print(f"[검증] 갱신된 토큰: {new_token}")
    print(f"[검증] 갱신 후 상태 - 토큰 만료 여부: {state.is_token_expired()}")
    
    current_state = state.get_state()
    if current_state["access_token"] == "NEW_ACCESS_TOKEN":
        print("결과: 상태 파일 내 액세스 토큰 갱신 성공")
    else:
        print("결과: 토큰 불일치 (실패)")

    # 서버 호출 여부 및 라즈베리파이 동기화 호출 여부 확인
    try:
        mock_client.get_access_token.assert_called_once()
        mock_client.send_access_token_to_rpi.assert_called_once_with("NEW_ACCESS_TOKEN")
        print("결과: 서버 API 및 RPI 동기화 메서드 정상 호출 완료")
    except AssertionError as e:
        print(f"결과: 메서드 호출 검증 실패 ({e})")

    # 테스트 후 임시 파일 정리
    if os.path.exists(state_file):
        os.remove(state_file)

if __name__ == "__main__":
    test_token_refresh()
