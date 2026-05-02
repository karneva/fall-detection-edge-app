"""
기기 상태 관리 모듈
장치의 등록 정보, 인증 토큰, 그룹 설정 등을 로컬 파일 시스템에 안전하게 저장하고 관리합니다.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DeviceState:
    """
    기기의 등록 상태 및 인증 정보를 파일 기반으로 관리하는 클래스입니다.
    애플리케이션 재시작 시에도 페어링 상태를 유지하기 위해 JSON 파일을 영속성 저장소로 사용합니다.
    """
    
    def __init__(self, state_file: str = "runs/device_state.json"):
        """
        DeviceState 인터페이스를 초기화합니다.
        
        Args:
            state_file: 상태 정보가 저장될 JSON 파일 경로
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """로컬 파일로부터 기기 상태 정보를 읽어옵니다."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                return {}
        return {}
    
    def _save(self):
        """현재 메모리에 있는 기기 상태 정보를 파일로 저장합니다."""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
            logger.info(f"State saved: {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def is_registered(self) -> bool:
        """기기가 서버에 성공적으로 등록(페어링)되어 있는지 확인합니다."""
        return "device_id" in self._state and "access_token" in self._state
    
    def register(self, device_id: str, access_token: str, refresh_token: str, 
                 group_id: int, serial_number: str, token_expiry: Optional[float] = None) -> bool:
        """
        서버로부터 받은 등록 정보를 저장합니다.
        
        Args:
            device_id: 기기 식별자
            access_token: API 접근을 위한 JWT 토큰
            refresh_token: 토큰 갱신을 위한 리프레시 토큰
            group_id: 장치가 소속된 그룹 ID
            serial_number: 장치의 고유 시리얼 번호
            token_expiry: 토큰 만료 시점 (Unix TS)
            
        Returns:
            저장 성공 여부
        """
        try:
            self._state["device_id"] = device_id
            self._state["access_token"] = access_token
            self._state["refresh_token"] = refresh_token
            self._state["group_id"] = group_id
            self._state["serial_number"] = serial_number
            self._state["registered_at"] = time.time()
            
            # 토큰 만료 시점 설정 (별도 지정 없으면 1년 뒤로 설정 - 장기 운용 고려)
            if token_expiry:
                self._state["token_expiry"] = token_expiry
            else:
                self._state["token_expiry"] = time.time() + 3600 * 24 * 365
            
            self._save()
            logger.info(f"Device registered: {device_id} (group: {group_id})")
            return True
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False
    
    def unregister(self) -> bool:
        """등록된 정보를 삭제하고 초기 상태로 되돌립니다."""
        try:
            self._state.clear()
            self._save()
            logger.info("Device unregistered")
            return True
        except Exception as e:
            logger.error(f"Unregistration failed: {e}")
            return False
    
    def get_device_id(self) -> Optional[str]:
        """현재 장치의 ID를 반환합니다."""
        return self._state.get("device_id")
    
    def get_access_token(self) -> Optional[str]:
        """현재 저장된 액세스 토큰을 반환합니다."""
        return self._state.get("access_token")
    
    def get_refresh_token(self) -> Optional[str]:
        """현재 저장된 리프레시 토큰을 반환합니다."""
        return self._state.get("refresh_token")
    
    def get_group_id(self) -> Optional[int]:
        """현재 장치가 속한 그룹 ID를 반환합니다."""
        return self._state.get("group_id")

    def is_token_expired(self) -> bool:
        """액세스 토큰이 만료되었는지 확인합니다."""
        token_expiry = self._state.get("token_expiry")
        if not token_expiry:
            return False
        return time.time() > token_expiry
    
    def refresh_tokens(self, new_access_token: str, new_refresh_token: str, token_expiry: Optional[float] = None) -> bool:
        """발급받은 새로운 토큰 번들로 상태 정보를 업데이트합니다."""
        try:
            self._state["access_token"] = new_access_token
            self._state["refresh_token"] = new_refresh_token
            if token_expiry:
                self._state["token_expiry"] = token_expiry
            else:
                self._state["token_expiry"] = time.time() + 3600 # 기본 만료 시간 연장
            self._save()
            logger.info("Tokens refreshed")
            return True
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """현재 모든 상태 정보를 담은 딕셔너리 복사본을 반환합니다."""
        return dict(self._state)
    
    def update(self, **kwargs) -> bool:
        """가변 인자를 통해 기기 상태 정보를 동적으로 업데이트합니다."""
        try:
            self._state.update(kwargs)
            self._save()
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False

    def ensure_valid_token(self, server_client) -> Optional[str]:
        """
        토큰의 유효성을 검사하고, 만료되었을 경우 리프레시 토큰을 사용하여 자동 갱신합니다.
        새롭게 발급된 토큰은 라즈베리파이 등 하위 장치와도 동기화합니다.
        
        Args:
            server_client: 서버와 통신할 서버 클라이언트 인스턴스
            
        Returns:
            유효한 액세스 토큰 (갱신 실패 시 None)
        """
        if not self.is_registered():
            return None

        if self.is_token_expired():
            logger.info("Access token expired. Attempting refresh...")
            refresh_token = self.get_refresh_token()
            if not refresh_token:
                logger.error("No refresh token available")
                return None
            
            new_data = server_client.get_access_token(refresh_token)
            if new_data:
                new_access_token = new_data.get("access_token")
                new_refresh_token = new_data.get("refresh_token")
                
                if new_access_token and new_refresh_token:
                    # 1. 로컬 저장소 업데이트
                    self.refresh_tokens(new_access_token, new_refresh_token)
                    
                    # 2. 하위 장치(라즈베리파이 등)로 갱신된 토큰 동기화 전송
                    result = server_client.send_access_token_to_rpi(new_access_token)
                    if result:
                        logger.info("Access token synchronized to RPI")
                    
                    return new_access_token
            
            return None
        
        return self.get_access_token()

# 전역에서 접근 가능한 가상 싱글톤 인스턴스 관리
_device_state: Optional[DeviceState] = None

def get_device_state() -> DeviceState:
    """DeviceState 클래스의 전역 싱글톤 인스턴스를 반환합니다."""
    global _device_state
    if _device_state is None:
        _device_state = DeviceState()
    return _device_state
