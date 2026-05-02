"""
서버 통신 클라이언트 모듈
백엔드 통합 서버 및 라즈베리파이(RPI) 하위 장치와의 모든 API 통신을 담당합니다.
페어링, 토큰 갱신, 이벤트 보고, 영상 클립 업로드 등의 기능을 수행합니다.
"""

import requests
import logging
from typing import Optional, Dict, Any, Tuple
import os

logger = logging.getLogger(__name__)


class ServerClient:
    """
    HTTP/REST 및 WebSocket을 통한 서버 통신 기능을 제공하는 클래스입니다.
    """
    
    def __init__(self, server_url: str = None, rpi_url: str = None, timeout: int = 10):
        """
        ServerClient를 초기화합니다.
        
        Args:
            server_url: 통합 백엔드 서버 주소
            rpi_url: 라즈베리파이(하위 장치) 제어 주소
            timeout: 요청 타임아웃 시간 (초)
        """
        from ..config import SERVER_URL, RPI_URL
        self.server_url = server_url or SERVER_URL
        self.rpi_url = rpi_url or RPI_URL
        self.timeout = timeout
    
    # ===== QR 페어링 및 장치 등록 =====

    def register_device(self, pairing_code: str, device_id: str) -> Optional[Dict[str, Any]]:
        """
        QR 코드를 통해 획득한 pairing_code를 기반으로 서버에 장치 등록을 요청합니다.
        성공 시 액세스 토큰 및 하위 장치 정보를 포함한 응답을 반환합니다.
        """
        try:
            url = f"{self.server_url}/api/iot/auth/pairing"
            # 마스터 장치(Jetson)와 함께 등록될 하위 장치 목록 구성
            child_devices = [
                {"serial_number": device_id, "device_type": "JETSON"},
                {"serial_number": "EEUM-R105", "device_type": "RPI"},
                {"serial_number": "EEUM-E105-1", "device_type": "ESP32"}
            ]

            payload = {
                "pairing_code": pairing_code,
                "master_serial": device_id,
                "devices": child_devices
            }
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            
            if result.get("status") == 200:
                logger.info(f"Device registration request successful: {device_id}")
            else:
                logger.warning(f"Registration failed with status: {result.get('status')}")
            
            return result
        except Exception as e:
            logger.error(f"Device registration failed: {e}")
            return None
    
    # ===== 하위 장치(RPI) 제어 =====

    def send_access_token_to_rpi(self, access_token: str) -> bool:
        """라즈베리파이에 인증 토큰을 전달하여 동기화합니다."""
        try:
            url = f"{self.rpi_url}/token"
            payload = {"token": access_token}
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            logger.info("Access token synchronized with RPI")
            return True
        except Exception as e:
            logger.error(f"Token sync with RPI failed: {e}")
            return False

    # ===== 토큰 관리 =====

    def get_access_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """리프레시 토큰을 사용하여 새로운 액세스 토큰을 발급받습니다."""
        try:
            url = f"{self.server_url}/api/iot/auth/refresh"
            payload = {
                "serial_number": os.getenv("DEVICE_ID", "EEUM-J105"),
                "refresh_Token": refresh_token
            }
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()

            if result.get("statusCode") == "200 OK":
                logger.info("Access token refreshed via server")
                return result.get("data", {})
            return None
        except Exception as e:
            logger.error(f"Token refresh request failed: {e}")
            return None

    # ===== 이벤트 및 사고 데이터 전송 =====
    
    def send_event_rpi(self, payload: Dict[str, Any]) -> bool:
        """낙상/입퇴실 이벤트를 라즈베리파이로 전송하여 알림 및 제어를 수행합니다."""
        try:
            url = f"{self.rpi_url}/event"
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            logger.info(f"Event notified to RPI: {payload.get('data', {}).get('event')}")
            return True
        except Exception as e:
            logger.error(f"RPI event notification failed: {e}")
            return False
        
    def send_event_server(self, payload: Dict[str, Any], access_token: str) -> Tuple[Optional[str], Optional[str]]:
        """
        사고 발생 이벤트를 통합 백엔드 서버로 전송합니다.
        서버로부터 사고 영상 업로드를 위한 Presigned URL과 저장 경로를 응답받습니다.
        """
        try:
            url = f"{self.server_url}/api/iot/device/falls/detection"
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            
            data = resp.json().get("data", {})
            presigned_url = data.get("presignedUrl")
            video_path = data.get("videoPath")
            
            logger.info(f"Fall event reported. videoPath: {video_path}")
            return presigned_url, video_path
        except Exception as e:
            logger.error(f"Fall event reporting failed: {e}")
            return None, None
    
    def upload_clip_via_presigned_put(self, presigned_url: str, clip_path: str, timeout: float = 120.0):
        """Pre-signed URL을 사용하여 가공된 사고 영상을 S3 등의 저장소로 직접 업로드합니다."""
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Clip file not found: {clip_path}")

        with open(clip_path, "rb") as f:
            resp = requests.put(
                presigned_url,
                data=f,
                headers={"Content-Type": "video/mp4"},
                timeout=timeout,
            )

        if resp.status_code not in (200, 201, 204):
            raise RuntimeError(f"Cloud storage upload failed: {resp.status_code} {resp.text}")

    def send_video_upload_success(self, video_path: str, access_token: str) -> bool:
        """영상 업로드가 성공적으로 완료되었음을 서버에 알려 DB 처리를 완료시킵니다."""
        try:
            url = f"{self.server_url}/api/iot/device/falls/upload-success"
            payload = {"videoPath": video_path}
            headers = {"Authorization": f"Bearer {access_token}"}
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            logger.info(f"Video finalization notified: {video_path}")
            return True
        except Exception as e:
            logger.error(f"Finalization notification failed: {e}")
            return False

    # ===== 데이터 수집 및 상태 확인 =====
    
    def ping(self) -> bool:
        """백엔드 서버와의 연결 상태를 점검합니다."""
        try:
            url = f"{self.server_url}/health"
            resp = requests.get(url, timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    # ===== WebSocket 스트리밍 제어 =====
    
    def connect_websocket(self, device_id: str) -> bool:
        """분석 영상을 실시간으로 송출하기 위한 WebSocket 연결을 수립합니다."""
        import websocket
        import json
        
        self.device_id = device_id
        try:
            ws_url = self.server_url.replace("http", "ws") + "/api/ws/stream"
            self.ws = websocket.WebSocket()
            self.ws.connect(ws_url, timeout=5)
            
            # 장치 활성화 상태 등록 메시지 전송
            msg = {"type": "REGISTER_DEVICE", "deviceId": device_id}
            self.ws.send(json.dumps(msg))
            logger.info(f"WebSocket streaming established for {device_id}")
            return True
        except Exception as e:
            logger.error(f"WebSocket establishment failed: {e}")
            self.ws = None
            return False

    def send_stream_frame(self, frame_bytes: bytes):
        """인코딩된 JPEG 프레임 데이터를 WebSocket을 통해 바이너리로 전송합니다."""
        # 연결이 끊겼을 경우 재연결 시도
        if not (hasattr(self, 'ws') and self.ws and self.ws.connected):
             if hasattr(self, 'device_id') and self.device_id:
                 if not self.connect_websocket(self.device_id): return

        if hasattr(self, 'ws') and self.ws and self.ws.connected:
            try:
                self.ws.send_binary(frame_bytes)
            except Exception as e:
                logger.error(f"Streaming frame send error: {e}")
                try: self.ws.close()
                except: pass
                self.ws = None
    
    def close_websocket(self):
        """실시간 영상 스트리밍 연결을 안전하게 종료합니다."""
        if hasattr(self, 'ws') and self.ws:
            try:
                self.ws.close()
            except: pass
            self.ws = None
            logger.info("WebSocket streaming closed")
