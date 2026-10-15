import threading
import time
import json
import logging
import websocket
from ..config import WS_SERVER_URL, DEVICE_ID, ENABLE_BACKEND_STREAM

logger = logging.getLogger(__name__)

class WebSocketStreamer:
    """
    백엔드 서버로 영상을 송출하는 WebSocket 스트리머 클래스입니다.
    
    주요 기능:
    - 백엔드 WebSocket 서버 연결 및 재연결 관리
    - AppController로부터 Raw 프레임을 받아 서버로 전송
    - 장치 등록(REGISTER_DEVICE) 메시지 전송
    """
    def __init__(self, controller):
        self.controller = controller
        self.stop_event = threading.Event()
        self.thread = None
        self.ws = None
        self.is_connected = False

    def start(self):
        """스트리밍 스레드를 시작합니다."""
        if not ENABLE_BACKEND_STREAM:
            logger.info("[STREAMER] Backend streaming disabled. Skipping WebSocket streamer start.")
            return
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("[STREAMER] WebSocket Streamer started")

    def stop(self):
        """스트리밍 스레드를 중지하고 연결을 종료합니다."""
        self.stop_event.set()
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("[STREAMER] WebSocket Streamer stopped")

    def _run(self):
        """WebSocket 연결 관리 및 재시도 루프"""
        retry_delay = 1
        max_delay = 60

        while not self.stop_event.is_set():
            try:
                logger.info(f"[STREAMER] Connecting to {WS_SERVER_URL}...")
                self.ws = websocket.create_connection(WS_SERVER_URL, timeout=5)
                
                # 핸드쉐이크: 장치 등록 메시지 전송
                register_msg = {
                    "type": "REGISTER_DEVICE",
                    "deviceId": DEVICE_ID
                }
                self.ws.send(json.dumps(register_msg))
                logger.info(f"[STREAMER] Registered as device: {DEVICE_ID}")
                
                self.is_connected = True
                retry_delay = 1 # 연결 성공 시 재시도 대기 시간 초기화
                
                self._stream_loop()

            except Exception as e:
                self.is_connected = False
                if not self.stop_event.is_set():
                    logger.warning(f"[STREAMER] Connection error: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_delay)
            finally:
                if self.ws:
                    self.ws.close()
                    self.ws = None

    def _stream_loop(self):
        """프레임 전송 루프"""
        last_count = 0
        sent_count = 0
        while not self.stop_event.is_set() and self.is_connected:
            try:
                # Raw 프레임 대기 (릴레이용으로는 오버레이 없는 원본 선호)
                jpg, current_count = self.controller.wait_for_raw_frame(last_count, timeout=1.0)
                
                if jpg is None:
                    # 프레임이 없으면 대기 (로그는 생략 가능)
                    continue
                
                last_count = current_count
                
                # 바이너리 프레임 전송
                self.ws.send_binary(jpg)
                sent_count += 1
                
                # if sent_count % 30 == 0:
                #     logger.info(f"[STREAMER] Successfully sent {sent_count} frames to backend")
                
            except Exception as e:
                logger.error(f"[STREAMER] Error in stream loop: {e}")
                self.is_connected = False
                break
