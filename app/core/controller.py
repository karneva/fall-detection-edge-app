import threading
import time
import logging
import cv2
import sys
from typing import Optional, Dict, Any, Tuple

from ..utils.camera import CameraManager
from ..api.server_client import ServerClient
from ..state.device_state import get_device_state
from ..config import DEVICE_ID, LOCATION_ID, RUNS_DIR, DETERMINISTIC, JPEG_QUALITY
from ..modes.base_mode import BaseMode
from ..utils import start_replay_thread

logger = logging.getLogger(__name__)

# 임시 테스트 등록 (Windows)
def _register_temp_for_windows():
    device_state = get_device_state()
    if sys.platform == "win32" and not device_state.is_registered():
        logger.info("[TEST] Windows environment - Creating temporary registration")
        device_state.register(
            device_id="TEST_DEVICE_001",
            access_token="test_access_token_xxx",
            refresh_token="test_refresh_token_xxx",
            group_id=1,
            serial_number="TEST-001"
        )

class AppController:
    """
    애플리케이션의 메인 컨트롤러 클래스입니다.
    카메라 제어, AI 모델 처리 루프 관리, 스트리밍용 프레임 버퍼링,
    재생(Replay) 및 녹화(Recording) 기능을 총괄합니다.

    - latest_jpeg_overlay: 스켈레톤/오버레이가 포함된 분석용 이미지
    - latest_jpeg_raw: 선이나 박스가 없는 원본 이미지
    """
    def __init__(self, model_factory):
        """
        컨트롤러를 초기화합니다.
        
        Args:
            model_factory: YOLO 모델을 생성하는 팩토리 함수
        """
        self.model_factory = model_factory

        self.camera_manager = CameraManager()
        self.server_client = ServerClient()
        self.device_state = get_device_state()

        self.stop_event = threading.Event()
        self.background_thread: Optional[threading.Thread] = None
        self.state_lock = threading.Lock()

        self.processing_mode: str = "initial"
        self.latest_obs: Optional[Dict[str, Any]] = None

        # 프레임 버퍼: 오버레이 버전과 원본 버전 분리
        self.latest_jpeg_overlay: Optional[bytes] = None
        self.latest_jpeg_raw: Optional[bytes] = None

        # 스트리밍 동기화를 위한 카운터와 Condition 객체
        self.frame_count_overlay = 0
        self.frame_count_raw = 0
        self.cond_overlay = threading.Condition()
        self.cond_raw = threading.Condition()

        self.current_mode: Optional[BaseMode] = None

        # 리플레이 설정
        self.replay_thread = None
        self.replay_stop_event = threading.Event()
        self.replay_running = False

        # 레코딩(JSONL) 설정
        self.record_lock = threading.Lock()
        self.record_fp = None
        self.record_path = None

        _register_temp_for_windows()

    def start(self):
        """백그라운드 처리 스레드와 카메라를 시작합니다."""
        if self.background_thread and self.background_thread.is_alive():
            return

        self.stop_event.clear()
        self.camera_manager.ensure_opened()

        logger.info("Starting background processing loop")
        self.background_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.background_thread.start()

    def stop(self):
        """카메라와 처리 루프를 중지하고 리소스를 정리합니다."""
        logger.info("Stopping background processing loop")
        self.stop_event.set()
        if self.background_thread:
            self.background_thread.join(timeout=5)

        self.stop_replay()

        if self.current_mode:
            self.current_mode.cleanup()
            self.current_mode = None

        self.camera_manager.release()

    def _initialize_mode_instance(self):
        """현재 상태(등록 여부 등)에 따라 적절한 실행 모드(QR 또는 Live)를 초기화합니다."""
        cap = self.camera_manager.ensure_opened() and self.camera_manager.get_cap()
        if not cap:
            logger.error("Camera setup failed")
            return None

        is_registered = self.device_state.is_registered()

        from ..modes import QRMode
        if QRMode is not None and not is_registered:
            logger.info("Device not registered. Switching to QRMode.")
            self.camera_manager.configure(width=640, height=480)
            self.camera_manager.try_autofocus()
            mode = QRMode(cap=cap, jpeg_quality=JPEG_QUALITY)
            self.set_processing_mode("qr")
            return mode

        logger.info("Switching to LiveMode")
        try:
            model = self.model_factory()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

        self.camera_manager.configure(width=1920, height=1080)
        from ..modes import LiveMode
        mode = LiveMode(model=model, cap=cap, jpeg_quality=JPEG_QUALITY)
        self.set_processing_mode("live")
        return mode

    def _processing_loop(self):
        """메인 프레임 처리 루프입니다. 현재 모드의 step()을 호출하고 결과를 버퍼링합니다."""
        while not self.stop_event.is_set():
            if self.replay_running:
                time.sleep(0.1)
                continue

            if self.current_mode is None:
                self.current_mode = self._initialize_mode_instance()
                if self.current_mode:
                    if not self.current_mode.setup():
                        logger.error("Mode setup failed")
                        self.current_mode = None
                        time.sleep(1.0)
                        continue

            if self.processing_mode == "qr" and self.device_state.is_registered():
                logger.info("Device registered in QRMode. Switching to LiveMode.")
                self.current_mode.cleanup()
                self.current_mode = None
                continue

            try:
                if self.current_mode:
                    # 현재 모드에서 프레임 분석 수행
                    obs, overlay_jpg, raw_frame, _ = self.current_mode.step()

                    # 원본 프레임을 JPEG로 인코딩 (스트리밍용)
                    raw_jpg = None
                    if raw_frame is not None:
                        ok, enc = cv2.imencode(
                            ".jpg",
                            raw_frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                        )
                        if ok:
                            raw_jpg = enc.tobytes()

                    with self.state_lock:
                        if obs is not None:
                            self.latest_obs = obs

                    # 갱신된 오버레이 이미지를 버퍼에 저장하고 대기 중인 스레드에 알림
                    if overlay_jpg is not None:
                        with self.state_lock:
                            self.latest_jpeg_overlay = overlay_jpg
                        with self.cond_overlay:
                            self.frame_count_overlay += 1
                            self.cond_overlay.notify_all()

                    # 갱신된 원본 이미지를 버퍼에 저장하고 대기 중인 스레드에 알림
                    if raw_jpg is not None:
                        with self.state_lock:
                            self.latest_jpeg_raw = raw_jpg
                        with self.cond_raw:
                            self.frame_count_raw += 1
                            self.cond_raw.notify_all()

                    if obs is not None:
                        self._write_record(obs)
                else:
                    time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.5)
                if self.current_mode:
                    self.current_mode.cleanup()
                    self.current_mode = None

            time.sleep(0.001)

    def set_processing_mode(self, mode: str):
        """현재 처리 모드 이름을 설정합니다."""
        with self.state_lock:
            self.processing_mode = mode

    def get_latest_obs(self):
        """가장 최근에 생성된 관측(Observation) 데이터를 반환합니다."""
        with self.state_lock:
            return self.latest_obs

    def wait_for_overlay_frame(self, last_count: int, timeout: float = 1.0) -> Tuple[Optional[bytes], int]:
        """새로운 오버레이 프레임이 생성될 때까지 대기합니다."""
        with self.cond_overlay:
            if self.frame_count_overlay <= last_count:
                if not self.cond_overlay.wait(timeout=timeout):
                    return None, last_count
        with self.state_lock:
            return self.latest_jpeg_overlay, self.frame_count_overlay

    def wait_for_raw_frame(self, last_count: int, timeout: float = 1.0) -> Tuple[Optional[bytes], int]:
        """새로운 원본 프레임이 생성될 때까지 대기합니다."""
        with self.cond_raw:
            if self.frame_count_raw <= last_count:
                if not self.cond_raw.wait(timeout=timeout):
                    return None, last_count
        with self.state_lock:
            return self.latest_jpeg_raw, self.frame_count_raw

    # --- 실시간 관측 데이터 기록 (JSONL) ---
    def start_recording(self):
        """관측 데이터 기록을 시작합니다."""
        import os
        os.makedirs(RUNS_DIR, exist_ok=True)
        fname = time.strftime("obs_%Y%m%d_%H%M%S.jsonl")
        path = os.path.join(RUNS_DIR, fname)

        with self.record_lock:
            if self.record_fp:
                self.record_fp.close()
            self.record_fp = open(path, "a", encoding="utf-8")
            self.record_path = path
        return path

    def stop_recording(self):
        """관측 데이터 기록을 중지합니다."""
        path = None
        with self.record_lock:
            if self.record_fp:
                self.record_fp.close()
                self.record_fp = None
            path = self.record_path
            self.record_path = None
        return path

    def _write_record(self, obs):
        """파일에 관측 데이터를 기록합니다."""
        with self.record_lock:
            if self.record_fp:
                import json
                self.record_fp.write(json.dumps(obs, ensure_ascii=False) + "\n")
                self.record_fp.flush()

    # --- 리플레이 제어 ---
    def start_replay(self, path: str, fps: float):
        """지정된 파일로부터 리플레이를 시작합니다."""
        if self.replay_running:
            return False

        self.stop_replay()
        self.replay_stop_event.clear()

        def on_frame(obs, jpg):
            """리플레이 스레드로부터 수신한 프레임을 버퍼에 갱신합니다."""
            with self.state_lock:
                self.latest_obs = obs
                self.latest_jpeg_overlay = jpg
            with self.cond_overlay:
                self.frame_count_overlay += 1
                self.cond_overlay.notify_all()

        from ..utils.replay import start_replay_thread
        self.replay_thread = start_replay_thread(
            path=path,
            fps=fps,
            jpeg_quality=JPEG_QUALITY,
            stop_event=self.replay_stop_event,
            on_frame=on_frame
        )
        self.replay_running = True
        self.set_processing_mode("replay")
        return True

    def stop_replay(self):
        """실행 중인 리플레이를 중지하고 이전 모드로 복귀합니다."""
        if self.replay_running:
            self.replay_stop_event.set()
            if self.replay_thread:
                self.replay_thread.join(timeout=2.0)
            self.replay_running = False

            # 원래 실행 중이던 모드로 상태 전환
            from ..modes.live_mode import LiveMode
            if self.current_mode:
                mode_name = "live" if isinstance(self.current_mode, LiveMode) else "qr"
                self.set_processing_mode(mode_name)
            else:
                self.set_processing_mode("initial")

            return True
        return False

    def get_status(self):
        """컨트롤러와 현재 모드, 장치의 상태 정보를 모아서 반환합니다."""
        mode_status = {}
        if self.current_mode:
            try:
                mode_status = self.current_mode.get_status()
            except:
                pass

        return {
            "device_id": DEVICE_ID,
            "location_id": LOCATION_ID,
            "processing_mode": self.processing_mode,
            "replay_running": self.replay_running,
            "is_registered": self.device_state.is_registered(),
            **mode_status
        }
