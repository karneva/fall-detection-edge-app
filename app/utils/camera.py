import os
import sys
import cv2
import time
from typing import Optional, Tuple
import logging

from ..config import CAM_INDEX, FRAME_W, FRAME_H

logger = logging.getLogger(__name__)

class CameraManager:
    """
    카메라 장치(VideoCapture)의 생명주기와 설정을 관리하는 클래스입니다.
    OS별 최적의 백엔드 선택 및 해상도, 오토포커스 등의 하드웨어 설정을 담당합니다.
    """
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_w = FRAME_W
        self.current_h = FRAME_H
        self.current_fps = 15
    
    def open_camera(self) -> cv2.VideoCapture:
        """
        현재 운영체제에 적합한 백엔드를 사용하여 카메라를 엽니다.
        Windows는 CAP_DSHOW/CAP_MSMF를, Linux는 CAP_V4L2를 우선 사용합니다.
        """
        if self.cap is not None and self.cap.isOpened():
            return self.cap

        cam_device = os.getenv("CAM_DEVICE", "").strip()
        cam_index = int(os.getenv("CAM_INDEX", str(CAM_INDEX)))

        logger.info(f"Opening camera: index={cam_index}, device={cam_device}, platform={sys.platform}")

        c = None
        if sys.platform.startswith("win"):
            # Windows: DSHOW 우선 시도 후 실패 시 MSMF 시도
            c = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            if not c.isOpened():
                c.release()
                c = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
        else:
            # Linux/Jetson: V4L2 백엔드 사용
            if cam_device:
                c = cv2.VideoCapture(cam_device, cv2.CAP_V4L2)
            else:
                c = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

        if c and c.isOpened():
            self.cap = c
            # 기본 해상도 및 FPS 설정 적용
            self.configure(self.current_w, self.current_h, self.current_fps)
            
            # 실제 적용된 해상도 로깅 (하드웨어 제약으로 다를 수 있음)
            aw = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Camera opened successfully. Request=({self.current_w}x{self.current_h}), Actual=({aw}x{ah})")
        else:
            logger.error("Failed to open camera.")
            self.cap = None

        return self.cap

    def ensure_opened(self) -> bool:
        """카메라가 열려 있는지 확인하고, 닫혀 있다면 재연결을 시도합니다."""
        if self.cap is not None and self.cap.isOpened():
            return True
        
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        
        self.open_camera()
        return self.cap is not None and self.cap.isOpened()

    def configure(self, width: int, height: int, fps: int = 15):
        """
        카메라의 해상도, 버퍼 크기 및 프레임 레이트를 설정합니다.
        """
        if self.cap is None or not self.cap.isOpened():
            return

        self.current_w = width
        self.current_h = height
        self.current_fps = fps

        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 지연 방지를 위해 버퍼 크기를 최소화
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if aw != width or ah != height:
            logger.warning(f"Camera config mismatch: req({width}x{height}) != actual({aw}x{ah})")
    
    def try_autofocus(self):
        """
        카메라의 오토포커스(Auto-focus)를 시도합니다 (QR 스캔 모드 등에 유용).
        V4L2 드라이버가 지원하는 장치에서만 유효하게 작동합니다.
        """
        if self.cap is None or not self.cap.isOpened():
            return
        
        try:
            # 자동 초점 활성화 시도
            af_supported = self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            if af_supported:
                logger.info("[Camera] Autofocus enabled")
            
            # 수동 초점을 중간/근거리 범위로 조정 시도 (장치마다 값의 범위가 다를 수 있음)
            focus_set = self.cap.set(cv2.CAP_PROP_FOCUS, 180)
            if focus_set:
                logger.info("[Camera] Manual focus set to 180 (close range)")
                
        except Exception as e:
            logger.warning(f"[Camera] Focus control failed: {e}")
    
    def release(self):
        """카메라 점유를 해제하고 자원을 반납합니다."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
        self.cap = None
        logger.info("Camera released.")

    def get_cap(self) -> Optional[cv2.VideoCapture]:
        """현재 사용 중인 비디오 캡처 객체를 반환합니다."""
        return self.cap
