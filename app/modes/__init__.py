"""modes module"""
import sys
import logging

logger = logging.getLogger(__name__)

from .base_mode import BaseMode
from .live_mode import LiveMode

# QRMode는 Linux 환경에서만 임포트 (pyzbar 라이브러리 의존성)
if sys.platform != "win32":
    try:
        from .qr_mode import QRMode
        logger.info("[INIT] QRMode imported successfully (Linux)")
    except (ImportError, OSError) as e:
        logger.warning(f"[INIT] QRMode import failed on Linux: {e}")
        QRMode = None
else:
    logger.info("[INIT] QRMode disabled (Windows)")
    QRMode = None
__all__ = ["BaseMode", "LiveMode", "QRMode"]
