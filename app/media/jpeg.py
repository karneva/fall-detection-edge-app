from typing import Optional

import cv2


def encode_jpeg(img, jpeg_quality: int) -> Optional[bytes]:
    """OpenCV 이미지 배열을 JPEG 바이트로 인코딩합니다. 실패하면 None."""
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return None
    return jpg.tobytes()
