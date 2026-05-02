import json
import time
import threading
from typing import Optional, Dict, Any, Callable

import cv2
from ..config import FRAME_W, FRAME_H

def encode_jpeg(img, jpeg_quality: int) -> Optional[bytes]:
    """OpenCV 이미지 배열을 JPEG 데이터로 인코딩합니다."""
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return None
    return jpg.tobytes()

def render_replay_frame(obs: Dict[str, Any], w: int = FRAME_W, h: int = FRAME_H):
    """
    관측 데이터(Observation)를 바탕으로 시각적인 리플레이 프레임을 생성합니다.
    검은 배경 위에 탐지된 인물의 바운딩 박스와 관절(키포인트)을 그립니다.
    """
    import numpy as np
    img = np.zeros((h, w, 3), dtype=np.uint8)

    tracks = obs.get("tracks", [])
    if not tracks:
        return img

    t0 = tracks[0]

    # 바운딩 박스 그리기
    if t0.get("bbox"):
        x1, y1, x2, y2 = t0["bbox"]
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

    # 관절(키포인트) 정보 표시
    for kp in t0.get("keypoints", []):
        if kp.get("conf", 0.0) < 0.2:
            continue
        cx = int(kp["x"] * w)
        cy = int(kp["y"] * h)
        cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)

    # 프레임 인덱스와 타임스탬프 텍스트 삽입
    ts = float(obs.get("ts", 0.0))
    txt = f"replay frame_index={obs.get('frame_index')} ts={ts:.3f}"
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    return img

def start_replay_thread(
    path: str,
    fps: float,
    jpeg_quality: int,
    stop_event: threading.Event,
    on_frame: Callable[[Dict[str, Any], bytes], None],
):
    """
    JSONL 형식의 로그 파일을 순차적으로 읽어 리플레이 영상을 생성하는 스레드를 시작합니다.
    분석 데이터 복원(obs)과 시각화된 JPEG를 콜백으로 전달합니다.
    """
    delay = 1.0 / max(1.0, float(fps))

    def worker():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if stop_event.is_set():
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obs = json.loads(line)
                        # 분석 결과 시각화
                        frame = render_replay_frame(obs, FRAME_W, FRAME_H)
                        jpg = encode_jpeg(frame, jpeg_quality)
                        if jpg is None:
                            continue
                        # 컨트롤러로 결과 전달
                        on_frame(obs, jpg)
                        time.sleep(delay)
                    except Exception:
                        continue
        finally:
            stop_event.clear()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t
