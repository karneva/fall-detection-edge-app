"""
Edge App 메인 엔트리 포인트

FastAPI 애플리케이션 초기화, 컨트롤러 및 스트리머 시작/종료 관리,
그리고 헬스 체크 및 데이터 스트리밍을 위한 API 라우트를 정의합니다.
"""

import os
import sys
import time
import logging
import threading
from typing import Optional, Dict, Any

import torch
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO

from .config import (
    MODEL_PATH, DETERMINISTIC, 
    RUNS_DIR, DEVICE_ID, LOCATION_ID
)
from .core.controller import AppController
from .core.streamer import WebSocketStreamer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 결정론적(Deterministic) / 시드 설정 ----------
if DETERMINISTIC:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- 모델 팩토리 ----------
def create_yolo_model():
    """YOLO 모델을 로드하고 설정을 적용하여 반환합니다."""
    logger.info(f"Loading YOLO model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    return model

# ---------- 컨트롤러 & 스트리머 초기화 ----------
controller = AppController(model_factory=create_yolo_model)
streamer = WebSocketStreamer(controller=controller)
app = FastAPI()

# ---------- 생명주기(Lifecycle) 이벤트 ----------
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 컨트롤러와 스트리머를 구동합니다."""
    logger.info("[STARTUP] Starting Application Controller and Streamer")
    controller.start()
    streamer.start()

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 리소스를 정리하고 스레드를 멈춥니다."""
    logger.info("[SHUTDOWN] Stopping Application Controller and Streamer")
    streamer.stop()
    controller.stop()

# ---------- API 라우트 ----------

@app.get("/health")
def health():
    """장치 상태 확인 엔드포인트"""
    return {
        "status": "ok",
        **controller.get_status()
    }

@app.get("/level1/status")
def level1_status():
    """
    현재 낙상 감지 상태를 반환합니다.
    LiveMode의 상세 상태(클립 레코더 등)와 마지막 이벤트 정보를 포함합니다.
    """
    return controller.get_status()

@app.post("/mode/live")
def set_mode_live():
    """Live 모드 전환 안내 (자동 전환됨)"""
    return {"status": "auto_mode_active", "info": "모드는 자동으로 선택됩니다 (QR 미등록 / Live 등록)"}

@app.post("/mode/replay")
def set_mode_replay():
    """Replay 모드 안내"""
    return {"status": "use /replay/start to enter replay mode"}

@app.post("/record/start")
def record_start():
    """JSONL 레코딩 시작"""
    path = controller.start_recording()
    return {"recording": True, "path": path}

@app.post("/record/stop")
def record_stop():
    """JSONL 레코딩 종료"""
    path = controller.stop_recording()
    return {"recording": False, "path": path}

@app.post("/replay/start")
def replay_start(path: str = Query(...), fps: float = Query(15.0)):
    """
    지정된 파일로 리플레이 모드를 시작합니다.
    
    Args:
        path (str): 재생할 jsonl 파일의 절대 경로
        fps (float): 재생 속도 (초당 프레임 수)
    """
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "file not found", "path": path})
    
    success = controller.start_replay(path, fps)
    if not success:
         return JSONResponse(status_code=409, content={"error": "replay already running"})
         
    return {"replay": True, "path": path, "fps": fps}

@app.post("/replay/stop")
def replay_stop():
    """리플레이를 중지하고 이전 모드로 복귀합니다."""
    stopped = controller.stop_replay()
    return {"replay": False, "stopped": stopped}

@app.get("/pose")
def pose():
    """최신 관측(Observation) 데이터를 반환합니다."""
    obs = controller.get_latest_obs()
    if obs is None:
        return JSONResponse(status_code=503, content={"error": "no observation yet"})
    return obs

@app.get("/api/iot/device/falls/stream_overlay")
def stream():
    """스켈레톤 오버레이가 포함된 MJPEG 스트림을 제공합니다."""
    boundary = "frame"

    def gen():
        last_count = 0
        while True:
            # 새로운 프레임이 올 때까지 대기 (최대 1초)
            jpg, current_count = controller.wait_for_overlay_frame(last_count, timeout=1.0)
            
            if jpg is None:
                continue
            
            last_count = current_count

            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                + jpg + b"\r\n"
            )

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/iot/device/falls/stream")
def stream_raw():
    """오버레이가 없는 원본 MJPEG 스트림을 제공합니다."""
    boundary = "frame"
    def gen():
        last = 0
        while True:
            jpg, last = controller.wait_for_raw_frame(last, timeout=1.0)
            if jpg is None:
                continue
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                + jpg + b"\r\n"
            )
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
