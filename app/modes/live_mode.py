"""
Live 모드
사람 감지, 낙상 감시, 이벤트 전송 및 영상 녹화 로직을 통합 관리합니다.
"""

import time
import logging
import cv2
import os
from typing import Optional, Tuple, Dict, Any

from .base_mode import BaseMode
from ..engine import LivePipeline
from ..core import Level1Params, Level1Engine, PresenceParams, PresenceEngine, ClipRecorder
from ..api.server_client import ServerClient
from ..state.device_state import get_device_state
from ..config import JPEG_QUALITY, DEVICE_ID, LOCATION_ID

logger = logging.getLogger(__name__)

class LiveMode(BaseMode):
    """
    실시간 분석 모드를 담당하는 클래스입니다.
    
    주요 역할:
    - YOLO 추론 파이프라인(LivePipeline) 실행
    - 재실 감지(PresenceEngine)를 통한 입퇴실 이벤트 관리
    - 낙상 감지(Level1Engine)를 통한 사고 판단 및 신뢰도 분석
    - 사고 전후 영상 기록(ClipRecorder) 및 서버 업로드
    - 백엔드 서버 및 라즈베리파이(RPI)로 이벤트 전송
    """
    
    def __init__(self, model, cap: cv2.VideoCapture, jpeg_quality: int = 80):
        super().__init__("LiveMode")
        self.model = model
        self.cap = cap
        self.jpeg_quality = jpeg_quality
        self.pipeline: Optional[LivePipeline] = None
        
        # 엔진 초기화
        from ..config import ABNORMAL_TIMEOUT_S
        self.presence_engine = PresenceEngine(PresenceParams(
            enter_hits=5, exit_hits=10, min_quality=0.10, cool_down_s=0.5
        ))
        self.level1_engine = Level1Engine(Level1Params(
            abnormal_timeout_s=ABNORMAL_TIMEOUT_S,
            ghost_timeout_s=10.0
        ))
        self.clip_recorder = ClipRecorder()
        
        # 클라이언트 및 상태 관리자
        self.server_client = ServerClient()
        self.device_state = get_device_state()
        
        # 내부 상태 변수
        self.last_level1_event: Optional[Dict[str, Any]] = None
        self.local_incident_ts: Optional[float] = None
    
    def setup(self) -> bool:
        """분석 파이프라인 및 엔진 상태를 초기화합니다."""
        try:
            logger.info("Initializing Live Mode Pipeline")
            self.pipeline = LivePipeline(
                model=self.model,
                cap=self.cap,
                jpeg_quality=self.jpeg_quality,
                source_id="cam0"
            )
            # 내부 엔진 상태 리셋
            self.level1_engine.reset_all()
            self.presence_engine.reset()
            self.last_level1_event = None
            self.local_incident_ts = None
            
            self.is_running = True
            return True
        except Exception as e:
            logger.error(f"Live Mode setup failed: {e}")
            return False
    
    def cleanup(self):
        """모드 종료 시 리소스를 정리합니다."""
        self.is_running = False
        self.pipeline = None
        logger.info("Live Mode cleaned up")

    def get_status(self) -> Dict[str, Any]:
        """현재 모드의 상태 정보를 반환합니다 (API 엔드포인트용)."""
        st = self.clip_recorder.status()
        return {
            "last_level1_event": self.last_level1_event,
            **st
        }

    def step(self) -> Tuple[Optional[Dict[str, Any]], Optional[bytes], Optional[Any], Optional[Any]]:
        """
        한 프레임에 대한 분석 루틴을 수행합니다.
        추론 -> 재실 판단 -> 낙상 판단 -> 영상 버퍼링 순으로 진행됩니다.
        """
        if self.pipeline is None:
            return None, None, None, None

        # 파이프라인 실행: 추론 결과(obs), 오버레이 이미지, 원본 프레임 획득
        obs, overlay_jpg, raw_frame = self.pipeline.step(overlay="smooth")

        if obs is None:
            return None, overlay_jpg, raw_frame, None

        ts_now = float(obs["ts"])

        # 1. 재실 이벤트 처리
        pe = self.presence_engine.step(obs)
        if pe is not None:
            self._handle_presence_event(pe, ts_now)

        # 2. 영상 레코더에 현재 프레임 푸시 (사고 발생 시 저장용)
        if raw_frame is not None:
            meta = {
                "frame_index": int(obs.get("frame_index", -1)),
                "state": self.level1_engine.state,
                "bbox": obs.get("tracks")[0].get("bbox") if obs.get("tracks") else None,
            }
            self.clip_recorder.push(ts_now, raw_frame, meta=meta)

        # 3. 낙상 및 이상행동 판단 처리
        ev = self.level1_engine.step(obs)
        if ev is not None:
            self._handle_fall_event(ev, obs, ts_now)

        return obs, overlay_jpg, raw_frame, None

    def _handle_presence_event(self, pe: Dict[str, Any], ts_now: float):
        """재실 감지(입/퇴실) 이벤트를 라즈베리파이로 전송합니다."""
        ptype = pe["data"]["event"]
        event_name = "enter" if ptype == "enter" else "exit"
        
        payload = {
            "kind": "vision",
            "device_id": DEVICE_ID,
            "data": {
                "location_id": LOCATION_ID,
                "event": event_name
            },
            "ts": ts_now,
        }
        self.server_client.send_event_rpi(payload)

    def _handle_fall_event(self, ev: Dict[str, Any], obs: Dict[str, Any], ts_now: float):
        """낙상 및 이상행동 발생 시 이벤트 전송 및 영상 업로드를 수행합니다."""
        et = ev.get("type")
        
        # 이상행동(의심) 시작 시 시점 기록
        if et == "abnormal_enter":
            self.local_incident_ts = float(ev.get("ts", ts_now))
            logger.info(f"[ABNORMAL START] ts={self.local_incident_ts:.2f}")

        # 정상 상태 복구 시 시점 초기화
        elif et == "abnormal_exit":
            self.local_incident_ts = None
            logger.info("[ABNORMAL END] Recovered")

        # 낙상 확정(LEVEL1) 이벤트 처리
        elif et == "level1":
            logger.warning(f"[FALL DETECTED] {ev}")
            self.last_level1_event = ev
            
            # 유효한 액세스 토큰 확보
            access_token = self.device_state.ensure_valid_token(self.server_client)
            if not access_token:
                logger.error("Token invalid. Cannot send fall event to Server.")
                access_token = ""

            t0 = (obs.get("tracks") or [{}])[0]
            payload = {
                "kind": "fall",
                "device_id": DEVICE_ID,
                "detected_at": float(ts_now),
                "data": {
                    "location_id": LOCATION_ID,
                    "event": "fall_detected",
                    "level": 1,
                    "has_person": bool(t0.get("has_person", False)),
                    "confidence": ev.get("values", {}).get("confidence_score", 0.0),
                },
            }
            
            # A. 라즈베리파이로 실시간 알림 전송
            try:
                self.server_client.send_event_rpi(payload)
            except Exception as e:
                logger.error(f"Failed to send to RPI: {e}")

            # B. 통합 서버로 이벤트 전송 및 업로드용 URL 획득
            presigned_url, video_path = None, None
            if access_token:
                try:
                    res = self.server_client.send_event_server(payload, access_token=access_token)
                    if res and isinstance(res, (tuple, list)) and len(res) >= 2:
                        presigned_url, video_path = res[0], res[1]
                except Exception as e:
                    logger.error(f"Failed to send to Server: {e}")

            # C. 사고 전후 영상 추출 및 서버 업로드
            if self.local_incident_ts is not None:
                from ..config import CLIP_PRE_SEC, CLIP_POST_SEC

                abn_ts = float(self.local_incident_ts)
                t_start = abn_ts - float(CLIP_PRE_SEC)
                t_end = abn_ts + float(CLIP_POST_SEC)

                clip_path = self.clip_recorder.save_range(
                    t_start=t_start,
                    t_end=t_end,
                    filename_prefix="incident",
                    subdir="fall"
                )

                if clip_path and presigned_url and video_path:
                    logger.info(f"Uploading clip: {clip_path} -> {video_path}")
                    try:
                        self.server_client.upload_clip_via_presigned_put(
                            presigned_url=presigned_url,
                            clip_path=clip_path,
                            timeout=120.0
                        )

                        # 업로드 완료 통보
                        if access_token:
                            self.server_client.send_video_upload_success(
                                video_path=video_path,
                                access_token=access_token
                            )
                        logger.info("Clip upload complete")
                    except Exception as e:
                        logger.error(f"Failed to upload clip: {e}")

            self.local_incident_ts = None

        # AI에 의해 반려된 오탐 의심 사례 저장 (Active Learning용)
        elif et == "ai_suppressed":
            logger.info(f"[AI SUPPRESSED RECORDING] {ev}")
            if self.local_incident_ts is not None:
                from ..config import CLIP_PRE_SEC, CLIP_POST_SEC
                abn_ts = float(self.local_incident_ts)
                t_start = abn_ts - float(CLIP_PRE_SEC)
                t_end = abn_ts + float(CLIP_POST_SEC)

                # not_fall 폴더에 저장 (서버 업로드는 하지 않음)
                self.clip_recorder.save_range(
                    t_start=t_start,
                    t_end=t_end,
                    filename_prefix="suppressed",
                    subdir="not_fall"
                )
            self.local_incident_ts = None
