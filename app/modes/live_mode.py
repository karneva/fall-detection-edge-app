"""
Live 모드
사람 감지, 낙상 감시, 이벤트 전송 및 영상 녹화 로직을 통합 관리합니다.
"""

import time
import logging
import cv2
import os
import dataclasses
from typing import Optional, Tuple, Dict, Any

from .base_mode import BaseMode
from ..engine import LivePipeline
from ..core import Level1Params, Level1Engine, PresenceParams, PresenceEngine, ClipRecorder
from ..api.server_client import ServerClient
from ..state.device_state import get_device_state
from ..config import JPEG_QUALITY, DEVICE_ID, LOCATION_ID, EVENTS_DIR, SAVE_EVENT_SIDECAR
from ..core.event_sidecar import write_sidecar

logger = logging.getLogger(__name__)

# 억제 클립을 AI 판정 시점까지 늘릴 때의 최대 추가 길이(초)
SUPPRESSED_CLIP_MAX_EXTRA_S = 20.0

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

        # 실측 처리 FPS (obs 간격의 EMA)
        self._last_obs_ts: Optional[float] = None
        self._fps_ema: Optional[float] = None

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

            # 첫 낙상 시점에 모델 로드가 지연되지 않도록 분류기를 미리 로드/워밍업
            self._warmup_classifier()
            
            self.is_running = True
            return True
        except Exception as e:
            logger.error(f"Live Mode setup failed: {e}")
            return False
    
    def _warmup_classifier(self):
        """AI 분류기를 미리 로드합니다. 실패해도 룰 기반 감지는 그대로 동작합니다."""
        if self.level1_engine.p.ai_mode == "off":
            logger.info("[AI] AI_MODE=off. classifier not loaded")
            return
        try:
            from ..ai_classifier.inference import get_classifier
            clf = get_classifier()
            ms = clf.warmup()
            logger.info(f"[AI] mode={self.level1_engine.p.ai_mode} status={clf.status()} warmup_ms={ms}")
        except Exception as e:
            logger.error(f"[AI] classifier warmup failed: {e!r} -> rule-based only")

    def _classifier_status(self) -> Dict[str, Any]:
        if self.level1_engine.p.ai_mode == "off":
            return {"mode": "off", "enabled": False}
        try:
            # 상태 조회(API 스레드)에서는 모델을 로드하지 않음
            from ..ai_classifier.inference import peek_classifier
            clf = peek_classifier()
            if clf is None:
                return {"mode": self.level1_engine.p.ai_mode, "enabled": None, "state": "not_loaded"}
            return {"mode": self.level1_engine.p.ai_mode, **clf.status()}
        except Exception as e:
            return {"mode": self.level1_engine.p.ai_mode, "enabled": False, "load_error": repr(e)}

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
            "ai_classifier": self._classifier_status(),
            "processing_fps": round(self._fps_ema, 2) if self._fps_ema else None,
            **st
        }

    def _update_fps(self, ts: float):
        if self._last_obs_ts is not None:
            dt = ts - self._last_obs_ts
            if 0.0 < dt < 2.0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema is None else (0.95 * self._fps_ema + 0.05 * inst)
        self._last_obs_ts = ts

    def _save_sidecar(self, ev: Dict[str, Any], clip_path: Optional[str], clip_range) -> Optional[str]:
        """판정 근거(사건 구간 키포인트, AI 입력, 파라미터 등)를 sidecar JSON으로 저장합니다."""
        if not SAVE_EVENT_SIDECAR:
            return None
        try:
            snapshot = self.level1_engine.get_incident_snapshot()
            params = dataclasses.asdict(self.level1_engine.p)
        except Exception:
            logger.exception("[SIDECAR] failed to collect incident snapshot")
            snapshot, params = None, None
        path = write_sidecar(
            EVENTS_DIR,
            ev,
            device_id=DEVICE_ID,
            location_id=LOCATION_ID,
            snapshot=snapshot,
            params=params,
            processing_fps=self._fps_ema,
            classifier=self._classifier_status(),
            clip={"path": clip_path, "t_start": clip_range[0] if clip_range else None,
                  "t_end": clip_range[1] if clip_range else None},
        )
        if path:
            logger.info(f"[SIDECAR] saved {path}")
        return path

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
        self._update_fps(ts_now)

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
            if self.local_incident_ts is None:
                self._save_sidecar(ev, None, None)
            else:
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
                self._save_sidecar(ev, clip_path, (t_start, t_end))

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
            if self.local_incident_ts is None:
                self._save_sidecar(ev, None, None)
            else:
                from ..config import CLIP_PRE_SEC, CLIP_POST_SEC
                abn_ts = float(self.local_incident_ts)
                t_start = abn_ts - float(CLIP_PRE_SEC)
                # AI가 실제로 본 구간(억제 시점 직전)까지 포함해야 검수할 수 있음
                # 단, 클립 인코딩이 처리 루프를 막으므로 길이에 상한을 둠
                t_end = min(
                    max(abn_ts + float(CLIP_POST_SEC), float(ev.get("ts", ts_now))),
                    abn_ts + float(CLIP_POST_SEC) + SUPPRESSED_CLIP_MAX_EXTRA_S,
                )

                # not_fall 폴더에 저장 (서버 업로드는 하지 않음, 검수 전까지 라벨 미확정)
                clip_path = self.clip_recorder.save_range(
                    t_start=t_start,
                    t_end=t_end,
                    filename_prefix="suppressed",
                    subdir="not_fall"
                )
                self._save_sidecar(ev, clip_path, (t_start, t_end))
            # 사건은 끝나지 않았음 (억제 후 감시 중): 이후 승격된 level1도 같은 사건 시점으로 클립을 저장하도록
            # local_incident_ts를 유지. 회복(abnormal_exit) 또는 level1 처리 시 초기화됨.
