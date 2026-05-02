import os
import logging
import cv2
import time
import numpy as np
import threading
import collections
import json
from datetime import datetime
from typing import Optional, Deque, Tuple, Any, Dict, List  
import subprocess
from pathlib import Path

from ..config import (
    FRAME_W, FRAME_H,
    CLIP_DIR, CLIP_FPS, CLIP_PRE_SEC, CLIP_POST_SEC,
    CLIP_COOLDOWN_S,
    CLIP_RESIZE_WIDTH, CLIP_RESIZE_HEIGHT,
    ABNORMAL_TIMEOUT_S, JPEG_QUALITY
)

logger = logging.getLogger(__name__)

os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(os.path.join(CLIP_DIR, "fall"), exist_ok=True)
os.makedirs(os.path.join(CLIP_DIR, "not_fall"), exist_ok=True)

class ClipRecorder:
    """
    이벤트 발생 시 과거(PRE) 프레임과 이후(POST) 프레임을 조합하여 영상을 저장하는 클래스입니다.
    
    특징:
    - 링버퍼(Ring Buffer)를 사용하여 항상 최신 일정 기간의 프레임을 유지합니다.
    - 메모리 절약을 위해 프레임을 JPEG로 인코딩하여 저장합니다.
    - FFmpeg를 통해 브라우저에서 재생 가능한 웹 친화적인 MP4 형식을 생성합니다.
    """

    def __init__(self):
        self.lock = threading.Lock()
        
        # [버퍼 크기 계산]
        # 사람 감지 유실 시 루프 속도가 매우 빨라질 수 있으므로(60fps+),
        # 안전하게 60fps 기준으로 일정 시간(약 52초) 분량의 버퍼를 확보합니다.
        required_duration = ABNORMAL_TIMEOUT_S + CLIP_PRE_SEC + CLIP_POST_SEC + 10.0
        safety_fps = 60.0 
        buf_size = int(safety_fps * required_duration)
        
        self.buffer: Deque[Tuple[float, bytes]] = collections.deque(maxlen=buf_size)
        self.last_push_ts = 0.0
        self.min_push_interval = 1.0 / 24.0 # 최대 24 FPS로 제한하여 버퍼 밀림 방지

        # 분석용 메타데이터 링버퍼 (ts, meta)
        self.meta_buffer: Deque[Tuple[float, Dict[str, Any]]] = collections.deque(maxlen=buf_size)

        self.recording = False
        self.writer = None
        self.post_remaining = 0
        self.clip_path: Optional[str] = None
        self.last_started_ts: float = 0.0
        
        # 중복 저장 방지용 변수
        self.last_saved_segment_ts: float = 0.0
        self.save_segment_cooldown: float = 3.0   # 3초 내 중복 저장 방지

    def push(self, ts: float, frame_bgr, meta: Optional[Dict[str, Any]] = None):
        """
        프레임과 메타데이터를 링버퍼에 추가합니다.
        
        Args:
            ts: 현재 프레임의 타임스탬프
            frame_bgr: BGR 포맷의 이미지 데이터
            meta: 해당 프레임의 분석 결과 메타데이터 (선택 사항)
        """
        with self.lock:
            if ts - self.last_push_ts < self.min_push_interval:
                return
            self.last_push_ts = ts

        if frame_bgr is None:
            return

        # 1. 속도 및 용량 최적화를 위한 리사이즈
        h, w = frame_bgr.shape[:2]
        if w != CLIP_RESIZE_WIDTH or h != CLIP_RESIZE_HEIGHT:
            frame_bgr = cv2.resize(frame_bgr, (CLIP_RESIZE_WIDTH, CLIP_RESIZE_HEIGHT))

        # 2. 메모리 점유율을 대폭 낮추기 위해 JPEG로 인코딩하여 저장
        ok, jpg_bytes = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        
        if ok:
            with self.lock:
                self.buffer.append((ts, jpg_bytes.tobytes()))
                if meta is not None:
                    # JSON 직렬화가 가능한 데이터만 선별하여 저장
                    self.meta_buffer.append((ts, meta))

    def _open_writer(self, path: str, fps: float, frame_shape: Tuple[int, int, int]):
        """
        동영상 저장을 위한 cv2.VideoWriter 객체를 생성합니다.
        여러 코덱을 순차적으로 시도하여 호환성을 확보합니다.
        """
        h, w = frame_shape[:2]
        
        if w <= 0 or h <= 0:
            logger.error(f"[CLIP] 유효하지 않은 해상도: {w}x{h}")
            raise ValueError(f"유효하지 않은 해상도: {w}x{h}")
        
        fps = float(fps)
        if fps <= 0 or fps > 240:
            logger.warning(f"[CLIP] FPS 값이 범위를 벗어남 ({fps}), [5, 240]으로 조정")
            fps = max(5.0, min(240.0, fps))
        
        # 호환성을 위한 코덱 시도 목록
        codecs = [
            ("mp4v", "mp4v"),
            ("MJPG", "mjpg"),
            ("XVID", "xvid"),
            ("DIVX", "divx"),
            ("MPEG", "mpeg"),
        ]
        
        for codec_name, codec_code in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_code)
                writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
                
                if writer.isOpened():
                    logger.info(f"[CLIP] VideoWriter가 '{codec_name}' 코덱으로 열림: {w}x{h} @ {fps}fps")
                    return writer
                else:
                    writer.release()
            except Exception as e:
                logger.warning(f"[CLIP] 코덱 '{codec_name}' 시도 중 오류: {e}")
                continue
        
        # 모드 코덱 실패 시 기본 설정 사용
        logger.error(f"[CLIP] 모든 코덱 시도가 실패함. 기본 엔진 사용.")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        return writer

    def status(self):
        """현재 레코더의 작동 상태와 버퍼 정보 등을 반환합니다."""
        with self.lock:
            return {
                "clip_recording": self.recording,
                "clip_post_frames_remaining": self.post_remaining,
                "clip_path": self.clip_path,
                "clip_last_started_ts": self.last_started_ts,
                "clip_last_saved_segment_ts": self.last_saved_segment_ts,
                "buffer_len": len(self.buffer),
                "meta_buffer_len": len(self.meta_buffer),
            }

    def _dump_jsonl(self, jsonl_path: str, records: List[Dict[str, Any]]) -> None:
        """분석 데이터(메타데이터)를 JSONL 파일로 저장합니다."""
        try:
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def save_segment(self, incident_ts: float, pre_sec: float, post_sec: float,
                 filename_prefix: str = "incident", subdir: Optional[str] = None) -> Optional[str]:
        """
        특정 사건 발생 시점(incident_ts)을 기준으로 전/후 구간을 동영상으로 저장합니다.
        
        Args:
            incident_ts: 사건 발생 시점 (abnormal_enter 등)
            pre_sec: 사건 이전 기록 시간(초)
            post_sec: 사건 이후 기록 시간(초)
            filename_prefix: 파일명 접두어
            
        Returns:
            저장된 영상의 경로 또는 None
        """
        with self.lock:
            if len(self.buffer) == 0:
                return None

            # 중복 저장 방지 (3초 쿨다운 적용)
            now = time.time()
            if (now - self.last_saved_segment_ts) < self.save_segment_cooldown:
                logger.warning(f"[CLIP] 중복 저장 시도 무시됨: {now:.2f}")
                return None
            
            self.last_saved_segment_ts = now

            t_start = incident_ts - float(pre_sec)
            t_end = incident_ts + float(post_sec)

            picked = []
            for ts, jpg_bytes in list(self.buffer):
                if t_start <= ts <= t_end:
                    picked.append((ts, jpg_bytes))

            if not picked:
                logger.warning(f"[CLIP] 해당 시간 범위에 데이터가 없음: [{t_start:.2f}, {t_end:.2f}]")
                return None

            # 실제 FPS 추정하여 영상 배속 방지
            span = max(1e-6, picked[-1][0] - picked[0][0])
            eff_fps = len(picked) / span
            eff_fps = max(1.0, min(float(CLIP_FPS), eff_fps))

            clip_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            base_dir = CLIP_DIR
            if subdir:
                base_dir = os.path.join(CLIP_DIR, subdir)
                os.makedirs(base_dir, exist_ok=True)
                
            mp4_path = os.path.join(base_dir, f"{filename_prefix}_{clip_id}.mp4")
            jsonl_path = os.path.join(base_dir, f"{filename_prefix}_{clip_id}.jsonl")

            # 첫 프레임 디코딩 (비디오 라이터 설정용)
            first_frame = cv2.imdecode(np.frombuffer(picked[0][1], np.uint8), cv2.IMREAD_COLOR)
            if first_frame is None:
                return None

            writer = self._open_writer(mp4_path, eff_fps, first_frame.shape)
            if not writer.isOpened():
                return None

            writer_w, writer_h = first_frame.shape[1], first_frame.shape[0]

        # 디코딩 및 쓰기 작업 (메인 루프 차단 방지를 위해 락 해제 후 수행)
        try:
            for _, jpg_bytes in picked:
                fr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if fr is None: continue
                if fr.shape[1] != writer_w or fr.shape[0] != writer_h:
                    fr = cv2.resize(fr, (writer_w, writer_h))
                writer.write(fr)
        finally:
            writer.release()

        # 웹 브라우저 재생 호환성을 위한 트랜스코딩
        mp4_path = self.transcode_mp4_for_web(mp4_path)
        
        # 메타데이터도 함께 저장
        with self.lock:
            meta_picked = []
            for mts, m in list(self.meta_buffer):
                if t_start <= mts <= t_end:
                    KEEP_KEYS = {"frame_index", "state", "bbox", "aspect", "vy", "drop", "is_still"}
                    rec = {k: m.get(k) for k in KEEP_KEYS if k in m}
                    rec["ts"] = float(mts)
                    meta_picked.append(rec)
                    
        if meta_picked:
            self._dump_jsonl(jsonl_path, meta_picked)

        return mp4_path

    def save_range(self, t_start: float, t_end: float, filename_prefix: str = "incident", subdir: Optional[str] = None) -> Optional[str]:
        """지정된 시간 범위 [t_start, t_end]의 영상을 저장합니다."""
        if t_end <= t_start: return None

        with self.lock:
            now = time.time()
            if (now - self.last_saved_segment_ts) < self.save_segment_cooldown: return None
            self.last_saved_segment_ts = now
            picked = [(ts, b) for ts, b in list(self.buffer) if t_start <= ts <= t_end]
            if not picked: return None

        # FPS 추정 및 파일 경로 설정
        span = max(1e-6, picked[-1][0] - picked[0][0])
        eff_fps = max(1.0, min(float(CLIP_FPS), len(picked) / span))
        clip_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        base_dir = CLIP_DIR
        if subdir:
            base_dir = os.path.join(CLIP_DIR, subdir)
            os.makedirs(base_dir, exist_ok=True)
            
        path = os.path.join(base_dir, f"{filename_prefix}_{clip_id}.mp4")

        # 프레임 쓰기
        first_frame = cv2.imdecode(np.frombuffer(picked[0][1], np.uint8), cv2.IMREAD_COLOR)
        if first_frame is None: return None
        writer = self._open_writer(path, eff_fps, first_frame.shape)
        try:
            for _, b in picked:
                fr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                if fr is not None: writer.write(fr)
        finally:
            writer.release()

        return self.transcode_mp4_for_web(path)

    def transcode_mp4_for_web(self, in_path: str) -> str:
        """FFmpeg를 호출하여 영상을 스트리밍에 최적화된 형식(H.264, Faststart)으로 변환합니다."""
        p = Path(in_path)
        out_path = str(p.with_suffix(".web.tmp.mp4"))
        cmd = ["ffmpeg", "-y", "-i", str(p), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", out_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.replace(out_path, str(p))
        except Exception:
            if os.path.exists(out_path): os.remove(out_path)
        return str(p)
