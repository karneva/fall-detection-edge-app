import os

# ---------------------------------------------------------
# 카메라 및 영상 품질 설정
# ---------------------------------------------------------
# 카메라 인덱스 (기본: 0)
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
# 처리 프레임 해상도
FRAME_W = int(os.getenv("FRAME_W", "1920"))
FRAME_H = int(os.getenv("FRAME_H", "1080"))
# 스트리밍 및 썸네일 JPEG 품질 (1-100)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
# YOLO 모델 추론 신뢰도 임계치
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.50"))

# ---------------------------------------------------------
# AI 모델 및 하드웨어 가속 설정
# ---------------------------------------------------------
# 결정론적 추론 활성화 (디버깅용)
DETERMINISTIC = os.getenv("DETERMINISTIC", "1") == "1"
# FP16 절반 정밀도 연산 사용 여부
USE_HALF = False if DETERMINISTIC else (os.getenv("USE_HALF", "1") == "1")
# YOLOv8 Pose 모델 파일 경로
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8s-pose.pt")

# ---------------------------------------------------------
# 장치 식별 및 네트워크 설정
# ---------------------------------------------------------
# 장치 시리얼 번호 및 이름
DEVICE_ID = os.getenv("DEVICE_ID", "JETSON-EDGE-001")
DEVICE_NAME = os.getenv("DEVICE_NAME", "Jetson-Orin-Nano")
# 설치 장소 에칭
LOCATION_ID = os.getenv("LOCATION_ID", "default-location")
# 통합 백엔드 API 서버 URL
SERVER_URL = os.getenv("SERVER_URL", "https://your-backend.example.com")
# 실시간 영상 전송용 WebSocket 서버 URL
WS_SERVER_URL = os.getenv("WS_SERVER_URL", SERVER_URL.replace("http", "ws") + "/api/ws/stream")
# 하위 제어 장치(Raspberry Pi) 주소
RPI_URL = os.getenv("RPI_URL", "http://your-rpi-host:8080/eeum")

# ---------------------------------------------------------
# 로그 및 동영상 저장 저장소
# ---------------------------------------------------------
RUNS_DIR = os.getenv("RUNS_DIR", "runs")
CLIP_DIR = os.getenv("CLIP_DIR", f"{RUNS_DIR}/clips")

# ---------------------------------------------------------
# 실시간 사고 영상(Clip) 녹화 파라미터
# ---------------------------------------------------------
CLIP_FPS = int(os.getenv("CLIP_FPS", "24"))
# 모바일 보기를 고려한 클립 리사이징 해상도
CLIP_RESIZE_WIDTH = int(os.getenv("CLIP_RESIZE_WIDTH", "640"))
CLIP_RESIZE_HEIGHT = int(os.getenv("CLIP_RESIZE_HEIGHT", "480"))
# 사고 시점 기준 전후 녹화 시간 (초)
CLIP_PRE_SEC = int(os.getenv("CLIP_PRE_SEC", "6"))
CLIP_POST_SEC = int(os.getenv("CLIP_POST_SEC", "6"))
# 전송 후 재녹화 방지 쿨다운 시간
CLIP_COOLDOWN_S = float(os.getenv("CLIP_COOLDOWN_S", "30.0"))

# ---------------------------------------------------------
# 낙상 감지 센서 튜닝 파라미터
# ---------------------------------------------------------
# 낙상 확정 대기 시간 (의심 발생 후 확정까지 걸리는 초)
ABNORMAL_TIMEOUT_S = float(os.getenv("ABNORMAL_TIMEOUT_S", "10.0"))

# ---------------------------------------------------------
# EMA(Exponential Moving Average) 스무딩 설정
# ---------------------------------------------------------
KP_EMA_ENABLE = os.getenv("KP_EMA_ENABLE", "1") == "1"
KP_EMA_ALPHA = float(os.getenv("KP_EMA_ALPHA", "0.5"))
# 신뢰도에 따른 스무딩 적용 최소 기준점
KP_MIN_CONF_FOR_SMOOTH = float(os.getenv("KP_MIN_CONF", "0.30"))

# 관절(Keypoint) ID별 스무딩 가중치(Alpha) 설정
# 값이 클수록 더 부드러워지며(지연 증가), 작을수록 반응성이 빨라짐
KP_ALPHA_BY_ID = {
    11: 0.55, 12: 0.55,  # 골반 (안정성 중시)
    5: 0.50, 6: 0.50,    # 어깨
    7: 0.45, 8: 0.45,    # 팔꿈치
    13: 0.45, 14: 0.45,  # 무릎
    9: 0.35, 10: 0.35,   # 손목 (빠른 움직임 대응)
    15: 0.35, 16: 0.35,  # 발목
}
KP_ALPHA_DEFAULT = 0.45

# 신뢰도 구간별 알파 보정 계수
KP_CONF_HOLD = 0.20   # 값을 유지(Hold)할 최저 신뢰도
KP_CONF_MID = 0.50
KP_ALPHA_MUL_LOW = 1.2   # 신뢰도 낮을 때 이전값 가중치 강화
KP_ALPHA_MUL_MID = 1.0
KP_ALPHA_MUL_HIGH = 0.9  # 신뢰도 높을 때 반응 속도 소폭 향상

# 속도 이상치(Jump) 감지 설정
KP_JUMP_ENABLE = True
KP_JUMP_RATIO = 0.12      # 바운딩 박스 대각선 대비 이동 거리 비율
KP_JUMP_RATIO_EXT = 0.18  # 손/발 등 끝단 관절의 허용치
KP_JUMP_ALPHA_MUL = 1.4   # 이상 이동 감지 시 스무딩 급격히 강화
KP_JUMP_HOLD_IF_LOWCONF = True
