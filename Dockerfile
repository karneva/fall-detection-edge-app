# 1. Jetson + CUDA + PyTorch 베이스
FROM dustynv/pytorch:2.7-r36.4.0-cu128-24.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. OpenCV 런타임에 필요한 시스템 라이브러리
RUN apt-get update -o Acquire::Check-Valid-Until=false \
 && apt-get install -y \
    python3-opencv \
    ffmpeg \
    v4l-utils \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxfixes3 \
 && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리
WORKDIR /app

# 4. 파이썬 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://pypi.org/simple

# 5. 소스 코드 복사
# 멀티 파일 구조라면 app/ 폴더를 통째로 복사
COPY . .

# 6. 컨테이너 실행 시 기본 명령
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
