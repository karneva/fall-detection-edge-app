"""
모델 성능 벤치마크 도구
PyTorch(.pt) 모델과 TensorRT(.engine) 모델 간의 추론 속도 및 FPS 성능을 비교합니다.
Jetson 에지 디바이스에서 최적화 효과를 측정하기 위해 사용됩니다.
"""

import os
import time
import torch
import numpy as np
import cv2
from ultralytics import YOLO

def benchmark_model(model_path, num_frames=100, warm_up=10):
    """특정 모델 파일에 대해 반복 추론을 수행하여 평균 지연 시간과 FPS를 측정합니다."""
    print(f"\n[성능 측정 시작] {model_path}")
    
    # 모델 로드 (PT 또는 Engine 자동 감지)
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None

    # 품질 측정을 위한 가상의 1080p 프레임 생성
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # 워밍업: 하드웨어 가속 및 메모리 할당을 안정화시킵니다.
    print(f"  - 워밍업 수행 중 ({warm_up} 프레임)...")
    for _ in range(warm_up):
        model.predict(dummy_frame, verbose=False, device=0)
    
    # 실제 구간 성능 측정
    print(f"  - 성능 측정 구간 실행 ({num_frames} 프레임)...")
    start_time = time.time()
    for _ in range(num_frames):
        # verbose=False로 콘솔 출력을 억제하여 순수 연산 속도만 측정
        model.predict(dummy_frame, verbose=False, device=0)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_frames) * 1000 # 밀리초(ms) 단위 변환
    fps = num_frames / total_time
    
    print(f"  - 결과: 평균 지연 시간 = {avg_latency:.2f}ms, 초당 프레임(FPS) = {fps:.2f}")
    
    # 메모리 정리: 젯슨 장비의 부족한 VRAM 자원을 회수합니다.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {"latency": avg_latency, "fps": fps}

def main():
    """기본 모델(PyTorch)과 최적화 모델(TensorRT)의 성능을 비교 분석합니다."""
    # 환경 변수 또는 기본 경로로부터 모델 파일 위치 확보
    pt_path = os.getenv("PT_MODEL_PATH", "yolov8n-pose.pt")
    engine_path = os.getenv("ENGINE_MODEL_PATH", "yolov8n-pose.engine")
    
    results = {}
    
    # 1. 원본 PyTorch 모델 측정
    if os.path.exists(pt_path):
        results["PyTorch (.pt)"] = benchmark_model(pt_path)
    else:
        print(f"주의: 원본 모델(.pt)을 찾을 수 없습니다: {pt_path}")

    # 2. 최적화 TensorRT 모델 측정
    if os.path.exists(engine_path):
        results["TensorRT (.engine)"] = benchmark_model(engine_path)
    else:
        print(f"주의: 최적화 엔진(.engine)을 찾을 수 없습니다: {engine_path}")
        print("참고: 먼저 'python export_tensorrt.py'를 실행하여 모델을 변환하세요.")

    # 3. 종합 성능 비교표 출력
    if len(results) >= 2:
        print("\n" + "="*50)
        print(" AI 모델 성능 비교 리포트 (PyTorch vs TensorRT) ")
        print("="*50)
        pt_res = results.get("PyTorch (.pt)")
        tr_res = results.get("TensorRT (.engine)")
        
        if pt_res and tr_res:
            speedup = pt_res['latency'] / tr_res['latency']
            fps_gain = tr_res['fps'] - pt_res['fps']
            
            print(f"PyTorch 기본 FPS    : {pt_res['fps']:.2f}")
            print(f"TensorRT 가속 FPS   : {tr_res['fps']:.2f}")
            print(f"속도 향상 배수(X)   : {speedup:.2f}배")
            print(f"프레임 이득(Gain)   : +{fps_gain:.2f} FPS")
        print("="*50)

if __name__ == "__main__":
    main()
