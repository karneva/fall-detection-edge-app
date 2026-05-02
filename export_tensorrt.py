"""
YOLOv8 TensorRT 엔진 내보내기 도구
PyTorch 모델(.pt)을 Jetson 하드웨어 가속에 최적화된 TensorRT 엔진(.engine)으로 변환합니다.
이 과정은 Jetson 내부 또는 동일한 아키텍처의 GPU PC에서 실행해야 최적의 성능을 낼 수 있습니다.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO
import torch

# 로깅 설정: 내보내기 진행 상황을 가독성 있게 출력합니다.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TensorRT-Exporter")


def setup_args():
    """명령줄 인자를 구성합니다."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 모델을 TensorRT 엔진으로 최적화 변환합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 기본 모델 경로 및 이미지 크기 설정
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="소스 PyTorch 모델 경로 (.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="모델 입력 이미지 크기 (정사각형)")
    
    # 최적화 관련 세부 파라미터
    parser.add_argument("--device", type=str, default="0", help="GPU 장치 인덱스 (기본: 0)")
    parser.add_argument("--half", action="store_true", default=True, help="FP16(반정밀도) 연산 활성화 (성능 향상)")
    parser.add_argument("--int8", action="store_true", help="INT8(저정밀도) 양자화 시도 (별도 캘리브레이션 데이터셋 필요)")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT 컴파일 시 할당할 최대 비디오 메모리(GB)")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset 버전 지정")
    parser.add_argument("--simplify", action="store_true", default=True, help="ONNX 그래프 구조 단순화 활성화")
    
    # 변환 후행 처리 및 검증
    parser.add_argument("--verify", action="store_true", help="변환 완료 후 엔진의 로드 및 추론 테스트를 수행합니다.")
    parser.add_argument("--test-image", type=str, help="검증 시 사용할 테스트 이미지 경로")
    
    # 하위 호환성용 플래그
    parser.add_argument("--fp32", action="store_true", help="FP16을 사용하지 않고 원래 정밀도(FP32)로 유지합니다.")
    
    return parser.parse_args()


def export_to_tensorrt(
    model_path: Union[str, Path],
    imgsz: int = 640,
    device: str = "0",
    half: bool = True,
    int8: bool = False,
    workspace: int = 4,
    opset: Optional[int] = None,
    simplify: bool = True
) -> Path:
    """
    구성된 설정에 따라 모델 변환을 수행하고 결과 엔진 경로를 반환합니다.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        raise FileNotFoundError(model_path)

    logger.info(f"🚀 TensorRT 변환 프로세스를 시작합니다: {model_path.name}")
    logger.info(f"⚙️ 설정 요약: 해상도={imgsz}, 정밀도={'INT8' if int8 else 'FP16' if half else 'FP32'}")

    # CUDA 환경 점검
    if device != 'cpu' and not torch.cuda.is_available():
        logger.warning("CUDA 가속을 사용할 수 없습니다. 일반 CPU에서 내보내기를 시도합니다.")
        device = 'cpu'

    start_time = time.time()
    try:
        model = YOLO(str(model_path))
        
        # Ultralytics 내보내기 엔진 호출
        engine_path_str = model.export(
            format="engine",
            imgsz=imgsz,
            device=device,
            half=half,
            int8=int8,
            workspace=workspace,
            opset=opset,
            simplify=simplify,
            verbose=False
        )
        
        engine_path = Path(engine_path_str)
        elapsed = time.time() - start_time
        
        # 최적화 결과 정보 요약 출력
        print(f"\n{'='*70}")
        logger.info(f"✅ 최적화 엔진 생성이 완료되었습니다! (소요 시간: {elapsed:.1f}초)")
        print(f"{'='*70}")
        print(f"  • 소스 모델   : {model_path}")
        print(f"  • 최적화 엔진 : {engine_path}")
        print(f"  • 엔진 용량   : {engine_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"  • 입력 크기   : {imgsz}x{imgsz}")
        print(f"  • 가속 정밀도 : {'INT8' if int8 else ('FP16' if half else 'FP32')}")
        print(f"{'='*70}")
        
        print("\n배포 가이드:")
        print(f"  1. 생성된 '{engine_path.name}' 파일을 Jetson 장비의 루트 디렉토리로 이동시키세요.")
        print(f"  2. 'app/config.py' 파일에서 MODEL_PATH 변수를 위 파일 이름으로 변경하세요.")
        print(f"  3. 'python -m app.main' 명령어로 애플리케이션을 구동하세요.\n")
        
        return engine_path

    except Exception as e:
        logger.error(f"❌ 변환 작업 중 오류가 발생했습니다: {e}")
        raise


def verify_engine(engine_path: Union[str, Path], test_image: Optional[str] = None):
    """생성된 TensorRT 엔진이 정상적으로 작동하는지 자가 진단합니다."""
    engine_path = Path(engine_path)
    logger.info(f"🔍 생성된 엔진 검증 중...")
    
    try:
        # 엔진 로드 테스트
        model = YOLO(str(engine_path), task='pose')
        logger.info("  - 엔진 파일 로드 성공")
        
        # 실제 추론 테스트 (선택 사항)
        if test_image and Path(test_image).exists():
            logger.info("  - 테스트 이미지 추론을 실행합니다.")
            results = model.predict(test_image, verbose=False)
            
            det_count = len(results[0].boxes) if results[0].boxes is not None else 0
            logger.info(f"  - 탐지 완료: {det_count}개의 객체가 발견되었습니다.")
            logger.info("  - 상태: 검증 통과(PASS)")
        else:
            logger.info("  - 상태: 기본 로드 검증 통과")
            
    except Exception as e:
        logger.error(f"❌ 엔진 검증 실패: {e}")
        raise


def main():
    args = setup_args()
    
    if args.fp32:
        args.half = False
    
    try:
        engine_path = export_to_tensorrt(
            model_path=args.model,
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            int8=args.int8,
            workspace=args.workspace,
            opset=args.opset,
            simplify=args.simplify
        )
        
        if args.verify or args.test_image:
            verify_engine(engine_path, args.test_image)
            
    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 강제 종료되었습니다.")
        return 1
    except Exception:
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
