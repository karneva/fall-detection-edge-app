"""
낙상 감지 및 신뢰도 점수 검증 테스트 스크립트
다양한 시나리오(정상, 낙상, 부분 신체 노출 등)를 모의하여 엔진의 상태 전이와 신뢰도 계산 산출물을 검증합니다.
"""

import sys
import os
from unittest.mock import MagicMock

# 애플리케이션 모듈 임포트를 위해 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# 하드웨어 의존성이 있는 라이브러리를 Mocking하여 일반 환경에서도 테스트 가능하게 설정
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()

from app.core.level1 import Level1Engine, Level1Params

def run_test():
    """주요 낙상 감지 케이스별 시뮬레이션 테스트를 수행합니다."""
    # 테스트용 엔진 초기화 (판단 시간을 단축하여 빠른 검증 수행)
    engine = Level1Engine(Level1Params(
        sustain_s=2.0,      # 상태 유지 시간 단축
        drop_window_s=1.0, 
        vy_th=1.0           # 속도 임계치 설정
    ))

    print("[테스트 1] 표준 낙상 시나리오 검토")
    engine.reset_all()

    # 프레임 0: 서 있는 상태 정의
    obs0 = {
        "ts": 0.0, "frame_index": 0,
        "tracks": [{"has_person": True, "quality_score": 0.9, "bbox": [0.45, 0.1, 0.55, 0.4]}]
    }
    engine.step(obs0)
    print(f"  - 초기 상태: {engine.state}")

    # 프레임 1: 급격한 하강 발생 (0.2초 만에 중심 y가 0.25에서 0.5로 이동)
    # 계산: vy = (0.5 - 0.25) / 0.2 = 1.25 (> 임계치 1.0)
    obs1 = {
        "ts": 0.2, "frame_index": 1,
        "tracks": [{"has_person": True, "quality_score": 0.9, "bbox": [0.45, 0.35, 0.55, 0.65]}]
    }
    res1 = engine.step(obs1) 
    print(f"  - 급강하 발생 후 상태: {engine.state} (예상: ABNORMAL)")

    # 이후 2초간 바닥에 누워 있는 상태 유지 시뮬레이션
    final_res = None
    for i in range(25):
        ts = 0.2 + (i+1)*0.1
        obs = {
            "ts": ts,
            "frame_index": 2 + i,
            "tracks": [{"has_person": True, "quality_score": 0.5, "bbox": [0.45, 0.8, 0.55, 0.9]}]
        }
        res = engine.step(obs)
        if res and res.get("type") == "level1":
            final_res = res
            break
            
    if final_res:
        print("  - 결과: 낙상 확정(Level 1) 발생!")
        vals = final_res.get("values", {})
        print(f"  - 신뢰도 점수: {vals.get('confidence_score')}")
        print(f"  - 상세 분석: {vals.get('confidence_detail')}")
    else:
        print("  - 결과: 낙상 감지 실패 (오류)")

    print("\n[테스트 2] 근거리 부분 신체 노출에 의한 오검출 방지 테스트")
    engine.reset_all()
    # 시나리오: 카메라에 아주 가까이 다가와 상체만 크게 보이는 경우 (y2가 하단 경계에 닿음)
    obs0 = {
        "ts": 10.0, "frame_index": 100,
        "tracks": [{"has_person": True, "quality_score": 0.9, "bbox": [0.3, 0.0, 0.7, 0.98]}]
    }
    engine.step(obs0)
    
    # 앉거나 숙여서 자세 비율이 변하지만 하강 속도는 낮은 경우
    obs1 = {
        "ts": 10.2, "frame_index": 101,
        "tracks": [{"has_person": True, "quality_score": 0.9, "bbox": [0.2, 0.0, 0.8, 0.99], 
                    "keypoints": [{"id": 0, "x": 0.5, "y": 0.2, "conf": 0.9}]}] 
    }
    engine.step(obs1)
    
    if engine.state == "NORMAL":
        print("  - 결과: 부분 신체 모드에서 오탐지 방지 성공 (NORMAL 유지)")
    else:
        print(f"  - 결과: 예기치 못한 상태 전이 발생 ({engine.state})")

    print("\n[테스트 3] 부분 신체 노출 상태에서의 '완만한 낙상' 감지 테스트")
    engine.reset_all()
    # 시나리오: 상체만 보이지만 머리 위치가 완만하게(임계치 이하이나 보정치 이상) 떨어지는 경우
    obs0 = {
        "ts": 20.0, "frame_index": 200,
        "tracks": [{"has_person": True, "quality_score": 0.9, "bbox": [0.3, 0.0, 0.7, 0.98],
                    "keypoints": [{"id": 0, "x": 0.5, "y": 0.2, "conf": 0.9}]}] 
    }
    engine.step(obs0)
    
    obs1 = {
        "ts": 20.2, "frame_index": 201, # dt = 0.2
        "tracks": [{"has_person": True, "quality_score": 0.9, "bbox": [0.3, 0.1, 0.7, 0.99],
                    "keypoints": [{"id": 0, "x": 0.5, "y": 0.3, "conf": 0.9}]}] # 머리가 0.2에서 0.3으로 하강
    }
    engine.step(obs1)
    
    if engine.state == "ABNORMAL":
        print("  - 결과: 부분 신체 모드 보정 로직을 통한 이상행동 감지 성공 (ABNORMAL)")
    else:
        print(f"  - 결과: 감지 실패 ({engine.state})")

    print("\n[테스트 4] 저자세 지속(천천히 쓰러짐) 감지 테스트")
    engine.reset_all()
    # 시나리오: 속도는 매우 낮지만 극단적인 저자세(바닥에 누움)가 15프레임 이상 지속될 때
    for i in range(16):
        obs = {
            "ts": 30.1 + i*0.1, "frame_index": 301+i,
            "tracks": [{"has_person": True, "bbox": [0.2, 0.8, 0.8, 0.95]}]
        }
        res = engine.step(obs)
        if res and res.get("type") == "abnormal_enter":
            print(f"  - 결과: 지속 시간 누적을 통한 이상행동 진입 성공 (프레임 {i+1})")
            break
            
    if engine.state != "ABNORMAL":
        print(f"  - 결과: 지속성 감지 실패 ({engine.state})")

if __name__ == "__main__":
    run_test()
