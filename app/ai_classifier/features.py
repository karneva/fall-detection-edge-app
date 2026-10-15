"""
1D CNN 입력 특징 (학습과 운영이 공유하는 단일 구현, numpy만 사용)

v1: 프레임마다 골반 중심을 원점으로 하는 상대좌표 (기존 학습 가중치와 비트 단위 동일)
    - 이전에는 dataset_builder.py, video_to_data.py, inference.py 세 곳에 같은 코드가 복제되어 있었음
    - tests/test_features_parity.py가 기존 구현과의 np.array_equal 동일성을 고정
v2: (예정) 타임스탬프 리샘플링, 스케일 정규화, 속도, 전역 궤적, 결측 마스크
"""
import numpy as np

FEATURE_VERSION_V1 = "v1"

NUM_KPS = 17
L_HIP, R_HIP, NOSE = 11, 12, 0


def normalize_frame_v1(kps: np.ndarray) -> np.ndarray:
    """
    골반(Hip Center)을 원점으로 하는 상대 좌표계로 변환합니다.
    kps: (17, 2) [x, y] 정규화 좌표. 반환: (34,)
    """
    l_hip = kps[L_HIP]
    r_hip = kps[R_HIP]

    # 두 골반의 신뢰도가 0일 경우(미검출)를 대비해 평균값 계산
    if np.any(l_hip) and np.any(r_hip):
        origin = (l_hip + r_hip) / 2
    elif np.any(l_hip):
        origin = l_hip
    elif np.any(r_hip):
        origin = r_hip
    else:
        # 골반이 없으면 Nose, 그것도 없으면 화면 중앙(0.5, 0.5)
        origin = kps[NOSE] if np.any(kps[NOSE]) else np.array([0.5, 0.5])

    return (kps - origin).flatten()


def normalize_sequence_v1(sequence) -> np.ndarray:
    """
    시퀀스의 각 프레임에 골반 중심 정규화를 적용합니다.
    sequence: (T, 34) 배열. 반환: (T, 34) float32
    """
    return np.array([normalize_frame_v1(frame.reshape(NUM_KPS, 2)) for frame in sequence], dtype=np.float32)
