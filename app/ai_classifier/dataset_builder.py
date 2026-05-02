import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
import glob
import random

# 설정
DATASET_ROOT = "dataset/URFD"
SAVE_PATH = "app/ai_classifier/data"
os.makedirs(SAVE_PATH, exist_ok=True)

MODEL_PATH = "yolov8s-pose.pt"
SEQ_LEN = 30
VAL_RATIO = 0.2

def normalize_keypoints(kps):
    """
    골반(Hip Center)을 원점으로 하는 상대 좌표계로 변환.
    kps: (17, 2) [x, y] 정규화 좌표
    """
    # 11: Left Hip, 12: Right Hip
    l_hip = kps[11]
    r_hip = kps[12]
    
    # 두 골반의 신뢰도가 0일 경우(미검출)를 대비해 평균값 계산
    if np.any(l_hip) and np.any(r_hip):
        origin = (l_hip + r_hip) / 2
    elif np.any(l_hip):
        origin = l_hip
    elif np.any(r_hip):
        origin = r_hip
    else:
        # 골반이 없으면 중심점(0.5, 0.5) 혹은 Nose(0) 등 차선책 사용
        origin = kps[0] if np.any(kps[0]) else np.array([0.5, 0.5])
    
    # 상대 좌표 변환
    norm_kps = kps - origin
    return norm_kps.flatten() # (34,)

def extract_keypoints(model, img_path):
    results = model(img_path, verbose=False)
    if not results or len(results[0].keypoints.data) == 0:
        return None
    
    # 첫 번째 감지된 사람의 키포인트 (xyn: 정규화 좌표)
    kps = results[0].keypoints.xyn[0].cpu().numpy() # (17, 2)
    return normalize_keypoints(kps)

def build_dataset():
    model = YOLO(MODEL_PATH)
    
    categories = {"Fall": 1, "ADL": 0}
    
    dataset = {"train": {"X": [], "y": []}, "val": {"X": [], "y": []}}
    
    for cat_name, label in categories.items():
        cat_path = os.path.join(DATASET_ROOT, cat_name, "unzipped")
        subdirs = sorted([d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))])
        
        # 영상(폴더) 단위로 Train/Val 분할 (데이터 누수 방지)
        random.seed(42)
        random.shuffle(subdirs)
        split_idx = int(len(subdirs) * (1 - VAL_RATIO))
        train_subdirs = subdirs[:split_idx]
        val_subdirs = subdirs[split_idx:]
        
        for phase, phase_subdirs in [("train", train_subdirs), ("val", val_subdirs)]:
            print(f"Processing {cat_name} - {phase}...")
            for subdir_name in tqdm(phase_subdirs):
                subdir = os.path.join(cat_path, subdir_name)
                img_files = sorted(glob.glob(os.path.join(subdir, "*.png")))
                if not img_files:
                    continue
                    
                sequence = []
                for img_path in img_files:
                    kps = extract_keypoints(model, img_path)
                    if kps is not None:
                        sequence.append(kps)
                    else:
                        if sequence:
                            sequence.append(sequence[-1])
                        else:
                            sequence.append(np.zeros(34))
                
                # 시퀀스 생성 (슬라이딩 윈도우)
                if len(sequence) < SEQ_LEN:
                    while len(sequence) < SEQ_LEN:
                        sequence.append(sequence[-1] if sequence else np.zeros(34))
                    dataset[phase]["X"].append(sequence)
                    dataset[phase]["y"].append(label)
                else:
                    # Train은 증강을 위해 간격을 좁게(5), Val은 중복 최소화를 위해 간격을 넓게(10)
                    step = 5 if phase == "train" else 10
                    for i in range(0, len(sequence) - SEQ_LEN + 1, step):
                        dataset[phase]["X"].append(sequence[i:i+SEQ_LEN])
                        dataset[phase]["y"].append(label)
                        
    for phase in ["train", "val"]:
        X = np.array(dataset[phase]["X"], dtype=np.float32)
        y = np.array(dataset[phase]["y"], dtype=np.int64)
        X = np.transpose(X, (0, 2, 1)) # (N, 34, 30)
        
        np.save(os.path.join(SAVE_PATH, f"X_{phase}.npy"), X)
        np.save(os.path.join(SAVE_PATH, f"y_{phase}.npy"), y)
        print(f"{phase} saved! X shape: {X.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    build_dataset()
