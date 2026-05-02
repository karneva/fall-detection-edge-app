import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
import glob

# 설정
MODEL_PATH = "yolov8s-pose.pt"
SEQ_LEN = 30

def normalize_keypoints(kps):
    """골반(Hip Center) 중심 정규화 (dataset_builder.py와 동일 로직)"""
    l_hip = kps[11]
    r_hip = kps[12]
    
    if np.any(l_hip) and np.any(r_hip):
        origin = (l_hip + r_hip) / 2
    elif np.any(l_hip):
        origin = l_hip
    elif np.any(r_hip):
        origin = r_hip
    else:
        origin = kps[0] if np.any(kps[0]) else np.array([0.5, 0.5])
    
    norm_kps = kps - origin
    return norm_kps.flatten()

def process_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    sequence = []
    
    # 영상 내 모든 프레임에서 키포인트 추출
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)
        if results and len(results[0].keypoints.data) > 0:
            kps = results[0].keypoints.xyn[0].cpu().numpy() # (17, 2)
            sequence.append(normalize_keypoints(kps))
        else:
            if sequence:
                sequence.append(sequence[-1]) # 미검출 시 유지
            else:
                sequence.append(np.zeros(34))
                
    cap.release()
    
    if len(sequence) < SEQ_LEN:
        return []
        
    # 슬라이딩 윈도우로 시퀀스 생성
    samples = []
    step = 5
    for i in range(0, len(sequence) - SEQ_LEN + 1, step):
        samples.append(sequence[i:i+SEQ_LEN])
        
    return samples

def collect_from_folder(folder_path, label):
    model = YOLO(MODEL_PATH)
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    
    X_list = []
    y_list = []
    
    print(f"Processing videos in {folder_path}...")
    for v_path in tqdm(video_files):
        samples = process_video(model, v_path)
        if samples:
            X_list.extend(samples)
            y_list.extend([label] * len(samples))
            
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

if __name__ == "__main__":
    # 예시: not_fall 폴더에서 ADL(0) 데이터 추출
    NOT_FALL_DIR = "runs/clips/not_fall"
    if os.path.exists(NOT_FALL_DIR):
        X_new, y_new = collect_from_folder(NOT_FALL_DIR, label=0)
        
        if len(X_new) > 0:
            # (N, 30, 34) -> (N, 34, 30)
            X_new = np.transpose(X_new, (0, 2, 1))
            
            save_dir = "app/ai_classifier/data/active_learning"
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "X_not_fall.npy"), X_new)
            np.save(os.path.join(save_dir, "y_not_fall.npy"), y_new)
            print(f"Extracted {len(X_new)} samples to {save_dir}")
        else:
            print("No valid samples found in video files.")
    else:
        print(f"Directory not found: {NOT_FALL_DIR}")
