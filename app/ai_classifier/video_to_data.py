import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
import glob
import argparse

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

def collect_from_folder(folder_path, label, model_path=MODEL_PATH):
    model = YOLO(model_path)
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


def save_samples(X, y, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"X_{prefix}.npy"), X)
    np.save(os.path.join(save_dir, f"y_{prefix}.npy"), y)
    print(f"Saved {len(X)} samples to {save_dir} with prefix '{prefix}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pose sequences from videos into NumPy arrays")
    parser.add_argument("--input-dir", default="runs/clips/not_fall", help="Folder containing source videos")
    parser.add_argument("--label", type=int, default=0, help="Class label: 0=ADL, 1=Fall")
    parser.add_argument("--prefix", default="not_fall", help="Output file prefix, e.g. not_fall or custom_fall")
    parser.add_argument("--save-dir", default="app/ai_classifier/data/active_learning", help="Directory to save .npy arrays")
    parser.add_argument("--model-path", default=MODEL_PATH, help="YOLO pose model path")
    args = parser.parse_args()

    if os.path.exists(args.input_dir):
        X_new, y_new = collect_from_folder(args.input_dir, label=args.label, model_path=args.model_path)

        if len(X_new) > 0:
            # (N, 30, 34) -> (N, 34, 30)
            X_new = np.transpose(X_new, (0, 2, 1))
            save_samples(X_new, y_new, args.save_dir, args.prefix)
        else:
            print("No valid samples found in video files.")
    else:
        print(f"Directory not found: {args.input_dir}")
