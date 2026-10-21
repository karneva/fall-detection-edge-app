"""
영상(mp4) 폴더 -> 30프레임 키포인트 시퀀스 배열 (v1 액티브 러닝용 추가 데이터, 이전 경로)

  python -m training.video_to_data --input-dir <라벨이 확인된 영상 폴더> --label 0 --prefix custom_adl

폴더 안 모든 영상에 같은 라벨을 붙이므로, 사람이 확인한 영상만 모은 폴더에 씁니다.
runs/clips/not_fall은 AI가 억제한 사건(실제 낙상 포함 가능)이라 그대로 쓰면 안 됨: 현장 클립은
training/review_queue.py로 검수하고 training.retrain_from_runs(v2)로 학습합니다.
"""
import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import sys
from pathlib import Path

if __package__ in (None, ""):
    # python training/<script>.py 로 직접 실행한 경우에도 저장소 루트의 app/, training/을 찾도록
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai_classifier.features import normalize_frame_v1 as normalize_keypoints
from training.paths import ACTIVE_DATA_DIR, YOLO_POSE_MODEL

# 설정
MODEL_PATH = YOLO_POSE_MODEL
SEQ_LEN = 30


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

def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract pose sequences from videos into NumPy arrays")
    # 기본값 없음: 검수 전 runs/clips/not_fall 전체에 라벨 0이 붙는 일을 막음
    parser.add_argument("--input-dir", required=True, help="Folder of human-verified source videos (one label)")
    parser.add_argument("--label", type=int, required=True, choices=(0, 1), help="Class label: 0=ADL, 1=Fall")
    parser.add_argument("--prefix", default="not_fall", help="Output file prefix, e.g. not_fall or custom_fall")
    parser.add_argument("--save-dir", default=str(ACTIVE_DATA_DIR), help="Directory to save .npy arrays")
    parser.add_argument("--model-path", default=MODEL_PATH, help="YOLO pose model path")
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    main()
