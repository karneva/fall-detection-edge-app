"""
v1 30프레임 분류기 액티브 러닝 재학습 (이전 경로, 런타임 기본 모델 아님)

  python -m training.train_active

app/ai_classifier/data의 v1 학습 배열과 active_learning/ 추가 배열(X_*.npy, y_*.npy)로 v1 모델을 미세조정합니다.
결과는 dataset/_models/v1_active/fall_model_active_best.pth (런타임 weights/ 밖). 현장 클립 재학습은
training.retrain_from_runs (검수 큐 -> v2)를 씁니다.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from pathlib import Path
from typing import List, Optional
import sys

if __package__ in (None, ""):
    # python training/<script>.py 로 직접 실행한 경우에도 저장소 루트의 app/, training/을 찾도록
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai_classifier.model import Fall1DCNN
from app.ai_classifier.model_meta import current_git_sha, make_meta, write_meta
from training.paths import ACTIVE_DATA_DIR, DATA_DIR, REPO_ROOT, WEIGHTS_DIR

# 설정
DATA_PATH = str(DATA_DIR)
ACTIVE_DATA_PATH = str(ACTIVE_DATA_DIR)
# 미세조정 시작점 (training/train.py가 저장한 v1 기본 모델)
BASE_WEIGHTS_PATH = str(WEIGHTS_DIR)
# v1 액티브 러닝 결과는 런타임 탐색 위치(weights/) 밖에 저장: weights/에 두면 배포 v2 가중치가 없는 장치에서
# 런타임이 이 모델을 조용히 로드함 (weights_resolver 탐색 순서 4). 써 보려면 AI_MODEL_PATH로 지정
MODEL_SAVE_PATH = str(REPO_ROOT / "dataset" / "_models" / "v1_active")

BATCH_SIZE = 32
LEARNING_RATE = 0.0005 # 데이터가 추가되었으므로 세밀한 학습을 위해 LR을 조금 낮춤
EPOCHS = 30

def load_data(phase):
    X = np.load(os.path.join(DATA_PATH, f"X_{phase}.npy"))
    y = np.load(os.path.join(DATA_PATH, f"y_{phase}.npy"))
    return X, y


def load_active_learning_arrays(active_dir: str, loaded_files: Optional[List[str]] = None):
    """
    active_learning 디렉터리 아래의 X_*.npy / y_*.npy 쌍을 모두 찾아 병합합니다.
    loaded_files가 주어지면 실제로 병합된 X 파일 경로를 추가합니다 (가중치 메타의 trained_on 기록용).
    예:
      X_not_fall.npy + y_not_fall.npy
      X_custom_adl.npy + y_custom_adl.npy
    """
    active_root = Path(active_dir)
    if not active_root.exists():
        return None, None

    x_parts = []
    y_parts = []

    for x_path in sorted(active_root.glob("X_*.npy")):
        suffix = x_path.name[2:]
        y_path = active_root / f"y_{suffix}"
        if not y_path.exists():
            print(f"Skipping {x_path.name}: matching {y_path.name} not found")
            continue

        x_arr = np.load(x_path)
        y_arr = np.load(y_path)

        if len(x_arr) != len(y_arr):
            print(f"Skipping {x_path.name}: X/Y length mismatch ({len(x_arr)} != {len(y_arr)})")
            continue

        print(f"Loaded active set: {x_path.name} ({len(x_arr)} samples)")
        if loaded_files is not None:
            loaded_files.append(str(x_path))
        x_parts.append(x_arr)
        y_parts.append(y_arr)

    if not x_parts:
        return None, None

    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)

def train_active():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    # 1. 기초 데이터 로드
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")
    
    # 2. 로컬 수집 데이터(Active Learning) 로드 및 병합
    active_files: List[str] = []
    X_active, y_active = load_active_learning_arrays(ACTIVE_DATA_PATH, active_files)
    if X_active is not None and y_active is not None:
        print(f"Merging Active Learning Data: {len(X_active)} samples")

        # Train 데이터에 병합
        X_train = np.concatenate([X_train, X_active], axis=0)
        y_train = np.concatenate([y_train, y_active], axis=0)
    else:
        print("No active learning data found. Training with base dataset only.")
    
    # 데이터셋 생성
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 초기화 (기존 가중치가 있다면 로드하여 Fine-tuning 가능)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Fall1DCNN().to(device)
    
    best_weights_path = os.path.join(BASE_WEIGHTS_PATH, "fall_model_best.pth")
    if os.path.exists(best_weights_path):
        print("Loading existing best weights for fine-tuning...")
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    # 가중치 옆에 남길 메타 (런타임이 입력 형식 호환 여부를 확인하는 데 사용)
    trained_on = [DATA_PATH]
    if X_active is not None:
        trained_on += active_files
    meta = make_meta(trained_on=trained_on, git_sha=current_git_sha())
    
    print(f"Starting Active Learning Train... Final Train Size: {len(X_train)}")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # 검증
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "fall_model_active_best.pth"))
            write_meta(os.path.join(MODEL_SAVE_PATH, "fall_model_active_best.pth"), meta)
            print(f"Saved Active Best Model (Acc: {val_acc:.2f}%)")

    print(f"Active learning training finished. Saved under {MODEL_SAVE_PATH} (not loaded by the runtime unless AI_MODEL_PATH points to it)")

if __name__ == "__main__":
    train_active()
