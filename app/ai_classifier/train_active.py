import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from .model import Fall1DCNN

# 설정
DATA_PATH = "app/ai_classifier/data"
ACTIVE_DATA_PATH = "app/ai_classifier/data/active_learning"
MODEL_SAVE_PATH = "app/ai_classifier/weights"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 0.0005 # 데이터가 추가되었으므로 세밀한 학습을 위해 LR을 조금 낮춤
EPOCHS = 30

def load_data(phase):
    X = np.load(os.path.join(DATA_PATH, f"X_{phase}.npy"))
    y = np.load(os.path.join(DATA_PATH, f"y_{phase}.npy"))
    return X, y

def train_active():
    # 1. 기초 데이터 로드
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")
    
    # 2. 로컬 수집 데이터(Active Learning) 로드 및 병합
    active_x_path = os.path.join(ACTIVE_DATA_PATH, "X_not_fall.npy")
    active_y_path = os.path.join(ACTIVE_DATA_PATH, "y_not_fall.npy")
    
    if os.path.exists(active_x_path):
        X_active = np.load(active_x_path)
        y_active = np.load(active_y_path)
        
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
    
    best_weights_path = os.path.join(MODEL_SAVE_PATH, "fall_model_best.pth")
    if os.path.exists(best_weights_path):
        print("Loading existing best weights for fine-tuning...")
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    
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
            print(f"Saved Active Best Model (Acc: {val_acc:.2f}%)")

    print("Active learning training finished.")

if __name__ == "__main__":
    train_active()
