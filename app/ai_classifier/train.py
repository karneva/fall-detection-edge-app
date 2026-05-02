import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from .model import Fall1DCNN

# 설정
DATA_PATH = "app/ai_classifier/data"
MODEL_SAVE_PATH = "app/ai_classifier/weights"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

def load_data(phase):
    X = np.load(os.path.join(DATA_PATH, f"X_{phase}.npy"))
    y = np.load(os.path.join(DATA_PATH, f"y_{phase}.npy"))
    return torch.from_numpy(X).float(), torch.from_numpy(y).long()

def train():
    # 데이터 로드 (이미 영상 단위로 나누어져 있음)
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Fall1DCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    
    print(f"Starting training... Train size: {len(X_train)}, Val size: {len(X_val)}")
    
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
            
        train_acc = 100 * correct / (total if total > 0 else 1)
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / (val_total if val_total > 0 else 1)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "fall_model_best.pth"))
            print(f"New best accuracy! Model saved. (Acc: {val_acc:.2f}%)")

    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "fall_model_last.pth"))
    print("Training finished.")

if __name__ == "__main__":
    if not os.path.exists(os.path.join(DATA_PATH, "X_train.npy")):
        print("Data not found. Please run dataset_builder.py first.")
    else:
        train()
