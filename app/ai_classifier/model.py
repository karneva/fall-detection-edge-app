import torch
import torch.nn as nn
import torch.nn.functional as F

class Fall1DCNN(nn.Module):
    def __init__(self, input_channels=34, seq_len=30, num_classes=2):
        super(Fall1DCNN, self).__init__()
        
        # 1D Convolution Layer 1: 로컬 패턴 추출
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # 1D Convolution Layer 2: 복합 패턴 추출
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # 1D Convolution Layer 3
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        # 출력 크기 계산
        # seq_len -> conv1 (seq_len) -> conv2 (seq_len) -> pool (seq_len // 2) -> conv3 (seq_len // 2)
        # 30 -> 30 -> 30 -> 15 -> 15
        flatten_size = 128 * (seq_len // 2)
        
        self.fc1 = nn.Linear(flatten_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, 34, 30)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # 간단한 테스트
    model = Fall1DCNN()
    dummy_input = torch.randn(8, 34, 30) # Batch size 8
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # (8, 2)
