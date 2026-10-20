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


class Fall1DCNNGAP(nn.Module):
    """
    배포 v2 모델 (E1): Fall1DCNN과 같은 conv 몸통(conv1 64 -> conv2 128 -> maxpool 2 -> conv3 128, 각 BN+ReLU)에
    flatten 대신 시간축 전역 평균 풀링(GAP)을 붙인 모델. 창 길이(T)에 의존하지 않습니다.
    입력 (N, 92, T) v2 특징(채널 표준화 후), 출력 (N, n_cls) logits.
    training/experiments/models.Fall1DCNNGAP는 이 클래스를 그대로 씀 (state_dict 키 동일, 92채널 100,738 파라미터).
    """

    def __init__(self, in_ch: int, n_cls: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_cls)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if x.size(-1) >= 2:  # 길이 1 창도 통과하도록 (풀링 생략)
            x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=-1)  # GAP: (N,128)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    # 간단한 테스트
    model = Fall1DCNN()
    dummy_input = torch.randn(8, 34, 30) # Batch size 8
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # (8, 2)
