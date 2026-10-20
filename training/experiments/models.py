"""
진행 단계 비교 실험(progression study)용 분류 모델 모음 (torch 필요)

- Fall1DCNN    : 운영(app/ai_classifier/model.py) 모델을 그대로 재사용 (단계 B/C/D, 입력 (N,34,30) 고정 길이)
- Fall1DCNNGAP : Fall1DCNN의 conv 몸통 + 전역 평균 풀링(GAP) 헤드 (단계 E1, 입력 길이 무관).
                 배포 모델이라 app/ai_classifier/model.py에 정의하고 여기서는 다시 내보냄 (저장된 E1_s*.pt 그대로 로드)
- FallTCN      : 1x1 stem -> 팽창(dilation 1/2/4/8) 잔차 블록 4개 -> 어텐션 풀링 -> 2-way 선형 (단계 E2)
                 92채널 입력, width=40에서 약 43k 파라미터

모든 모델은 (N, C, T) 입력에 (N, n_cls) logits를 반환합니다 (softmax는 호출 쪽에서).
운영 동작에는 영향이 없습니다 (app/는 import만 함).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.ai_classifier.model import Fall1DCNN, Fall1DCNNGAP

__all__ = ["Fall1DCNN", "Fall1DCNNGAP", "FallTCN", "count_params", "build_model"]


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """학습 가능한(기본) 파라미터 수. BatchNorm running 통계 같은 buffer는 세지 않습니다."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only))


class _TCNBlock(nn.Module):
    """팽창 conv(k=3) 2개 + BN + ReLU + dropout, 잔차 연결. 길이 보존(same padding, 비인과)."""

    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, dilation=dilation, padding=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3, dilation=dilation, padding=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.relu(self.bn1(self.conv1(x))))
        h = self.drop(self.bn2(self.conv2(h)))
        return F.relu(x + h)


class FallTCN(nn.Module):
    """
    stem(1x1 conv in_ch -> width, BN, ReLU) -> 잔차 블록 4개(dilation 1/2/4/8, 수용 영역 61프레임 = 15Hz에서 약 4초)
    -> 어텐션 풀링(시간축 softmax 가중 평균) -> Linear(width, n_cls).
    in_ch=92, width=40 -> 약 43k 파라미터. 창 길이(T)에 의존하지 않습니다.
    """

    DILATIONS = (1, 2, 4, 8)

    def __init__(self, in_ch: int, width: int = 40, n_cls: int = 2, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=1),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_TCNBlock(width, d, dropout) for d in self.DILATIONS])
        self.attn = nn.Conv1d(width, 1, kernel_size=1)  # 시점별 점수
        self.head = nn.Linear(width, n_cls)

    def attention_weights(self, x):
        """(N, T) 시간축 어텐션 가중치 (해석/디버깅용)"""
        h = self.blocks(self.stem(x))
        return torch.softmax(self.attn(h).squeeze(1), dim=-1)

    def forward(self, x):
        h = self.blocks(self.stem(x))                      # (N, W, T)
        w = torch.softmax(self.attn(h).squeeze(1), dim=-1)  # (N, T)
        pooled = torch.bmm(h, w.unsqueeze(-1)).squeeze(-1)  # (N, W)
        return self.head(pooled)


def build_model(name: str, in_ch: int, seq_len: int = 30, n_cls: int = 2) -> nn.Module:
    """이름으로 모델 생성: 'cnn'(Fall1DCNN, 고정 길이 seq_len), 'cnn_gap', 'tcn'"""
    key = name.lower()
    if key in ("cnn", "fall1dcnn"):
        return Fall1DCNN(input_channels=in_ch, seq_len=seq_len, num_classes=n_cls)
    if key in ("cnn_gap", "fall1dcnngap", "gap"):
        return Fall1DCNNGAP(in_ch, n_cls=n_cls)
    if key in ("tcn", "falltcn"):
        return FallTCN(in_ch, n_cls=n_cls)
    raise ValueError(f"알 수 없는 모델 이름: {name}")
