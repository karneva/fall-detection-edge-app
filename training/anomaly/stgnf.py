"""
"nf": STG-NF (Hirschorn & Avidan, "Normalizing Flows for Human Pose Anomaly Detection", ICCV 2023,
arXiv 2211.10946) 직접 구현 (torch 필요). 공식 코드(github.com/orhir/STG-NF)는 옮기지 않고 논문 서술대로 작성했습니다.

핵심 아이디어
  정상 포즈 세그먼트 x (C=2, T=24, V=17)를 가역 변환 f로 표준 정규 잠재 z = f(x)에 보냅니다.
  변수 변환 공식: log p_X(x) = log p_Z(f(x)) + log|det(df/dx)|
  f = f_K o ... o f_1 이면 야코비안 행렬식은 곱이 되므로 log|det| = sum_k log|det(df_k/dh_{k-1})|.
  각 층은 야코비안이 대각/삼각/블록 대각이라 행렬식을 닫힌 꼴로 계산합니다.
  사전분포 p_Z = N(0, I) (원소 D = C*T*V = 816개):  log p_Z(z) = -1/2 * sum(z^2) - D/2 * log(2*pi)
  손실/점수 = 원소당 음의 로그 우도 NLL(x) = -(log p_Z(f(x)) + log|det|) / D  (nats/원소, 클수록 이상)

흐름 한 단계 (K = 8번 반복): ActNorm -> 가역 1x1 conv -> affine coupling
  ActNorm: y = (x + b) * exp(s)  (채널별 b, s; 데이터 의존 초기화: 첫 배치에서 출력이 채널별 평균 0, 표준편차 1)
    야코비안은 대각 exp(s_c)가 원소마다 반복되므로 log|det| = T*V * sum_c s_c
  가역 1x1 conv: 각 (t, v)에서 2채널 벡터에 같은 2x2 행렬 W를 곱함 (y = W x). 무작위 회전 행렬로 초기화
    야코비안은 W가 T*V번 반복된 블록 대각이라 log|det| = T*V * log|det W| (2x2라 닫힌 꼴 |w00 w11 - w01 w10|)
  affine coupling: 채널을 (x_a | x_b) = (x, y 좌표가 W로 섞인 두 채널 중 0번 | 1번)으로 나눔
    (log_s, t) = NN(x_a),  log_s = a * tanh(h / a) (a = 2, 폭주 방지용 부드러운 제한)
    y_a = x_a,  y_b = x_b * exp(log_s) + t
    y_a가 x_a 그대로이고 y_b가 x_b에 원소별로만 의존하므로 야코비안은 삼각 행렬, 대각 = (1, exp(log_s))
    -> log|det| = sum(log_s).  역변환: x_b = (y_b - t) * exp(-log_s) (NN은 y_a = x_a만 보므로 그대로 계산)
    NN = 작은 ST-GCN 블록: 1x1 conv(1 -> H) -> 공간 그래프 합성(정규화 인접 행렬 A 곱) -> ReLU
         -> 시간 conv(kernel 9, same padding) -> ReLU -> 1x1 conv(H -> 2), 마지막 층 0 초기화(시작 시 항등 변환)
    인접 행렬: COCO-17 골격 A + I의 대칭 정규화 (ST-GCN 'uniform' 1분할, segments.coco_adjacency)

입력 세그먼트 (segments.window_segments)
  v2 창 (92, 60) -> 프레임별 절대 좌표 rel_j + (hip_dx, hip_dy) (몸통 길이 단위) -> 24스텝 세그먼트, 간격 6 (7개)
  -> 무효 관절은 세그먼트 안 선형 보간, 없으면 골반 좌표 -> 세그먼트 평균 골반 위치를 뺌
  관절 유효 비율 < 0.5 세그먼트는 학습에서 빼고 점수도 내지 않음. 창 점수 = 남은 세그먼트 NLL의 최댓값 (없으면 NaN)

학습: Adam lr 5e-4, batch 256, 최대 30 epoch, 검증 정상 세그먼트 NLL 기준 early stopping (patience 5), seed 고정.
  학습 배치에만 좌표 가우스 잡음 sigma = 0.01 (몸통 길이 단위, 약 0.5~1px) 추가.

논문과 다른 점 (재현 수준: 구조와 목적 함수는 논문대로, 데이터 처리와 일부 세부는 이 프로젝트에 맞춤)
  - 관절: COCO-17 (논문은 AlphaPose 18관절 형식). 영상 15Hz 리샘플링 (논문은 원본 fps, ShanghaiTech 약 24~30fps)
  - 세그먼트 24스텝 = 1.6초 (논문 ShanghaiTech tau = 24프레임 약 1초, UBnormal 16). v2 창 4초 안에서 간격 6으로 자름
  - 정규화: 몸통 길이 단위 + 세그먼트 평균 골반 중심화 (논문은 세그먼트마다 평균 0, 분산 1). 분산 정규화를 하면
    낙상의 하강 폭 같은 크기 정보가 사라지므로 v2의 몸통 길이 스케일을 그대로 씀
  - 결측 관절: 세그먼트 안 선형 보간, 없으면 골반 좌표 (논문은 추적기 출력 그대로, 결측 처리 언급 없음).
    채운 값이 정확히 같은 궤적에 놓여 밀도가 무한대로 퇴화하지 않도록 학습 때만 작은 잡음을 더함 (논문에 없음,
    논문은 잡음을 강건성 실험에서만 씀)
  - 사전분포: 비지도 설정만 씀 (G는 낙상 라벨을 쓰지 않음). 논문은 N(3, I), 여기는 N(0, I):
    ActNorm 편향과 1x1 conv가 상수 이동을 흡수하므로 표현력은 같음 (지도 설정의 GMM 사전분포는 구현하지 않음)
  - 인접 행렬: 논문 보충 자료는 모든 관절이 같은 비중인 'unit adjacency'가 가장 좋았다고 함. 스펙에 따라 골격
    인접 행렬을 기본으로 쓰고 비교용으로 adjacency="full"을 둠. fold 0 seed 0 검증 정상 NLL(nats/원소)은
    skeleton 0.112, full 0.868로 골격 쪽이 훨씬 낮아 기본값을 유지 (선택은 검증 정상 손실로만 함).
    이 비교는 머리-몸통 연결 (3, 5), (4, 6)을 추가하기 전(그래프가 머리와 몸으로 나뉜 상태)에 잰 값이며 다시 재지 않음
  - 학습 예산: 5-fold 실행에서 거의 모든 run의 best epoch가 상한(30)이라 더 학습하면 나아질 여지가 있음
  - coupling 스케일: tanh 제한 + 마지막 층 0 초기화 (Glow 계열 구현의 sigmoid(h + 2)와 달리 시작이 정확히 항등).
    coupling NN 안에 BatchNorm을 두지 않음 (학습 중에도 표본별 밀도가 배치와 무관하도록)
  - 학습 길이: 논문 8 epoch 고정, 여기는 최대 30 epoch + 검증 NLL early stopping
  - 점수 집계: 논문은 프레임마다 사람별 log p의 최솟값, 여기는 한 사람의 창 안 세그먼트 NLL 최댓값
  - 크기: H = 8에서 약 5k 파라미터 (논문 약 1k)
"""
import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from training.anomaly import segments as SG
from training.anomaly._loop import EVAL_BATCH, train_unsupervised
from training.anomaly.base import Detector, as_windows

__all__ = ["ActNorm", "InvConv1x1", "STGCNBlock", "AffineCoupling", "FlowStep", "STGNF",
           "STGNFDetector", "fit_nf", "NF_RECIPE", "NF_ARCH", "TRAIN_NOISE"]

NF_RECIPE = dict(optimizer="adam", lr=5e-4, weight_decay=0.0, batch_size=256, max_epochs=30, patience=5)
NF_ARCH = dict(K=8, hidden=8, adjacency="skeleton", scale_clamp=2.0, tcn_kernel=9)
TRAIN_NOISE = 0.01
_LOG_2PI = math.log(2 * math.pi)


class ActNorm(nn.Module):
    """
    채널별 affine y = (x + b) * exp(s).  log|det| = (T*V) * sum_c s_c
    데이터 의존 초기화(init_from): 첫 배치 기준 b = -평균, s = -log(표준편차) -> 출력 채널별 평균 0, 표준편차 1
    """

    def __init__(self, channels: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def init_from(self, x: torch.Tensor) -> None:
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True)
        self.bias.data.copy_(-mean)
        self.logs.data.copy_(-torch.log(std + 1e-6))
        self.initialized.fill_(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = (x + self.bias) * torch.exp(self.logs)
        logdet = self.logs.sum() * (x.shape[2] * x.shape[3])
        return y, logdet.expand(x.shape[0])

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * torch.exp(-self.logs) - self.bias


class InvConv1x1(nn.Module):
    """
    가역 1x1 conv: 모든 (t, v)에서 y = W x (W: C x C).  log|det| = (T*V) * log|det W|
    초기값은 무작위 직교 행렬 (QR, 행렬식 부호와 무관하게 |det| = 1이라 시작 log|det| = 0)
    """

    def __init__(self, channels: int):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(channels, channels))
        self.weight = nn.Parameter(q)

    def logabsdet(self) -> torch.Tensor:
        w = self.weight
        if w.shape[0] == 2:  # 2x2는 닫힌 꼴 (CUDA slogdet은 작은 행렬에서도 느림)
            return torch.log(torch.abs(w[0, 0] * w[1, 1] - w[0, 1] * w[1, 0]))
        return torch.linalg.slogdet(w)[1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.einsum("ij,bjtv->bitv", self.weight, x)
        return y, (self.logabsdet() * (x.shape[2] * x.shape[3])).expand(x.shape[0])

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bjtv->bitv", torch.linalg.inv(self.weight), y)


class STGCNBlock(nn.Module):
    """
    coupling의 NN: (B, c_in, T, V) -> (B, c_out, T, V)
    1x1 conv(c_in -> H) -> 공간 그래프 합성 h[..., w] = sum_v h[..., v] A[v, w] -> ReLU
    -> 시간 conv (k, 1) same padding -> ReLU -> 1x1 conv(H -> c_out, 0 초기화)
    """

    def __init__(self, c_in: int, c_out: int, A: np.ndarray, hidden: int = 8, kernel: int = 9):
        super().__init__()
        self.register_buffer("A", torch.as_tensor(np.asarray(A), dtype=torch.float32))
        self.gcn = nn.Conv2d(c_in, hidden, 1)
        self.tcn = nn.Conv2d(hidden, hidden, (kernel, 1), padding=(kernel // 2, 0))
        self.out = nn.Conv2d(hidden, c_out, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("bctv,vw->bctw", self.gcn(x), self.A.to(x.dtype))
        h = torch.relu(h)
        h = torch.relu(self.tcn(h))
        return self.out(h)


class AffineCoupling(nn.Module):
    """
    y_a = x_a,  y_b = x_b * exp(log_s) + t,  (log_s, t) = NN(x_a),  log_s = a * tanh(h / a)
    log|det| = sum(log_s) (표본별)
    """

    def __init__(self, channels: int, A: np.ndarray, hidden: int = 8, kernel: int = 9, scale_clamp: float = 2.0):
        super().__init__()
        self.c_a = channels // 2
        self.c_b = channels - self.c_a
        self.clamp = float(scale_clamp)
        self.nn = STGCNBlock(self.c_a, 2 * self.c_b, A, hidden=hidden, kernel=kernel)

    def _st(self, x_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.nn(x_a)
        log_s = self.clamp * torch.tanh(h[:, :self.c_b] / self.clamp)
        return log_s, h[:, self.c_b:]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_a, x_b = x[:, :self.c_a], x[:, self.c_a:]
        log_s, t = self._st(x_a)
        y_b = x_b * torch.exp(log_s) + t
        return torch.cat([x_a, y_b], dim=1), log_s.flatten(1).sum(1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_a, y_b = y[:, :self.c_a], y[:, self.c_a:]
        log_s, t = self._st(y_a)
        return torch.cat([y_a, (y_b - t) * torch.exp(-log_s)], dim=1)


class FlowStep(nn.Module):
    """ActNorm -> 가역 1x1 conv -> affine coupling. log|det| = 세 층의 합"""

    def __init__(self, channels: int, A: np.ndarray, hidden: int, kernel: int, scale_clamp: float):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.invconv = InvConv1x1(channels)
        self.coupling = AffineCoupling(channels, A, hidden=hidden, kernel=kernel, scale_clamp=scale_clamp)

    def forward(self, x):
        x, ld1 = self.actnorm(x)
        x, ld2 = self.invconv(x)
        x, ld3 = self.coupling(x)
        return x, ld1 + ld2 + ld3

    def inverse(self, y):
        return self.actnorm.inverse(self.invconv.inverse(self.coupling.inverse(y)))


class STGNF(nn.Module):
    """
    K개 FlowStep. forward(x (B, C, T, V)) -> (z, log|det| (B,)), inverse(z) -> x
    nll(x) -> (B,) 원소당 NLL = -(log N(z; 0, I) + log|det|) / (C*T*V)
    """

    def __init__(self, channels: int = 2, K: int = 8, hidden: int = 8, adjacency: str = "skeleton",
                 scale_clamp: float = 2.0, tcn_kernel: int = 9):
        super().__init__()
        A = SG.coco_adjacency(adjacency)
        self.arch = dict(channels=channels, K=K, hidden=hidden, adjacency=adjacency, scale_clamp=scale_clamp,
                         tcn_kernel=tcn_kernel)
        self.steps = nn.ModuleList([FlowStep(channels, A, hidden, tcn_kernel, scale_clamp) for _ in range(K)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for st in self.steps:
            x, ld = st(x)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        for st in reversed(self.steps):
            z = st.inverse(z)
        return z

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """(B,) 세그먼트 전체 log p_X(x) (nats)"""
        z, logdet = self(x)
        d = z[0].numel()
        log_pz = -0.5 * (z ** 2).flatten(1).sum(1) - 0.5 * d * _LOG_2PI
        return log_pz + logdet

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x) / x[0].numel()

    @torch.no_grad()
    def data_init(self, x: torch.Tensor) -> None:
        """ActNorm 데이터 의존 초기화: x를 단계별로 통과시키며 각 ActNorm을 그 입력으로 초기화"""
        for st in self.steps:
            st.actnorm.init_from(x)
            x, _ = st(x)


def _nll_loss(model: STGNF, xb: torch.Tensor) -> torch.Tensor:
    return model.nll(xb)


class STGNFDetector(Detector):
    kind = "nf"

    def __init__(self, model: STGNF, device: str = "cpu", seg_len: int = SG.SEG_LEN, stride: int = SG.SEG_STRIDE,
                 min_valid: float = SG.MIN_SEG_VALID):
        super().__init__()
        self.model = model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.seg_len, self.stride, self.min_valid = int(seg_len), int(stride), float(min_valid)
        self.n_params = int(sum(p.numel() for p in model.parameters()))

    def to(self, device) -> "STGNFDetector":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def segments(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """창 (N, 92, T) -> 세그먼트 (N, S, 2, L, 17), 사용 여부 (N, S)"""
        seg, keep, _ = SG.window_segments(as_windows(X), self.seg_len, self.stride, self.min_valid)
        return seg, keep

    def segment_nll(self, seg: np.ndarray) -> np.ndarray:
        """(M, 2, L, 17) -> (M,) float64 원소당 NLL"""
        if len(seg) == 0:
            return np.zeros(0)
        xt = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32))
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(xt), EVAL_BATCH):
                out.append(self.model.nll(xt[i:i + EVAL_BATCH].to(self.device)).double().cpu())
        return torch.cat(out).numpy()

    def score(self, X) -> np.ndarray:
        X = as_windows(X)
        seg, keep = self.segments(X)
        n, s = keep.shape
        out = np.full(n, np.nan)
        if n == 0 or s == 0 or not keep.any():
            return out
        nll = np.full((n, s), -np.inf)
        nll[keep] = self.segment_nll(seg[keep])
        has = keep.any(axis=1)
        out[has] = nll[has].max(axis=1)
        return out

    def state(self) -> Dict[str, Any]:
        return {"kind": self.kind, "arch": dict(self.model.arch),
                "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "seg": {"seg_len": self.seg_len, "stride": self.stride, "min_valid": self.min_valid},
                "info": self.info}

    @classmethod
    def from_state(cls, st: Dict[str, Any], device: str = "cpu") -> "STGNFDetector":
        model = STGNF(**st["arch"])
        model.load_state_dict(st["state_dict"])
        det = cls(model, device=device, **st.get("seg", {}))
        det.info = dict(st.get("info") or {})
        return det


def _kept_segments(det: STGNFDetector, X: np.ndarray) -> np.ndarray:
    seg, keep = det.segments(X)
    return seg[keep] if keep.size else np.zeros((0, 2, det.seg_len, SG.NUM_KPS), dtype=np.float32)


def fit_nf(X_tr, X_va, seed: int = 0, device: str = "cpu", verbose: bool = False,
           noise: float = TRAIN_NOISE, arch: Optional[Dict[str, Any]] = None, **overrides) -> STGNFDetector:
    """
    정상 학습 창 (N, 92, T) -> 사용 세그먼트로 STG-NF 학습. 검증 정상 창의 세그먼트 NLL로 early stopping.
    arch: NF_ARCH 덮어쓰기, overrides: NF_RECIPE 덮어쓰기 (점검용)
    """
    from training.experiments.trainer import set_seed

    rc = dict(NF_RECIPE, **overrides)
    ar = dict(NF_ARCH, **(arch or {}))
    t0 = time.time()
    set_seed(int(seed))
    model = STGNF(channels=2, K=ar["K"], hidden=ar["hidden"], adjacency=ar["adjacency"],
                  scale_clamp=ar["scale_clamp"], tcn_kernel=ar["tcn_kernel"])
    det = STGNFDetector(model, device="cpu")
    S_tr = torch.from_numpy(_kept_segments(det, as_windows(X_tr)))
    S_va = torch.from_numpy(_kept_segments(det, as_windows(X_va)))
    if len(S_tr) == 0:
        raise ValueError("학습에 쓸 세그먼트가 없습니다 (관절 유효 비율 >= 0.5 세그먼트 0개)")
    dev = torch.device(device)
    model.to(dev)
    # ActNorm 데이터 의존 초기화: seed로 고른 학습 세그먼트 최대 4096개
    g = torch.Generator()
    g.manual_seed(int(seed))
    idx = torch.randperm(len(S_tr), generator=g)[:4096]
    model.data_init(S_tr[idx].to(dev))

    def aug(xb, gen):
        if noise <= 0:
            return xb
        return xb + noise * torch.randn(xb.shape, generator=gen, device=xb.device, dtype=xb.dtype)

    hist = train_unsupervised(model, S_tr, S_va, _nll_loss, seed=int(seed), device=dev, train_aug=aug,
                              verbose=verbose, **rc)
    det.to(dev)
    hist.update({"recipe": rc, "arch": ar, "noise": noise, "n_train_segments": int(len(S_tr)),
                 "n_val_segments": int(len(S_va)), "n_train_windows": int(len(as_windows(X_tr))),
                 "n_val_windows": int(len(as_windows(X_va))), "time_s": round(time.time() - t0, 2),
                 "device": str(dev)})
    det.info = hist
    return det
