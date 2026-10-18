"""
정상 데이터만 쓰는 비지도 모델(오토인코더, STG-NF)의 공통 학습 루프 (torch 필요)

- 표본별 손실 함수 loss_fn(model, xb) -> (B,) 텐서. 학습은 배치 평균, 검증은 표본 평균
- epoch마다 검증 정상 손실을 재고 최저 epoch 가중치를 복원 (patience epoch 동안 개선 없으면 중단)
- 셔플 순서는 seed로 고정. 학습 손실이 NaN/inf가 되면 그 epoch에서 멈추고 그 전 최저 가중치로 돌아감
"""
import copy
import math
import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

__all__ = ["train_unsupervised", "eval_loss", "EVAL_BATCH"]

EVAL_BATCH = 4096


def eval_loss(model: nn.Module, X: torch.Tensor, loss_fn: Callable, device, batch: int = EVAL_BATCH) -> float:
    """표본 평균 손실 (eval 모드). 빈 집합이면 nan"""
    n = len(X)
    if n == 0:
        return float("nan")
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for i in range(0, n, batch):
            tot += float(loss_fn(model, X[i:i + batch].to(device)).double().sum().item())
    return tot / n


def train_unsupervised(model: nn.Module, X_tr: torch.Tensor, X_va: torch.Tensor, loss_fn: Callable, *,
                       optimizer: str, lr: float, weight_decay: float, batch_size: int, max_epochs: int,
                       patience: int, seed: int, device, train_aug: Optional[Callable] = None,
                       clip_grad: Optional[float] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    model을 학습하고 최저 검증 손실 epoch 가중치를 복원합니다 (검증이 비면 마지막 epoch).
    train_aug(xb, gen) -> xb: 학습 배치에만 적용 (예: STG-NF 잡음). gen은 seed로 만든 장치 난수 생성기.
    반환 history: {"epochs": [{epoch, train_loss, val_loss}], best_epoch, best_val_loss, stopped_epoch,
                   n_train, n_val, time_s, diverged}
    """
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"알 수 없는 optimizer: {optimizer}")
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    dev = torch.device(device)
    aug_gen = torch.Generator(device=dev)
    aug_gen.manual_seed(int(seed) + 1)
    n = len(X_tr)
    has_val = len(X_va) > 0
    hist: Dict[str, Any] = {"epochs": [], "best_epoch": None, "best_val_loss": None, "stopped_epoch": None,
                            "n_train": int(n), "n_val": int(len(X_va)), "diverged": False}
    best_state, best, since = None, math.inf, 0
    t0 = time.time()
    for ep in range(1, int(max_epochs) + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot, cnt = 0.0, 0
        for i in range(0, n, int(batch_size)):
            idx = perm[i:i + int(batch_size)]
            if len(idx) < 2:
                continue  # BatchNorm이 표본 1개로는 통계를 못 냄
            xb = X_tr[idx].to(dev, non_blocking=True)
            if train_aug is not None:
                xb = train_aug(xb, aug_gen)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, xb).mean()
            if not torch.isfinite(loss):
                hist["diverged"] = True
                break
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(params, float(clip_grad))
            opt.step()
            tot += float(loss.item()) * len(idx)
            cnt += len(idx)
        if hist["diverged"]:
            hist["stopped_epoch"] = ep
            break
        tr = tot / cnt if cnt else float("nan")
        va = eval_loss(model, X_va, loss_fn, dev) if has_val else float("nan")
        hist["epochs"].append({"epoch": ep, "train_loss": tr, "val_loss": va})
        if verbose:
            print(f"ep {ep:3d} train {tr:.5f} val {va:.5f}")
        if not has_val:
            continue
        if math.isfinite(va) and va < best:
            best, since = va, 0
            best_state = copy.deepcopy(model.state_dict())
            hist["best_epoch"], hist["best_val_loss"] = ep, va
        else:
            since += 1
            if since >= int(patience):
                hist["stopped_epoch"] = ep
                break
    if hist["stopped_epoch"] is None:
        hist["stopped_epoch"] = len(hist["epochs"])
    if best_state is not None:
        model.load_state_dict(best_state)
    elif hist["epochs"]:
        last = hist["epochs"][-1]
        hist["best_epoch"], hist["best_val_loss"] = last["epoch"], last["val_loss"]
    hist["time_s"] = round(time.time() - t0, 2)
    model.eval()
    return hist
