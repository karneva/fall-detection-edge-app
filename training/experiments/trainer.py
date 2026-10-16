"""
진행 단계 비교 실험용 범용 torch 학습기

레시피
  original : 9e40e48 train.py 재현. Adam lr 1e-3, 50 epoch, 일반 CE, batch 32, 검증 정확도 최고 epoch 선택
             (동률이면 먼저 나온 epoch, 원본의 `val_acc > best_val_acc`와 같음)
  improved : AdamW lr 1e-3 wd 1e-4, cosine 감쇠, 클래스 가중 CE, batch 64, 최대 60 epoch,
             검증 손실 기준 early stopping(patience 8) 후 최저 검증 손실 epoch 복원

결정성: seed로 random/numpy/torch(+CUDA) 시드와 셔플 순서를 고정하고 cuDNN deterministic을 켭니다.
CPU에서는 같은 seed면 같은 가중치가 나옵니다. GPU에서는 일부 커널이 비결정적일 수 있어 근사적으로만 같습니다.

데이터는 CPU 텐서로 두고 배치마다 device로 옮깁니다 (수십만 창도 GPU 메모리를 넘지 않도록).
"""
import copy
import math
import random
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

__all__ = ["RECIPES", "get_recipe", "resolve_device", "set_seed", "class_weights",
           "train_classifier", "predict_proba"]

RECIPES: Dict[str, Dict[str, Any]] = {
    "original": dict(
        name="original",
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        epochs=50,
        batch_size=32,
        scheduler=None,
        class_weighted=False,
        select="val_acc",
        patience=None,
    ),
    "improved": dict(
        name="improved",
        optimizer="adamw",
        lr=1e-3,
        weight_decay=1e-4,
        epochs=60,
        batch_size=64,
        scheduler="cosine",
        class_weighted=True,
        select="val_loss",
        patience=8,
    ),
}

EVAL_BATCH = 1024


def get_recipe(recipe: Union[str, Dict[str, Any]], **overrides) -> Dict[str, Any]:
    """
    레시피 이름 또는 dict -> 완성된 레시피 dict.
    dict에 'name'이 있으면 그 기본 레시피 위에 나머지 키를 덮어씁니다 (예: {"name": "improved", "epochs": 5}).
    """
    if isinstance(recipe, str):
        base, extra = recipe, {}
    else:
        extra = dict(recipe)
        base = extra.pop("name", "improved")
    if base not in RECIPES:
        raise ValueError(f"알 수 없는 레시피: {base}")
    out = dict(RECIPES[base])
    out.update(extra)
    out.update(overrides)
    out["name"] = base
    return out


def resolve_device(device=None) -> torch.device:
    """None/'auto' -> CUDA가 있으면 cuda, 없으면 cpu"""
    if device is None or (isinstance(device, str) and device == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights(y: np.ndarray, n_cls: int = 2) -> np.ndarray:
    """균형 가중치 n / (n_cls * count_c). 없는 클래스는 0 (손실에 기여하지 않음)."""
    y = np.asarray(y).astype(np.int64)
    counts = np.bincount(y, minlength=n_cls).astype(np.float64)
    w = np.zeros(n_cls, dtype=np.float64)
    nz = counts > 0
    w[nz] = len(y) / (n_cls * counts[nz])
    return w.astype(np.float32)


def _as_tensors(X, y=None):
    Xt = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32))
    yt = None if y is None else torch.as_tensor(np.asarray(y).astype(np.int64))
    return Xt, yt


def _n_classes(y_tr: np.ndarray) -> int:
    n = int(np.max(y_tr)) + 1 if len(y_tr) else 2
    return max(2, n)


def _evaluate(model, Xt, yt, device, criterion) -> Tuple[float, float]:
    """(평균 손실, 정확도) — 표본 가중 평균. 빈 집합이면 (nan, nan)"""
    n = len(Xt)
    if n == 0:
        return float("nan"), float("nan")
    model.eval()
    loss_sum, correct, w_sum = 0.0, 0, 0.0
    with torch.no_grad():
        for i in range(0, n, EVAL_BATCH):
            xb = Xt[i:i + EVAL_BATCH].to(device)
            yb = yt[i:i + EVAL_BATCH].to(device)
            out = model(xb)
            # 가중 CE의 reduction='mean'은 가중치 합으로 나누므로, 배치 간에도 같은 방식으로 합산
            if criterion.weight is not None:
                bw = criterion.weight[yb].sum().item()
            else:
                bw = float(len(yb))
            if bw > 0:  # 가중치 0 클래스만 든 배치는 CE가 0/0=nan이라 합산에서 제외 (정확도는 셈)
                loss_sum += criterion(out, yb).item() * bw
                w_sum += bw
            correct += (out.argmax(1) == yb).sum().item()
    loss = loss_sum / w_sum if w_sum > 0 else float("nan")
    return loss, correct / n


def train_classifier(model: nn.Module, X_tr, y_tr, X_va, y_va, recipe="improved", seed: int = 0,
                     device=None, verbose: bool = False) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    model을 학습하고 선택 기준(레시피의 select)으로 고른 epoch의 가중치를 복원해 반환합니다.

    X_*: (N, C, T) float, y_*: (N,) int. 검증 집합이 비어 있으면 마지막 epoch를 사용합니다.
    seed는 모델 가중치 초기화에는 관여하지 않습니다 (모델은 호출 쪽에서 만듦) — 결정적 초기화가 필요하면
    모델 생성 전에 set_seed(seed)를 부르세요 (run_progression이 그렇게 함).
    반환: (model(eval 모드, device 위), history)
      history = {"recipe", "seed", "device", "epochs": [{epoch, train_loss, train_acc, val_loss, val_acc, lr}],
                 "best_epoch", "best_val_loss", "best_val_acc", "stopped_epoch", "class_weights", "n_train", "n_val"}
    """
    rc = get_recipe(recipe)
    dev = resolve_device(device)
    set_seed(seed)

    y_tr = np.asarray(y_tr).astype(np.int64)
    y_va = np.asarray(y_va).astype(np.int64) if y_va is not None else np.zeros(0, dtype=np.int64)
    if X_va is None:
        X_va = np.zeros((0,) + tuple(np.shape(X_tr)[1:]), dtype=np.float32)
    Xt, yt = _as_tensors(X_tr, y_tr)
    Xv, yv = _as_tensors(X_va, y_va)
    n_cls = _n_classes(y_tr)

    model = model.to(dev)
    cw = class_weights(y_tr, n_cls) if rc["class_weighted"] else None
    weight_t = torch.as_tensor(cw, device=dev) if cw is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_t)

    if rc["optimizer"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=rc["lr"], weight_decay=rc["weight_decay"])
    elif rc["optimizer"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=rc["lr"], weight_decay=rc["weight_decay"])
    else:
        raise ValueError(f"알 수 없는 optimizer: {rc['optimizer']}")
    epochs = int(rc["epochs"])
    sched = None
    if rc["scheduler"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    gen = torch.Generator()
    gen.manual_seed(seed)
    bs = int(rc["batch_size"])
    n = len(Xt)
    has_val = len(Xv) > 0
    select = rc["select"]
    patience = rc.get("patience")

    hist: Dict[str, Any] = {
        "recipe": rc, "seed": seed, "device": str(dev), "epochs": [],
        "class_weights": None if cw is None else [float(v) for v in cw],
        "n_train": int(n), "n_val": int(len(Xv)),
        "best_epoch": None, "best_val_loss": None, "best_val_acc": None, "stopped_epoch": None,
    }
    best_state = None
    best_key = None
    since_best = 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        loss_sum, w_sum, correct = 0.0, 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) * Xt.shape[-1] < 2:
                continue  # BatchNorm이 값 1개로는 통계를 못 냄 (T=1 창의 표본 1개 배치만 해당)
            xb = Xt[idx].to(dev, non_blocking=True)
            yb = yt[idx].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            bw = weight_t[yb].sum().item() if weight_t is not None else float(len(yb))
            loss_sum += loss.item() * bw
            w_sum += bw
            correct += (out.argmax(1) == yb).sum().item()
        lr_now = opt.param_groups[0]["lr"]
        if sched is not None:
            sched.step()

        tr_loss = loss_sum / w_sum if w_sum > 0 else float("nan")
        tr_acc = correct / n if n else float("nan")
        va_loss, va_acc = _evaluate(model, Xv, yv, dev, criterion)
        hist["epochs"].append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                                   val_loss=va_loss, val_acc=va_acc, lr=lr_now))
        if verbose:
            print(f"[{rc['name']}] ep {ep:3d} loss {tr_loss:.4f} acc {tr_acc:.3f} "
                  f"| val loss {va_loss:.4f} acc {va_acc:.3f}")

        if not has_val:
            continue
        # 선택 기준: 클수록 좋은 값으로 통일 (strict > 이므로 동률이면 먼저 나온 epoch 유지)
        key = va_acc if select == "val_acc" else -va_loss
        if math.isnan(key):
            key = -math.inf
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            hist["best_epoch"], hist["best_val_loss"], hist["best_val_acc"] = ep, va_loss, va_acc
            since_best = 0
        else:
            since_best += 1
            if patience is not None and since_best >= int(patience):
                hist["stopped_epoch"] = ep
                break

    if hist["stopped_epoch"] is None:
        hist["stopped_epoch"] = len(hist["epochs"])
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        last = hist["epochs"][-1] if hist["epochs"] else {}
        hist["best_epoch"] = last.get("epoch")
        hist["best_val_loss"], hist["best_val_acc"] = last.get("val_loss"), last.get("val_acc")
    model.eval()
    return model, hist


def predict_proba(model: nn.Module, X, device=None, batch_size: int = EVAL_BATCH) -> np.ndarray:
    """
    softmax 낙상 확률 p_fall (클래스 1), (N,) float32. 모델은 eval 모드로 전환됩니다.
    device=None이면 모델이 지금 있는 장치를 그대로 씁니다 (검증기와 공유하는 모델을 다른 장치로 옮기지 않도록).
    """
    if device is None:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
    else:
        dev = resolve_device(device)
    Xt, _ = _as_tensors(X)
    if len(Xt) == 0:
        return np.zeros(0, dtype=np.float32)
    model = model.to(dev)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            logits = model(Xt[i:i + batch_size].to(dev))
            outs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
    return torch.cat(outs).numpy().astype(np.float32)
