"""
배포 v2 분류기(Fall1DCNNGAP, 사건 구간 4초 창) -> ONNX 내보내기 (opset 17, 배치 크기 동적)

입력  "x"      float32 (B, 92, 60): features.window_v2 특징 (표준화 전 원래 값; --no-std면 표준화한 값)
출력  "logits" float32 (B, 2), "p_fall" float32 (B,) = softmax(logits)[:, 1]

채널 표준화 (x - mu) / sd
- 기본: 그래프 안에 Sub/Div 연산으로 넣음 (mu/sd는 상수). 런타임은 원래 특징만 넘기면 됨.
  conv1 가중치에 접어 넣지(fold) 않는 이유: conv1은 zero padding이라, 표준화한 입력의 0 패딩은 원래 값 mu에 해당해
  접어 넣으면 창 양 끝 프레임의 출력이 달라짐 (정확히 같은 계산이 아님).
- --no-std: 모델만 내보내고 표준화는 밖에서 (app.ai_classifier.event_window.Standardizer, 메타의 mu/sd 사용).
창 길이 60은 고정 (GAP라 길이와 무관하지만 길이 1 창의 풀링 생략 분기가 내보낼 때 상수로 고정되기 때문).

검증: --check면 PyTorch(CPU)와 ONNX Runtime(CPU)의 출력을 무작위 창으로 비교 (최대 절대 오차 < 1e-4).
옆에 <out>.meta.json (가중치 메타 + format/standardize/io 정보)을 씁니다.

실행 (저장소 루트에서)
  python -m training.export_onnx --weights dataset/_models/fall_cnn_gap_v2.pt [--out ....onnx] [--no-std] [--check]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from app.ai_classifier import model_meta as MM
from app.ai_classifier.model import Fall1DCNNGAP

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "dataset" / "_models" / "fall_cnn_gap_v2.pt"
OPSET = 17
IN_CH = MM.V2_IN_CHANNELS
SEQ_LEN = int(round(MM.V2_WINDOW_S * MM.V2_HZ))  # 60
PARITY_ATOL = 1e-4

__all__ = ["load_checkpoint", "ExportWrapper", "export_onnx", "onnx_parity", "main"]


def load_checkpoint(path) -> Tuple[Fall1DCNNGAP, np.ndarray, np.ndarray]:
    """{"arch", "state_dict", "mu", "sd"} 체크포인트 -> (eval 모델 CPU, mu (92,), sd (92,)). 신뢰하는 파일만 (pickle)."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise ValueError(f"{path}: not a v2 checkpoint dict (state_dict/mu/sd)")
    model = Fall1DCNNGAP(IN_CH)
    model.load_state_dict(ck["state_dict"], strict=True)
    mu = np.asarray(ck["mu"], dtype=np.float32).reshape(-1)
    sd = np.asarray(ck["sd"], dtype=np.float32).reshape(-1)
    if mu.shape != (IN_CH,) or sd.shape != (IN_CH,):
        raise ValueError(f"mu/sd shape {mu.shape}/{sd.shape} != ({IN_CH},)")
    return model.eval(), mu, sd


class ExportWrapper(nn.Module):
    """(선택) 채널 표준화 + Fall1DCNNGAP + softmax. 출력 (logits, p_fall)."""

    def __init__(self, model: nn.Module, mu: Optional[np.ndarray] = None, sd: Optional[np.ndarray] = None):
        super().__init__()
        self.model = model
        self.standardize = mu is not None
        if self.standardize:
            self.register_buffer("mu", torch.from_numpy(np.asarray(mu, np.float32).reshape(1, -1, 1)))
            self.register_buffer("sd", torch.from_numpy(np.asarray(sd, np.float32).reshape(1, -1, 1)))

    def forward(self, x):
        if self.standardize:
            x = (x - self.mu) / self.sd
        logits = self.model(x)
        return logits, torch.softmax(logits, dim=1)[:, 1]


def export_onnx(weights, out, standardize: bool = True, opset: int = OPSET) -> Dict[str, Any]:
    """weights(.pt) -> out(.onnx) + out.meta.json. 반환: 메타 dict."""
    model, mu, sd = load_checkpoint(weights)
    wrapper = ExportWrapper(model, mu, sd) if standardize else ExportWrapper(model)
    wrapper.eval()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, IN_CH, SEQ_LEN, dtype=torch.float32)
    with warnings.catch_warnings():
        # TorchScript 기반 내보내기 사용 경고, 길이 분기(상수로 고정, 의도한 동작) 경고
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper, (dummy,), str(out), opset_version=opset, dynamo=False,
            input_names=["x"], output_names=["logits", "p_fall"],
            dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}, "p_fall": {0: "batch"}},
        )
    try:
        meta = MM.load_meta(weights) or MM.make_meta_v2()
    except ValueError:
        meta = MM.make_meta_v2()
    meta = dict(meta)
    meta.update({
        "format": "onnx",
        "onnx_opset": opset,
        "source_weights": str(Path(weights).name),
        "standardize": "in-graph (x - mu) / sd" if standardize else "outside: apply checkpoint mu/sd before input",
        "io": {"input": f"x float32 (batch, {IN_CH}, {SEQ_LEN})", "outputs": ["logits (batch, 2)", "p_fall (batch,)"]},
    })
    if not standardize:
        meta["mu"] = [float(v) for v in mu]
        meta["sd"] = [float(v) for v in sd]
    MM.write_meta(out, meta)
    return meta


def onnx_parity(weights, onnx_path, X: np.ndarray, standardize: bool = True) -> Dict[str, float]:
    """PyTorch(CPU) vs ONNX Runtime(CPU) 최대 절대 오차 {"logits", "p_fall"}. X (B, 92, 60) 원래 특징."""
    import onnxruntime as ort

    model, mu, sd = load_checkpoint(weights)
    X = np.ascontiguousarray(X, dtype=np.float32)
    feed = X if standardize else ((X - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)
    wrapper = (ExportWrapper(model, mu, sd) if standardize else ExportWrapper(model)).eval()
    with torch.no_grad():
        lt, pt = wrapper(torch.from_numpy(feed))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    lo, po = sess.run(["logits", "p_fall"], {"x": feed})
    return {"logits": float(np.max(np.abs(lt.numpy() - lo))), "p_fall": float(np.max(np.abs(pt.numpy() - po)))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="export v2 event-window classifier (Fall1DCNNGAP) to ONNX")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--out", default=None, help="기본: 가중치와 같은 이름의 .onnx")
    ap.add_argument("--no-std", action="store_true", help="표준화를 그래프에 넣지 않음 (밖에서 mu/sd 적용)")
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--check", action="store_true", help="ONNX Runtime과 출력 비교 (onnxruntime 필요)")
    args = ap.parse_args(argv)

    weights = Path(args.weights)
    out = Path(args.out) if args.out else weights.with_suffix(".onnx")
    std = not args.no_std
    export_onnx(weights, out, standardize=std, opset=args.opset)
    print(f"saved {out} (+ {MM.meta_path_for(out).name}), opset {args.opset}, standardize "
          f"{'in-graph' if std else 'outside'}")
    if args.check:
        X = np.random.default_rng(0).normal(size=(16, IN_CH, SEQ_LEN)).astype(np.float32)
        d = onnx_parity(weights, out, X, standardize=std)
        ok = max(d.values()) < PARITY_ATOL
        print(f"parity max abs diff {json.dumps(d)} -> {'OK' if ok else 'FAIL'} (< {PARITY_ATOL})")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
