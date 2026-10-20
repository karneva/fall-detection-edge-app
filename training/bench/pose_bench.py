"""
포즈 모델 최적화 비교 (노트북 측정, Jetson 아님)

형식/정밀도별 포즈 모델의 프레임당 지연과 FP32 운영 모델 대비 출력 일치도, 회전 폴백 비용, 배포 v2 분류기의
AI 호출 지연을 재서 dataset/_experiments/bench_v1/results.json, summary.md로 씁니다.

- 지연: imgsz 640, 실제 프레임, 프레임마다 model.predict 한 번 (전처리 + 추론 + 후처리 + 결과를 CPU numpy로,
  live.py와 같은 conf 0.5). 워밍업 20장 뒤 200장. GPU 변형은 GPU, CPU 변형은 스레드 4개(저사양 엣지 대용)와
  라이브러리 기본 (PyTorch는 ultralytics가 첫 predict에서 8로 바꿈, ORT는 자체 기본).
  2회(round)를 순서를 뒤집어 재고 두 회의 표본을 합쳐 통계를 냄. 회차 p50 차이가 15%를 넘으면 '불안정'으로 표시하고
  FP32 대비 비율을 내지 않음. all은 GPU 변형 전부(2회)를 CPU 변형보다 먼저 재고, 실행마다 GPU/CPU 유휴 대기 + 안정 대기.
- 해상도: 프레임은 데이터셋 해상도(최대 1280x720). 운영 카메라 기본 1920x1080의 영향은 HIRES_VARIANTS만 같은 300장을
  1920x1080으로 키워 따로 잼 (run --resize 1920x1080).
- 일치도: FP32 운영 모델(yolov8s-pose PyTorch GPU) 출력과의 일치이지 정답 대비 정확도가 아님.
  프레임 집합 = 포즈 평가셋 낙상 후 150장(dataset/_pose/eval_v1/images/val, 초안 라벨은 정답으로 쓰지 않음)
  + 평가 데이터셋 낙상 클립의 낙상 전 서 있는 구간(walk/standing)에서 클립당 1장 150장 (seed 0).
  (a) conf 0.5 사람 검출률 (b) 둘 다 검출한 프레임의 OKS 유사 관절 일치도 (COCO sigma, s^2 = 기준 상자 넓이,
  기준 관절 conf >= 0.5만) (c) 상자 IoU. 기준은 최고 신뢰도 사람, 비교 모델은 그 상자와 IoU가 가장 큰 사람.
- 회전 폴백: v3_fb 키포인트(n_cand, present, fallback)로 폴백 시도 프레임(원래 방향 후보 0명, 직전 5초 안에 대상
  봤음)과 성공 프레임을 다시 세고, 시도 1번 = 90/270도 2장 묶음 predict 1번의 측정 지연으로 실효 FPS를 추정.
- 분류기: fall_cnn_gap_v2.pt, AI 호출 1번 = 4초 창 16개. 운영 경로(창마다 배치 1) PyTorch CPU/GPU,
  묶음(배치 16), ONNX Runtime CPU (training/export_onnx.py로 내보냄).

모델 파일(ONNX, INT8, 복사한 .pt)은 bench_v1/models/에 만듭니다 (dataset/_models는 건드리지 않음).
onnxruntime-gpu, TensorRT는 공유 venv를 바꾸지 않도록 따로 설치한 폴더를 PYTHONPATH로 넘길 때만 잽니다
(--ort-gpu-path, --trt-path). ultralytics 자동 설치는 끕니다 (YOLO_AUTOINSTALL=False).

명령 (저장소 루트에서, numpy/torch/ultralytics가 있는 파이썬)
  python -m training.bench.pose_bench frames                  # 프레임 집합 (낙상 후 150 + 서 있음 150)
  python -m training.bench.pose_bench export [--trt-path P]   # ONNX / INT8 / (TensorRT) / 분류기 ONNX
  python -m training.bench.pose_bench run --variant s_pt_fp32_gpu [--threads 4] [--round 1] [--dump] [--resize WxH]
  python -m training.bench.pose_bench classifier [--threads 4]
  python -m training.bench.pose_bench fallback-stats
  python -m training.bench.pose_bench report                  # raw/*.json -> results.json, summary.md
  python -m training.bench.pose_bench all [--ort-gpu-path P] [--trt-path P]   # 위 전부 (하위 프로세스)
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "dataset" / "_experiments" / "bench_v1"
MODELS_DIR = OUT_DIR / "models"
RAW_DIR = OUT_DIR / "raw"
FRAMES_JSON = OUT_DIR / "frames.json"
UPRIGHT_DIR = OUT_DIR / "frames" / "upright"
EVAL_POSE_DIR = REPO_ROOT / "dataset" / "_pose" / "eval_v1"
CALIB_DIR = REPO_ROOT / "dataset" / "_pose" / "v1" / "images" / "train"
MANIFEST = REPO_ROOT / "dataset" / "_manifest" / "manifest_v1.jsonl"
KP_DIR = REPO_ROOT / "dataset" / "_keypoints" / "v3_fb"
SRC_MODELS = REPO_ROOT / "dataset" / "_models"
CLS_WEIGHTS = SRC_MODELS / "fall_cnn_gap_v2.pt"

IMGSZ = 640
CONF = 0.5                  # app.config.DEFAULT_CONF
WARMUP = 20
N_TIMED = 200
FB_WARMUP = 5
FB_TIMED = 100
CLS_WINDOWS = 16
CLS_WARMUP = 10
CLS_TIMED = 100
N_CALIB = 200
SEED = 0
FALLBACK_RECENT_S = 5.0     # app.engine.rotate_fallback.FALLBACK_RECENT_S
VIS_TH = 0.5                # OKS에 쓰는 기준 관절 conf 하한
EVAL_DATASETS = ("le2i", "caucafall", "gmdcsa24", "mcfd", "urfd")
# 서 있는 프레임 데이터셋별 수 (포즈 평가셋 낙상 후 프레임과 같은 분포)
UPRIGHT_QUOTA = {"le2i": 35, "caucafall": 25, "gmdcsa24": 30, "mcfd": 40, "urfd": 20}
UPRIGHT_CLS = ("walk", "standing")
UPRIGHT_MARGIN_S = 0.2
LEAD_IN_END_S = 0.5         # 라벨 없는 앞부분을 쓸 때 낙상 시작 이 시간 전까지만
REFERENCE = "s_pt_fp32_gpu"
UNSTABLE_REL = 0.15         # 두 회차 p50 상대 차이가 이보다 크면 불안정 (비율을 내지 않음)
HIRES_WH = (1920, 1080)     # 운영 카메라 기본 (app.config FRAME_W/FRAME_H)
HIRES_VARIANTS = ("s_pt_fp32_gpu", "s_trt_fp16_gpu")
SQUARE_CONTROL = "s_pt_fp32_gpu_sq"   # 정적 640x640 입력의 PyTorch FP32 (ONNX/TensorRT와 입력 모양이 같음)
# COCO 17 관절 sigma (pycocotools)
COCO_SIGMAS = (0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
               0.107, 0.107, 0.087, 0.087, 0.089, 0.089)

# 변형: model(s|n), fmt(pt|onnx|engine), prec, device(gpu|cpu), needs(ort_gpu|trt|None), 모델 파일 이름(MODELS_DIR)
VARIANTS: Dict[str, Dict[str, Any]] = {
    "s_pt_fp32_gpu": {"model": "s", "fmt": "pt", "prec": "fp32", "device": "gpu", "file": "yolov8s-pose.pt",
                      "desc": "yolov8s-pose PyTorch FP32 (운영 기본)"},
    "s_pt_fp32_gpu_sq": {"model": "s", "fmt": "pt", "prec": "fp32", "device": "gpu", "file": "yolov8s-pose.pt",
                         "rect": False, "desc": "yolov8s-pose PyTorch FP32, 정적 640x640 입력 (rect=False, 입력 모양 대조군)"},
    "s_pt_fp16_gpu": {"model": "s", "fmt": "pt", "prec": "fp16", "device": "gpu", "file": "yolov8s-pose.pt",
                      "desc": "yolov8s-pose PyTorch FP16 (half=True)"},
    "s_ort_fp32_gpu": {"model": "s", "fmt": "onnx", "prec": "fp32", "device": "gpu", "needs": "ort_gpu",
                       "file": "yolov8s-pose.onnx", "desc": "yolov8s-pose ONNX Runtime CUDA EP FP32"},
    "s_trt_fp16_gpu": {"model": "s", "fmt": "engine", "prec": "fp16", "device": "gpu", "needs": "trt",
                       "file": "yolov8s-pose_fp16.engine", "desc": "yolov8s-pose TensorRT FP16"},
    "s_trt_int8_gpu": {"model": "s", "fmt": "engine", "prec": "int8", "device": "gpu", "needs": "trt",
                       "file": "yolov8s-pose_int8.engine", "desc": "yolov8s-pose TensorRT INT8"},
    "n_pt_fp32_gpu": {"model": "n", "fmt": "pt", "prec": "fp32", "device": "gpu", "file": "yolov8n-pose.pt",
                      "desc": "yolov8n-pose PyTorch FP32 (크기 효과)"},
    "n_pt_fp16_gpu": {"model": "n", "fmt": "pt", "prec": "fp16", "device": "gpu", "file": "yolov8n-pose.pt",
                      "desc": "yolov8n-pose PyTorch FP16"},
    "s_pt_fp32_cpu": {"model": "s", "fmt": "pt", "prec": "fp32", "device": "cpu", "file": "yolov8s-pose.pt",
                      "desc": "yolov8s-pose PyTorch FP32 CPU"},
    "s_ort_fp32_cpu": {"model": "s", "fmt": "onnx", "prec": "fp32", "device": "cpu", "file": "yolov8s-pose.onnx",
                       "desc": "yolov8s-pose ONNX Runtime CPU FP32"},
    "s_ort_int8_cpu": {"model": "s", "fmt": "onnx", "prec": "int8", "device": "cpu",
                       "file": "yolov8s-pose_int8.onnx",
                       "desc": "yolov8s-pose ONNX Runtime CPU INT8 (정적 양자화, 활성 S8, ultralytics 기본)"},
    "s_ort_int8u8_cpu": {"model": "s", "fmt": "onnx", "prec": "int8", "device": "cpu",
                         "file": "yolov8s-pose_int8_u8.onnx", "desc": "yolov8s-pose ONNX Runtime CPU INT8 (활성 U8)"},
    "s_ort_int8u8pc_cpu": {"model": "s", "fmt": "onnx", "prec": "int8", "device": "cpu",
                           "file": "yolov8s-pose_int8_u8pc.onnx",
                           "desc": "yolov8s-pose ONNX Runtime CPU INT8 (활성 U8, 가중치 per-channel)"},
    "n_ort_fp32_cpu": {"model": "n", "fmt": "onnx", "prec": "fp32", "device": "cpu", "file": "yolov8n-pose.onnx",
                       "desc": "yolov8n-pose ONNX Runtime CPU FP32"},
}
# 회전 폴백 비용을 재는 변형 (운영 후보)
FALLBACK_COST_VARIANTS = ("s_pt_fp32_gpu", "s_pt_fp16_gpu")
CPU_THREADS = (4, 0)        # 0 = 라이브러리 기본 (PyTorch는 ultralytics가 8로 맞춤)
# GPU 블록 순서: n과 s를 번갈아 (2회째는 뒤집음)
GPU_ORDER = ("s_pt_fp32_gpu", "n_pt_fp32_gpu", "s_pt_fp16_gpu", "n_pt_fp16_gpu", "s_pt_fp32_gpu_sq",
             "s_ort_fp32_gpu", "s_trt_fp16_gpu", "s_trt_int8_gpu")

__all__ = ["VARIANTS", "COCO_SIGMAS", "latency_stats", "box_iou", "oks", "frame_agreement", "summarize_agreement",
           "fallback_attempts", "effective_ms", "pick_upright_time", "md_table", "round_spread", "parse_wh",
           "main"]


# ---------------------------------------------------------------------------
# 순수 함수 (표준 라이브러리만, 테스트 대상)
# ---------------------------------------------------------------------------

def _percentile(xs: Sequence[float], q: float) -> float:
    """numpy.percentile 기본(linear)과 같은 보간"""
    s = sorted(float(x) for x in xs)
    if not s:
        raise ValueError("empty")
    pos = (len(s) - 1) * float(q) / 100.0
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def latency_stats(ms: Sequence[float]) -> Dict[str, Any]:
    """지연 표본(ms) -> n, mean, p50, p95, FPS (= 1000 / mean)"""
    xs = [float(x) for x in ms]
    if not xs:
        raise ValueError("no samples")
    mean = sum(xs) / len(xs)
    return {"n": len(xs), "mean_ms": round(mean, 3), "p50_ms": round(_percentile(xs, 50), 3),
            "p95_ms": round(_percentile(xs, 95), 3), "fps": round(1000.0 / mean, 2) if mean > 0 else None}


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """[x1, y1, x2, y2] 두 상자의 IoU"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def oks(ref_kps: Sequence[Sequence[float]], kps: Sequence[Sequence[float]], area: float,
        sigmas: Sequence[float] = COCO_SIGMAS, vis_th: float = VIS_TH) -> Optional[float]:
    """
    OKS 유사 일치도: mean_i exp(-d_i^2 / (2 * area * (2 sigma_i)^2)), 기준 관절 conf >= vis_th인 관절만.
    관절은 (x, y, conf) 픽셀. area는 기준 상자 넓이 (COCO는 분할 넓이라 이 값이 조금 관대함). 쓸 관절이 없으면 None.
    """
    if area <= 0:
        return None
    vals = []
    for (rx, ry, rc), (x, y, _), s in zip(ref_kps, kps, sigmas):
        if rc < vis_th:
            continue
        d2 = (x - rx) ** 2 + (y - ry) ** 2
        vals.append(math.exp(-d2 / (2.0 * area * (2.0 * s) ** 2)))
    return sum(vals) / len(vals) if vals else None


def frame_agreement(ref: Dict[str, Any], var: Dict[str, Any]) -> Dict[str, Any]:
    """
    프레임 하나의 일치도. ref/var = {"boxes": [[x1,y1,x2,y2]...], "conf": [...], "kps": [[[x,y,c]*17]...]} (conf 하한 적용됨)
    기준 = ref의 최고 신뢰도 사람, 짝 = var 중 그 상자와 IoU가 가장 큰 사람.
    """
    rd, vd = len(ref.get("boxes") or []) > 0, len(var.get("boxes") or []) > 0
    out: Dict[str, Any] = {"ref_det": rd, "var_det": vd, "iou": None, "oks": None}
    if not (rd and vd):
        return out
    i = max(range(len(ref["conf"])), key=lambda k: ref["conf"][k])
    rb = ref["boxes"][i]
    ious = [box_iou(rb, b) for b in var["boxes"]]
    j = max(range(len(ious)), key=lambda k: ious[k])
    area = max(0.0, rb[2] - rb[0]) * max(0.0, rb[3] - rb[1])
    out["iou"] = ious[j]
    out["oks"] = oks(ref["kps"][i], var["kps"][j], area)
    return out


def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in xs if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def summarize_agreement(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """frame_agreement 결과 목록 -> 검출률(기준/비교), 검출 여부 일치율, 둘 다 검출한 프레임의 평균 OKS/IoU"""
    n = len(records)
    if n == 0:
        return {"n": 0}
    both = [r for r in records if r["ref_det"] and r["var_det"]]
    return {
        "n": n,
        "ref_det_rate": round(sum(r["ref_det"] for r in records) / n, 4),
        "var_det_rate": round(sum(r["var_det"] for r in records) / n, 4),
        "det_agree_rate": round(sum(r["ref_det"] == r["var_det"] for r in records) / n, 4),
        "n_both": len(both),
        "oks_mean": _mean(r["oks"] for r in both),
        "iou_mean": _mean(r["iou"] for r in both),
        "oks_ge_0p9_rate": (round(sum((r["oks"] or 0) >= 0.9 for r in both) / len(both), 4) if both else None),
    }


def fallback_attempts(present: Sequence[bool], n_cand: Sequence[int], fps: float,
                      recent_s: float = FALLBACK_RECENT_S) -> List[bool]:
    """
    프레임별 회전 폴백 시도 여부 (추출기/live.py와 같은 조건): 원래 방향 후보 0명이고, 마지막으로 대상을 본
    프레임(폴백으로 찾은 프레임 포함 = present)이 recent_s초 안. present는 폴백 결과까지 반영된 값이어야 함.
    """
    out: List[bool] = []
    last_seen: Optional[int] = None
    for i, (p, c) in enumerate(zip(present, n_cand)):
        out.append(int(c) == 0 and last_seen is not None and (i - last_seen) / float(fps) <= recent_s)
        if p:
            last_seen = i
    return out


def effective_ms(base_ms: float, fb_ms: float, attempt_rate: float) -> Dict[str, float]:
    """프레임당 평균 지연 = 기본 + 시도율 x 시도 1번 지연 -> ms, FPS"""
    ms = float(base_ms) + float(attempt_rate) * float(fb_ms)
    return {"ms": round(ms, 3), "fps": round(1000.0 / ms, 2) if ms > 0 else None}


def pick_upright_time(segments: Sequence[Dict[str, Any]], fall_start_s: float, rng: random.Random,
                      classes: Sequence[str] = UPRIGHT_CLS, margin_s: float = UPRIGHT_MARGIN_S,
                      lead_in_end_s: float = LEAD_IN_END_S) -> Tuple[Optional[float], Optional[str]]:
    """
    낙상 시작 전에 끝나는 서 있는 구간(classes)에서 시각 하나: 구간 길이에 비례해 구간을 고르고 그 안에서 균등.
    구간 양 끝 margin_s는 뺌. 반환 (t, "segment").
    라벨 구간이 모두 낙상 시작 이후인 클립(URFD: 앞부분 라벨 없음)은 [margin_s, fall_start - lead_in_end_s]에서
    균등, 반환 (t, "lead_in"). 쓸 구간이 없으면 (None, None).
    """
    cands = []
    for s in segments:
        if s.get("cls") not in classes:
            continue
        a, b = float(s["start_s"]) + margin_s, min(float(s["end_s"]), float(fall_start_s)) - margin_s
        if b > a:
            cands.append((a, b))
    src = "segment"
    if not cands and all(float(s["start_s"]) >= float(fall_start_s) - 1e-6 for s in segments):
        a, b = margin_s, float(fall_start_s) - lead_in_end_s
        if b > a:
            cands, src = [(a, b)], "lead_in"
    if not cands:
        return None, None
    total = sum(b - a for a, b in cands)
    u = rng.uniform(0.0, total)
    for a, b in cands:
        if u <= b - a:
            return round(a + u, 3), src
        u -= b - a
    return round(cands[-1][1], 3), src


def round_spread(p50s: Sequence[float]) -> Optional[float]:
    """회차별 p50의 상대 차이 (max - min) / min. 회차가 2개 미만이면 None"""
    xs = [float(x) for x in p50s]
    if len(xs) < 2 or min(xs) <= 0:
        return None
    return round((max(xs) - min(xs)) / min(xs), 4)


def parse_wh(s: str) -> Tuple[int, int]:
    """'1920x1080' -> (1920, 1080)"""
    w, h = (int(v) for v in str(s).lower().split("x"))
    if w <= 0 or h <= 0:
        raise ValueError(s)
    return w, h


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """마크다운 표 (None은 -)"""
    def cell(v):
        return "-" if v is None else str(v)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 환경 정보
# ---------------------------------------------------------------------------

def gpu_state() -> Optional[Dict[str, Any]]:
    """nvidia-smi 한 번 (사용률, 메모리, 온도, SM 클럭, 전력, P-state). 실패하면 None"""
    q = "utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw,pstate"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    except Exception:
        return None
    v = [x.strip() for x in out.split(",")]

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None
    return {"util_pct": num(v[0]), "mem_mb": num(v[1]), "temp_c": num(v[2]), "sm_mhz": num(v[3]),
            "power_w": num(v[4]), "pstate": v[5]}


def wait_gpu_idle(max_wait_s: float = 600, util_th: float = 10.0, mem_th_mb: float = 500.0) -> Dict[str, Any]:
    """GPU 사용률/메모리가 낮아질 때까지 30초 간격으로 기다림. 반환: 마지막 상태 + busy(시간 초과 여부)"""
    t0 = time.time()
    while True:
        st = gpu_state() or {}
        busy = (st.get("util_pct") or 0) > util_th or (st.get("mem_mb") or 0) > mem_th_mb
        if not busy or time.time() - t0 > max_wait_s:
            st["busy"] = bool(busy)
            st["waited_s"] = round(time.time() - t0, 1)
            return st
        print(f"[bench] GPU busy {st}, waiting...", flush=True)
        time.sleep(30)


def host_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": sys.version.split()[0], "cpu_count_logical": os.cpu_count()}
    try:
        import platform
        info["platform"] = platform.platform()
        info["cpu"] = platform.processor()
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                info["cpu"] = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        pass
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
        b = psutil.sensors_battery()
        info["power_plugged"] = None if b is None else bool(b.power_plugged)
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        info["gpu"] = out
    except Exception:
        pass
    return info


def _cpu_busy_pct(interval: float = 1.0) -> Optional[float]:
    try:
        import psutil
        return float(psutil.cpu_percent(interval=interval))
    except Exception:
        return None


def wait_cpu_idle(max_wait_s: float = 1800, busy_th: float = 10.0) -> Dict[str, Any]:
    """전체 CPU 사용률(5초 평균)이 busy_th % 이하가 될 때까지 15초 간격으로 기다림. psutil이 없으면 바로 반환"""
    t0 = time.time()
    while True:
        b = _cpu_busy_pct(5.0)
        busy = b is not None and b > busy_th
        if not busy or time.time() - t0 > max_wait_s:
            return {"cpu_busy_pct": b, "busy": bool(busy), "waited_s": round(time.time() - t0, 1)}
        print(f"[bench] CPU busy {b}%, waiting...", flush=True)
        time.sleep(15)


def _write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 프레임 집합
# ---------------------------------------------------------------------------

def _read_frame(video: Path, t: float, fps: Optional[float] = None, crop_px=None):
    """영상의 t초 프레임 (fps가 없으면 컨테이너 fps, 클립 끝을 넘으면 마지막 프레임). crop_px가 있으면 자름"""
    import cv2
    cap = cv2.VideoCapture(str(video))
    f = float(fps) if fps else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = int(round(float(t) * f))
    if n > 0:
        target = min(target, n - 1)
    img, i = None, 0
    while i <= target:
        ok, frame = cap.read()
        if not ok:
            break
        img = frame
        i += 1
    cap.release()
    if img is not None and crop_px:
        x1, y1, x2, y2 = (int(v) for v in crop_px)
        img = img[y1:y2, x1:x2]
    return img


def cmd_frames(args) -> int:
    """낙상 후 150장(포즈 평가셋 이미지 그대로) + 낙상 전 서 있는 150장(영상에서 꺼내 JPEG 95) -> frames.json"""
    import cv2

    frames: List[Dict[str, Any]] = []
    ev = _read_json(EVAL_POSE_DIR / "frames.json")
    val_dir = EVAL_POSE_DIR / "images" / "val"
    names = sorted(p.name for p in val_dir.glob("*.jpg"))
    if len(names) != len(ev):
        raise RuntimeError(f"eval_v1 images {len(names)} != frames.json {len(ev)}")
    for name, r in zip(names, ev):
        frames.append({"set": "fallen", "path": (val_dir / name).relative_to(REPO_ROOT).as_posix(),
                       "clip_id": r["clip_id"], "dataset": r["dataset"], "t": r["t"]})
    rows = [json.loads(line) for line in open(MANIFEST, encoding="utf-8") if line.strip()]
    rng = random.Random(SEED)
    UPRIGHT_DIR.mkdir(parents=True, exist_ok=True)
    for old in UPRIGHT_DIR.glob("*.jpg"):  # 다시 만들 때 이전 번호 파일이 남지 않게
        old.unlink()
    k = 0
    for ds, n in UPRIGHT_QUOTA.items():
        cands = sorted((r for r in rows if r.get("dataset") == ds and r.get("label") == "fall"
                        and r.get("fall_start_s")), key=lambda r: r["clip_id"])
        rng.shuffle(cands)
        got = 0
        for r in cands:
            if got >= n:
                break
            t, src = pick_upright_time(r.get("segments") or [], float(r["fall_start_s"]), rng)
            if t is None:
                continue
            img = _read_frame(REPO_ROOT / r["video"], t, r.get("fps"), r.get("crop_px"))
            if img is None:
                continue
            out = UPRIGHT_DIR / f"{k:04d}_{r['clip_id'].replace('/', '__')}.jpg"
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames.append({"set": "upright", "path": out.relative_to(REPO_ROOT).as_posix(), "clip_id": r["clip_id"],
                           "dataset": ds, "t": t, "source": src})
            k += 1
            got += 1
        if got < n:
            print(f"[frames] {ds}: only {got}/{n} upright frames", flush=True)
    _write_json(FRAMES_JSON, {"seed": SEED, "upright_classes": list(UPRIGHT_CLS), "lead_in_end_s": LEAD_IN_END_S,
                              "frames": frames})
    print(f"[frames] {sum(f['set'] == 'fallen' for f in frames)} fallen + "
          f"{sum(f['set'] == 'upright' for f in frames)} upright -> {FRAMES_JSON}")
    return 0


def _load_frames():
    import cv2
    fr = _read_json(FRAMES_JSON)["frames"]
    imgs = []
    for f in fr:
        im = cv2.imread(str(REPO_ROOT / f["path"]))
        if im is None:
            raise FileNotFoundError(f["path"])
        imgs.append(im)
    return fr, imgs


def _timing_order(n: int) -> List[int]:
    """지연 측정 순서: 낙상 후/서 있음이 섞이도록 고정 seed로 섞은 인덱스"""
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    return idx


# ---------------------------------------------------------------------------
# 모델 내보내기
# ---------------------------------------------------------------------------

def _letterbox_input(img):
    """ultralytics 정적 640 입력과 같은 전처리 (LetterBox auto=False, BGR->RGB, /255, NCHW float32)"""
    import numpy as np
    from ultralytics.data.augment import LetterBox
    im = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False, stride=32)(image=img)
    im = im[..., ::-1].transpose(2, 0, 1)
    return (np.ascontiguousarray(im, dtype=np.float32) / 255.0)[None]


def quantize_int8(src: Path, dst: Path, calib_paths: Sequence[Path], per_channel: bool = False,
                  symmetric: bool = False, act_u8: bool = False) -> Dict[str, Any]:
    """
    ONNX Runtime 정적 양자화 (QDQ, 가중치/활성 int8, MinMax 보정). ultralytics onnx_int8_quantize와 같은 규칙:
    Conv/Gemm/MatMul만 양자화하고 head 디코딩(상자 픽셀 ~640과 확률 0~1이 한 텐서)은 float로 둠.
    symmetric=True: TensorRT용. 활성도 대칭(zero point 0)이고 bias는 float로 둠 (TensorRT는 0이 아닌 zero point와
      int32 bias DequantizeLinear를 받지 않음).
    act_u8=True: 활성 uint8 (ORT x86 CPU 권장 U8S8 조합).
    """
    import cv2
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self, paths):
            self.it = iter(paths)

        def get_next(self):
            for p in self.it:
                im = cv2.imread(str(p))
                if im is not None:
                    return {"images": _letterbox_input(im)}
            return None

    model = onnx.load(str(src))
    exclude = [n.name for n in model.graph.node if n.op_type not in {"Conv", "Gemm", "MatMul"}]
    quantize_static(str(src), str(dst), Reader(calib_paths), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8 if act_u8 else QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=per_channel,
                    nodes_to_exclude=exclude, extra_options={"ActivationSymmetric": bool(symmetric),
                                                             "WeightSymmetric": True,
                                                             "QuantizeBias": not symmetric})
    # ultralytics가 읽는 메타데이터(kpt_shape, imgsz, names 등)를 옮김
    q = onnx.load(str(dst))
    del q.metadata_props[:]
    for p in model.metadata_props:
        q.metadata_props.add(key=p.key, value=p.value)
    onnx.save(q, str(dst))
    return {"per_channel": per_channel, "format": "QDQ", "activation": "uint8" if act_u8 else "int8", "weight": "int8",
            "activation_symmetric": bool(symmetric), "weight_symmetric": True, "quantize_bias": not symmetric,
            "calibrate": "MinMax", "n_calib": len(calib_paths), "excluded_nodes": len(exclude),
            "rule": "Conv/Gemm/MatMul only (ultralytics onnx_int8_quantize)"}


def cmd_export(args) -> int:
    """bench_v1/models/에 .pt 복사(n은 내려받기), ONNX(정적 640, opset 기본, simplify 끔), INT8, (TensorRT), 분류기 ONNX"""
    import shutil
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    info: Dict[str, Any] = {}
    s_pt = MODELS_DIR / "yolov8s-pose.pt"
    if not s_pt.exists():
        shutil.copy2(SRC_MODELS / "yolov8s-pose.pt", s_pt)
    n_pt = MODELS_DIR / "yolov8n-pose.pt"
    if not n_pt.exists():
        attempt_download_asset(str(n_pt))
    for m in ("s", "n"):
        pt = MODELS_DIR / f"yolov8{m}-pose.pt"
        onnx_path = pt.with_suffix(".onnx")
        if not onnx_path.exists():
            YOLO(str(pt)).export(format="onnx", imgsz=IMGSZ, dynamic=False, simplify=False, half=False,
                                 device="cpu", batch=1)
        info[f"{m}_onnx"] = {"imgsz": IMGSZ, "dynamic": False, "simplify": False, "batch": 1}
    int8 = MODELS_DIR / "yolov8s-pose_int8.onnx"
    calib = sorted(CALIB_DIR.glob("*.jpg"))
    calib = random.Random(SEED).sample(calib, min(N_CALIB, len(calib)))
    if not int8.exists() or args.force_int8:
        info["s_int8"] = quantize_int8(MODELS_DIR / "yolov8s-pose.onnx", int8, calib,
                                       per_channel=bool(args.int8_per_channel))
        info["s_int8"]["calib_dir"] = CALIB_DIR.relative_to(REPO_ROOT).as_posix()
        _write_json(MODELS_DIR / "int8_info.json", info["s_int8"])
    int8u8 = MODELS_DIR / "yolov8s-pose_int8_u8.onnx"
    if not int8u8.exists():
        qi = quantize_int8(MODELS_DIR / "yolov8s-pose.onnx", int8u8, calib, act_u8=True)
        qi["calib_dir"] = CALIB_DIR.relative_to(REPO_ROOT).as_posix()
        _write_json(MODELS_DIR / "int8_u8_info.json", qi)
    int8u8pc = MODELS_DIR / "yolov8s-pose_int8_u8pc.onnx"
    if not int8u8pc.exists():
        qi = quantize_int8(MODELS_DIR / "yolov8s-pose.onnx", int8u8pc, calib, per_channel=True, act_u8=True)
        qi["calib_dir"] = CALIB_DIR.relative_to(REPO_ROOT).as_posix()
        _write_json(MODELS_DIR / "int8_u8pc_info.json", qi)
    for key, fn in (("s_int8_u8", "int8_u8_info.json"), ("s_int8_u8pc", "int8_u8pc_info.json")):
        if (MODELS_DIR / fn).exists():
            info[key] = _read_json(MODELS_DIR / fn)
    if (MODELS_DIR / "int8_info.json").exists():
        info["s_int8"] = _read_json(MODELS_DIR / "int8_info.json")
    if args.trt:
        info["trt"] = _export_trt()
    # 분류기 ONNX (training/export_onnx.py, 표준화는 그래프 안)
    from training.export_onnx import export_onnx
    cls_onnx = MODELS_DIR / "fall_cnn_gap_v2.onnx"
    if not cls_onnx.exists():
        export_onnx(CLS_WEIGHTS, cls_onnx)
    for p in sorted(MODELS_DIR.glob("yolov8*")):
        info.setdefault("sizes_mb", {})[p.name] = round(p.stat().st_size / 2 ** 20, 2)
    _write_json(RAW_DIR / "export_info.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


def _export_trt() -> Dict[str, Any]:
    """
    TensorRT 엔진 (ultralytics onnx2engine, 정적 640, 배치 1). TensorRT 11은 정밀도 플래그와 INT8 보정기가 없어
    (strongly typed) ultralytics는 ModelOpt로 ONNX를 바꾸는데, ModelOpt 없이 하려고 정밀도를 ONNX에 미리 넣어 줌:
    - FP16: ONNX를 FP16으로 변환 (onnxruntime.transformers.float16, 입출력은 FP32 유지, ultralytics ONNX half와 같음)
    - INT8: ORT 정적 양자화 QDQ ONNX를 활성 대칭으로 따로 만들어(yolov8s-pose_int8_sym.onnx, 보정 이미지는 CPU INT8과
      같음) 그대로 빌드 (명시적 양자화, 양자화하지 않은 층은 FP32)
    실패는 기록만.
    """
    import onnx
    from onnxruntime.transformers import float16
    from ultralytics.nn.backends.base import BaseBackend
    from ultralytics.utils.export.engine import onnx2engine
    src = MODELS_DIR / "yolov8s-pose.onnx"
    meta = BaseBackend.read_metadata(src)
    info_path = MODELS_DIR / "trt_info.json"
    out: Dict[str, Any] = _read_json(info_path) if info_path.exists() else {}
    for prec in ("fp16", "int8"):
        dst = MODELS_DIR / f"yolov8s-pose_{prec}.engine"
        if dst.exists() and str(out.get(prec, "")).startswith("ok"):
            continue
        try:
            if prec == "fp16":
                f16 = MODELS_DIR / "yolov8s-pose_fp16.onnx"
                if not f16.exists():
                    m = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
                    onnx.save(m, str(f16))
                onnx_in = f16
            else:
                onnx_in = MODELS_DIR / "yolov8s-pose_int8_sym.onnx"
                if not onnx_in.exists():
                    calib = sorted(CALIB_DIR.glob("*.jpg"))
                    calib = random.Random(SEED).sample(calib, min(N_CALIB, len(calib)))
                    qi = quantize_int8(src, onnx_in, calib, symmetric=True)
                    _write_json(MODELS_DIR / "int8_sym_info.json", qi)
            onnx2engine(str(onnx_in), dst, None, None, False, (1, 3, IMGSZ, IMGSZ), metadata=meta)
            out[prec] = f"ok (from {onnx_in.name})"
        except Exception as e:  # 설치/변환 실패는 표에 '측정 안 함'으로
            out[prec] = f"failed: {e!r}"[:500]
    import tensorrt
    out["tensorrt_version"] = tensorrt.__version__
    _write_json(info_path, out)
    return out


# ---------------------------------------------------------------------------
# 포즈 변형 측정 (하위 프로세스 하나 = 변형 하나 x 스레드 설정 하나)
# ---------------------------------------------------------------------------

def _limit_threads(n: int):
    """
    CPU 스레드 제한: torch, ONNX Runtime(세션 옵션 주입). n=0이면 기본값 유지.
    cv2는 건드리지 않음: ultralytics import가 cv2.setNumThreads(0)(단일 스레드)으로 두고 운영도 같음.
    ultralytics는 CPU predict 첫 호출(select_device)에서 torch 스레드를 8로 바꾸므로 cmd_run이 그 뒤에 다시 맞춤.
    """
    import torch
    if n > 0:
        torch.set_num_threads(n)
        import onnxruntime as ort
        base = ort.InferenceSession

        class _Limited(base):  # ultralytics ONNXBackend가 onnxruntime.InferenceSession을 부름
            def __init__(self, path_or_bytes, sess_options=None, providers=None, **kw):
                so = sess_options or ort.SessionOptions()
                so.intra_op_num_threads = n
                so.inter_op_num_threads = 1
                super().__init__(path_or_bytes, so, providers=providers, **kw)
        ort.InferenceSession = _Limited


def _result_arrays(r) -> Dict[str, Any]:
    """ultralytics 결과 -> {"boxes", "conf", "kps"} (CPU numpy, live.py select_best_person/_predict_arrays와 같이)"""
    import numpy as np
    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
        return {"boxes": np.zeros((0, 4)), "conf": np.zeros(0), "kps": np.zeros((0, 17, 3))}
    kxy = r.keypoints.xy.cpu().numpy()
    kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
    return {"boxes": r.boxes.xyxy.cpu().numpy(), "conf": r.boxes.conf.cpu().numpy(),
            "kps": np.concatenate([kxy, kc[..., None]], axis=-1)}


def _to_jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    return {"boxes": [[round(float(v), 2) for v in b] for b in d["boxes"]],
            "conf": [round(float(c), 4) for c in d["conf"]],
            "kps": [[[round(float(k[0]), 2), round(float(k[1]), 2), round(float(k[2]), 4)] for k in p]
                    for p in d["kps"]]}


def _providers(model) -> Optional[List[str]]:
    try:
        return list(model.predictor.model.session.get_providers())
    except Exception:
        return None


def cmd_run(args) -> int:
    v = VARIANTS[args.variant]
    threads = int(args.threads)
    if v["device"] == "cpu":
        _limit_threads(threads)
    import torch
    from ultralytics import YOLO

    path = MODELS_DIR / v["file"]
    if not path.exists():
        raise FileNotFoundError(path)
    fr, imgs = _load_frames()
    resize = parse_wh(args.resize) if getattr(args, "resize", None) else None
    if resize:
        import cv2
        imgs = [cv2.resize(im, resize, interpolation=cv2.INTER_LINEAR) for im in imgs]
    model = YOLO(str(path), task="pose")
    device = 0 if v["device"] == "gpu" else "cpu"
    kw = {"imgsz": IMGSZ, "conf": CONF, "device": device, "half": v["fmt"] == "pt" and v["prec"] == "fp16",
          "verbose": False}
    if "rect" in v:
        kw["rect"] = bool(v["rect"])
    import cv2
    # 준비 predict 1번 (모델 적재, select_device). ultralytics가 CPU면 torch 스레드를 8로 바꾸므로 그 뒤 다시 맞춤
    model.predict(imgs[0], **kw)
    if v["device"] == "cpu" and threads > 0:
        torch.set_num_threads(threads)
    gpu_before = gpu_state()
    cpu_busy = _cpu_busy_pct()
    rec: Dict[str, Any] = {"variant": args.variant, "threads": threads, "round": int(args.round),
                           "gpu_before": gpu_before, "cpu_busy_pct_before": cpu_busy, "model_file": path.name,
                           "model_size_mb": round(path.stat().st_size / 2 ** 20, 2),
                           "frame_wh": list(resize) if resize else "dataset"}
    if args.dump and not resize:
        preds = []
        for im in imgs:
            preds.append(_to_jsonable(_result_arrays(model.predict(im, **kw)[0])))
        _write_json(RAW_DIR / f"pred_{args.variant}.json", {"variant": args.variant, "conf": CONF, "preds": preds})
    order = _timing_order(len(imgs))
    lat, speed = [], []
    try:
        import psutil
        psutil.cpu_percent(interval=None)  # 다음 호출이 측정 구간 평균
    except Exception:
        psutil = None
    for i in range(WARMUP + N_TIMED):
        im = imgs[order[i % len(order)]]
        t0 = time.perf_counter()
        r = model.predict(im, **kw)[0]
        _result_arrays(r)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= WARMUP:
            lat.append(dt)
            speed.append(dict(r.speed))
    # 실제로 쓴 스레드 수는 측정 뒤에 기록 (자기 프로세스 포함 CPU 사용률도)
    rec["cpu_busy_pct_during"] = float(psutil.cpu_percent(interval=None)) if psutil else None
    rec["torch_threads"] = torch.get_num_threads()
    rec["cv2_threads"] = cv2.getNumThreads()
    rec["ort_intra_op_threads"] = _ort_intra_threads(model)
    if v["device"] == "cpu" and threads > 0:
        assert rec["torch_threads"] == threads, (rec["torch_threads"], threads)
        if v["fmt"] == "onnx":
            assert rec["ort_intra_op_threads"] == threads, (rec["ort_intra_op_threads"], threads)
    rec["providers"] = _providers(model)
    rec["backend_fp16"] = bool(getattr(model.predictor.model, "fp16", False))
    if v["fmt"] == "onnx" and v["device"] == "gpu" and "CUDAExecutionProvider" not in (rec["providers"] or []):
        rec["error"] = f"CUDAExecutionProvider not used (providers={rec['providers']})"
    rec["latency_ms"] = [round(x, 3) for x in lat]
    rec["stats"] = latency_stats(lat)
    rec["speed_mean_ms"] = {k: round(sum(s[k] for s in speed) / len(speed), 3) for k in ("preprocess", "inference",
                                                                                         "postprocess")}
    if args.variant in FALLBACK_COST_VARIANTS and not resize:
        rec["fallback_cost"] = _fallback_cost(model, imgs, order, kw)
    rec["gpu_after"] = gpu_state()
    if resize:
        out = RAW_DIR / f"hires_{args.variant}_{resize[0]}x{resize[1]}_r{args.round}.json"
    else:
        out = RAW_DIR / f"lat_{args.variant}_t{threads}_r{args.round}.json"
    _write_json(out, rec)
    print(json.dumps({k: rec[k] for k in ("variant", "threads", "round", "stats", "speed_mean_ms", "providers")}
                     | ({"fallback_cost": rec["fallback_cost"]["stats"]} if "fallback_cost" in rec else {})))
    return 0


def _ort_intra_threads(model) -> Optional[int]:
    """ONNX Runtime 세션의 intra-op 스레드 설정 (0 = ORT 기본). ORT 백엔드가 아니면 None"""
    try:
        return int(model.predictor.model.session.get_session_options().intra_op_num_threads)
    except Exception:
        return None


def _fallback_cost(model, imgs, order, kw) -> Dict[str, Any]:
    """회전 폴백 시도 1번 = 90/270도 회전 2장 묶음 predict 1번 (live.py _predict_arrays) + 결과를 CPU로"""
    from app.engine.rotate_fallback import FALLBACK_ANGLES, rotate_image
    lat = []
    for i in range(FB_WARMUP + FB_TIMED):
        im = imgs[order[i % len(order)]]
        t0 = time.perf_counter()
        rs = model.predict([rotate_image(im, a) for a in FALLBACK_ANGLES], **kw)
        for r in rs:
            _result_arrays(r)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= FB_WARMUP:
            lat.append(dt)
    return {"angles": list(FALLBACK_ANGLES), "latency_ms": [round(x, 3) for x in lat], "stats": latency_stats(lat)}


# ---------------------------------------------------------------------------
# 회전 폴백 빈도 (v3_fb 키포인트)
# ---------------------------------------------------------------------------

def cmd_fallback_stats(args) -> int:
    import numpy as np
    rows = {json.loads(line)["clip_id"]: json.loads(line) for line in open(MANIFEST, encoding="utf-8") if line.strip()}
    agg: Dict[str, Dict[str, int]] = {}
    mismatch = 0
    n_clips = 0

    def add(key, **kv):
        a = agg.setdefault(key, {"clips": 0, "frames": 0, "attempts": 0, "successes": 0, "meta_fallback": 0})
        for k, v in kv.items():
            a[k] += int(v)

    for p in sorted(KP_DIR.rglob("*.npz")):
        d = np.load(p, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        if d["fallback"].shape != d["present"].shape:
            continue
        fps = float(meta["native_fps"])
        att = fallback_attempts(d["present"].tolist(), d["n_cand"].tolist(), fps)
        fb = d["fallback"].tolist()
        # 성공 프레임은 반드시 시도 프레임 (재구성 확인)
        mismatch += sum(1 for a, s in zip(att, fb) if s and not a)
        ds = meta["dataset"]
        row = rows.get(meta["clip_id"], {})
        n = len(att)
        kv = {"clips": 1, "frames": n, "attempts": sum(att), "successes": sum(fb),
              "meta_fallback": int(meta.get("n_fallback_frames") or 0)}
        group = "eval" if ds in EVAL_DATASETS else "train_only"
        for key in (ds, group, "all"):
            add(key, **kv)
        # 낙상 클립의 낙상 시작 이후 / ADL 클립 (평가 데이터셋)
        if ds in EVAL_DATASETS:
            if row.get("label") == "fall" and row.get("fall_start_s") is not None:
                i0 = min(n, max(0, int(math.floor(float(row["fall_start_s"]) * fps))))
                add("eval_fall_after_onset", clips=1, frames=n - i0, attempts=sum(att[i0:]), successes=sum(fb[i0:]),
                    meta_fallback=0)
                add("eval_fall_before_onset", clips=1, frames=i0, attempts=sum(att[:i0]), successes=sum(fb[:i0]),
                    meta_fallback=0)
            elif row.get("label") == "adl":
                add("eval_adl", clips=1, frames=n, attempts=sum(att), successes=sum(fb), meta_fallback=0)
        n_clips += 1
    for a in agg.values():
        a["attempt_rate"] = round(a["attempts"] / a["frames"], 5) if a["frames"] else None
        a["success_rate_per_frame"] = round(a["successes"] / a["frames"], 5) if a["frames"] else None
        a["success_per_attempt"] = round(a["successes"] / a["attempts"], 4) if a["attempts"] else None
    out = {"source": KP_DIR.relative_to(REPO_ROOT).as_posix(), "clips": n_clips, "recent_s": FALLBACK_RECENT_S,
           "success_not_attempt_frames": mismatch, "groups": agg,
           "note": "시도 = 원래 방향 후보 0명이고 직전 5초 안에 대상을 본 프레임 (n_cand/present로 재구성). "
                   "추출은 8프레임 묶음이었지만 시도/성공은 프레임 단위라 운영과 같음."}
    _write_json(RAW_DIR / "fallback_stats.json", out)
    print(json.dumps({k: {kk: v[kk] for kk in ("frames", "attempts", "successes", "attempt_rate",
                                                 "success_per_attempt")} for k, v in agg.items()}, indent=1))
    print("success_not_attempt_frames", mismatch)
    return 0


# ---------------------------------------------------------------------------
# 분류기 (v2) AI 호출 지연
# ---------------------------------------------------------------------------

def _classifier_windows(n: int = CLS_WINDOWS):
    """실제 낙상 클립에서 낙상 시작 1.5초 전부터 0.5초 간격 4초 창 n개 (window_v2, 표준화 전) + 특징 계산 시간"""
    import numpy as np
    from app.ai_classifier.features import window_v2
    clip_id = "le2i/Coffee_room_01/video_6"
    rows = {json.loads(line)["clip_id"]: json.loads(line) for line in open(MANIFEST, encoding="utf-8") if line.strip()}
    d = np.load(KP_DIR / "le2i" / "Coffee_room_01" / "video_6.npz", allow_pickle=False)
    t0 = max(0.0, float(rows[clip_id]["fall_start_s"]) - 1.5)
    args = (d["ts"], d["kps"], d["bbox"], d["present"], d["frame_wh"])
    lat = []
    for _ in range(3 + 20):
        s = time.perf_counter()
        X = np.stack([window_v2(*args, t0 + 0.5 * i, t0 + 0.5 * i + 4.0) for i in range(n)]).astype(np.float32)
        lat.append((time.perf_counter() - s) * 1000.0)
    return X, clip_id, latency_stats(lat[3:])


def _time_calls(fn: Callable[[], Any], warmup: int = CLS_WARMUP, n: int = CLS_TIMED) -> Dict[str, Any]:
    lat = []
    for i in range(warmup + n):
        s = time.perf_counter()
        fn()
        if i >= warmup:
            lat.append((time.perf_counter() - s) * 1000.0)
    return latency_stats(lat)


def cmd_classifier(args) -> int:
    threads = int(args.threads)
    _limit_threads(threads)
    import numpy as np
    import onnxruntime as ort
    import torch
    from app.ai_classifier.inference import FallClassifier
    from app.ai_classifier.scoring import nn_batch_scorer

    X, clip_id, feat_stats = _classifier_windows()
    rec: Dict[str, Any] = {"threads": threads, "torch_threads": torch.get_num_threads(), "windows": int(X.shape[0]),
                           "window_shape": list(X.shape[1:]), "clip": clip_id, "feature_build_ms": feat_stats,
                           "cpu_busy_pct_before": _cpu_busy_pct(), "gpu_before": gpu_state(), "results": {}}
    ps: Dict[str, Any] = {}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() and threads == 0 else [])
    for dev in devices:
        clf = FallClassifier(model_path=str(CLS_WEIGHTS), device=torch.device(dev))
        if not clf.is_v2:
            raise RuntimeError(f"classifier not v2: {clf.load_error}")
        rec["results"][f"torch_{dev}_per_window"] = _time_calls(lambda: clf.score_windows(X))
        ps[f"torch_{dev}"] = clf.score_windows(X)
        batched = nn_batch_scorer(clf.model, device=torch.device(dev), preprocess=clf.std, batched=True)
        rec["results"][f"torch_{dev}_batch16"] = _time_calls(lambda: batched(X))
    onnx_path = MODELS_DIR / "fall_cnn_gap_v2.onnx"
    so = ort.SessionOptions()
    if threads > 0:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    def ort_per_window():
        return np.asarray([sess.run(["p_fall"], {"x": x[None]})[0][0] for x in X])

    def ort_batch():
        return sess.run(["p_fall"], {"x": X})[0]
    rec["results"]["ort_cpu_per_window"] = _time_calls(ort_per_window)
    rec["results"]["ort_cpu_batch16"] = _time_calls(ort_batch)
    p_ort = ort_per_window()
    rec["max_abs_diff_p_torch_cpu_vs_ort_cpu"] = float(np.max(np.abs(np.asarray(ps["torch_cpu"]) - p_ort)))
    rec["p_fall_max"] = float(np.max(p_ort))
    rec["onnx_size_kb"] = round(onnx_path.stat().st_size / 1024, 1)
    _write_json(RAW_DIR / f"classifier_t{threads}.json", rec)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------------------
# 전체 실행과 보고
# ---------------------------------------------------------------------------

def _sub_env(needs: Optional[str], ort_gpu_path: Optional[str], trt_path: Optional[str]) -> Dict[str, str]:
    env = dict(os.environ)
    env["YOLO_AUTOINSTALL"] = "False"
    env["PYTHONIOENCODING"] = "utf-8"
    extra = [str(REPO_ROOT)]
    if needs == "ort_gpu" and ort_gpu_path:
        extra.insert(0, ort_gpu_path)
    if needs == "trt" and trt_path:
        extra.insert(0, trt_path)
    env["PYTHONPATH"] = os.pathsep.join(extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def _available(name: str, ort_gpu_path: Optional[str], trt_path: Optional[str]) -> Optional[str]:
    """측정 못 하는 이유 (측정 가능하면 None)"""
    v = VARIANTS[name]
    if v.get("needs") == "ort_gpu" and not ort_gpu_path:
        return "onnxruntime-gpu 경로 없음 (--ort-gpu-path)"
    if v.get("needs") == "trt" and not trt_path:
        return "TensorRT 없음 (--trt-path)"
    if not (MODELS_DIR / v["file"]).exists():
        return f"모델 파일 없음 ({v['file']})"
    return None


def cmd_all(args) -> int:
    py = sys.executable
    base = [py, "-m", "training.bench.pose_bench"]
    env0 = _sub_env(None, None, None)
    if not FRAMES_JSON.exists():
        subprocess.run(base + ["frames"], check=True, cwd=REPO_ROOT, env=env0)
    exp = base + ["export"] + (["--trt"] if args.trt_path else [])
    subprocess.run(exp, check=True, cwd=REPO_ROOT, env=_sub_env("trt", None, args.trt_path))
    subprocess.run(base + ["fallback-stats"], check=True, cwd=REPO_ROOT, env=env0)
    names = [n for n in (args.variants or list(VARIANTS))]
    skipped = {n: r for n in names if (r := _available(n, args.ort_gpu_path, args.trt_path))}
    run_names = [n for n in names if n not in skipped]
    if REFERENCE not in run_names and not (RAW_DIR / f"pred_{REFERENCE}.json").exists():
        raise RuntimeError(f"reference predictions missing: run {REFERENCE} first")
    # GPU 블록(2회) 전부를 CPU 블록보다 먼저: CPU 전 코어 측정 뒤 열/전력 상태가 GPU 측정에 섞이지 않게
    gpu_names = sorted((n for n in run_names if VARIANTS[n]["device"] == "gpu"),
                       key=lambda n: GPU_ORDER.index(n) if n in GPU_ORDER else len(GPU_ORDER))
    cpu_names = [n for n in run_names if VARIANTS[n]["device"] == "cpu"]
    hires = [] if args.no_hires else [n for n in HIRES_VARIANTS if n in gpu_names]
    jobs: List[Tuple[str, int, int, Optional[str]]] = []   # (변형, 스레드, 회차, resize)
    for rnd in range(1, int(args.rounds) + 1):
        order = gpu_names if rnd % 2 == 1 else list(reversed(gpu_names))
        jobs += [(n, 0, rnd, None) for n in order]
        h = hires if rnd % 2 == 1 else list(reversed(hires))
        jobs += [(n, 0, rnd, f"{HIRES_WH[0]}x{HIRES_WH[1]}") for n in h]
    for rnd in range(1, int(args.rounds) + 1):
        order = cpu_names if rnd % 2 == 1 else list(reversed(cpu_names))
        jobs += [(n, th, rnd, None) for n in order for th in CPU_THREADS]
    for n, th, rnd, resize in jobs:
        v = VARIANTS[n]
        # 안정 대기 뒤 유휴 확인 (대기 중에 다른 작업이 시작된 경우를 거름)
        time.sleep(float(args.settle_s) if v["device"] == "gpu" else float(args.cooldown_s))
        st = wait_gpu_idle()
        cst = wait_cpu_idle()
        cmd = base + ["run", "--variant", n, "--threads", str(th), "--round", str(rnd)]
        if resize:
            cmd += ["--resize", resize]
        elif rnd == 1 and not (RAW_DIR / f"pred_{n}.json").exists() and                 th == (0 if v["device"] == "gpu" else CPU_THREADS[0]):
            cmd.append("--dump")   # 예측은 없을 때만 저장 (기준 예측을 다시 쓰지 않음)
        print(f"[bench] round {rnd} {n} threads={th} resize={resize} gpu={st} cpu={cst}", flush=True)
        p = subprocess.run(cmd, cwd=REPO_ROOT, env=_sub_env(v.get("needs"), args.ort_gpu_path, args.trt_path))
        if p.returncode != 0:
            _write_json(RAW_DIR / f"fail_{n}_t{th}_r{rnd}.json", {"variant": n, "threads": th, "round": rnd,
                                                                 "resize": resize, "returncode": p.returncode})
    if not args.skip_classifier:
        for th in CPU_THREADS:
            wait_gpu_idle()
            subprocess.run(base + ["classifier", "--threads", str(th)], cwd=REPO_ROOT, env=env0)
    prev = _read_json(RAW_DIR / "skipped.json") if (RAW_DIR / "skipped.json").exists() else {}
    prev = {k: r for k, r in prev.items() if k not in names}
    _write_json(RAW_DIR / "skipped.json", {**prev, **skipped})
    return cmd_report(args)


def _pool_latency(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    lat = [x for r in recs for x in r["latency_ms"]]
    out = latency_stats(lat)
    out["rounds_p50_ms"] = [r["stats"]["p50_ms"] for r in sorted(recs, key=lambda r: r["round"])]
    out["round_spread"] = round_spread(out["rounds_p50_ms"])
    out["unstable"] = out["round_spread"] is not None and out["round_spread"] > UNSTABLE_REL
    out["cpu_busy_pct_during"] = [r.get("cpu_busy_pct_during") for r in sorted(recs, key=lambda r: r["round"])]
    out["speed_mean_ms"] = {k: round(sum(r["speed_mean_ms"][k] for r in recs) / len(recs), 3)
                            for k in ("preprocess", "inference", "postprocess")}
    return out


def _agreement(name: str, frames: List[Dict[str, Any]], ref_name: str = REFERENCE) -> Optional[Dict[str, Any]]:
    ref_p, var_p = RAW_DIR / f"pred_{ref_name}.json", RAW_DIR / f"pred_{name}.json"
    if not (ref_p.exists() and var_p.exists()):
        return None
    ref, var = _read_json(ref_p)["preds"], _read_json(var_p)["preds"]
    recs = [frame_agreement(a, b) for a, b in zip(ref, var)]
    out = {"all": summarize_agreement(recs)}
    for s in ("fallen", "upright"):
        out[s] = summarize_agreement([r for r, f in zip(recs, frames) if f["set"] == s])
    return out


def _f(v, nd=1):
    return None if v is None else f"{v:.{nd}f}"


def cmd_report(args) -> int:
    frames = _read_json(FRAMES_JSON)["frames"]
    skipped = _read_json(RAW_DIR / "skipped.json") if (RAW_DIR / "skipped.json").exists() else {}
    export_info = _read_json(RAW_DIR / "export_info.json") if (RAW_DIR / "export_info.json").exists() else {}
    pose: Dict[str, Any] = {}
    for name, v in VARIANTS.items():
        recs = [_read_json(p) for p in sorted(RAW_DIR.glob(f"lat_{name}_t*_r*.json"))]
        fails = [_read_json(p) for p in sorted(RAW_DIR.glob(f"fail_{name}_t*_r*.json"))]
        entry: Dict[str, Any] = {k: v[k] for k in ("desc", "model", "fmt", "prec", "device")}
        errs = [r["error"] for r in recs if r.get("error")]
        good = [r for r in recs if not r.get("error")]
        if not good:
            entry["status"] = "not_measured"
            entry["reason"] = skipped.get(name) or (errs[0] if errs else ("run failed" if fails else "not run"))
            pose[name] = entry
            continue
        entry["status"] = "ok"
        entry["protocol_v2"] = all("cv2_threads" in r for r in good)
        entry["model_size_mb"] = good[0]["model_size_mb"]
        entry["providers"] = good[0].get("providers")
        entry["latency"] = {}
        for th in sorted({r["threads"] for r in good}):
            key = "gpu" if v["device"] == "gpu" else ("t4" if th == 4 else f"t{th}" if th else "all")
            sel = [r for r in good if r["threads"] == th]
            entry["latency"][key] = _pool_latency(sel)
            # 측정 뒤 기록한 실제 스레드 (옛 기록은 측정 전 값이라 표시하지 않음)
            measured_after = all("cv2_threads" in r for r in sel)
            entry["latency"][key]["torch_threads"] = sel[0]["torch_threads"] if measured_after else None
            entry["latency"][key]["cv2_threads"] = sel[0].get("cv2_threads")
            entry["latency"][key]["ort_intra_op_threads"] = sel[0].get("ort_intra_op_threads")
        fb = [r["fallback_cost"] for r in good if "fallback_cost" in r]
        if fb:
            lat = [x for c in fb for x in c["latency_ms"]]
            entry["fallback_cost"] = latency_stats(lat)
        entry["gpu_state"] = [{"round": r["round"], "threads": r["threads"], "before": r.get("gpu_before"),
                               "after": r.get("gpu_after"), "cpu_busy_pct_before": r.get("cpu_busy_pct_before")}
                              for r in good]
        if name != REFERENCE:
            entry["agreement_vs_fp32"] = _agreement(name, frames)
            if v["fmt"] != "pt":  # 같은 정적 640 입력의 PyTorch FP32 대비 (형식/정밀도만의 차이)
                entry["agreement_vs_fp32_square"] = _agreement(name, frames, SQUARE_CONTROL)
        else:
            entry["agreement_vs_fp32"] = None
            a = _agreement(name, frames)
            entry["detect_rate"] = {k: a[k]["ref_det_rate"] for k in ("all", "fallen", "upright")} if a else None
        pose[name] = entry
    hires: Dict[str, Any] = {}
    for name in HIRES_VARIANTS:
        recs = [_read_json(p) for p in sorted(RAW_DIR.glob(f"hires_{name}_*_r*.json"))]
        e = pose.get(name, {})
        if recs and e.get("status") == "ok":
            hires[name] = {"frame_wh": recs[0]["frame_wh"], "latency": _pool_latency(recs),
                           "native": e["latency"]["gpu"]}
    fbs = _read_json(RAW_DIR / "fallback_stats.json") if (RAW_DIR / "fallback_stats.json").exists() else None
    fallback: Dict[str, Any] = {"stats": fbs, "effective": {}}
    if fbs:
        rates = {k: fbs["groups"][k]["attempt_rate"] for k in ("all", "eval", "eval_adl", "eval_fall_after_onset")
                 if k in fbs["groups"]}
        rates["worst_every_frame"] = 1.0
        for name in FALLBACK_COST_VARIANTS:
            e = pose.get(name, {})
            if e.get("status") != "ok" or "fallback_cost" not in e:
                continue
            base_ms, fb_ms = e["latency"]["gpu"]["mean_ms"], e["fallback_cost"]["mean_ms"]
            fallback["effective"][name] = {"base": effective_ms(base_ms, fb_ms, 0.0), "fb_attempt_ms": fb_ms,
                                           "by_rate": {k: dict(effective_ms(base_ms, fb_ms, r), rate=r)
                                                       for k, r in rates.items()}}
    cls = {}
    for p in sorted(RAW_DIR.glob("classifier_t*.json")):
        r = _read_json(p)
        cls["t4" if r["threads"] == 4 else "all"] = r
    results = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": host_info(),
        "protocol": {"imgsz": IMGSZ, "conf": CONF, "warmup": WARMUP, "timed": N_TIMED,
                     "rounds": "2 (2회째는 변형 순서를 뒤집음), 두 회 표본을 합쳐 통계",
                     "latency": "model.predict 1회 (전처리+추론+후처리) + 결과 CPU numpy 변환, 프레임 디코딩 제외",
                     "rounds_order": "GPU 변형 2회를 먼저 (n/s 번갈아), 그 뒤 CPU 변형 2회. 실행마다 GPU 30초 / CPU 10초 "
                                     "대기 뒤 GPU/CPU 유휴 확인 (CPU 5초 평균 10% 이하). pose.*.protocol_v2=false인 변형은 이 순서 도입 전 "
                                     "측정 (CPU 블록과 GPU 블록이 섞인 순서, 대기 5초)",
                     "unstable": f"회차 p50 상대 차이 > {UNSTABLE_REL:.0%}면 불안정, FP32 대비 비율 생략",
                     "cpu_threads": "t4 = torch intra-op 4 (첫 predict 뒤 다시 맞춤) / ORT intra-op 4, inter 1. "
                                    "all = 라이브러리 기본: torch는 ultralytics가 8로 맞춤, ORT는 자체 기본. "
                                    "cv2는 모든 변형에서 ultralytics가 setNumThreads(0)으로 둔 값(운영과 같음). "
                                    "실제 값은 측정 뒤 기록 (pose.*.latency.*.torch_threads 등)",
                     "frames": "낙상 후 150 (eval_v1 val) + 낙상 전 서 있음 150, 지연은 섞은 순서의 앞 220장. "
                               "데이터셋 해상도 그대로 (최대 1280x720, 대부분 720x480/320x240)",
                     "hires": f"{HIRES_VARIANTS}만 같은 300장을 {HIRES_WH[0]}x{HIRES_WH[1]}로 키워 따로 잼 "
                              "(운영 카메라 기본 해상도, 지연만)",
                     "agreement": "FP32 운영 모델(s_pt_fp32_gpu) 출력과의 일치 (정답 대비 정확도 아님)",
                     "oks": "COCO sigma, s^2 = 기준 상자 넓이, 기준 관절 conf >= 0.5만",
                     "onnx_input": "ONNX/TensorRT는 정적 640x640 (PyTorch는 rect letterbox, 예: 640x480)"},
        "frames": {"n": len(frames), "fallen": sum(f["set"] == "fallen" for f in frames),
                   "upright": sum(f["set"] == "upright" for f in frames)},
        "export": export_info,
        "pose": pose,
        "hires": hires,
        "fallback": fallback,
        "classifier": cls,
    }
    _write_json(OUT_DIR / "results.json", results)
    md = render_summary(results)
    with open(OUT_DIR / "summary.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(md)
    return 0


def _lat(e: Dict[str, Any], key: str = "gpu") -> Optional[Dict[str, Any]]:
    if not e or e.get("status") != "ok":
        return None
    return (e.get("latency") or {}).get(key)


def _rounds(s: Optional[Dict[str, Any]]) -> Optional[str]:
    """회차 p50 'a / b' (+ 불안정 표시)"""
    if not s:
        return None
    txt = " / ".join(_f(x) for x in s.get("rounds_p50_ms") or [])
    if s.get("unstable"):
        txt += f" (불안정, 차이 {100 * s['round_spread']:.0f}%)"
    return txt


def _cmp_word(a: float, b: float, tol: float = 0.1) -> str:
    """a가 b보다 빠름/느림/비슷함 (상대 tol 안이면 비슷함)"""
    r = a / b
    return "비슷함" if abs(r - 1.0) <= tol else ("빠름" if r < 1.0 else "느림")


def _threads_cell(e: Dict[str, Any]) -> str:
    """CPU 표의 실제 스레드 '4스레드 / 기본'"""
    out = []
    for key in ("t4", "all"):
        s = (e.get("latency") or {}).get(key) or {}
        if e["fmt"] == "pt":
            out.append(str(s["torch_threads"]) if s.get("torch_threads") else "?")
        else:   # ORT 세션 옵션 (0 = ORT 기본). 옛 기록은 설정값
            v = s.get("ort_intra_op_threads")
            out.append("ORT 기본" if key == "all" and not v else str(v or 4))
    return " / ".join(out)


def render_summary(res: Dict[str, Any]) -> str:
    pose = res["pose"]
    ref = pose.get(REFERENCE, {})
    ref_lat = _lat(ref) or {}
    ref_ms = ref_lat.get("mean_ms")
    h = res.get("host", {})
    L = ["# 추론 최적화 비교 (노트북, bench_v1)", "",
         f"측정 {res['created_at']}. {h.get('gpu', '')} / {h.get('cpu', '')} (논리 {h.get('cpu_count_logical')}, "
         f"물리 {h.get('cpu_count_physical')}) / RAM {h.get('ram_gb')} GB / 전원 연결 {h.get('power_plugged')}.",
         "Jetson 값이 아니라 같은 노트북에서 모델 사이를 비교한 값입니다. "
         "imgsz 640, conf 0.5, 실제 프레임, 워밍업 20장 뒤 200장 x 2회(순서 뒤집음). "
         "GPU 변형 2회를 CPU 변형보다 먼저 재고, 실행마다 GPU 30초 / CPU 10초 기다린 뒤 GPU/CPU 유휴(CPU 5초 평균 10% 이하)를 확인.", "",
         "## 포즈 모델 지연 (GPU)", "",
         f"회차 p50 차이가 {UNSTABLE_REL:.0%}를 넘으면 불안정으로 보고 FP32 대비 속도를 내지 않음.", ""]
    rows = []
    for n, e in pose.items():
        if e["device"] != "gpu":
            continue
        if e.get("status") != "ok":
            rows.append([n, e["desc"]] + [None] * 6 + [f"측정 안 함: {e.get('reason')}"])
            continue
        s = e["latency"]["gpu"]
        sp = s["speed_mean_ms"]
        if s.get("unstable") or ref_lat.get("unstable"):
            ratio = "생략 (불안정)"
        else:
            ratio = _f(ref_ms / s["mean_ms"], 2) if ref_ms else None
        rows.append([n, e["desc"], _f(s["mean_ms"]), _f(s["p50_ms"]), _f(s["p95_ms"]), _f(s["fps"]), _rounds(s),
                     ratio, f"전/추/후 {sp['preprocess']:.1f}/{sp['inference']:.1f}/{sp['postprocess']:.1f}, "
                            f"{e['model_size_mb']} MB"])
    L += [md_table(["변형", "설명", "평균 ms", "p50", "p95", "FPS", "회차 p50 (1 / 2)", "FP32 대비 속도", "비고"],
                   rows), ""]
    hi = res.get("hires") or {}
    if hi:
        L += [f"## 해상도 영향 (같은 300장을 {HIRES_WH[0]}x{HIRES_WH[1]}로 키움, 운영 카메라 기본)", "",
              "프레임 집합은 데이터셋 해상도(최대 1280x720)라 운영 1080p보다 전처리(letterbox 축소, 복사)가 가벼움.", ""]
        rows = []
        for n, x in hi.items():
            a0, a1 = x["native"], x["latency"]
            rows.append([n, _f(a0["mean_ms"]), _f(a0["p50_ms"]), _f(a1["mean_ms"]), _f(a1["p50_ms"]), _rounds(a1),
                         _f(a1["mean_ms"] - a0["mean_ms"], 2),
                         f"{a0['speed_mean_ms']['preprocess']:.2f} -> {a1['speed_mean_ms']['preprocess']:.2f}"])
        L += [md_table(["변형", "데이터셋 해상도 평균 ms", "p50", "1080p 평균 ms", "p50", "1080p 회차 p50",
                        "평균 차이 ms", "전처리 ms (데이터셋 -> 1080p)"], rows), ""]
    L += ["## 포즈 모델 지연 (CPU)", "",
          "기본 = 라이브러리 기본 스레드: PyTorch는 ultralytics가 첫 predict에서 8로 맞춤, ONNX Runtime은 자체 기본. "
          "실제 스레드는 측정 뒤 기록한 값.", ""]
    rows = []
    for n, e in pose.items():
        if e["device"] != "cpu":
            continue
        if e.get("status") != "ok":
            rows.append([n, e["desc"]] + [None] * 10 + [f"측정 안 함: {e.get('reason')}"])
            continue
        r = [n, e["desc"]]
        notes = []
        for key, lab in (("t4", "4스레드"), ("all", "기본")):
            s = e["latency"].get(key)
            r += [_f(s["mean_ms"]), _f(s["p50_ms"]), _f(s["p95_ms"]), _f(s["fps"])] if s else [None] * 4
            if s and s.get("unstable"):
                notes.append(f"{lab} 불안정 (회차 p50 {_rounds(s)})")
        r += [_threads_cell(e), f"{e['model_size_mb']} MB", "; ".join(notes) or None]
        rows.append(r)
    L += [md_table(["변형", "설명", "4스레드 평균", "p50", "p95", "FPS", "기본 평균", "p50", "p95", "FPS",
                    "실제 스레드 (4 / 기본)", "크기", "비고"], rows), ""]
    L += ["## FP32 운영 모델과의 출력 일치도 (정답 대비 정확도 아님)", "",
          "기준 = yolov8s-pose PyTorch FP32 GPU 출력. 검출률은 conf 0.5 이상 사람이 1명 이상인 프레임 비율.",
          "OKS/IoU는 둘 다 검출한 프레임에서 기준 최고 신뢰도 사람과 그 상자에 가장 많이 겹치는 사람.", ""]
    det = ref.get("detect_rate") or {}
    rows = [[REFERENCE, _f(det.get("fallen"), 3), _f(det.get("upright"), 3), "1.000", "-", "-", "-", "-"]]
    for n, e in pose.items():
        a = e.get("agreement_vs_fp32")
        if not a:
            continue
        rows.append([n, _f(a["fallen"]["var_det_rate"], 3), _f(a["upright"]["var_det_rate"], 3),
                     _f(a["all"]["det_agree_rate"], 3), a["all"]["n_both"], _f(a["all"]["oks_mean"], 3),
                     _f(a["all"]["oks_ge_0p9_rate"], 3), _f(a["all"]["iou_mean"], 3)])
    L += [md_table(["변형", "검출률 낙상 후", "검출률 서 있음", "검출 여부 일치", "둘 다 검출", "평균 OKS",
                    "OKS>=0.9 비율", "평균 IoU"], rows), ""]
    rows = []
    for n, e in pose.items():
        a = e.get("agreement_vs_fp32_square")
        if a:
            rows.append([n, _f(a["fallen"]["var_det_rate"], 3), _f(a["all"]["det_agree_rate"], 3), a["all"]["n_both"],
                         _f(a["all"]["oks_mean"], 3), _f(a["all"]["oks_ge_0p9_rate"], 3), _f(a["all"]["iou_mean"], 3)])
    if rows:
        L += [f"ONNX/TensorRT는 입력이 정적 640x640이라, 같은 입력의 PyTorch FP32({SQUARE_CONTROL}) 대비로도 봄 "
              "(형식/정밀도만의 차이):", "",
              md_table(["변형", "검출률 낙상 후", "검출 여부 일치", "둘 다 검출", "평균 OKS", "OKS>=0.9 비율",
                        "평균 IoU"], rows), ""]
    fb = res.get("fallback") or {}
    st = (fb.get("stats") or {}).get("groups") or {}
    if st:
        L += ["## 회전 폴백 비용", "",
              "v3_fb 키포인트에서 다시 센 폴백 시도(원래 방향 후보 0명, 직전 5초 안에 대상을 봄)와 성공 프레임.", ""]
        rows = []
        for k, lab in (("all", "전체 2,726 클립"), ("eval", "평가 데이터셋"), ("eval_adl", "평가 ADL 클립"),
                       ("eval_fall_before_onset", "평가 낙상 클립, 낙상 전"),
                       ("eval_fall_after_onset", "평가 낙상 클립, 낙상 시작 뒤"), ("train_only", "학습 전용(Figshare)"),
                       *((d, f"  {d}") for d in EVAL_DATASETS)):
            g = st.get(k)
            if g:
                rows.append([lab, f"{g['frames']:,}", f"{g['attempts']:,}", _f(100 * g["attempt_rate"], 2),
                             f"{g['successes']:,}", _f(g["success_per_attempt"], 3)])
        L += [md_table(["구간", "프레임", "시도", "시도율 %", "성공", "시도당 성공"], rows), ""]
        rows = []
        for n, e in (fb.get("effective") or {}).items():
            by = e["by_rate"]
            rows.append([n, _f(e["base"]["ms"]), _f(e["fb_attempt_ms"])]
                        + [f"{_f(by[k]['ms'])} ms / {_f(by[k]['fps'])}" for k in
                           ("all", "eval_adl", "eval_fall_after_onset", "worst_every_frame") if k in by])
        L += ["시도 1번 = 90/270도 회전 2장을 한 번에 predict (live.py와 같음). 실효 = 기본 + 시도율 x 시도 지연.", "",
              md_table(["변형", "기본 ms", "시도 1번 ms", "실효 (전체 시도율)", "실효 (평가 ADL)",
                        "실효 (낙상 시작 뒤)", "최악 (매 프레임 시도)"], rows), ""]
    cls = res.get("classifier") or {}
    if cls:
        L += ["## 배포 분류기 (Fall1DCNN-GAP v2) AI 호출 1번 지연", "",
              "AI 호출 1번 = 4초 창 16개 점수 (E*max 프로토콜). 운영 경로는 창마다 배치 1 (per_window).", ""]
        rows = []
        keys = ["torch_cpu_per_window", "torch_cpu_batch16", "ort_cpu_per_window", "ort_cpu_batch16",
                "torch_cuda_per_window", "torch_cuda_batch16"]
        for k in keys:
            r = [k]
            for th in ("t4", "all"):
                s = (cls.get(th) or {}).get("results", {}).get(k)
                r += [_f(s["mean_ms"], 2), _f(s["p95_ms"], 2)] if s else [None, None]
            rows.append(r)
        L += [md_table(["경로", "4스레드 평균 ms", "p95", "전체 평균 ms", "p95"], rows), ""]
        c = cls.get("all") or cls.get("t4")
        L += [f"특징 계산(window_v2 16창, numpy): 평균 {_f(c['feature_build_ms']['mean_ms'], 2)} ms. "
              f"PyTorch CPU와 ONNX Runtime CPU의 p_fall 최대 차이 {c['max_abs_diff_p_torch_cpu_vs_ort_cpu']:.2e}. "
              f"ONNX 크기 {c['onnx_size_kb']} KB.", ""]
    L += ["## 관찰 (이번 측정의 해석, 가설은 가설로 표시)", ""] + _observations(res) + [""]
    L += ["## 주의", "",
          "- 노트북(RTX 4050 Laptop 6GB, Core Ultra 7 155H) 값이며 Jetson 지연이 아님. 형식/정밀도 사이 상대 비교로만 씀.",
          "- 같은 변형도 측정 세션에 따라 절대값이 달라짐: 이 순서(대기 뒤 측정) 도입 전 첫 측정에서는 s PyTorch FP32 GPU "
          "p50이 9.2 / 9.6 ms, PyTorch CPU 8스레드가 약 80 ms였음 (대기 없이 연달아 잰 경우). 표 안의 행끼리만 비교하고 "
          "절대값은 대략으로 봄. 분류기 표는 첫 측정 세션 값.",
          "- 노트북 GPU는 전원/온도에 따라 클럭이 바뀜. 측정 전후 GPU 상태와 측정 중 CPU 사용률(자기 프로세스 포함)은 "
          "results.json pose.*.gpu_state, pose.*.latency.*.cpu_busy_pct_during에 있음. GPU 사용률은 실행 전후 "
          "한 번씩만 잰 값이라 병목 판단 근거로 쓰지 않음.",
          "- CPU 4스레드는 저사양 엣지 대용일 뿐이며 하이브리드 CPU(P/E 코어) 배정은 OS가 정함. cv2는 모든 변형에서 "
          "ultralytics가 setNumThreads(0)으로 둔 값(운영 live.py와 같음)이고 4스레드 제한 대상이 아님.",
          "- 일치도는 FP32 운영 모델과 같은지를 본 값이지 정답 대비 정확도가 아님 (평가셋 초안 라벨은 쓰지 않음).",
          "- ONNX/TensorRT는 정적 640x640 입력, PyTorch는 rect letterbox(예: 640x480)라 입력 모양이 다름. "
          "ONNX FP32의 작은 불일치에는 이 차이가 포함됨.",
          "- INT8 = ONNX Runtime 정적 양자화, Conv/Gemm/MatMul만, 가중치 per-tensor(s_ort_int8u8pc_cpu만 per-channel), "
          "MinMax 보정 200장, 나머지 층 FP32 (TensorRT 엔진도 FP16 대체 없이 FP32). INT8 정확도와 속도는 이 레시피의 "
          "값이며 INT8 일반의 상한이 아님.",
          "- 프레임은 데이터셋 해상도(최대 1280x720, 대부분 720x480과 320x240)이고 운영 카메라 기본은 1920x1080. "
          "1080p에서는 letterbox 축소가 더 오래 걸리고(위 해상도 영향 표), 16:9 입력이면 PyTorch rect 입력은 항상 "
          "640x384인데 이번 프레임 대부분은 640x448/640x480이라 PyTorch 행은 운영보다 조금 무겁게, 고정 640x640인 "
          "ONNX/TensorRT의 속도 이득은 조금 낙관적으로 나왔을 수 있음.",
          "- 지연에 프레임 디코딩, 대상 선택, EMA, 규칙 엔진, JPEG 인코딩은 들어가지 않음.", ""]
    return "\n".join(L)


def _observations(res: Dict[str, Any]) -> List[str]:
    """관찰 문장 (결과 값으로 채움, 없는 값은 건너뜀)"""
    pose = res["pose"]
    O: List[str] = []

    def g(name, key="gpu"):
        return _lat(pose.get(name, {}), key)

    s32, s16, n32, trt16, trt8 = (g("s_pt_fp32_gpu"), g("s_pt_fp16_gpu"), g("n_pt_fp32_gpu"), g("s_trt_fp16_gpu"),
                                  g("s_trt_int8_gpu"))
    if s32 and n32:
        rs, rn = s32["rounds_p50_ms"], n32["rounds_p50_ms"]
        O.append(f"- yolov8n PyTorch FP32는 s와 속도가 {_cmp_word(n32['p50_ms'], s32['p50_ms'])} "
                 f"(회차 p50 n {' / '.join(_f(x) for x in rn)} ms, s {' / '.join(_f(x) for x in rs)} ms).")
    if s32 and n32 and s16 and trt16:
        inf = {k: v["speed_mean_ms"]["inference"] for k, v in (("n", n32), ("s", s32), ("s16", s16), ("t", trt16))}
        O.append(f"- 가설: GPU PyTorch 배치 1은 연산보다 커널 실행 준비(CPU 쪽)에 묶여 있을 가능성. 근거는 간접적임: "
                 f"ultralytics 추론 구간이 n {inf['n']:.1f} ms, s {inf['s']:.1f} ms, s FP16 {inf['s16']:.1f} ms이고 "
                 f"TensorRT FP16은 {inf['t']:.1f} ms. GPU 사용률은 실행 뒤 한 번만 잰 값이라 근거로 쓰지 않음 "
                 "(확인하려면 측정 중 사용률 표본이나 CUDA 이벤트 시간이 필요).")
    O.append("- ONNX Runtime/TensorRT FP16의 FP32 대비 불일치는 대부분 입력 모양(정적 640x640) 때문: 같은 입력의 PyTorch "
             "대조군과 비교하면 거의 같음.")

    def det_fallen(name):
        a = pose.get(name, {}).get("agreement_vs_fp32")
        return a["fallen"]["var_det_rate"] if a else None

    sq, pt8, pc8 = det_fallen(SQUARE_CONTROL), det_fallen("s_ort_int8u8_cpu"), det_fallen("s_ort_int8u8pc_cpu")
    if sq is not None and pt8 is not None:
        txt = (f"- INT8은 이 레시피(가중치 per-tensor, MinMax)에서 같은 입력 대조군 대비로도 관절 일치도가 떨어지고 "
               f"낙상 후(누운 자세) 검출률이 {pt8:.3f}로 대조군 {sq:.3f}보다 낮음.")
        if pc8 is not None:
            txt += f" 가중치만 per-channel로 바꾸면 {pc8:.3f}."
        if trt8 and trt16:
            txt += (f" TensorRT INT8({trt8['mean_ms']:.1f} ms)이 FP16({trt16['mean_ms']:.1f} ms)보다 "
                    f"{_cmp_word(trt8['mean_ms'], trt16['mean_ms'], 0.05)}: 양자화하지 않은 층을 FP32로 둔 이 빌드의 "
                    "값이며 INT8 자체의 한계로 보지 않음.")
        O.append(txt)
    c8, cu8, cpc, c32 = (g("s_ort_int8_cpu", "all"), g("s_ort_int8u8_cpu", "all"), g("s_ort_int8u8pc_cpu", "all"),
                         g("s_ort_fp32_cpu", "all"))
    if c8 and cu8 and c32:
        txt = (f"- ONNX Runtime CPU INT8(기본 스레드)은 활성 S8(ultralytics 기본)이면 {c8['mean_ms']:.1f} ms로 FP32 "
               f"{c32['mean_ms']:.1f} ms보다 느렸고, 활성 U8이면 {cu8['mean_ms']:.1f} ms (x86 CPU의 U8S8 경로)")
        txt += (f", U8 + per-channel은 {cpc['mean_ms']:.1f} ms." if cpc else ".")
        O.append(txt)
    p4, p0 = g("s_pt_fp32_cpu", "t4"), g("s_pt_fp32_cpu", "all")
    if p4 and p0:
        O.append(f"- PyTorch CPU 회차 p50: torch {p4.get('torch_threads') or '?'}스레드 {_rounds(p4)} ms, "
                 f"ultralytics 기본 {p0.get('torch_threads') or '?'}스레드 {_rounds(p0)} ms. 4스레드는 OS가 어느 코어"
                 "(P/E)에 두느냐에 따라 회차마다 크게 달라질 수 있음. ONNX Runtime CPU도 스레드 수에 따라 크게 바뀜.")
    hi = res.get("hires") or {}
    if hi:
        parts = [f"{n} 평균 {x['native']['mean_ms']:.1f} -> {x['latency']['mean_ms']:.1f} ms "
                 f"(전처리 {x['native']['speed_mean_ms']['preprocess']:.2f} -> "
                 f"{x['latency']['speed_mean_ms']['preprocess']:.2f}, 추론 구간 "
                 f"{x['native']['speed_mean_ms']['inference']:.1f} -> {x['latency']['speed_mean_ms']['inference']:.1f})"
                 for n, x in hi.items()]
        O.append("- 1920x1080으로 키운 프레임: " + ", ".join(parts) + ". 전처리는 늘지만 PyTorch는 16:9에서 rect 입력이 "
                 "640x384로 작아져 추론 구간이 줄어 상쇄되고, 고정 640x640인 TensorRT는 전처리/후처리 증가가 그대로 남음. "
                 "운영 해상도에서는 TensorRT의 이득이 이 표보다 조금 작음.")
    O += ["- 분류기 운영 경로(창마다 배치 1)는 CUDA가 CPU보다 느림 (작은 모델이라 호출마다 GPU 왕복이 큼). AI 호출 1번에서 "
          "특징 계산(numpy)이 점수 계산보다 오래 걸림. AI 호출은 사건 판정 때만 하므로 프레임 처리율에는 거의 영향 없음.",
          "- 회전 폴백 시도율은 데이터셋마다 크게 다름 (MCFD는 원거리/광각이라 사람을 자주 놓쳐 시도율이 높음)."]
    return O


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="포즈 모델 최적화 비교 (노트북)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("frames")
    p = sub.add_parser("export")
    p.add_argument("--trt", action="store_true", help="TensorRT 엔진도 만듦 (tensorrt import 가능할 때)")
    p.add_argument("--force-int8", action="store_true")
    p.add_argument("--int8-per-channel", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--dump", action="store_true", help="프레임 집합 예측을 raw/pred_<variant>.json에 저장")
    p.add_argument("--resize", default=None, help="프레임을 WxH로 키워 지연만 잼 (raw/hires_*.json)")
    p = sub.add_parser("classifier")
    p.add_argument("--threads", type=int, default=0)
    sub.add_parser("fallback-stats")
    sub.add_parser("report")
    p = sub.add_parser("all")
    p.add_argument("--ort-gpu-path", default=None)
    p.add_argument("--trt-path", default=None)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--cooldown-s", type=float, default=10.0, help="CPU 실행 전 대기")
    p.add_argument("--settle-s", type=float, default=30.0, help="GPU 실행 전 대기 (CPU 유휴 확인 뒤)")
    p.add_argument("--no-hires", action="store_true", help="1920x1080 확대 프레임 측정을 건너뜀")
    p.add_argument("--skip-classifier", action="store_true")
    args = ap.parse_args(argv)
    fn = {"frames": cmd_frames, "export": cmd_export, "run": cmd_run, "classifier": cmd_classifier,
          "fallback-stats": cmd_fallback_stats, "report": cmd_report, "all": cmd_all}[args.cmd]
    return int(fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
