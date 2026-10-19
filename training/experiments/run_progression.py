"""
진행 단계 비교 실험(progression study) 실행기 (CLI)

  A 규칙만 -> B 1D CNN + URFD(원래 방식) -> C + 데이터 추가 -> D + 라벨/학습-운영 정렬
  -> E + 모델/특징 변경 (E0 HistGradientBoosting, E1 Fall1DCNN-GAP, E2 FallTCN) -> F + 임계값 조정
  (E*: 저장된 E 모델을 진입 이후 여러 창에 적용, 재학습 없음; L: 학습 낙상 클립 10/25/50%만 쓴 E0/E1;
   G: 양성 창 없이 정상 창만으로 학습한 이상 탐지)
을 같은 5-fold held-out 클립, 같은 이벤트 단위 프로토콜(replay.py: 15Hz obs, EMA, 20초 정지 꼬리)로 평가합니다.

실행 예 (저장소 루트에서)
  python -m training.experiments.run_progression --stages A,B,C,D,E,F --folds 0-4 --seeds 3 --jobs 5
  python -m training.experiments.run_progression --stages Es --folds 0-4 --seeds 3 --jobs 5   (E 모델 필요)
  python -m training.experiments.run_progression --stages L --folds 0-4 --seeds 3 --jobs 5
  python -m training.experiments.run_progression --stages G --folds 0-4 --seeds 3 --jobs 5   (GnfE0는 E0 모델 필요)
  python -m training.experiments.run_progression --summarize-only [--fa-ref E1max]

출력 (--out, 기본 dataset/_experiments/progression_v1/)
  fold{k}/{variant}.json : 변형(A, B, C, D, E0, E1, E2, E0max.., F0, F1, F2, E0p10.., Gmd..)별 fold 결과
                           runs[seed]: 클립 결과(outcomes), 요약(summary), 창 단위 지표(window), 학습 기록 요약
  fold{k}/models/        : E 모델 (F, Es 단계가 다시 읽음), L 모델 ({variant}_s{seed}.pkl/.pt, E와 같은 형식),
                           G 탐지기 (G{kind}_s{seed}.pt: {"kind", "detector": Detector.state(), "calibrator"})
  logs/fold{k}.log       : --jobs > 1일 때 fold 프로세스 로그
  summary.json, summary.md : 전체/데이터셋별 표

집계 규칙
- pooled: 5개 fold의 test 클립을 합쳐 계산 (seed별로 계산 후 seed 평균, 개수도 seed 평균이라 소수가 될 수 있음)
- fold mean ± std: fold마다 seed 평균을 낸 뒤 fold 간 평균과 표본 표준편차(ddof=1)
- AI 호출 AUC: 규칙이 LEVEL1 판정까지 간 클립의 첫 p_fall과 클립 라벨 (pooled, seed 평균)
- 창 AUC: 각 단계 자신의 창 생성기로 만든 test fold 창 (fold mean ± std)

F 임계값: E 모델마다(seed별) 검증 클립(학습 fold 안 15% 그룹, 평가 데이터셋만)을 shadow 모드로 재생해
  - F0/F1/F2 (스펙): LEVEL1 판정에 도달한 낙상 클립의 첫 p_fall 최솟값 - margin(기본 0.05)을 억제 임계값으로 씀
    (검증 낙상 recall 1.0 유지, 0~0.95로 자름, 도달한 낙상이 없으면 0.4)
  - F0b/F1b/F2b (참고): 검증 balanced(recall과 1 - ADL 오탐률의 조화평균)가 가장 큰 임계값
  온도 보정(temperature scaling)은 단조 변환이라 임계값을 다시 고르면 판정이 같으므로 하지 않습니다.

E* (단계 키 Es, 재학습 없음, 임계값 0.4 그대로): verifiers.SlidingWindowVerifier
  - E0max/E1max/E2max: 진입 사건 구간(E와 같은 첫 창)부터 LEVEL1 판정 시점 - 4초까지 0.5초 간격 4초 창들의 최대 p_fall
    (진입 직후의 이른 창이 p~0이어도 같은 에피소드 안의 뒤 창이 낙상이면 통과)
  - E0last/E1last/E2last (참고): 판정 시점 직전 [ts - 4, ts] 한 창. 정지 꼬리에서는 같은 프레임 반복이라
    E 모델이 학습 때 보지 못한 입력
  E와 같은 fold/seed 모델끼리 클립 단위로 비교한 paired 표를 summary에 함께 씁니다 (회복된 억제 낙상, 새 ADL 알람).

L (라벨 효율): E0p10/E0p25/E0p50, E1p10/E1p25/E1p50
  - E와 같은 학습 창(같은 증강 rng)에서 낙상 클립을 clip_id 단위로 줄임: 남긴 낙상 클립은 그 클립의 창을 모두
    (양성 창과 낙상 전 음성 창), 뺀 낙상 클립은 낙상 전 음성 창까지 모두 뺌 (낙상 전 구간도 fall_start 라벨로 자르므로).
    ADL 클립 창은 모두 사용. 중첩 표본: 전체 -> 50% -> 25% -> 10% 순서로 바로 위 집합에서
    random.Random("{fold}-{seed}-{f}")로 뽑음 (10% ⊂ 25% ⊂ 50%). 검증 창은 모두 그대로.
    E0는 E와 같은 _make_hgb (early_stopping 명시). 판정은 E*max와 같은 SlidingWindowVerifier mode max, 0.4
  - 라벨 효율 표: 0% = 이상 탐지 G 변형 (결과가 있으면), 10/25/50% = L, 100% = E0max/E1max

G (이상 탐지, training/anomaly): Gmd (Mahalanobis), Gae (1D conv 오토인코더), Gnf (STG-NF), GnfE0
  - 학습: E와 같은 학습 창(같은 증강 rng)에서 y == 0 창만 (양성 창은 학습/보정에 쓰지 않음). 다만 정상 집합에는
    낙상 클립의 낙상 전 창(pre_fall, entry_pre_fall)이 들어가고, 이 창은 fall_start 라벨로 자른 것이라
    낙상 시각 라벨을 전혀 안 쓰는 것은 아님. GnfS는 ADL 클립의 창만으로 학습·보정해 낙상 라벨을 전혀 쓰지 않음.
    검증 정상 창(va_rows, 증강 없음, y == 0; GnfS는 ADL 클립 창)으로 early stopping과 보정
    (Calibrator q=0.8 -> p 0.4, 사전 등록)
  - 판정: 보정한 의사 p_fall로 E*max와 같은 SlidingWindowVerifier mode max, 0.4. 점수가 NaN인 창(STG-NF에서
    관절 유효 50% 이상 세그먼트가 없는 창 등)은 건너뛰고, 모든 창이 NaN이면 None (fail-open): NanSafeSlidingVerifier
  - GnfE0: 창마다 p = (p_Gnf + p_E0) / 2 (같은 fold/seed의 저장된 E0_s{seed}.pkl), p_Gnf가 NaN이면 p_E0, 그 뒤 max
  - 창 AUC: test 창의 원시 이상 점수 (GnfE0는 결합 p), NaN 창은 빼고 계산 (window.n_nan에 개수)

같은 오탐률 비교 (matched FA, oracle): p_fall이 있는 변형마다 seed별로 fold를 합쳐, 첫 AI 호출 p_fall 억제 임계값을
  test 결과로 골라 근사 ADL 오탐률을 기준 변형(--fa-ref, 기본 E0max)의 근사 오탐률에 맞춘 뒤 recall을 비교.
  근사는 억제 뒤 재진입/승격 알람을 무시함. test로 고른 임계값이라 모델 순위 비교용일 뿐 성능 추정이 아님.

시간 기준 지표 (replay.summarize): 제때 recall(엄격/느슨), 제때 balanced, 첫 LEVEL1이 정지 꼬리 안인지(in_tail).
  timing 필드가 없는 옛 결과 파일은 '-'로 표시합니다.

이 모듈의 집계/표 함수는 표준 라이브러리만 씁니다 (numpy/torch는 fold 실행 함수 안에서 import).
"""
import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "dataset" / "_experiments" / "progression_v1"
DEFAULT_MANIFEST = REPO_ROOT / "dataset" / "_manifest" / "manifest_v1.jsonl"

STAGES = ("A", "B", "C", "D", "E", "Es", "F", "L", "G", "R")
STAGE_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "A": ("A",), "B": ("B",), "C": ("C",), "D": ("D",),
    "E": ("E0", "E1", "E2"), "Es": ("E0max", "E1max", "E2max", "E0last", "E1last", "E2last"),
    "F": ("F0", "F1", "F2", "F0b", "F1b", "F2b"),
    "L": ("E0p10", "E0p25", "E0p50", "E1p10", "E1p25", "E1p50"),
    "G": ("Gmd", "Gae", "Gnf", "GnfS", "GnfE0"),
    "R": ("AR1", "E0maxR1", "E1maxR1"),
}
# E* 변형 -> (기준 E 변형, SlidingWindowVerifier mode)
ES_BASE: Dict[str, Tuple[str, str]] = {v: (v[:2], v[2:]) for v in STAGE_VARIANTS["Es"]}
# L 변형 -> (기준 E 변형, 남기는 양성 학습 클립 비율)
L_BASE: Dict[str, Tuple[str, float]] = {v: (v[:2], int(v[3:]) / 100.0) for v in STAGE_VARIANTS["L"]}
L_FRACTIONS: Tuple[float, ...] = tuple(sorted({f for _, f in L_BASE.values()}))
# G 변형 -> 이상 탐지기 종류 (training.anomaly.KINDS), GnfE0 -> (탐지기, 결합할 E 변형)
G_KIND: Dict[str, str] = {"Gmd": "md", "Gae": "ae", "Gnf": "nf"}
# 엄격한 정상 집합 변형 -> 탐지기 종류: ADL 클립의 창만으로 학습·보정 (낙상 클립의 낙상 전 구간 제외, 낙상 라벨을 전혀 안 씀)
G_STRICT: Dict[str, str] = {"GnfS": "nf"}
G_COMBO: Dict[str, Tuple[str, str]] = {"GnfE0": ("nf", "E0")}
G_PARAMS_UNIT: Dict[str, str] = {"md": "statistics", "ae": "weights", "nf": "weights"}
# 라벨 효율 표의 행: (낙상 라벨 비율, 변형들). 등록되지 않았거나 결과가 없는 변형은 표에서 빠짐.
# 0%: 정상 창만으로 학습한 이상 탐지 (단계 G). GnfE0는 100% 라벨로 학습한 E0를 섞으므로 100% 행에 둠.
LABEL_EFF_ROWS: Tuple[Tuple[float, Tuple[str, ...]], ...] = (
    (0.0, ("Gmd", "Gae", "Gnf", "GnfS")),
    (0.10, ("E0p10", "E1p10")),
    (0.25, ("E0p25", "E1p25")),
    (0.50, ("E0p50", "E1p50")),
    (1.0, ("E0max", "E1max", "GnfE0")),
)
# 같은 오탐률 비교(matched FA)의 기준 변형 (CLI --fa-ref)
DEFAULT_FA_REF = "E0max"
VARIANTS = tuple(v for s in STAGES for v in STAGE_VARIANTS[s])
VARIANT_INFO: Dict[str, Dict[str, str]] = {
    "A": {"title": "규칙만", "change": "rules only (ai_mode=off)", "model": "-"},
    "AR1": {"title": "규칙만 + 짧은 미검출 유예 (R1)",
            "change": "rules only, lost_grace_s=0.5 (keep warmup/accumulators, continue timeout over short gaps)",
            "model": "-"},
    "E0maxR1": {"title": "E0max + 짧은 미검출 유예 (R1)", "change": "E0max verifier (saved E0), rules with lost_grace_s=0.5",
                "model": "HistGradientBoosting"},
    "E1maxR1": {"title": "E1max + 짧은 미검출 유예 (R1)", "change": "E1max verifier (saved E1), rules with lost_grace_s=0.5",
                "model": "Fall1DCNN-GAP"},
    "B": {"title": "1D CNN + URFD (원래 방식)", "change": "+ Fall1DCNN, URFD only, whole-video labels, 30fps",
          "model": "Fall1DCNN (v1 30f)"},
    "C": {"title": "+ 데이터 추가", "change": "+ all datasets (+figshare), same recipe", "model": "Fall1DCNN (v1 30f)"},
    "D": {"title": "+ 라벨/학습-운영 정렬", "change": "+ 15Hz EMA runtime windows, post-fall labels, improved recipe",
          "model": "Fall1DCNN (v1 30f)"},
    "E0": {"title": "+ 사건 구간 v2 (고전 ML)", "change": "+ ABNORMAL-entry window v2, handcrafted stats",
           "model": "HistGradientBoosting"},
    "E1": {"title": "+ 사건 구간 v2 (CNN-GAP)", "change": "+ ABNORMAL-entry window v2", "model": "Fall1DCNN-GAP"},
    "E2": {"title": "+ 사건 구간 v2 (TCN)", "change": "+ ABNORMAL-entry window v2", "model": "FallTCN"},
    "E0max": {"title": "E* 진입 이후 최대 p_fall (E0)", "change": "E0 model, max p_fall over 4s windows entry..LEVEL1",
              "model": "HistGradientBoosting"},
    "E1max": {"title": "E* 진입 이후 최대 p_fall (E1)", "change": "E1 model, max p_fall over 4s windows entry..LEVEL1",
              "model": "Fall1DCNN-GAP"},
    "E2max": {"title": "E* 진입 이후 최대 p_fall (E2)", "change": "E2 model, max p_fall over 4s windows entry..LEVEL1",
              "model": "FallTCN"},
    "E0last": {"title": "E* 판정 시점 최신 창 (E0, 참고)", "change": "E0 model, last 4s window before LEVEL1",
               "model": "HistGradientBoosting"},
    "E1last": {"title": "E* 판정 시점 최신 창 (E1, 참고)", "change": "E1 model, last 4s window before LEVEL1",
               "model": "Fall1DCNN-GAP"},
    "E2last": {"title": "E* 판정 시점 최신 창 (E2, 참고)", "change": "E2 model, last 4s window before LEVEL1",
               "model": "FallTCN"},
    "F0": {"title": "+ 임계값 조정 (E0)", "change": "+ val-tuned suppress threshold", "model": "HistGradientBoosting"},
    "F1": {"title": "+ 임계값 조정 (E1)", "change": "+ val-tuned suppress threshold", "model": "Fall1DCNN-GAP"},
    "F2": {"title": "+ 임계값 조정 (E2)", "change": "+ val-tuned suppress threshold", "model": "FallTCN"},
    "F0b": {"title": "+ 임계값 조정 (E0, 검증 balanced 최대)", "change": "+ val balanced-max threshold",
            "model": "HistGradientBoosting"},
    "F1b": {"title": "+ 임계값 조정 (E1, 검증 balanced 최대)", "change": "+ val balanced-max threshold",
            "model": "Fall1DCNN-GAP"},
    "F2b": {"title": "+ 임계값 조정 (E2, 검증 balanced 최대)", "change": "+ val balanced-max threshold",
            "model": "FallTCN"},
    "E0p10": {"title": "라벨 효율: 학습 낙상 클립 10% (E0, E*max 판정)",
              "change": "E0 recipe, 10% of positive train clips (nested by clip_id), max p_fall",
              "model": "HistGradientBoosting"},
    "E0p25": {"title": "라벨 효율: 학습 낙상 클립 25% (E0, E*max 판정)",
              "change": "E0 recipe, 25% of positive train clips (nested by clip_id), max p_fall",
              "model": "HistGradientBoosting"},
    "E0p50": {"title": "라벨 효율: 학습 낙상 클립 50% (E0, E*max 판정)",
              "change": "E0 recipe, 50% of positive train clips (nested by clip_id), max p_fall",
              "model": "HistGradientBoosting"},
    "E1p10": {"title": "라벨 효율: 학습 낙상 클립 10% (E1, E*max 판정)",
              "change": "E1 recipe, 10% of positive train clips (nested by clip_id), max p_fall",
              "model": "Fall1DCNN-GAP"},
    "E1p25": {"title": "라벨 효율: 학습 낙상 클립 25% (E1, E*max 판정)",
              "change": "E1 recipe, 25% of positive train clips (nested by clip_id), max p_fall",
              "model": "Fall1DCNN-GAP"},
    "E1p50": {"title": "라벨 효율: 학습 낙상 클립 50% (E1, E*max 판정)",
              "change": "E1 recipe, 50% of positive train clips (nested by clip_id), max p_fall",
              "model": "Fall1DCNN-GAP"},
    "Gmd": {"title": "이상 탐지: Mahalanobis (정상 창만, E*max 판정)",
            "change": "normal windows only, handcrafted + LedoitWolf Mahalanobis, val q0.8 -> p 0.4, max p",
            "model": "Mahalanobis (LedoitWolf)"},
    "Gae": {"title": "이상 탐지: 1D conv 오토인코더 (정상 창만, E*max 판정)",
            "change": "normal windows only, mask-weighted reconstruction error, val q0.8 -> p 0.4, max p",
            "model": "Conv1D AE"},
    "Gnf": {"title": "이상 탐지: STG-NF (정상 창만, E*max 판정)",
            "change": "normal windows only, pose-segment flow NLL (max over segments), val q0.8 -> p 0.4, max p",
            "model": "STG-NF"},
    "GnfS": {"title": "이상 탐지: STG-NF, ADL 클립만 (낙상 라벨 전혀 없음)",
             "change": "like Gnf but train/calibrate on windows of ADL clips only (no pre-fall windows of fall clips)",
             "model": "STG-NF"},
    "GnfE0": {"title": "STG-NF + E0 평균 (E*max 판정)",
              "change": "per-window mean of calibrated Gnf p and E0 p_fall (same fold/seed), max p",
              "model": "STG-NF + HistGradientBoosting"},
}
EVAL_DATASETS_ORDER = ("le2i", "caucafall", "gmdcsa24", "mcfd", "urfd")

DEFAULT_TH = 0.4
# R1: 짧은 미검출 유예 (초, Level1Params.lost_grace_s). 사전에 정한 값 (Day 5 분석: 한 프레임 끊김에 워밍업 리셋)
LOST_GRACE_R1 = 0.5
F_MARGIN = 0.05
F_TH_MAX = 0.95
VAL_FRAC = 0.15
VAL_SPLIT_SEED = 0

__all__ = [
    "STAGES", "VARIANTS", "parse_folds", "parse_stages", "select_threshold", "auc_mann_whitney",
    "first_p_fall", "val_points", "sim_balanced", "select_threshold_balanced",
    "positive_clip_ids", "nested_clip_subsets", "keep_positive_mask", "ai_points", "sim_rates",
    "G_KIND", "G_COMBO", "NanSafeSlidingVerifier", "combine_mean_p",
    "match_fa_threshold", "matched_fa", "label_efficiency_rows",
    "aggregate_variant", "paired_compare", "build_summary", "render_markdown", "run_fold", "main",
]


# ---------------------------------------------------------------------------
# 인자 해석, 순수 계산 함수 (표준 라이브러리만)
# ---------------------------------------------------------------------------

def parse_folds(spec: str) -> List[int]:
    """'0-4' / '0,2,3' / '3' -> 정렬된 fold 번호 목록"""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"잘못된 fold 범위: {part}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("fold가 비어 있습니다")
    return sorted(out)


def parse_stages(spec: str) -> List[str]:
    """
    'A,B,E' -> 단계 목록 (STAGES 순서, 대소문자 무시: 'es' -> 'Es').
    F와 Es는 E 모델이 필요합니다 (같은 실행에 E가 없으면 저장된 모델을 읽음).
    """
    got = {s.strip().upper() for s in str(spec).split(",") if s.strip()}
    bad = got - {s.upper() for s in STAGES}
    if bad:
        raise ValueError(f"알 수 없는 단계: {sorted(bad)}")
    return [s for s in STAGES if s.upper() in got]


def select_threshold(fall_p: Sequence[float], margin: float = F_MARGIN, default: float = DEFAULT_TH,
                     th_max: float = F_TH_MAX) -> float:
    """
    검증 낙상 클립의 첫 p_fall 목록 -> 억제 임계값 (p_fall <= th면 억제).
    모든 검증 낙상이 통과하는(recall 1.0) 가장 큰 임계값 min(p)에서 margin을 뺀 값, [0, th_max]로 자름.
    값이 없으면 default.
    """
    vals = [float(p) for p in fall_p if p is not None and not math.isnan(float(p))]
    if not vals:
        return float(default)
    return float(min(max(min(vals) - float(margin), 0.0), float(th_max)))


def auc_mann_whitney(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """ROC-AUC = P(양성 점수 > 음성 점수) + 0.5 P(같음). 한쪽이 비면 None."""
    pos, neg = list(pos), list(neg)
    if not pos or not neg:
        return None
    pairs = sorted([(float(v), 1) for v in pos] + [(float(v), 0) for v in neg])
    # 순위 합 (동점은 평균 순위)
    rank_sum_pos = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        rank_sum_pos += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1] == 1)
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def first_p_fall(outcome: Dict[str, Any]) -> Optional[float]:
    """클립에서 처음 AI가 낸 p_fall (규칙이 LEVEL1 판정까지 간 클립만). 없으면 None."""
    ps = outcome.get("p_fall") or []
    return float(ps[0]) if ps else None


def _mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def _std(vals: Iterable[Optional[float]]) -> Optional[float]:
    """표본 표준편차 (ddof=1). 값이 2개 미만이면 0 또는 None."""
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _mean_std(vals: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = list(vals)
    return {"mean": _mean(vals), "std": _std(vals), "n": sum(1 for x in vals if x is not None)}


def _summarize(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    from training.experiments.replay import summarize  # numpy import는 여기서만 (replay가 numpy 사용)
    return summarize(outcomes)


_SUM_KEYS = ("n_fall", "n_adl", "n_detected", "n_adl_alarm", "recall", "adl_fa_rate", "balanced",
             "suppressed_true_falls", "n_fall_suppressed", "n_adl_suppressed", "rules_ceiling",
             "n_fall_reached", "adl_rules_fa_rate", "n_adl_reached", "precision", "f1",
             "n_timely", "timely_recall", "timely_balanced", "n_timely_loose", "timely_recall_loose")
_FOLD_KEYS = ("recall", "adl_fa_rate", "balanced", "suppressed_true_falls", "rules_ceiling",
              "timely_recall", "timely_balanced")
_TAIL_KEYS = ("fall_live", "fall_tail", "adl_live", "adl_tail", "unknown")


def _avg_stats(stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """seed별 요약 dict 목록 -> 키마다 평균 (None 제외)"""
    return {k: _mean(s.get(k) for s in stats) for k in _SUM_KEYS}


def _tail_split(outcomes: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    알람(첫 LEVEL1)이 실제 영상 구간(live)인지 정지 꼬리(tail) 안인지 낙상/ADL별로 셈.
    unknown: in_tail이 없는 알람 (옛 결과 파일).
    """
    out = {k: 0 for k in _TAIL_KEYS}
    for o in outcomes:
        if not o.get("alarm"):
            continue
        it = o.get("in_tail")
        if it is None:
            out["unknown"] += 1
            continue
        out[("fall_" if o.get("label") == "fall" else "adl_") + ("tail" if it else "live")] += 1
    return out


def _ai_call_auc(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    pos, neg = [], []
    for o in outcomes:
        p = first_p_fall(o)
        if p is None:
            continue
        (pos if o.get("label") == "fall" else neg).append(p)
    return {"roc_auc": auc_mann_whitney(pos, neg), "n_fall": len(pos), "n_adl": len(neg)}


def aggregate_variant(fold_results: List[Dict[str, Any]], summarize_fn: Optional[Callable] = None
                      ) -> Dict[str, Any]:
    """
    한 변형의 fold 결과 목록 -> 집계.
    fold 결과: {"fold", "params", "runs": [{"seed", "outcomes", "window": {...}, "threshold", "train": {...}}], "time_s"}
    """
    summarize_fn = summarize_fn or _summarize
    fold_results = sorted(fold_results, key=lambda r: r["fold"])
    n_runs = min(len(r["runs"]) for r in fold_results) if fold_results else 0
    # pooled: run 번호(seed 순서)별로 fold를 합침
    pooled_runs = []
    ai_runs = []
    tail_runs: List[Dict[str, int]] = []
    for i in range(n_runs):
        outs = [o for r in fold_results for o in r["runs"][i]["outcomes"]]
        pooled_runs.append(summarize_fn(outs))
        ai_runs.append(_ai_call_auc(outs))
        tail_runs.append(_tail_split(outs))
    pooled = _avg_stats(pooled_runs)
    datasets = sorted({d for s in pooled_runs for d in s.get("by_dataset", {})})
    pooled["by_dataset"] = {d: _avg_stats([s["by_dataset"][d] for s in pooled_runs if d in s.get("by_dataset", {})])
                            for d in datasets}
    # fold 단위: seed 평균 후 fold 간 mean ± std
    per_fold: Dict[str, List[Optional[float]]] = {k: [] for k in _FOLD_KEYS}
    win_fold: Dict[str, List[Optional[float]]] = {"roc_auc": [], "pr_auc": []}
    for r in fold_results:
        stats = [summarize_fn(run["outcomes"]) for run in r["runs"][:n_runs]]
        for k in _FOLD_KEYS:
            per_fold[k].append(_mean(s.get(k) for s in stats))
        wins = [run.get("window") or {} for run in r["runs"][:n_runs]]
        for k in win_fold:
            win_fold[k].append(_mean(w.get(k) for w in wins))
    thresholds = [run.get("threshold") for r in fold_results for run in r["runs"][:n_runs]]
    thresholds = [t for t in thresholds if t is not None]
    train_times = [run.get("train", {}).get("time_s") for r in fold_results for run in r["runs"][:n_runs]]
    first = fold_results[0] if fold_results else {}
    return {
        "n_folds": len(fold_results),
        "folds": [r["fold"] for r in fold_results],
        "n_runs_per_fold": n_runs,
        "params": first.get("params"),
        "params_unit": first.get("params_unit", "weights"),
        "pooled": pooled,
        "fold": {k: _mean_std(v) for k, v in per_fold.items()},
        "fold_values": per_fold,
        "window": {k: _mean_std(v) for k, v in win_fold.items()},
        "ai_call_auc": _mean(a["roc_auc"] for a in ai_runs),
        "tail_split": {k: _mean(t[k] for t in tail_runs) for k in _TAIL_KEYS},
        "ai_call_n": {"fall": _mean(a["n_fall"] for a in ai_runs), "adl": _mean(a["n_adl"] for a in ai_runs)},
        "threshold": {"mean": _mean(thresholds), "std": _std(thresholds),
                      "min": min(thresholds) if thresholds else None,
                      "max": max(thresholds) if thresholds else None},
        "train_time_s": _mean(train_times),
        "time_s_total": sum(float(r.get("time_s") or 0.0) for r in fold_results),
        "val_balanced": _mean(run.get("val", {}).get("balanced_at_th")
                              for r in fold_results for run in r["runs"][:n_runs]),
        # L: 학습에 남긴 양성 클립 수 (run 평균, 다른 변형은 None)
        "train_pos_clips": _mean((run.get("train") or {}).get("n_pos_clips")
                                 for r in fold_results for run in r["runs"][:n_runs]),
    }


def _load_fold_results(out_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    res: Dict[str, List[Dict[str, Any]]] = {}
    for fd in sorted(out_dir.glob("fold*")):
        if not fd.is_dir():
            continue
        for v in VARIANTS:
            p = fd / f"{v}.json"
            if p.exists():
                res.setdefault(v, []).append(json.loads(p.read_text(encoding="utf-8")))
    return res


_PAIR_KEYS = ("base_supp_true", "recovered", "recovered_timely", "recovered_tail", "fall_gained", "fall_lost",
              "adl_new_alarm", "adl_removed")
_DELTA_KEYS = ("balanced", "recall", "adl_fa_rate")


def _runs_by_seed(fold_result: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    """fold 결과의 runs -> {seed (없으면 순번): run}"""
    return {(r.get("seed") if r.get("seed") is not None else i): r for i, r in enumerate(fold_result["runs"])}


def paired_compare(base_folds: List[Dict[str, Any]], new_folds: List[Dict[str, Any]],
                   summarize_fn: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
    """
    같은 fold, 같은 seed끼리 두 변형 비교 (new - base). 공통 fold가 없으면 None.

    delta[지표]: fold별 seed 평균 차이 목록(values), mean ± std, n_better/n_worse (adl_fa_rate는 줄어야 좋아짐)
    clips (clip_id로 맞춤, fold 합 후 seed 평균):
      base_supp_true: base에서 AI가 억제해 끝내 놓친 낙상, recovered: 그중 new에서 알람
      recovered_timely / recovered_tail: 회복된 낙상 중 new에서 제때(엄격) / 첫 LEVEL1이 정지 꼬리 안 (timing 필드 없으면 0)
      fall_gained / fall_lost: 낙상 알람이 new에서 새로 생김 / 사라짐
      adl_new_alarm / adl_removed: ADL 알람이 new에서 새로 생김 / 사라짐
    """
    summarize_fn = summarize_fn or _summarize
    base = {r["fold"]: r for r in base_folds}
    new = {r["fold"]: r for r in new_folds}
    common = sorted(set(base) & set(new))
    if not common:
        return None
    deltas: Dict[str, List[Optional[float]]] = {k: [] for k in _DELTA_KEYS}
    counts: Dict[Any, Dict[str, int]] = {}
    for k in common:
        b_runs, n_runs = _runs_by_seed(base[k]), _runs_by_seed(new[k])
        seeds = [s for s in n_runs if s in b_runs]
        b_stats = [summarize_fn(b_runs[s]["outcomes"]) for s in seeds]
        n_stats = [summarize_fn(n_runs[s]["outcomes"]) for s in seeds]
        for key in _DELTA_KEYS:
            b, n = _mean(x.get(key) for x in b_stats), _mean(x.get(key) for x in n_stats)
            deltas[key].append(None if b is None or n is None else n - b)
        for s in seeds:
            c = counts.setdefault(s, {key: 0 for key in _PAIR_KEYS})
            b_out = {o["clip_id"]: o for o in b_runs[s]["outcomes"]}
            for o in n_runs[s]["outcomes"]:
                bo = b_out.get(o["clip_id"])
                if bo is None:
                    continue
                b_alarm, n_alarm = bool(bo.get("alarm")), bool(o.get("alarm"))
                if o.get("label") == "fall":
                    supp = bool(bo.get("suppressed")) and not b_alarm
                    c["base_supp_true"] += supp
                    c["recovered"] += supp and n_alarm
                    c["recovered_timely"] += supp and n_alarm and bool(o.get("timely_strict"))
                    c["recovered_tail"] += supp and n_alarm and bool(o.get("in_tail"))
                    c["fall_gained"] += (not b_alarm) and n_alarm
                    c["fall_lost"] += b_alarm and not n_alarm
                else:
                    c["adl_new_alarm"] += (not b_alarm) and n_alarm
                    c["adl_removed"] += b_alarm and not n_alarm
    eps = 1e-9
    delta = {}
    for key, vals in deltas.items():
        sign = -1.0 if key == "adl_fa_rate" else 1.0
        got = [sign * x for x in vals if x is not None]
        delta[key] = dict(_mean_std(vals), values=vals, n_better=sum(1 for x in got if x > eps),
                          n_worse=sum(1 for x in got if x < -eps))
    return {"folds": common, "delta": delta, "n_seeds": len(counts),
            "clips": {key: _mean(c[key] for c in counts.values()) for key in _PAIR_KEYS}}


# ---------------------------------------------------------------------------
# L: 양성 클립 부분 표본 (표준 라이브러리만)
# ---------------------------------------------------------------------------

def positive_clip_ids(meta: Sequence[Dict[str, Any]], y: Sequence[int]) -> List[str]:
    """학습 창 meta/y -> 양성 창(y == 1)이 하나라도 있는 clip_id 목록 (정렬)"""
    return sorted({str(m.get("clip_id")) for m, t in zip(meta, y) if int(t) == 1})


def _n_keep(frac: float, n: int) -> int:
    """비율 frac로 남길 클립 수: round(frac * n), 0 < frac이면 1개 이상, n 이하"""
    if n <= 0 or frac <= 0:
        return 0
    return int(min(n, max(1, round(float(frac) * n))))


def nested_clip_subsets(clip_ids: Iterable[str], fractions: Iterable[float], fold: int, seed: int
                        ) -> Dict[float, List[str]]:
    """
    양성 클립 목록 -> {비율: 남길 clip_id 목록 (정렬)}. 클립 단위라 한 낙상 클립의 창은 함께 남거나 함께 빠짐.
    중첩: 큰 비율부터 차례로 바로 위 집합에서 뽑음 (전체 -> 50% -> 25% -> 10%), 그래서 10% ⊂ 25% ⊂ 50%.
    비율 f의 표본은 random.Random("{fold}-{seed}-{f}")로 바로 위 집합에서 뽑음 (같은 입력이면 실행마다 같음).
    클립 수는 전체 클립 수 기준 _n_keep(f, n). 비율 1.0 이상은 전체.
    요청 비율 집합이 달라도 결과가 같도록 L_FRACTIONS의 비율은 항상 모두 거쳐 뽑음.
    """
    import random

    pool = sorted(set(str(c) for c in clip_ids))
    n = len(pool)
    want = {float(x) for x in fractions}
    out: Dict[float, List[str]] = {}
    parent = pool
    for f in sorted(want | set(L_FRACTIONS), reverse=True):
        if f >= 1.0:
            sub = list(pool)
        else:
            k = min(_n_keep(f, n), len(parent))
            rng = random.Random(f"{int(fold)}-{int(seed)}-{f:.4f}")
            sub = sorted(rng.sample(parent, k))
        parent = sub
        if f in want:
            out[f] = list(sub)
    return out


def keep_positive_mask(meta: Sequence[Dict[str, Any]], y: Sequence[int], kept_clips: Iterable[str]) -> List[bool]:
    """
    낙상 라벨 예산 f: 낙상 클립(meta label "fall")은 kept_clips에 있을 때만 그 클립의 창을 모두 남기고
    (양성 창, 낙상 전 음성 창 pre_fall/entry_pre_fall 모두), 그 밖의 낙상 클립 창은 모두 뺌.
    낙상 전 음성 창도 라벨(fall_start)로 잘라 낸 것이라, 쓰지 않는 낙상 클립에서 남기면 라벨을 쓰는 셈이 되기 때문.
    ADL 클립 창은 모두 남김. meta에 label이 없으면 y로 판단 (y == 1이면 낙상 클립 창).
    """
    kept = set(str(c) for c in kept_clips)

    def is_fall_clip(m, t):
        lab = m.get("label")
        return lab == "fall" if lab is not None else int(t) == 1
    return [(not is_fall_clip(m, t)) or str(m.get("clip_id")) in kept for m, t in zip(meta, y)]


# ---------------------------------------------------------------------------
# 같은 오탐률 비교 (matched FA, 표준 라이브러리만)
# ---------------------------------------------------------------------------

NO_VETO = -1.0  # 아무것도 억제하지 않는 임계값 (p_fall >= 0)


def ai_points(outcomes: Iterable[Dict[str, Any]]) -> List[Tuple[str, bool, Optional[float]]]:
    """
    veto 재생 클립 결과 -> (label, 규칙이 LEVEL1 판정까지 감(reached_ai), 첫 AI 호출 p_fall) 목록.
    veto 결과의 alarm은 AI 억제 뒤의 값이라 규칙 알람은 reached_ai로 대신함 (shadow 재생의 alarm과 같은 뜻).
    첫 AI 호출은 억제 전이라 첫 p_fall은 억제 임계값과 무관.
    """
    return [(str(o.get("label")), bool(o.get("reached_ai")), first_p_fall(o)) for o in outcomes]


def sim_rates(points: Sequence[Tuple[str, bool, Optional[float]]], th: float
              ) -> Tuple[Optional[float], Optional[float]]:
    """
    임계값 th의 veto를 근사한 (recall, ADL 오탐률): 첫 AI 호출 p_fall <= th면 억제, p_fall이 없으면(판단 불가)
    fail-open으로 알람. 억제 뒤 재진입/승격 알람은 무시하는 근사. 한쪽 클립이 없으면 그쪽은 None.
    """
    n_f = n_a = det = fa = 0
    for lab, alarm, p in points:
        hit = bool(alarm) and (p is None or p > th)
        if lab == "fall":
            n_f += 1
            det += hit
        else:
            n_a += 1
            fa += hit
    return (det / n_f if n_f else None), (fa / n_a if n_a else None)


def match_fa_threshold(points: Sequence[Tuple[str, bool, Optional[float]]], target_fa: float,
                       eps: float = 1e-9) -> Dict[str, Any]:
    """
    근사 ADL 오탐률이 target_fa 이하가 되는 가장 작은 임계값(= recall이 가장 큰 쪽)과 그때의 recall/오탐률.
    후보 = 억제 없음(NO_VETO)과 클립 첫 p_fall 값들. 모든 p를 억제해도 target을 못 맞추면(판단 불가 알람이
    많을 때) 가장 큰 후보를 쓰고 matched=False.
    """
    cands = [NO_VETO] + sorted({float(p) for _, _, p in points if p is not None})
    th, r, fa = NO_VETO, None, None
    for th in cands:
        r, fa = sim_rates(points, th)
        if fa is None or fa <= float(target_fa) + eps:
            return {"threshold": th, "recall": r, "adl_fa_rate": fa, "matched": True}
    return {"threshold": th, "recall": r, "adl_fa_rate": fa, "matched": False}


def _has_p_fall(fold_results: List[Dict[str, Any]]) -> bool:
    return any(o.get("p_fall") for r in fold_results for run in r.get("runs", []) for o in run.get("outcomes", []))


def _actual_rates(outcomes: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """실제 재생 결과의 (recall, ADL 오탐률)"""
    f = [o for o in outcomes if o.get("label") == "fall"]
    a = [o for o in outcomes if o.get("label") != "fall"]
    return (sum(bool(o.get("alarm")) for o in f) / len(f) if f else None,
            sum(bool(o.get("alarm")) for o in a) / len(a) if a else None)


def matched_fa(results: Dict[str, List[Dict[str, Any]]], ref: str = DEFAULT_FA_REF) -> Dict[str, Any]:
    """
    같은 오탐률 비교 (oracle: test 결과로 임계값을 맞추므로 모델 순위 비교용, 운영 임계값이 아님).
    p_fall이 있는 변형마다 seed별로 기준 변형(ref)과 공통 fold의 test 클립을 합쳐
      target = 기준 변형의 근사 ADL 오탐률 (sim_rates, 기준 변형 자신의 억제 임계값; 실제 오탐률도 함께 기록),
      target에 맞춘 임계값(match_fa_threshold)에서의 recall, 자기 실제/근사 recall·오탐률.
    seed 평균 (threshold는 평균/최소/최대). 반환: {"ref", "variants": {variant: ...}} (기준 결과가 없으면 빔)
    """
    out: Dict[str, Any] = {"ref": ref, "variants": {}}
    ref_folds = {r["fold"]: r for r in results.get(ref) or []}
    if not ref_folds:
        return out
    for v in VARIANTS:
        frs = results.get(v) or []
        if not frs or not _has_p_fall(frs):
            continue
        new_folds = {r["fold"]: r for r in frs}
        common = sorted(set(ref_folds) & set(new_folds))
        if not common:
            continue
        ref_runs = {k: _runs_by_seed(ref_folds[k]) for k in common}
        new_runs = {k: _runs_by_seed(new_folds[k]) for k in common}
        seeds = [s for s in new_runs[common[0]] if all(s in ref_runs[k] and s in new_runs[k] for k in common)]
        per: List[Dict[str, Any]] = []
        for s in seeds:
            ref_out = [o for k in common for o in ref_runs[k][s]["outcomes"]]
            new_out = [o for k in common for o in new_runs[k][s]["outcomes"]]
            ref_th = _mean(ref_runs[k][s].get("threshold") for k in common)
            own_th = _mean(new_runs[k][s].get("threshold") for k in common)
            _, target = sim_rates(ai_points(ref_out), DEFAULT_TH if ref_th is None else ref_th)
            if target is None:
                continue
            pts = ai_points(new_out)
            m = match_fa_threshold(pts, target)
            own_r, own_fa = _actual_rates(new_out)
            sim_r, sim_fa = sim_rates(pts, DEFAULT_TH if own_th is None else own_th)
            per.append({"seed": s, "target_fa": target, "ref_fa_actual": _actual_rates(ref_out)[1],
                        "threshold": m["threshold"], "recall_at_fa": m["recall"], "fa_at_th": m["adl_fa_rate"],
                        "matched": m["matched"], "own_recall": own_r, "own_fa": own_fa,
                        "own_sim_recall": sim_r, "own_sim_fa": sim_fa})
        if not per:
            continue
        ths = [p["threshold"] for p in per]
        row: Dict[str, Any] = {k: _mean(p[k] for p in per) for k in _MFA_KEYS}
        row.update({"folds": common, "n_seeds": len(per), "all_matched": all(p["matched"] for p in per),
                    "threshold": {"mean": _mean(ths), "min": min(ths), "max": max(ths)},
                    "recall_at_fa_seeds": [p["recall_at_fa"] for p in per]})
        out["variants"][v] = row
    return out


_MFA_KEYS = ("target_fa", "ref_fa_actual", "recall_at_fa", "fa_at_th", "own_recall", "own_fa",
             "own_sim_recall", "own_sim_fa")


def label_efficiency_rows(variants: Dict[str, Any], matched: Optional[Dict[str, Any]] = None
                          ) -> List[Dict[str, Any]]:
    """
    summary variants -> 라벨 효율 표 행 (LABEL_EFF_ROWS 순서, 결과가 있는 변형만).
    100% 행(E*max) 말고 다른 행이 하나도 없으면 빈 목록.
    """
    full = {v for frac, vs in LABEL_EFF_ROWS if frac >= 1.0 for v in vs if v in ES_BASE}
    if not any(v in variants for _, vs in LABEL_EFF_ROWS for v in vs if v not in full):
        return []
    mv = (matched or {}).get("variants") or {}
    rows = []
    for frac, vs in LABEL_EFF_ROWS:
        for v in vs:
            s = variants.get(v)
            if not s:
                continue
            fd = s.get("fold") or {}
            rows.append({"frac": frac, "variant": v, "model": s.get("model", ""),
                         "train_pos_clips": s.get("train_pos_clips"),
                         "recall": fd.get("recall"), "adl_fa_rate": fd.get("adl_fa_rate"),
                         "balanced": fd.get("balanced"), "ai_call_auc": s.get("ai_call_auc"),
                         "window_auc": (s.get("window") or {}).get("roc_auc"),
                         "matched_recall": (mv.get(v) or {}).get("recall_at_fa")})
    return rows


def build_summary(results: Dict[str, List[Dict[str, Any]]], summarize_fn: Optional[Callable] = None,
                  meta: Optional[Dict[str, Any]] = None, fa_ref: str = DEFAULT_FA_REF) -> Dict[str, Any]:
    """
    {variant: [fold 결과]} -> summary dict (variants 순서 유지). 'best_f': 검증 balanced가 가장 높은 F 변형.
    'paired': E* 변형마다 기준 E 변형과의 paired_compare (둘 다 있을 때).
    'matched_fa': 기준 변형 fa_ref의 오탐률에 맞춘 recall (matched_fa), 'label_efficiency': 라벨 효율 표 행.
    """
    variants = {}
    for v in VARIANTS:
        if v in results and results[v]:
            variants[v] = dict(VARIANT_INFO.get(v, {}), **aggregate_variant(results[v], summarize_fn))
    best_f = None
    f_cands = [(variants[v].get("val_balanced"), v) for v in STAGE_VARIANTS["F"] if v in variants]
    f_cands = [c for c in f_cands if c[0] is not None]
    if f_cands:
        best_f = max(f_cands)[1]
    paired = {}
    for v, (ev, _) in ES_BASE.items():
        if results.get(v) and results.get(ev):
            pc = paired_compare(results[ev], results[v], summarize_fn)
            if pc is not None:
                paired[v] = dict(pc, base=ev)
    mfa = matched_fa(results, ref=fa_ref)
    return {"version": 1, "meta": dict(meta or {}), "variants": variants, "best_f": best_f, "paired": paired,
            "matched_fa": mfa, "label_efficiency": label_efficiency_rows(variants, mfa)}


def _fmt(x: Optional[float], nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _fmt_ms(d: Optional[Dict[str, Any]], nd: int = 3) -> str:
    if not d or d.get("mean") is None:
        return "-"
    return f"{d['mean']:.{nd}f} ± {(d.get('std') or 0.0):.{nd}f}"


def _fmt_count(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "-"
    sa = f"{a:.0f}" if abs(a - round(a)) < 1e-9 else f"{a:.1f}"
    return f"{sa}/{b:.0f}"


def _fmt_rate_count(r: Optional[float], a: Optional[float], b: Optional[float]) -> str:
    """'0.600 (3/5)', 비율이 없으면 '-'"""
    return "-" if r is None else f"{_fmt(r)} ({_fmt_count(a, b)})"


def _fmt_params(v: Dict[str, Any]) -> str:
    p = v.get("params")
    if p is None:
        return "-"
    unit = v.get("params_unit", "weights")
    return f"{int(p):,}" if unit == "weights" else f"{int(p):,} {unit}"


def render_markdown(summary: Dict[str, Any]) -> str:
    """summary -> markdown 표 (전체, fold mean ± std, 데이터셋별)"""
    V = summary.get("variants", {})
    lines: List[str] = []
    meta = summary.get("meta") or {}
    lines.append("# Progression study 결과 (progression_v1)")
    lines.append("")
    if meta:
        lines.append(f"- 평가: {meta.get('protocol', '')}")
        if meta.get("folds") is not None:
            lines.append(f"- folds: {meta.get('folds')}, seeds per NN stage: {meta.get('seeds')}")
    lines.append("- pooled = 5개 test fold를 합친 값 (seed 평균), fold = fold별 값의 평균 ± 표본 표준편차")
    lines.append("- 억제된 실제 낙상 = AI가 거부해 끝내 알람이 없었던 낙상 클립 수 (pooled, seed 평균)")
    lines.append("- 규칙 상한 = 규칙 엔진이 LEVEL1 판정까지 간 낙상 클립 비율 (AI는 거부권만 있어 recall은 이 값을 넘지 못함)")
    if summary.get("best_f"):
        lines.append(f"- 검증 balanced 기준 최선 F 변형: {summary['best_f']}")
    lines.append("")
    lines.append("## 전체 (이벤트 단위)")
    lines.append("")
    lines.append("| 단계 | 변경 | 모델 | 파라미터 | Recall (pooled) | Recall (fold) | ADL 오탐률 (pooled) | "
                 "ADL 오탐률 (fold) | Balanced (pooled) | Balanced (fold) | 제때 Recall (pooled) | "
                 "제때 Balanced (pooled) | 억제된 실제 낙상 | 규칙 상한 | "
                 "AI 호출 AUC | 창 ROC-AUC | 창 PR-AUC | 억제 임계값 |")
    lines.append("|" + "---|" * 18)
    for v, s in V.items():
        p = s["pooled"]
        th = s.get("threshold") or {}
        if v == "A":
            th_s = "-"
        elif th.get("mean") is None:
            th_s = "-"
        elif (th.get("max") or 0) - (th.get("min") or 0) < 1e-9:
            th_s = f"{th['mean']:.2f}"
        else:
            th_s = f"{th['mean']:.2f} ({th['min']:.2f}~{th['max']:.2f})"
        lines.append(
            f"| {v} | {s.get('title', '')} | {s.get('model', '')} | {_fmt_params(s)} | "
            f"{_fmt(p.get('recall'))} ({_fmt_count(p.get('n_detected'), p.get('n_fall'))}) | "
            f"{_fmt_ms(s['fold'].get('recall'))} | "
            f"{_fmt(p.get('adl_fa_rate'))} ({_fmt_count(p.get('n_adl_alarm'), p.get('n_adl'))}) | "
            f"{_fmt_ms(s['fold'].get('adl_fa_rate'))} | {_fmt(p.get('balanced'))} | "
            f"{_fmt_ms(s['fold'].get('balanced'))} | "
            f"{_fmt_rate_count(p.get('timely_recall'), p.get('n_timely'), p.get('n_fall'))} | "
            f"{_fmt(p.get('timely_balanced'))} | {_fmt(p.get('suppressed_true_falls'), 1)} | "
            f"{_fmt(p.get('rules_ceiling'))} | {_fmt(s.get('ai_call_auc'))} | "
            f"{_fmt_ms((s.get('window') or {}).get('roc_auc'))} | {_fmt_ms((s.get('window') or {}).get('pr_auc'))} | "
            f"{th_s} |")
    lines.append("")
    lines.append("## 데이터셋별 (pooled, seed 평균): Recall (탐지/낙상) / ADL 오탐률 (알람/ADL)")
    lines.append("")
    ds = [d for d in EVAL_DATASETS_ORDER if any(d in s["pooled"].get("by_dataset", {}) for s in V.values())]
    lines.append("| 단계 | " + " | ".join(f"{d} recall | {d} ADL FA" for d in ds) + " |")
    lines.append("|---|" + "---|---|" * len(ds))
    for v, s in V.items():
        cells = []
        for d in ds:
            b = s["pooled"].get("by_dataset", {}).get(d) or {}
            cells.append(f"{_fmt(b.get('recall'))} ({_fmt_count(b.get('n_detected'), b.get('n_fall'))})")
            cells.append(f"{_fmt(b.get('adl_fa_rate'))} ({_fmt_count(b.get('n_adl_alarm'), b.get('n_adl'))})")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 규칙 상한과 AI 억제 (pooled, seed 평균)")
    lines.append("")
    lines.append("| 단계 | LEVEL1 도달 낙상 | LEVEL1 도달 ADL (AI 없을 때 오탐) | AI 억제된 낙상 클립 | AI 억제된 ADL 클립 | "
                 "학습 시간/run (s) | 총 시간 (s, 5 fold) |")
    lines.append("|---|---|---|---|---|---|---|")
    for v, s in V.items():
        p = s["pooled"]
        lines.append(
            f"| {v} | {_fmt_count(p.get('n_fall_reached'), p.get('n_fall'))} | "
            f"{_fmt_count(p.get('n_adl_reached'), p.get('n_adl'))} | {_fmt(p.get('n_fall_suppressed'), 1)} | "
            f"{_fmt(p.get('n_adl_suppressed'), 1)} | {_fmt(s.get('train_time_s'), 1)} | {_fmt(s.get('time_s_total'), 0)} |")
    lines.append("")
    lines.extend(_render_timing(V))
    lines.extend(_render_paired(summary.get("paired") or {}))
    lines.extend(_render_label_efficiency(summary.get("label_efficiency") or [],
                                          (summary.get("matched_fa") or {}).get("ref")))
    lines.extend(_render_matched_fa(summary.get("matched_fa") or {}))
    return "\n".join(lines)


def _render_timing(V: Dict[str, Any]) -> List[str]:
    """제때 탐지 지표와 첫 LEVEL1의 정지 꼬리 여부 표 (timing 필드가 있는 변형만 값, 없으면 '-')."""
    lines = ["## 제때 탐지와 정지 꼬리 (pooled, seed 평균)", "",
             "- 제때(엄격): ABNORMAL 진입이 [fall_start - 2, fall_end + 3] 안이고 LEVEL1이 fall_start 이후, "
             "느슨: 진입 또는 LEVEL1 중 하나가 그 창 안",
             "- 실제/꼬리: 첫 LEVEL1이 마지막 실제 프레임 이전인지, 붙인 정지 꼬리(마지막 프레임 반복) 안인지",
             "",
             "| 단계 | 제때 Recall 엄격 (pooled) | 제때 Recall 엄격 (fold) | 제때 Recall 느슨 (pooled) | "
             "제때 Balanced (pooled) | 제때 Balanced (fold) | 탐지 낙상 실제/꼬리 | ADL 알람 실제/꼬리 |",
             "|---|---|---|---|---|---|---|---|"]
    for v, s in V.items():
        p = s["pooled"]
        ts = s.get("tail_split") or {}
        known = any((ts.get(k) or 0) > 0 for k in ("fall_live", "fall_tail", "adl_live", "adl_tail"))

        def pair(a, b):
            if not known:
                return "-"
            return f"{_fmt(ts.get(a), 1)} / {_fmt(ts.get(b), 1)}"

        lines.append(
            f"| {v} | {_fmt_rate_count(p.get('timely_recall'), p.get('n_timely'), p.get('n_fall'))} | "
            f"{_fmt_ms(s['fold'].get('timely_recall'))} | "
            f"{_fmt_rate_count(p.get('timely_recall_loose'), p.get('n_timely_loose'), p.get('n_fall'))} | "
            f"{_fmt(p.get('timely_balanced'))} | {_fmt_ms(s['fold'].get('timely_balanced'))} | "
            f"{pair('fall_live', 'fall_tail')} | {pair('adl_live', 'adl_tail')} |")
    lines.append("")
    return lines


def _render_paired(paired: Dict[str, Any]) -> List[str]:
    """E* 변형과 기준 E 변형의 같은 fold/seed paired 비교 표."""
    if not paired:
        return []
    lines = ["## E* 대 E (같은 fold, 같은 seed 모델의 paired 차이)", "",
             "- Δ = E* - E, fold별 seed 평균 차이의 평균 ± 표본 표준편차, 좋아진 fold 수/공통 fold 수 "
             "(ADL 오탐률은 줄면 좋아짐)",
             "- 클립 수는 fold 합, seed 평균. 회복 = E에서 AI가 억제해 놓친 낙상 중 E*에서 알람 "
             "(그중 제때(엄격) / 첫 LEVEL1이 정지 꼬리 안)",
             "",
             "| 변형 | 기준 | ΔBalanced | 개선 fold | ΔRecall | 개선 fold | ΔADL 오탐률 | 개선 fold | "
             "E 억제 낙상 회복 | 회복 중 제때/꼬리 | 새로 탐지된 낙상 | 잃은 낙상 | 새 ADL 알람 | 사라진 ADL 알람 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        pc = paired.get(v)
        if not pc:
            continue
        d, c = pc["delta"], pc["clips"]
        nf = len(pc.get("folds") or [])

        def dms(key):
            x = d.get(key) or {}
            if x.get("mean") is None:
                return "-", "-"
            return f"{x['mean']:+.3f} ± {(x.get('std') or 0.0):.3f}", f"{x.get('n_better', 0)}/{nf}"

        cells = [x for key in _DELTA_KEYS for x in dms(key)]
        rec = c.get("recovered")
        base_s = c.get("base_supp_true")
        lines.append(
            f"| {v} | {pc.get('base', '')} | " + " | ".join(cells) + " | "
            f"{_fmt(rec, 1)}/{_fmt(base_s, 1)} | {_fmt(c.get('recovered_timely'), 1)}/{_fmt(c.get('recovered_tail'), 1)} | "
            f"{_fmt(c.get('fall_gained'), 1)} | {_fmt(c.get('fall_lost'), 1)} | "
            f"{_fmt(c.get('adl_new_alarm'), 1)} | {_fmt(c.get('adl_removed'), 1)} |")
    lines.append("")
    return lines


def _fmt_th(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if x < 0:
        return "억제 없음"
    return f"{x:.1e}" if 0 < x < 5e-4 else f"{x:.3f}"  # HGB p_fall은 1e-4 단위까지 내려감


def _render_matched_fa(mfa: Dict[str, Any]) -> List[str]:
    """같은 오탐률 비교 표 (기준 변형 결과가 있을 때만)."""
    V = mfa.get("variants") or {}
    if not V:
        return []
    ref = mfa.get("ref", "")
    lines = [f"## 같은 오탐률에서의 recall (기준 {ref}, oracle 비교: 모델 순위용)", "",
             f"- seed마다 5개 test fold를 합쳐, 첫 AI 호출 p_fall의 억제 임계값을 test 결과로 골라 근사 ADL 오탐률을 "
             f"{ref}(같은 seed)의 근사 오탐률 이하로 맞춤. test로 고른 임계값이라 운영 임계값이나 성능 추정이 아님",
             "- 근사: 첫 AI 호출 p_fall <= 임계값이면 억제, p_fall이 없으면 알람(fail-open). 억제 뒤 재진입/승격 알람은 "
             "무시하므로 실제 재생 값과 다를 수 있음 (자기 근사 vs 실제 열 참고)",
             "- 값은 seed 평균. 임계값은 평균 (최소~최대), '억제 없음'은 억제하지 않아도 이미 오탐률이 기준 이하인 경우",
             "",
             "| 변형 | 기준 오탐률 (근사 / 실제) | 맞춘 임계값 | 같은 오탐률 Recall | 그때 ADL 오탐률 | "
             "자기 Recall / 오탐률 (실제) | 자기 Recall / 오탐률 (근사) |",
             "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        r = V.get(v)
        if not r:
            continue
        th = r.get("threshold") or {}
        th_s = _fmt_th(th.get("mean"))
        if th.get("min") is not None and (th.get("max") or 0) - th["min"] > 1e-9:
            th_s += f" ({_fmt_th(th['min'])}~{_fmt_th(th['max'])})"
        flag = "" if r.get("all_matched", True) else " (못 맞춤)"
        lines.append(
            f"| {v} | {_fmt(r.get('target_fa'))} / {_fmt(r.get('ref_fa_actual'))} | {th_s} | "
            f"{_fmt(r.get('recall_at_fa'))}{flag} | {_fmt(r.get('fa_at_th'))} | "
            f"{_fmt(r.get('own_recall'))} / {_fmt(r.get('own_fa'))} | "
            f"{_fmt(r.get('own_sim_recall'))} / {_fmt(r.get('own_sim_fa'))} |")
    lines.append("")
    return lines


def _render_label_efficiency(rows: List[Dict[str, Any]], ref: Optional[str]) -> List[str]:
    """라벨 효율 표 (L/G 변형 결과가 있을 때만)."""
    if not rows:
        return []
    lines = ["## 라벨 효율 (학습 낙상 라벨 비율별, E*max 판정)", "",
             "- 0%: 양성 창 없이 정상 창만으로 학습한 이상 탐지 (G). 정상 집합에 낙상 클립의 낙상 전 창이 들어가 "
             "fall_start 라벨은 쓰며, GnfS만 ADL 클립 창만 써서 낙상 라벨을 전혀 쓰지 않음",
             "- 10/25/50%: 학습 fold의 낙상 클립을 clip_id 단위로 중첩 표본 (10% ⊂ 25% ⊂ 50%). 뺀 낙상 클립은 낙상 전 "
             "음성 창까지 모두 뺌, ADL 클립 창은 모두 사용. 100%: E0max/E1max (GnfE0는 100% 라벨 E0와 결합)",
             "- Recall/오탐률/Balanced는 fold mean ± std, AI 호출 AUC는 pooled seed 평균, "
             f"같은 오탐률 Recall은 아래 oracle 비교 (기준 {ref or '-'})",
             "",
             "| 낙상 라벨 | 변형 | 모델 | 학습 양성 클립 | Recall (fold) | ADL 오탐률 (fold) | Balanced (fold) | "
             "AI 호출 AUC | 창 ROC-AUC | 같은 오탐률 Recall |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['frac'] * 100:.0f}% | {r['variant']} | {r.get('model', '')} | {_fmt(r.get('train_pos_clips'), 1)} | "
            f"{_fmt_ms(r.get('recall'))} | {_fmt_ms(r.get('adl_fa_rate'))} | {_fmt_ms(r.get('balanced'))} | "
            f"{_fmt(r.get('ai_call_auc'))} | {_fmt_ms(r.get('window_auc'))} | {_fmt(r.get('matched_recall'))} |")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# fold 실행 (numpy / torch / sklearn 필요)
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"not JSON serializable: {type(o)}")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _window_metrics(y, p, meta=None) -> Dict[str, Any]:
    """창 단위 ROC-AUC / PR-AUC (전체 + 데이터셋별). 한 클래스뿐이면 None."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)

    def one(mask):
        yy, pp = y[mask], p[mask]
        d = {"n": int(len(yy)), "n_pos": int(yy.sum())}
        if 0 < yy.sum() < len(yy):
            d["roc_auc"] = float(roc_auc_score(yy, pp))
            d["pr_auc"] = float(average_precision_score(yy, pp))
        else:
            d["roc_auc"] = d["pr_auc"] = None
        return d

    out = one(np.ones(len(y), dtype=bool))
    if meta is not None and len(meta) == len(y):
        ds = np.asarray([str(m.get("dataset")) for m in meta])
        out["by_dataset"] = {d: one(ds == d) for d in sorted(set(ds.tolist()))}
    return out


class HandcraftedVerifier:
    """E0: 사건 구간 v2 -> 손으로 만든 통계 특징 -> sklearn 분류기 p_fall. 판단 불가면 None (fail-open)."""

    def __init__(self, clf, model_id: str = "E0-hgb"):
        self.clf = clf
        self.model_id = model_id
        self._engine = None
        self.calls = 0
        self.n_none = 0

    def bind_engine(self, engine) -> None:
        self._engine = engine

    def __call__(self, seq):
        from training.experiments.verifiers import event_window_to_v2
        from training.experiments.windows import handcrafted
        self.calls += 1
        if self._engine is None:
            self.n_none += 1
            return None
        snap = self._engine.get_incident_snapshot() or {}
        x = event_window_to_v2(snap.get("event_window"))
        if x is None:
            self.n_none += 1
            return None
        return float(self.clf.predict_proba(handcrafted(x[None]))[0, 1])


class _Standardizer:
    """채널별 표준화 (학습 창의 N, T 축 평균/표준편차). EventWindowVerifier.preprocess로도 사용."""

    def __init__(self, mu, sd):
        import numpy as np
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sd = np.asarray(sd, dtype=np.float32)

    @classmethod
    def fit(cls, X):
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(axis=(0, 2))
        sd = X.std(axis=(0, 2))
        sd = np.where(sd > 1e-6, sd, 1.0)
        return cls(mu, sd)

    def __call__(self, x):
        import numpy as np
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            return (x - self.mu[:, None]) / self.sd[:, None]
        return (x - self.mu[None, :, None]) / self.sd[None, :, None]


class NanSafeSlidingVerifier:
    """
    G: SlidingWindowVerifier(mode "max")와 같은 창들에 점수 함수를 적용하되, 점수가 NaN/inf인 창은 건너뜀.
    (이상 탐지기는 쓸 입력이 없는 창에 NaN을 냄. 원래 검증기의 np.argmax는 NaN을 최댓값으로 고름)
    모든 창이 None 또는 NaN이면 None (fail-open). 점수가 모두 유한하면 SlidingWindowVerifier max와 같은 값.
    창 생성(_windows)과 진단 카운터는 내부 SlidingWindowVerifier의 것을 그대로 씀.
    """

    def __init__(self, score_fn: Callable[[Any], Any], model_id: Optional[str] = None):
        from training.experiments.verifiers import SlidingWindowVerifier
        self._inner = SlidingWindowVerifier(score_fn, mode="max", model_id=model_id)
        self.model_id = self._inner.model_id
        self.n_nan_windows = 0   # 점수가 NaN이라 건너뛴 창 수 합
        self.n_all_nan = 0       # 창은 있었지만 모두 NaN이라 None을 낸 호출 수

    @property
    def calls(self) -> int:
        return self._inner.calls

    @property
    def n_none(self) -> int:
        return self._inner.n_none

    def bind_engine(self, engine) -> None:
        self._inner.bind_engine(engine)

    def stats(self) -> Dict[str, Any]:
        return dict(self._inner.stats(), nan_windows=self.n_nan_windows, all_nan_calls=self.n_all_nan)

    def __call__(self, seq) -> Optional[float]:
        import numpy as np

        v = self._inner
        v.calls += 1
        v.last_ps = None
        engine = v._engine_ref() if v._engine_ref is not None else v._engine
        if engine is None:
            return v._done(None)
        xs = v._windows(engine)
        ok = [i for i, x in enumerate(xs) if x is not None]
        ps = np.zeros(0)
        if ok:
            ps = np.asarray(v.score_fn(np.stack([xs[i] for i in ok])), dtype=np.float64).reshape(-1)
            v.last_ps = [float(p) for p in ps]
            fin = np.isfinite(ps)
            self.n_nan_windows += int((~fin).sum())
            if not fin.all():
                if not fin.any():
                    self.n_all_nan += 1
                ok, ps = [i for i, f in zip(ok, fin) if f], ps[fin]
        v.n_windows_skipped += len(xs) - len(ok)
        if not ok:
            return v._done(None)
        v.n_windows += len(ok)
        j = int(np.argmax(ps))
        if ok[j] != 0:
            v.n_max_later += 1
        return v._done(float(ps[j]))


def combine_mean_p(p_a, p_b):
    """GnfE0: 창별 (p_a + p_b) / 2, p_a가 NaN이면 p_b (p_b도 NaN이면 NaN)"""
    import numpy as np

    a = np.asarray(p_a, dtype=np.float64)
    b = np.asarray(p_b, dtype=np.float64)
    return np.where(np.isnan(a), b, np.where(np.isnan(b), a, (a + b) / 2.0))


def _window_metrics_nan(y, s, meta=None) -> Dict[str, Any]:
    """_window_metrics + NaN 점수 창 제외 (G: 쓸 입력이 없는 창). n_nan에 뺀 창 수, n은 남은 창 수."""
    import numpy as np

    s = np.asarray(s, dtype=np.float64)
    fin = np.isfinite(s)
    y = np.asarray(y)[fin]
    m = [mm for mm, f in zip(meta, fin) if f] if meta is not None and len(meta) == len(fin) else None
    out = _window_metrics(y, s[fin], m)
    out["n_nan"] = int((~fin).sum())
    return out


def _evaluate(test_rows, clips, params, verifier=None) -> List[Dict[str, Any]]:
    """test 행을 운영 obs로 재생해 클립 결과 목록 (obs는 매번 다시 만듦: 메모리 절약, 클립당 수십 ms)"""
    from training.experiments import replay as R
    outs = []
    for row in test_rows:
        clip = clips.get(row["clip_id"])
        if clip is None:
            raise FileNotFoundError(f"keypoints missing for {row['clip_id']}")
        obs = R.clip_to_obs(clip)
        events, _ = R.run_clip(obs, params, verifier=verifier)
        outs.append(R.clip_outcome(row, events, end_s=R.clip_end_s(obs)))
    return outs


def _train_summary(hist: Dict[str, Any], t_s: float, y_tr) -> Dict[str, Any]:
    import numpy as np
    y_tr = np.asarray(y_tr)
    return {
        "best_epoch": hist.get("best_epoch"), "stopped_epoch": hist.get("stopped_epoch"),
        "best_val_loss": hist.get("best_val_loss"), "best_val_acc": hist.get("best_val_acc"),
        "n_train": hist.get("n_train"), "n_val": hist.get("n_val"),
        "train_pos_frac": float(y_tr.mean()) if len(y_tr) else None,
        "class_weights": hist.get("class_weights"), "time_s": round(t_s, 2),
        "val_loss_curve": [e.get("val_loss") for e in hist.get("epochs", [])],
    }


def run_fold(k: int, rows: List[Dict[str, Any]], folds: Dict[str, int], clips: Dict[str, Dict[str, Any]],
             stages: Sequence[str], seeds: int, out_dir: Path, device: str = "cuda",
             epochs: Optional[int] = None, skip_existing: bool = False, f_margin: float = F_MARGIN,
             hgb_iter: int = 300) -> Dict[str, Dict[str, Any]]:
    """
    fold k 하나의 모든 요청 단계를 실행하고 fold{k}/{variant}.json을 씁니다. 반환: {variant: 결과}
    epochs: 학습 epoch 상한 덮어쓰기 (빠른 점검용, 기본은 레시피 그대로)
    """
    import pickle

    import numpy as np
    import torch

    from training.experiments import folds as FO
    from training.experiments import models as M
    from training.experiments import replay as R
    from training.experiments import trainer as T
    from training.experiments import windows as W
    from training.experiments.verifiers import EventWindowVerifier, V1SeqVerifier

    out_dir = Path(out_dir)
    fdir = out_dir / f"fold{k}"
    mdir = fdir / "models"
    fdir.mkdir(parents=True, exist_ok=True)

    train_rows, test_rows = FO.split_fold(rows, folds, k)
    tr_rows, va_rows = FO.split_train_val(train_rows, frac=VAL_FRAC, seed=VAL_SPLIT_SEED)
    va_eval_rows = [r for r in va_rows if r.get("dataset") in FO.EVAL_DATASETS]
    p_off = R.study_params("off")
    p_veto = R.study_params("veto")
    seed_list = list(range(int(seeds)))
    _log(f"fold {k}: train {len(train_rows)} (fit {len(tr_rows)}, val {len(va_rows)}), test {len(test_rows)} "
         f"(fall {sum(r['label'] == 'fall' for r in test_rows)}, adl {sum(r['label'] == 'adl' for r in test_rows)})")

    def recipe(name):
        rc = {"name": name}
        if epochs is not None:
            rc["epochs"] = int(epochs)
        return rc

    results: Dict[str, Dict[str, Any]] = {}

    def done(variant) -> bool:
        p = fdir / f"{variant}.json"
        if skip_existing and p.exists():
            results[variant] = json.loads(p.read_text(encoding="utf-8"))
            _log(f"fold {k}: {variant} exists, skipped")
            return True
        return False

    def save(variant, res):
        res.update({"variant": variant, "fold": k, "n_test": len(test_rows)})
        _write_json(fdir / f"{variant}.json", res)
        results[variant] = res
        s = R.summarize([o for run in res["runs"] for o in run["outcomes"]])
        _log(f"fold {k}: {variant} done in {res['time_s']:.0f}s | recall {s['recall']:.3f} "
             f"adl_fa {s['adl_fa_rate']:.3f} balanced {s['balanced']:.3f} supp_true {s['suppressed_true_falls']} "
             f"(runs pooled)")

    # ---------------- A: 규칙만 ----------------
    if "A" in stages and not done("A"):
        t0 = time.time()
        outs = _evaluate(test_rows, clips, p_off)
        save("A", {"params": None, "runs": [{"seed": None, "outcomes": outs, "summary": R.summarize(outs),
                                             "threshold": None}], "time_s": time.time() - t0})

    # ---------------- B / C: 원래 방식 ----------------
    if any(s in stages for s in ("B", "C")):
        X_te_o, y_te_o, m_te_o = W.windows_original_urfd(test_rows, clips, return_meta=True)
        for variant in ("B", "C"):
            if variant not in stages or done(variant):
                continue
            t0 = time.time()
            sel = (lambda r: r.get("dataset") == "urfd") if variant == "B" else (lambda r: True)
            X_tr, y_tr = W.windows_original_urfd([r for r in tr_rows if sel(r)], clips)
            X_va, y_va = W.windows_original_urfd([r for r in va_rows if sel(r)], clips)
            _log(f"fold {k}: {variant} windows train {X_tr.shape} pos {y_tr.mean():.3f}, val {X_va.shape}")
            runs = []
            n_params = None
            for seed in seed_list:
                T.set_seed(seed)
                model = M.Fall1DCNN(input_channels=34, seq_len=30, num_classes=2)
                n_params = M.count_params(model)
                ts = time.time()
                model, hist = T.train_classifier(model, X_tr, y_tr, X_va, y_va, recipe("original"),
                                                 seed=seed, device=device)
                tr_s = time.time() - ts
                p_te = T.predict_proba(model, X_te_o)
                ver = V1SeqVerifier(model, device=device, model_id=f"{variant}-s{seed}")
                outs = _evaluate(test_rows, clips, p_veto, ver)
                runs.append({"seed": seed, "outcomes": outs, "summary": R.summarize(outs), "threshold": DEFAULT_TH,
                             "window": _window_metrics(y_te_o, p_te, m_te_o),
                             "train": _train_summary(hist, tr_s, y_tr),
                             "verifier": {"calls": ver.calls, "none": ver.n_none}})
                _log(f"fold {k}: {variant} seed {seed} best ep {hist['best_epoch']} train {tr_s:.0f}s "
                     f"win auc {runs[-1]['window']['roc_auc']}")
            save(variant, {"params": n_params, "runs": runs, "time_s": time.time() - t0,
                           "n_windows": {"train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te_o))}})

    # ---------------- D: 운영 정렬 v1 ----------------
    if "D" in stages and not done("D"):
        t0 = time.time()
        X_tr, y_tr = W.windows_runtime_v1(tr_rows, clips, p_off)
        X_va, y_va = W.windows_runtime_v1(va_rows, clips, p_off)
        X_te, y_te, m_te = W.windows_runtime_v1(test_rows, clips, p_off, return_meta=True)
        _log(f"fold {k}: D windows train {X_tr.shape} pos {y_tr.mean():.3f}, val {X_va.shape}, test {X_te.shape} "
             f"({time.time() - t0:.0f}s)")
        runs = []
        n_params = None
        for seed in seed_list:
            T.set_seed(seed)
            model = M.Fall1DCNN(input_channels=34, seq_len=30, num_classes=2)
            n_params = M.count_params(model)
            ts = time.time()
            model, hist = T.train_classifier(model, X_tr, y_tr, X_va, y_va, recipe("improved"),
                                             seed=seed, device=device)
            tr_s = time.time() - ts
            p_te = T.predict_proba(model, X_te)
            ver = V1SeqVerifier(model, device=device, model_id=f"D-s{seed}")
            outs = _evaluate(test_rows, clips, p_veto, ver)
            runs.append({"seed": seed, "outcomes": outs, "summary": R.summarize(outs), "threshold": DEFAULT_TH,
                         "window": _window_metrics(y_te, p_te, m_te), "train": _train_summary(hist, tr_s, y_tr),
                         "verifier": {"calls": ver.calls, "none": ver.n_none}})
            _log(f"fold {k}: D seed {seed} best ep {hist['best_epoch']}/{hist['stopped_epoch']} train {tr_s:.0f}s "
                 f"win auc {runs[-1]['window']['roc_auc']}")
        save("D", {"params": n_params, "runs": runs, "time_s": time.time() - t0,
                   "n_windows": {"train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te))}})

    # ---------------- E: 사건 구간 v2 ----------------
    # 학습된 E 모델: {(variant, seed): verifier 생성 함수}, {(variant, seed): (종류, 모델, 표준화)} (Es가 다시 씀)
    e_makers: Dict[Tuple[str, int], Callable[[], Any]] = {}
    e_models: Dict[Tuple[str, int], Tuple[str, Any, Any]] = {}

    def _nn_maker(model, std, vid):
        return lambda: EventWindowVerifier(model, device=device, model_id=vid, preprocess=std)

    def _hgb_maker(clf, vid):
        return lambda: HandcraftedVerifier(clf, model_id=vid)

    e_needed = [v for v in STAGE_VARIANTS["E"] if "E" in stages and not done(v)]
    if e_needed:
        t_stage = time.time()
        X_va, y_va = W.windows_event_v2(va_rows, clips, p_off, augment=False)
        X_te, y_te, m_te = W.windows_event_v2(test_rows, clips, p_off, augment=False, return_meta=True)
        runs: Dict[str, List[Dict[str, Any]]] = {v: [] for v in e_needed}
        n_params: Dict[str, Any] = {}
        t_var = {v: 0.0 for v in e_needed}
        n_win = {}
        for seed in seed_list:
            ts = time.time()
            X_tr, y_tr, m_tr = W.windows_event_v2(tr_rows, clips, p_off, rng=np.random.default_rng(1000 + seed),
                                                  augment=True, return_meta=True)
            t_build = time.time() - ts
            n_win = {"train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te)),
                     "train_kinds": W.count_windows(m_tr, by=("kind", "y"))}
            _log(f"fold {k}: E seed {seed} windows train {X_tr.shape} pos {y_tr.mean():.3f}, val {X_va.shape}, "
                 f"test {X_te.shape} ({t_build:.0f}s)")
            std = _Standardizer.fit(X_tr)
            for v in e_needed:
                ts = time.time()
                if v == "E0":
                    clf = _make_hgb(seed, hgb_iter)
                    clf.fit(W.handcrafted(X_tr), y_tr)
                    tr_s = time.time() - ts
                    p_te = clf.predict_proba(W.handcrafted(X_te))[:, 1] if len(X_te) else np.zeros(0)
                    n_params[v] = int(sum(len(pr.nodes) for it in clf._predictors for pr in it))
                    train = {"time_s": round(tr_s, 2), "n_train": int(len(y_tr)),
                             "train_pos_frac": float(y_tr.mean()), "n_iter": int(clf.n_iter_)}
                    mdir.mkdir(parents=True, exist_ok=True)
                    with open(mdir / f"{v}_s{seed}.pkl", "wb") as f:
                        pickle.dump(clf, f)
                    maker = _hgb_maker(clf, f"{v}-s{seed}")
                    e_models[(v, seed)] = ("hgb", clf, None)
                else:
                    T.set_seed(seed)
                    model = M.Fall1DCNNGAP(W.V2_CHANNELS) if v == "E1" else M.FallTCN(W.V2_CHANNELS)
                    n_params[v] = M.count_params(model)
                    model, hist = T.train_classifier(model, std(X_tr), y_tr, std(X_va), y_va, recipe("improved"),
                                                     seed=seed, device=device)
                    tr_s = time.time() - ts
                    p_te = T.predict_proba(model, std(X_te))
                    train = _train_summary(hist, tr_s, y_tr)
                    mdir.mkdir(parents=True, exist_ok=True)
                    torch.save({"arch": v, "state_dict": model.state_dict(), "mu": std.mu, "sd": std.sd},
                               mdir / f"{v}_s{seed}.pt")
                    maker = _nn_maker(model, std, f"{v}-s{seed}")
                    e_models[(v, seed)] = ("nn", model, std)
                e_makers[(v, seed)] = maker
                ver = maker()
                outs = _evaluate(test_rows, clips, p_veto, ver)
                runs[v].append({"seed": seed, "outcomes": outs, "summary": R.summarize(outs),
                                "threshold": DEFAULT_TH, "window": _window_metrics(y_te, p_te, m_te), "train": train,
                                "verifier": {"calls": ver.calls, "none": ver.n_none}})
                t_var[v] += time.time() - ts + t_build / len(e_needed)
                _log(f"fold {k}: {v} seed {seed} train {tr_s:.0f}s win auc {runs[v][-1]['window']['roc_auc']}")
        for v in e_needed:
            save(v, {"params": n_params.get(v), "params_unit": "tree nodes" if v == "E0" else "weights",
                     "runs": runs[v], "time_s": t_var[v], "n_windows": n_win})
        _log(f"fold {k}: E total {time.time() - t_stage:.0f}s")

    # ---------------- E*: 저장된 E 모델로 진입 이후 여러 창 판정 (재학습 없음) ----------------
    es_needed = [v for v in STAGE_VARIANTS["Es"] if "Es" in stages and not done(v)]
    for v in es_needed:
        from training.experiments.verifiers import SLIDE_STEP_S, SlidingWindowVerifier
        ev, mode = ES_BASE[v]
        t0 = time.time()
        e_res = results.get(ev)
        if e_res is None and (fdir / f"{ev}.json").exists():
            e_res = json.loads((fdir / f"{ev}.json").read_text(encoding="utf-8"))
        runs_s = []
        for seed in seed_list:
            entry = e_models.get((ev, seed)) or _load_e_model(mdir, ev, seed)
            if entry is None:
                raise FileNotFoundError(f"fold {k}: {ev} seed {seed} model not found (run stage E first)")
            e_models[(ev, seed)] = entry
            ver = SlidingWindowVerifier(_e_scorer(entry, device), mode=mode, model_id=f"{v}-s{seed}")
            outs = _evaluate(test_rows, clips, p_veto, ver)
            e_run = None
            if e_res is not None:
                e_run = next((r for r in e_res["runs"] if r.get("seed") == seed), None)
            # 창 단위 지표/학습 기록은 같은 모델(E)의 것
            runs_s.append({"seed": seed, "outcomes": outs, "summary": R.summarize(outs), "threshold": DEFAULT_TH,
                           "window": (e_run or {}).get("window"), "train": (e_run or {}).get("train", {}),
                           "verifier": ver.stats()})
            st = ver.stats()
            _log(f"fold {k}: {v} seed {seed} calls {st['calls']} none {st['none']} windows {st['windows']} "
                 f"skipped {st['windows_skipped']} max_later {st['max_later']}")
        save(v, {"params": (e_res or {}).get("params"), "params_unit": (e_res or {}).get("params_unit", "weights"),
                 "runs": runs_s, "time_s": time.time() - t0, "base": ev, "mode": mode, "step_s": SLIDE_STEP_S})

    # ---------------- F: 검증 기반 임계값 ----------------
    # F{i}: 검증 낙상 recall 1.0 유지 (스펙), F{i}b: 검증 balanced 최대 (참고 변형). 같은 검증 재생을 공유.
    if "F" in stages:
        for ev in ("E0", "E1", "E2"):
            idx = ev[1]
            fv_s, fv_b = f"F{idx}", f"F{idx}b"
            todo = [fv for fv in (fv_s, fv_b) if not done(fv)]
            if not todo:
                continue
            t0 = time.time()
            runs_f: Dict[str, List[Dict[str, Any]]] = {fv: [] for fv in todo}
            e_res = results.get(ev)
            if e_res is None and (fdir / f"{ev}.json").exists():
                e_res = json.loads((fdir / f"{ev}.json").read_text(encoding="utf-8"))
            for seed in seed_list:
                maker = e_makers.get((ev, seed)) or _load_e_maker(mdir, ev, seed, device)
                if maker is None:
                    raise FileNotFoundError(f"fold {k}: {ev} seed {seed} model not found (run stage E first)")
                # 검증 클립 shadow 재생 (억제 없음) -> 클립별 첫 p_fall
                val_outs = _evaluate(va_eval_rows, clips, R.study_params("shadow"), maker())
                pts = val_points(val_outs)
                fall_p = [p for lab, _, p in pts if lab == "fall" and p is not None]
                ths = {fv_s: select_threshold(fall_p, margin=f_margin), fv_b: select_threshold_balanced(pts)}
                e_run = None
                if e_res is not None:
                    e_run = next((r for r in e_res["runs"] if r.get("seed") == seed), None)
                for fv in todo:
                    th = ths[fv]
                    val_info = {"n_val_clips": len(val_outs), "n_val_fall_reached": len(fall_p),
                                "min_fall_p": min(fall_p) if fall_p else None,
                                "balanced_at_th": sim_balanced(pts, th),
                                "balanced_at_default": sim_balanced(pts, DEFAULT_TH)}
                    outs = _evaluate(test_rows, clips, R.study_params("veto", ai_suppress_pfall_th=th), maker())
                    runs_f[fv].append({"seed": seed, "outcomes": outs, "summary": R.summarize(outs), "threshold": th,
                                       "val": val_info, "window": (e_run or {}).get("window"),
                                       "train": (e_run or {}).get("train", {})})
                    _log(f"fold {k}: {fv} seed {seed} th {th:.3f} (val min fall p {val_info['min_fall_p']}, "
                         f"val balanced {val_info['balanced_at_th']})")
            for fv in todo:
                save(fv, {"params": (e_res or {}).get("params"),
                          "params_unit": (e_res or {}).get("params_unit", "weights"),
                          "runs": runs_f[fv], "time_s": (time.time() - t0) / len(todo), "base": ev,
                          "rule": "val_recall_1" if fv == fv_s else "val_balanced_max", "margin": f_margin})

    # ---------------- L: 라벨 효율 (양성 학습 클립 일부만, E*max 판정) ----------------
    l_needed = [v for v in STAGE_VARIANTS["L"] if "L" in stages and not done(v)]
    if l_needed:
        from training.experiments.verifiers import SLIDE_STEP_S, SlidingWindowVerifier
        t_stage = time.time()
        X_va, y_va = W.windows_event_v2(va_rows, clips, p_off, augment=False)
        X_te, y_te, m_te = W.windows_event_v2(test_rows, clips, p_off, augment=False, return_meta=True)
        runs_l: Dict[str, List[Dict[str, Any]]] = {v: [] for v in l_needed}
        n_params_l: Dict[str, Any] = {}
        t_var = {v: 0.0 for v in l_needed}
        fracs = sorted({L_BASE[v][1] for v in l_needed})
        for seed in seed_list:
            ts = time.time()
            # E와 같은 학습 창 (같은 증강 rng) -> 양성 창만 clip_id 단위로 줄임
            X_tr, y_tr, m_tr = W.windows_event_v2(tr_rows, clips, p_off, rng=np.random.default_rng(1000 + seed),
                                                  augment=True, return_meta=True)
            t_build = time.time() - ts
            pos_all = positive_clip_ids(m_tr, y_tr)
            kept = nested_clip_subsets(pos_all, fracs, fold=k, seed=seed)
            _log(f"fold {k}: L seed {seed} windows train {X_tr.shape} pos clips {len(pos_all)} "
                 f"kept {[len(kept[f]) for f in fracs]} ({t_build:.0f}s)")
            for v in l_needed:
                ts = time.time()
                base, frac = L_BASE[v]
                keep = np.asarray(keep_positive_mask(m_tr, y_tr, kept[frac]), dtype=bool)
                Xs, ys = X_tr[keep], y_tr[keep]
                ms = [m for m, kk in zip(m_tr, keep) if kk]
                entry, n_par, train = _fit_supervised(base, Xs, ys, X_va, y_va, seed, device, recipe("improved"),
                                                      hgb_iter)
                n_params_l[v] = n_par
                tr_s = time.time() - ts
                mdir.mkdir(parents=True, exist_ok=True)
                _save_e_model(entry, base, mdir / f"{v}_s{seed}")
                kind, model, std = entry
                if not len(X_te):
                    p_te = np.zeros(0)
                elif kind == "hgb":
                    p_te = model.predict_proba(W.handcrafted(X_te))[:, 1]
                else:
                    p_te = T.predict_proba(model, std(X_te))
                pos_ds: Dict[str, int] = {}
                for c in kept[frac]:
                    d = c.split("/", 1)[0]
                    pos_ds[d] = pos_ds.get(d, 0) + 1
                train.update({"label_frac": frac, "n_pos_clips": len(kept[frac]), "n_pos_clips_total": len(pos_all),
                              "n_pos_windows": int(ys.sum()), "n_neg_windows": int(len(ys) - ys.sum()),
                              "pos_clips_by_prefix": pos_ds, "train_kinds": W.count_windows(ms, by=("kind", "y"))})
                ver = SlidingWindowVerifier(_e_scorer(entry, device), mode="max", model_id=f"{v}-s{seed}")
                outs = _evaluate(test_rows, clips, p_veto, ver)
                runs_l[v].append({"seed": seed, "outcomes": outs, "summary": R.summarize(outs),
                                  "threshold": DEFAULT_TH, "window": _window_metrics(y_te, p_te, m_te),
                                  "train": train, "verifier": ver.stats()})
                t_var[v] += time.time() - ts + t_build / len(l_needed)
                _log(f"fold {k}: {v} seed {seed} pos clips {len(kept[frac])}/{len(pos_all)} train {tr_s:.0f}s "
                     f"win auc {runs_l[v][-1]['window']['roc_auc']}")
        for v in l_needed:
            base, frac = L_BASE[v]
            save(v, {"params": n_params_l.get(v), "params_unit": "tree nodes" if base == "E0" else "weights",
                     "runs": runs_l[v], "time_s": t_var[v], "base": base, "label_frac": frac, "mode": "max",
                     "step_s": SLIDE_STEP_S, "sampling": "positive train clips by clip_id, nested 50>25>10, "
                     "random.Random(f'{fold}-{seed}-{frac}')",
                     "n_windows": {"val": int(len(y_va)), "test": int(len(y_te))}})
        _log(f"fold {k}: L total {time.time() - t_stage:.0f}s")

    # ---------------- G: 정상 창만으로 학습한 이상 탐지 (보정 p, E*max 판정) ----------------
    g_needed = [v for v in STAGE_VARIANTS["G"] if "G" in stages and not done(v)]
    if g_needed:
        from training import anomaly as AN
        from training.experiments.verifiers import SLIDE_STEP_S
        t_stage = time.time()
        X_va, y_va, m_va = W.windows_event_v2(va_rows, clips, p_off, augment=False, return_meta=True)
        va_neg = np.asarray(y_va) == 0
        X_va_n = X_va[va_neg]
        # 엄격한 정상 집합 (GnfS): ADL 클립의 창만
        va_adl = np.asarray([m.get("label") != "fall" for m in m_va], dtype=bool) & va_neg
        X_va_s = X_va[va_adl]
        X_te, y_te, m_te = W.windows_event_v2(test_rows, clips, p_off, augment=False, return_meta=True)
        # 학습할 탐지기: 요청된 G{kind} + GnfE0용 nf (Gnf를 다시 돌리지 않으면 저장된 모델을 먼저 찾음)
        fit_kinds = [kd for kd in AN.KINDS if any(G_KIND.get(v) == kd for v in g_needed)]
        strict_needed = [v for v in g_needed if v in G_STRICT]
        g_trainers = [v for v in g_needed if v in G_KIND or v in G_STRICT]
        combo_kinds = sorted({G_COMBO[v][0] for v in g_needed if v in G_COMBO})
        runs_g: Dict[str, List[Dict[str, Any]]] = {v: [] for v in g_needed}
        n_params_g: Dict[str, Any] = {}
        t_var = {v: 0.0 for v in g_needed}
        n_win: Dict[str, Any] = {}

        def g_eval(v, seed, score_fn, s_te, train, cal_params, extra=None):
            ts_e = time.time()
            ver = NanSafeSlidingVerifier(score_fn, model_id=f"{v}-s{seed}")
            outs = _evaluate(test_rows, clips, p_veto, ver)
            win = _window_metrics_nan(y_te, s_te, m_te)
            run = {"seed": seed, "outcomes": outs, "summary": R.summarize(outs), "threshold": DEFAULT_TH,
                   "window": win, "train": train, "calibrator": cal_params, "verifier": ver.stats()}
            run.update(extra or {})
            runs_g[v].append(run)
            st = ver.stats()
            _log(f"fold {k}: {v} seed {seed} win auc {win['roc_auc']} (nan {win['n_nan']}) calls {st['calls']} "
                 f"none {st['none']} nan windows {st['nan_windows']} all-nan calls {st['all_nan_calls']} "
                 f"eval {time.time() - ts_e:.0f}s")
            return time.time() - ts_e

        for seed in seed_list:
            # E와 같은 학습 창 (같은 증강 rng) -> 음성(정상) 창만. 탐지기를 실제로 학습할 때만 만듦 (저장된 모델만
            # 쓰는 GnfE0 단독 실행이면 만들지 않음)
            n_win = {"val_normal": int(len(X_va_n)), "val_normal_adl": int(len(X_va_s)), "test": int(len(y_te)),
                     "test_pos": int(np.sum(y_te))}
            tr_cache: Dict[str, Any] = {}

            def get_tr(seed=seed, tr_cache=tr_cache, n_win=n_win):
                if not tr_cache:
                    ts_b = time.time()
                    X_tr, y_tr, m_tr = W.windows_event_v2(tr_rows, clips, p_off,
                                                          rng=np.random.default_rng(1000 + seed),
                                                          augment=True, return_meta=True)
                    neg = np.asarray(y_tr) == 0
                    adl = np.asarray([m.get("label") != "fall" for m in m_tr], dtype=bool) & neg
                    tr_cache["X_n"] = X_tr[neg]
                    tr_cache["X_s"] = X_tr[adl]
                    m_tr_n = [m for m, ok in zip(m_tr, neg) if ok]
                    m_tr_s = [m for m, ok in zip(m_tr, adl) if ok]
                    del X_tr
                    tr_cache["t_build"] = time.time() - ts_b
                    n_win.update({"train_normal": int(neg.sum()), "train_normal_adl": int(adl.sum()),
                                  "train_dropped_pos": int((~neg).sum()),
                                  "train_kinds": W.count_windows(m_tr_n, by=("kind", "y")),
                                  "train_kinds_adl": W.count_windows(m_tr_s, by=("kind", "y"))})
                    _log(f"fold {k}: G seed {seed} normal windows train {tr_cache['X_n'].shape} "
                         f"(ADL clips only {tr_cache['X_s'].shape}), val {X_va_n.shape} ({len(X_va_s)}), "
                         f"test {X_te.shape} ({tr_cache['t_build']:.0f}s)")
                return tr_cache

            def build_share():
                # 창 생성 시간은 이번 실행에서 실제로 학습하는 변형들(GnfE0 제외)이 나눠 가짐
                return tr_cache.get("t_build", 0.0) / max(1, len(g_trainers))

            fitted: Dict[str, Tuple[Any, Any]] = {}
            for kind in fit_kinds + [kd for kd in combo_kinds if kd not in fit_kinds]:
                v = f"G{kind}"
                if v not in g_needed:
                    got = _load_g_model(mdir, kind, seed, device)
                    if got is not None:
                        fitted[kind] = got
                        _log(f"fold {k}: {v} seed {seed} loaded from {mdir} (for {combo_kinds})")
                        continue
                X_tr_n = get_tr()["X_n"]
                ts = time.time()
                over = {"max_epochs": int(epochs)} if epochs is not None and kind != "md" else {}
                det = AN.fit_detector(kind, X_tr_n, X_va_n, seed=seed, device=device, **over)
                s_va = det.score(X_va_n)
                cal = AN.Calibrator.fit(s_va)
                tr_s = time.time() - ts
                mdir.mkdir(parents=True, exist_ok=True)
                _save_g_model(det, cal, mdir / f"{v}_s{seed}.pt")
                fitted[kind] = (det, cal)
                n_params_g[v] = det.n_params
                _log(f"fold {k}: {v} seed {seed} fit {tr_s:.0f}s best ep {det.info.get('best_epoch')}/"
                     f"{det.info.get('stopped_epoch')} val thr {cal.threshold:.4g} "
                     f"(val nan {int(np.isnan(s_va).sum())}/{len(s_va)})")
                if v not in g_needed:
                    continue
                s_te = det.score(X_te)
                train = _g_train_summary(det, tr_s, s_va, n_win)
                t_eval = g_eval(v, seed, AN.calibrated_scorer(det, cal), s_te, train, cal.params(),
                                {"window_veto": _veto_rates(cal(s_te), y_te)})
                t_var[v] += tr_s + t_eval + build_share()
            for v in strict_needed:
                kind = G_STRICT[v]
                X_tr_s = get_tr()["X_s"]
                ts = time.time()
                over = {"max_epochs": int(epochs)} if epochs is not None and kind != "md" else {}
                det = AN.fit_detector(kind, X_tr_s, X_va_s, seed=seed, device=device, **over)
                s_va = det.score(X_va_s)
                cal = AN.Calibrator.fit(s_va)
                tr_s = time.time() - ts
                mdir.mkdir(parents=True, exist_ok=True)
                _save_g_model(det, cal, mdir / f"{v}_s{seed}.pt")
                n_params_g[v] = det.n_params
                _log(f"fold {k}: {v} seed {seed} fit {tr_s:.0f}s on ADL-clip windows {X_tr_s.shape} "
                     f"best ep {det.info.get('best_epoch')}/{det.info.get('stopped_epoch')} val thr {cal.threshold:.4g}")
                s_te = det.score(X_te)
                train = _g_train_summary(det, tr_s, s_va, n_win, strict=True)
                t_eval = g_eval(v, seed, AN.calibrated_scorer(det, cal), s_te, train, cal.params(),
                                {"window_veto": _veto_rates(cal(s_te), y_te)})
                t_var[v] += tr_s + t_eval + build_share()
            for v in (vv for vv in g_needed if vv in G_COMBO):
                kind, ev = G_COMBO[v]
                ts = time.time()
                entry = e_models.get((ev, seed)) or _load_e_model(mdir, ev, seed)
                if entry is None:
                    raise FileNotFoundError(f"fold {k}: {ev} seed {seed} model not found (run stage E first)")
                e_models[(ev, seed)] = entry
                det, cal = fitted[kind]
                g_fn, e_fn = AN.calibrated_scorer(det, cal), _e_scorer(entry, device)

                def comb_fn(X, g_fn=g_fn, e_fn=e_fn):
                    return combine_mean_p(g_fn(X), e_fn(X))

                p_te = comb_fn(X_te) if len(X_te) else np.zeros(0)
                n_params_g[v] = det.n_params
                train = {"time_s": None, "n_pos_clips": None, "detector": f"G{kind}_s{seed}",
                         "e_model": f"{ev}_s{seed}", "combine": "mean(p_G, p_E), p_G NaN -> p_E"}
                t_eval = g_eval(v, seed, comb_fn, p_te, train, cal.params(), {"window_veto": _veto_rates(p_te, y_te)})
                t_var[v] += time.time() - ts
        for v in g_needed:
            kind = G_KIND.get(v) or G_STRICT.get(v) or G_COMBO[v][0]
            fit_on = ("val windows of ADL clips (va_rows, label adl)" if v in G_STRICT
                      else "val normal windows (va_rows, y == 0, incl. pre-fall windows of fall clips)")
            res = {"params": n_params_g.get(v), "params_unit": G_PARAMS_UNIT[kind], "runs": runs_g[v],
                   "time_s": t_var[v], "kind": kind, "mode": "max", "step_s": SLIDE_STEP_S, "label_frac": 0.0,
                   "normal_set": "adl_clips_only" if v in G_STRICT else "negative_windows",
                   "calibration": {"q": 0.8, "p_at_q": DEFAULT_TH, "fit_on": fit_on},
                   "n_windows": n_win}
            if v in G_COMBO:
                res.update({"label_frac": 1.0, "base": G_COMBO[v][1], "params_unit": "weights (+E0 trees)"})
            save(v, res)
        _log(f"fold {k}: G total {time.time() - t_stage:.0f}s")

    # ---------------- R: 규칙 엔진 변경 (R1: 짧은 미검출 유예) ----------------
    if "R" in stages and not done("AR1"):
        t0 = time.time()
        outs = _evaluate(test_rows, clips, R.study_params("off", lost_grace_s=LOST_GRACE_R1))
        save("AR1", {"params": None, "lost_grace_s": LOST_GRACE_R1,
                     "runs": [{"seed": None, "outcomes": outs, "summary": R.summarize(outs), "threshold": None}],
                     "time_s": time.time() - t0})
    r_needed = [vv for vv in ("E0maxR1", "E1maxR1") if "R" in stages and not done(vv)]
    if r_needed:
        from training.experiments.verifiers import SlidingWindowVerifier as _SWV
    for v in r_needed:
        ev = v[:2]
        if any(_load_e_model(mdir, ev, sd) is None for sd in seed_list):
            # 다른 키포인트 추출본(--kp-dir)으로 규칙만 잴 때처럼 E 모델이 없으면 이 변형은 건너뜀
            _log(f"fold {k}: {v} skipped ({ev} models not found in {mdir})")
            continue
        t0 = time.time()
        runs_r = []
        for seed in seed_list:
            entry = _load_e_model(mdir, ev, seed)
            ver = _SWV(_e_scorer(entry, device), mode="max", model_id=f"{v}-s{seed}")
            outs = _evaluate(test_rows, clips, R.study_params("veto", lost_grace_s=LOST_GRACE_R1), ver)
            runs_r.append({"seed": seed, "outcomes": outs, "summary": R.summarize(outs), "threshold": DEFAULT_TH,
                           "verifier": ver.stats()})
        save(v, {"params": None, "lost_grace_s": LOST_GRACE_R1, "base": f"{ev}max", "runs": runs_r,
                 "time_s": time.time() - t0})
    return results


def _veto_rates(p, y) -> Dict[str, Any]:
    """창 단위 억제 비율 (p <= 0.4): 정상 창 중 억제, 양성 창 중 통과. NaN p는 판단 불가(통과)로 셈."""
    import numpy as np

    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(int)
    veto = np.isfinite(p) & (p <= DEFAULT_TH)
    neg, pos = y == 0, y == 1
    return {"neg_vetoed": float(veto[neg].mean()) if neg.any() else None,
            "pos_passed": float((~veto[pos]).mean()) if pos.any() else None,
            "n_nan": int((~np.isfinite(p)).sum())}


def _g_train_summary(det, t_s: float, s_va, n_win: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """
    이상 탐지기 학습 기록 요약 (JSON용): epoch 곡선, best epoch, 검증 점수 통계, 낙상 라벨 0개.
    strict: ADL 클립 창만으로 학습한 변형 (학습/검증 창 수와 창 종류 분포를 그 집합 기준으로 기록)
    """
    import numpy as np

    info = det.info or {}
    eps = info.get("epochs") or []
    s_va = np.asarray(s_va, dtype=np.float64)
    fin = s_va[np.isfinite(s_va)]
    out = {"time_s": round(t_s, 2), "n_pos_clips": 0, "label_frac": 0.0,
           "n_train": n_win.get("train_normal_adl" if strict else "train_normal"),
           "n_val": n_win.get("val_normal_adl" if strict else "val_normal"),
           "best_epoch": info.get("best_epoch"), "stopped_epoch": info.get("stopped_epoch"),
           "best_val_loss": info.get("best_val_loss"), "diverged": info.get("diverged"),
           "train_loss_curve": [e.get("train_loss") for e in eps], "val_loss_curve": [e.get("val_loss") for e in eps],
           "val_score": {"n_nan": int(len(s_va) - len(fin)),
                         "mean": float(fin.mean()) if len(fin) else None,
                         "q50": float(np.quantile(fin, 0.5)) if len(fin) else None,
                         "q80": float(np.quantile(fin, 0.8)) if len(fin) else None},
           "train_kinds": n_win.get("train_kinds_adl" if strict else "train_kinds")}
    for key in ("n_train_segments", "n_val_segments", "noise", "recipe", "arch", "device", "shrinkage",
                "n_features", "train_loss", "val_loss"):
        if key in info:
            out[key] = info[key]
    return out


def _save_g_model(det, cal, path: Path) -> Path:
    """이상 탐지기 + 보정기 -> path (.pt, torch.save): {"kind", "detector": state(), "calibrator": params()}"""
    import torch

    torch.save({"kind": det.kind, "detector": det.state(), "calibrator": cal.params()}, path)
    return path


def _load_g_model(mdir: Path, kind: str, seed: int, device: str = "cpu") -> Optional[Tuple[Any, Any]]:
    """저장된 G{kind}_s{seed}.pt -> (Detector, Calibrator), 없으면 None"""
    import torch

    from training import anomaly as AN

    p = Path(mdir) / f"G{kind}_s{seed}.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return AN.load_detector(kind, ck["detector"], device=device), AN.Calibrator.from_params(ck["calibrator"])


def _make_hgb(seed: int, hgb_iter: int = 300):
    """
    E0 분류기. early_stopping=True를 명시: sklearn 기본값 'auto'는 학습 표본이 1만 개를 넘을 때만 켜져,
    양성 클립을 줄인 L 변형(1만 개 미만)과 E0(약 1.2만 개)의 학습 방식이 달라짐. 1만 개 초과에서는 'auto'와 같은 모델.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=int(hgb_iter), learning_rate=0.1, class_weight="balanced",
                                          early_stopping=True, validation_fraction=0.1, random_state=seed)


def _fit_supervised(base: str, X_tr, y_tr, X_va, y_va, seed: int, device: str, recipe: Dict[str, Any],
                    hgb_iter: int = 300) -> Tuple[Tuple[str, Any, Any], Any, Dict[str, Any]]:
    """
    단계 E와 같은 레시피로 E0 (HistGradientBoosting, 손설계 특징) 또는 E1/E2 (표준화 v2 + NN) 학습 (L 단계용).
    반환: (모델 항목 ("hgb", clf, None) / ("nn", model, 표준화), 파라미터 수, 학습 기록 요약)
    """
    import numpy as np

    from training.experiments import models as M
    from training.experiments import trainer as T
    from training.experiments import windows as W

    ts = time.time()
    if base == "E0":
        clf = _make_hgb(seed, hgb_iter)
        clf.fit(W.handcrafted(X_tr), y_tr)
        n_par = int(sum(len(pr.nodes) for it in clf._predictors for pr in it))
        train = {"time_s": round(time.time() - ts, 2), "n_train": int(len(y_tr)),
                 "train_pos_frac": float(np.mean(y_tr)) if len(y_tr) else None, "n_iter": int(clf.n_iter_)}
        return ("hgb", clf, None), n_par, train
    std = _Standardizer.fit(X_tr)
    T.set_seed(seed)
    model = M.Fall1DCNNGAP(W.V2_CHANNELS) if base == "E1" else M.FallTCN(W.V2_CHANNELS)
    n_par = M.count_params(model)
    model, hist = T.train_classifier(model, std(X_tr), y_tr, std(X_va), y_va, recipe, seed=seed, device=device)
    return ("nn", model, std), n_par, _train_summary(hist, time.time() - ts, y_tr)


def _save_e_model(entry: Tuple[str, Any, Any], base: str, stem: Path) -> Path:
    """모델 항목 -> stem.pkl (hgb) / stem.pt (nn, E와 같은 형식: arch, state_dict, mu, sd). _load_e_model이 읽음."""
    import pickle

    import torch

    kind, model, std = entry
    if kind == "hgb":
        path = stem.with_suffix(".pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return path
    path = stem.with_suffix(".pt")
    torch.save({"arch": base, "state_dict": model.state_dict(), "mu": std.mu, "sd": std.sd}, path)
    return path


def val_points(val_outs: List[Dict[str, Any]]) -> List[Tuple[str, bool, Optional[float]]]:
    """shadow 재생 클립 결과 -> (label, 규칙 알람 여부, 첫 p_fall) 목록"""
    return [(str(o.get("label")), bool(o.get("alarm")), first_p_fall(o)) for o in val_outs]


def sim_balanced(points: Sequence[Tuple[str, bool, Optional[float]]], th: float) -> Optional[float]:
    """
    shadow 재생 결과로 임계값 th의 veto 결과를 근사 (첫 AI 호출 p_fall <= th면 억제, 억제 후 재진입은 무시).
    p_fall이 없으면(판단 불가) 규칙 결과 유지. 반환: recall과 (1 - ADL 오탐률)의 조화평균 (한쪽 클립이 없으면 None).
    """
    n_f = n_a = det = fa = 0
    for lab, alarm, p in points:
        hit = bool(alarm) and (p is None or p > th)
        if lab == "fall":
            n_f += 1
            det += hit
        else:
            n_a += 1
            fa += hit
    if n_f == 0 or n_a == 0:
        return None
    r, sp = det / n_f, 1.0 - fa / n_a
    return 0.0 if r + sp <= 0 else 2 * r * sp / (r + sp)


def select_threshold_balanced(points: Sequence[Tuple[str, bool, Optional[float]]], default: float = DEFAULT_TH,
                              th_max: float = F_TH_MAX) -> float:
    """
    검증 balanced(sim_balanced)가 가장 큰 임계값. 후보 = 0과 검증 클립 p_fall 값들 (th_max 이하).
    동률이면 가장 작은 임계값 (덜 억제 = recall 쪽으로 안전). 계산할 수 없으면 default.
    """
    cands = sorted({0.0} | {float(p) for _, _, p in points if p is not None and float(p) <= th_max})
    best_th, best = None, None
    for th in cands:
        b = sim_balanced(points, th)
        if b is None:
            continue
        if best is None or b > best + 1e-12:
            best_th, best = th, b
    return float(default) if best_th is None else float(best_th)


def _load_e_model(mdir: Path, variant: str, seed: int) -> Optional[Tuple[str, Any, Any]]:
    """
    저장된 E 모델 -> ("hgb", clf, None) 또는 ("nn", model, 표준화) (없으면 None).
    아키텍처는 이름 앞 두 글자 (E0/E1/E2; L 변형 E0p10, E1p25 등도 같은 형식으로 저장됨).
    """
    import pickle

    import torch

    from training.experiments import models as M
    from training.experiments import windows as W

    arch = variant[:2]
    if arch == "E0":
        p = mdir / f"{variant}_s{seed}.pkl"
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return ("hgb", pickle.load(f), None)
    p = mdir / f"{variant}_s{seed}.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model = M.Fall1DCNNGAP(W.V2_CHANNELS) if arch == "E1" else M.FallTCN(W.V2_CHANNELS)
    model.load_state_dict(ck["state_dict"])
    return ("nn", model, _Standardizer(ck["mu"], ck["sd"]))


def _load_e_maker(mdir: Path, variant: str, seed: int, device: str):
    """저장된 E 모델 -> 검증기 생성 함수 (없으면 None)"""
    from training.experiments.verifiers import EventWindowVerifier

    entry = _load_e_model(mdir, variant, seed)
    if entry is None:
        return None
    kind, model, std = entry
    if kind == "hgb":
        return lambda: HandcraftedVerifier(model, model_id=f"{variant}-s{seed}")
    return lambda: EventWindowVerifier(model, device=device, model_id=f"{variant}-s{seed}", preprocess=std)


def _e_scorer(entry: Tuple[str, Any, Any], device: str) -> Callable[[Any], Any]:
    """E 모델 -> 묶음 점수 함수 X (B, 92, L) -> p_fall (B,) (E*의 SlidingWindowVerifier용, E와 같은 전처리)"""
    from training.experiments.verifiers import nn_batch_scorer
    from training.experiments.windows import handcrafted

    kind, model, std = entry
    if kind == "hgb":
        return lambda X: model.predict_proba(handcrafted(X))[:, 1]
    return nn_batch_scorer(model, device=device, preprocess=std)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PROTOCOL = ("15Hz runtime replay (nearest frame), EMA smoothing, 20s freeze tail, Level1Engine(study_params), "
            "event level: fall clip detected = any level1, ADL false alarm = any level1")


def summarize_out(out_dir: Path, meta: Optional[Dict[str, Any]] = None, fa_ref: str = DEFAULT_FA_REF
                  ) -> Dict[str, Any]:
    """out_dir의 fold 결과를 모아 summary.json / summary.md를 씁니다. fa_ref: 같은 오탐률 비교의 기준 변형."""
    results = _load_fold_results(out_dir)
    meta = dict(meta or {})
    if "stages" in meta:
        meta["requested_stages"] = meta.pop("stages")
    meta["stages"] = [s for s in STAGES if any(v in results for v in STAGE_VARIANTS[s])]
    summary = build_summary(results, meta=dict({"protocol": PROTOCOL}, **meta), fa_ref=fa_ref)
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _spawn_folds(args, fold_list: List[int], out_dir: Path) -> int:
    """fold마다 하위 프로세스 (GPU는 공유, 작은 모델이라 동시 실행이 빠름). 실패한 fold 수 반환."""
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    threads = max(1, (os.cpu_count() or 4) // max(1, args.jobs))
    pending = list(fold_list)
    running: List[Tuple[int, subprocess.Popen, Any]] = []
    failed = 0
    while pending or running:
        while pending and len(running) < args.jobs:
            k = pending.pop(0)
            cmd = [sys.executable, "-m", "training.experiments.run_progression",
                   "--stages", ",".join(args.stages_list), "--folds", str(k), "--seeds", str(args.seeds),
                   "--out", str(out_dir), "--device", args.device, "--jobs", "1", "--no-summary",
                   "--threads", str(threads), "--f-margin", str(args.f_margin), "--hgb-iter", str(args.hgb_iter),
                   "--manifest", str(args.manifest), "--folds-file", str(args.folds_file)]
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]
            if args.kp_dir:
                cmd += ["--kp-dir", str(args.kp_dir)]
            if args.switch_policy != "all":
                cmd += ["--switch-policy", args.switch_policy]
            if args.no_fallback:
                cmd += ["--no-fallback"]
            if args.skip_existing:
                cmd += ["--skip-existing"]
            fh = open(log_dir / f"fold{k}.log", "a", encoding="utf-8")
            env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS=str(threads))
            running.append((k, subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT,
                                                env=env), fh))
            _log(f"fold {k} started (log {log_dir / f'fold{k}.log'})")
        time.sleep(5)
        still = []
        for k, proc, fh in running:
            rc = proc.poll()
            if rc is None:
                still.append((k, proc, fh))
                continue
            fh.close()
            _log(f"fold {k} finished rc={rc}")
            failed += int(rc != 0)
        running = still
    return failed


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="progression study: A rules -> B/C/D/E/F classifier stages")
    ap.add_argument("--stages", default="A,B,C,D,E,Es,F")
    ap.add_argument("--folds", default="0-4")
    ap.add_argument("--seeds", type=int, default=3, help="NN 단계 seed 수 (0..n-1)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--folds-file", default=None, help="기본 dataset/_experiments/folds_v1.json (없으면 생성)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--jobs", type=int, default=1, help="fold 병렬 프로세스 수")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU 스레드 수")
    ap.add_argument("--epochs", type=int, default=None, help="epoch 상한 덮어쓰기 (점검용)")
    ap.add_argument("--hgb-iter", type=int, default=300)
    ap.add_argument("--f-margin", type=float, default=F_MARGIN)
    ap.add_argument("--skip-existing", action="store_true", help="이미 있는 fold/변형 결과는 다시 돌리지 않음")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--fa-ref", default=DEFAULT_FA_REF, help="같은 오탐률(matched FA) 비교의 기준 변형")
    ap.add_argument("--switch-policy", default="all", choices=("all", "multi"),
                    help="대상 전환 정책 (all: 운영과 같음, multi: 후보 2명 이상일 때만 전환, 클립에 n_cand 필요)")
    ap.add_argument("--no-fallback", action="store_true",
                    help="회전 폴백으로 찾은 프레임을 미검출로 재생 (폴백 켜고 추출한 클립으로 폴백 없는 결과 재현)")
    ap.add_argument("--kp-dir", default=None,
                    help="키포인트 클립 폴더 (기본 dataset/_keypoints/v1). 다른 추출본으로 같은 실험을 돌릴 때는 --out도 따로 둘 것")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    args.stages_list = parse_stages(args.stages)
    fold_list = parse_folds(args.folds)
    if args.fa_ref not in VARIANTS:
        raise SystemExit(f"--fa-ref: 알 수 없는 변형 {args.fa_ref!r}")
    meta = {"stages": args.stages_list, "folds": fold_list, "seeds": args.seeds, "f_margin": args.f_margin,
            "epochs_override": args.epochs, "fa_ref": args.fa_ref}

    if args.summarize_only:
        # 요약만 다시 만들 때는 실행 설정(단계, seed, epoch)이 이번 명령의 기본값일 뿐이라 기록하지 않음
        summarize_out(out_dir, {"fa_ref": args.fa_ref}, fa_ref=args.fa_ref)
        _log(f"summary written to {out_dir}")
        return 0

    # 엔진 INFO 로그(AI VERIFY 등)는 수만 줄이라 끔
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("app").setLevel(logging.WARNING)

    from training.experiments import folds as FO
    if args.folds_file is None:
        args.folds_file = str(FO.DEFAULT_FOLDS_PATH)

    t_all = time.time()
    if args.jobs > 1 and len(fold_list) > 1:
        failed = _spawn_folds(args, fold_list, out_dir)
        if failed:
            _log(f"{failed} fold process(es) failed; see {out_dir / 'logs'}")
    else:
        import torch

        from training.datasets.manifest import load_manifest
        from training.experiments import trainer as T
        from training.experiments import windows as W

        if args.threads:
            torch.set_num_threads(int(args.threads))
        device = str(T.resolve_device(None if args.device == "auto" else args.device))
        rows = load_manifest(args.manifest)
        fpath = Path(args.folds_file)
        if fpath.exists():
            folds = FO.load_folds(fpath)
        else:
            folds = FO.build_folds(rows)
            FO.save_folds(fpath, folds, meta={"n_folds": 5, "seed": 0, "manifest": str(args.manifest)})
        t0 = time.time()
        from training.experiments import replay as _R
        _R.SWITCH_POLICY = args.switch_policy
        _R.USE_FALLBACK = not args.no_fallback
        clips = W.load_clips(rows, kp_dir=args.kp_dir)
        _log(f"loaded {len(clips)} clips in {time.time() - t0:.1f}s, device {device}")
        failed = 0
        for k in fold_list:
            tk = time.time()
            run_fold(k, rows, folds, clips, args.stages_list, args.seeds, out_dir, device=device,
                     epochs=args.epochs, skip_existing=args.skip_existing, f_margin=args.f_margin,
                     hgb_iter=args.hgb_iter)
            _log(f"fold {k} total {time.time() - tk:.0f}s")
    _log(f"all folds done in {time.time() - t_all:.0f}s")
    if not args.no_summary:
        summarize_out(out_dir, meta, fa_ref=args.fa_ref)
        _log(f"summary written to {out_dir / 'summary.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
