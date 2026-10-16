"""
진행 단계 비교 실험(progression study) 실행기 (CLI)

  A 규칙만 -> B 1D CNN + URFD(원래 방식) -> C + 데이터 추가 -> D + 라벨/학습-운영 정렬
  -> E + 모델/특징 변경 (E0 HistGradientBoosting, E1 Fall1DCNN-GAP, E2 FallTCN) -> F + 임계값 조정
  (E*: 저장된 E 모델을 진입 이후 여러 창에 적용, 재학습 없음)
을 같은 5-fold held-out 클립, 같은 이벤트 단위 프로토콜(replay.py: 15Hz obs, EMA, 20초 정지 꼬리)로 평가합니다.

실행 예 (저장소 루트에서)
  python -m training.experiments.run_progression --stages A,B,C,D,E,F --folds 0-4 --seeds 3 --jobs 5
  python -m training.experiments.run_progression --stages Es --folds 0-4 --seeds 3 --jobs 5   (E 모델 필요)
  python -m training.experiments.run_progression --summarize-only

출력 (--out, 기본 dataset/_experiments/progression_v1/)
  fold{k}/{variant}.json : 변형(A, B, C, D, E0, E1, E2, E0max.., F0, F1, F2)별 fold 결과
                           runs[seed]: 클립 결과(outcomes), 요약(summary), 창 단위 지표(window), 학습 기록 요약
  fold{k}/models/        : E 모델 (F, Es 단계가 다시 읽음)
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

STAGES = ("A", "B", "C", "D", "E", "Es", "F")
STAGE_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "A": ("A",), "B": ("B",), "C": ("C",), "D": ("D",),
    "E": ("E0", "E1", "E2"), "Es": ("E0max", "E1max", "E2max", "E0last", "E1last", "E2last"),
    "F": ("F0", "F1", "F2", "F0b", "F1b", "F2b"),
}
# E* 변형 -> (기준 E 변형, SlidingWindowVerifier mode)
ES_BASE: Dict[str, Tuple[str, str]] = {v: (v[:2], v[2:]) for v in STAGE_VARIANTS["Es"]}
VARIANTS = tuple(v for s in STAGES for v in STAGE_VARIANTS[s])
VARIANT_INFO: Dict[str, Dict[str, str]] = {
    "A": {"title": "규칙만", "change": "rules only (ai_mode=off)", "model": "-"},
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
}
EVAL_DATASETS_ORDER = ("le2i", "caucafall", "gmdcsa24", "mcfd", "urfd")

DEFAULT_TH = 0.4
F_MARGIN = 0.05
F_TH_MAX = 0.95
VAL_FRAC = 0.15
VAL_SPLIT_SEED = 0

__all__ = [
    "STAGES", "VARIANTS", "parse_folds", "parse_stages", "select_threshold", "auc_mann_whitney",
    "first_p_fall", "val_points", "sim_balanced", "select_threshold_balanced",
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


def build_summary(results: Dict[str, List[Dict[str, Any]]], summarize_fn: Optional[Callable] = None,
                  meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    {variant: [fold 결과]} -> summary dict (variants 순서 유지). 'best_f': 검증 balanced가 가장 높은 F 변형.
    'paired': E* 변형마다 기준 E 변형과의 paired_compare (둘 다 있을 때).
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
    return {"version": 1, "meta": dict(meta or {}), "variants": variants, "best_f": best_f, "paired": paired}


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
                    from sklearn.ensemble import HistGradientBoostingClassifier
                    clf = HistGradientBoostingClassifier(max_iter=int(hgb_iter), learning_rate=0.1,
                                                         class_weight="balanced", random_state=seed)
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
    return results


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
    """저장된 E 모델 -> ("hgb", clf, None) 또는 ("nn", model, 표준화) (없으면 None)"""
    import pickle

    import torch

    from training.experiments import models as M
    from training.experiments import windows as W

    if variant == "E0":
        p = mdir / f"{variant}_s{seed}.pkl"
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return ("hgb", pickle.load(f), None)
    p = mdir / f"{variant}_s{seed}.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model = M.Fall1DCNNGAP(W.V2_CHANNELS) if variant == "E1" else M.FallTCN(W.V2_CHANNELS)
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


def summarize_out(out_dir: Path, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """out_dir의 fold 결과를 모아 summary.json / summary.md를 씁니다."""
    results = _load_fold_results(out_dir)
    summary = build_summary(results, meta=dict({"protocol": PROTOCOL}, **(meta or {})))
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
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    args.stages_list = parse_stages(args.stages)
    fold_list = parse_folds(args.folds)
    meta = {"stages": args.stages_list, "folds": fold_list, "seeds": args.seeds, "f_margin": args.f_margin,
            "epochs_override": args.epochs}

    if args.summarize_only:
        summarize_out(out_dir, meta)
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
        clips = W.load_clips(rows)
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
        summarize_out(out_dir, meta)
        _log(f"summary written to {out_dir / 'summary.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
