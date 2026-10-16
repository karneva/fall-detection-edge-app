"""
교차 검증 fold 배정 (event_group 단위, (dataset, label) 층화)

- 평가 데이터셋(EVAL_DATASETS): le2i, caucafall, gmdcsa24, mcfd, urfd. 같은 사건의 다른 시점(event_group)은 같은 fold.
- 학습 전용(TRAIN_ONLY_DATASETS): figshare_ursul29. 피험자 단위 그룹("figshare/SBJ_xx")으로 fold에 배정하지만
  평가(test)에는 쓰지 않습니다 (클립 단위 폴더 라벨, 약 1.9초 클립).
- sklearn StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed). 저장: dataset/_experiments/folds_v1.json
- 학습 fold 안의 검증 분리(split_train_val)는 그룹 단위, (dataset, label) 층별로 frac만큼 (표준 라이브러리 난수)
"""
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

EVAL_DATASETS = ("le2i", "caucafall", "gmdcsa24", "mcfd", "urfd")
TRAIN_ONLY_DATASETS = ("figshare_ursul29",)
FOLDS_VERSION = 1
DEFAULT_FOLDS_PATH = Path(__file__).resolve().parents[2] / "dataset" / "_experiments" / "folds_v1.json"

_SBJ_RE = re.compile(r"SBJ_(\d+)", re.IGNORECASE)

__all__ = [
    "EVAL_DATASETS", "TRAIN_ONLY_DATASETS", "DEFAULT_FOLDS_PATH",
    "fold_group", "stratum", "build_folds", "save_folds", "load_folds", "split_fold", "split_train_val",
]


def fold_group(row: Dict[str, Any]) -> str:
    """분할 그룹. 학습 전용 데이터(figshare)는 피험자 단위 'figshare/SBJ_xx', 나머지는 event_group."""
    if row.get("dataset") in TRAIN_ONLY_DATASETS:
        m = _SBJ_RE.search(str(row.get("clip_id", "")))
        sid = m.group(1) if m else row.get("subject")
        if sid is not None and str(sid).strip():
            s = str(sid).strip()
            return f"figshare/SBJ_{int(s):02d}" if s.isdigit() else f"figshare/SBJ_{s}"
    return str(row.get("event_group") or row.get("clip_id"))


def stratum(row: Dict[str, Any]) -> str:
    return f"{row.get('dataset')}|{row.get('label')}"


def build_folds(rows: Iterable[Dict[str, Any]], n_folds: int = 5, seed: int = 0) -> Dict[str, int]:
    """
    {clip_id: fold(0..n_folds-1)}. 그룹(fold_group)은 한 fold에만 들어가고, (dataset, label) 비율을 맞춥니다.
    같은 입력과 seed면 같은 결과입니다.
    """
    import warnings

    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    rows = sorted(rows, key=lambda r: r["clip_id"])  # 입력 순서와 무관하게 결정적
    if not rows:
        return {}
    y = np.array([stratum(r) for r in rows])
    groups = np.array([fold_group(r) for r in rows])
    sgkf = StratifiedGroupKFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
    folds: Dict[str, int] = {}
    with warnings.catch_warnings():
        # 그룹 수가 fold 수보다 적은 층(예: mcfd adl)은 경고만 나고 일부 fold에만 들어감
        warnings.simplefilter("ignore", UserWarning)
        for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(rows)), y, groups)):
            for i in test_idx:
                folds[rows[i]["clip_id"]] = int(k)
    return folds


def save_folds(path, folds: Dict[str, int], meta: Optional[Dict[str, Any]] = None) -> Path:
    """fold 배정을 JSON으로 저장합니다. {"version", "created_at", "meta", "folds": {clip_id: fold}}"""
    if isinstance(path, dict) and not isinstance(folds, dict):
        path, folds = folds, path  # save_folds(folds, path) 순서도 허용
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": FOLDS_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_folds": (max(folds.values()) + 1) if folds else 0,
        "meta": dict(meta or {}),
        "folds": {k: int(v) for k, v in sorted(folds.items())},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load_folds(path=DEFAULT_FOLDS_PATH) -> Dict[str, int]:
    """save_folds로 저장한 파일 -> {clip_id: fold}"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(data.get("version", 0)) > FOLDS_VERSION:
        raise ValueError(f"folds version {data.get('version')} is newer than supported {FOLDS_VERSION}")
    return {str(k): int(v) for k, v in data["folds"].items()}


def split_fold(rows: Iterable[Dict[str, Any]], folds: Dict[str, int], k: int,
               eval_datasets: Iterable[str] = EVAL_DATASETS) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    fold k를 test로: (train 행 = fold != k 전체(학습 전용 포함), test 행 = fold == k 이면서 평가 데이터셋).
    fold 배정이 없는 행은 KeyError (manifest와 fold 파일 불일치를 조용히 넘기지 않음).
    """
    eval_set = set(eval_datasets)
    train, test = [], []
    for r in rows:
        f = folds[r["clip_id"]]
        if f == k:
            if r.get("dataset") in eval_set:
                test.append(r)
        else:
            train.append(r)
    return train, test


def split_train_val(rows_train: Iterable[Dict[str, Any]], frac: float = 0.15, seed: int = 0
                    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    학습 행을 그룹(fold_group) 단위로 (train, val)로 나눕니다. 층((dataset, 다수 label))마다 그룹의 frac만큼
    (반올림, 그룹이 2개 이상이면 최소 1개, 최소 1개는 train에 남김)을 val로 보냅니다. 결정적입니다.
    """
    rows_train = list(rows_train)
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows_train:
        by_group[fold_group(r)].append(r)
    strata: Dict[str, List[str]] = defaultdict(list)
    for g, rs in by_group.items():
        strata[Counter(stratum(r) for r in rs).most_common(1)[0][0]].append(g)

    rng = random.Random(int(seed))
    val_groups = set()
    for key in sorted(strata):
        gs = sorted(strata[key])
        rng.shuffle(gs)
        n = len(gs)
        if n < 2:
            continue
        n_val = min(n - 1, max(1, int(round(frac * n))))
        val_groups.update(gs[:n_val])
    train = [r for r in rows_train if fold_group(r) not in val_groups]
    val = [r for r in rows_train if fold_group(r) in val_groups]
    return train, val
