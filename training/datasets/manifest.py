"""
통합 학습 manifest (JSONL, 영상 클립 1개당 1줄) 읽기/쓰기/검증 (표준 라이브러리만 사용)

행 형식 (docs/DATASETS.md, 만드는 방법은 training/datasets/build_manifest.py):
  {"clip_id": "le2i/Coffee_room_01/video_14",   # "<dataset>/<경로>", 전체에서 유일
   "dataset": "le2i", "video": "<저장소 루트 기준 상대 경로>",
   "label": "fall" | "adl",                      # 클립 단위: fall 구간이 하나라도 있으면 fall
   "segments": [{"cls": "fall", "start_s": 3.1, "end_s": 4.0}, ...],   # OmniFall 16클래스 이름
   "fall_start_s": 3.1 | null, "fall_end_s": 4.0 | null,              # 첫 번째 fall 구간
   "subject": "1", "cam": "1", "event_group": "...",  # 같은 사건의 모든 시점은 같은 event_group
   "label_source": "omnifall" | "urfd_csv" | "folder" | "review", "license": "..."}

- segments가 비어 있으면 구간 라벨이 없는 클립 (폴더 단위 라벨, label_source "folder"만 fall 허용)
- 분할(train/val/test)은 event_group 단위로 나눠야 다중 시점 누수가 생기지 않음
- label_source "review": 현장 클립을 사람이 검수한 라벨 (training/review_queue.py export, dataset "field_runs")
"""
import json
import os
from pathlib import Path

MANIFEST_VERSION = 1

# OmniFall labels/label2id.csv 순서 (id -> 이름)
OMNIFALL_CLASSES = (
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up",
    "standing", "other", "kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump",
)

LABELS = ("fall", "adl")
LABEL_SOURCES = ("omnifall", "urfd_csv", "folder", "review")
FIELDS = ("clip_id", "dataset", "video", "label", "segments", "fall_start_s", "fall_end_s",
          "subject", "cam", "event_group", "label_source", "license")


def make_row(dataset: str, rel_id: str, video: str, segments: list, label_source: str, license: str,
             subject=None, cam=None, event_group=None, label=None, crop_px=None, fps=None) -> dict:
    """
    한 클립의 manifest 행을 만듭니다.
    segments는 시작 시각 순으로 정렬하고, label을 주지 않으면 fall 구간 유무로 정합니다.
    event_group을 주지 않으면 clip_id와 같게 둡니다 (단일 시점 데이터).
    crop_px: 영상에서 실제로 쓸 영역 [x1, y1, x2, y2] 픽셀 (선택). 예: URFD는 깊이|RGB 합성 영상이라 RGB 절반만 사용
    fps: 라벨 시각의 기준이 되는 실제 촬영 fps (선택). 컨테이너 헤더 fps가 틀린 데이터에서 추출기가 이 값을 우선 사용
         (예: MCFD avi 헤더는 120fps지만 실제 촬영과 OmniFall 라벨은 30fps)
    """
    clip_id = f"{dataset}/{rel_id}"
    segs = sorted(({"cls": s["cls"], "start_s": float(s["start_s"]), "end_s": float(s["end_s"])}
                   for s in segments), key=lambda s: (s["start_s"], s["end_s"]))
    falls = [s for s in segs if s["cls"] == "fall"]
    if label is None:
        label = "fall" if falls else "adl"
    row = {
        "clip_id": clip_id,
        "dataset": dataset,
        "video": video,
        "label": label,
        "segments": segs,
        "fall_start_s": falls[0]["start_s"] if falls else None,
        "fall_end_s": falls[0]["end_s"] if falls else None,
        "subject": None if subject is None else str(subject),
        "cam": None if cam is None else str(cam),
        "event_group": event_group or clip_id,
        "label_source": label_source,
        "license": license,
    }
    if crop_px is not None:
        row["crop_px"] = [int(v) for v in crop_px]
    if fps is not None:
        row["fps"] = float(fps)
    return row


def crop_error(crop) -> str:
    """crop_px 형식 오류 메시지 (정상이면 빈 문자열)."""
    if not isinstance(crop, (list, tuple)) or len(crop) != 4:
        return "crop_px must be [x1, y1, x2, y2]"
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in crop):
        return "crop_px values must be integers"
    x1, y1, x2, y2 = crop
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        return "crop_px must satisfy 0 <= x1 < x2 and 0 <= y1 < y2"
    return ""


def validate_rows(rows) -> list:
    """
    manifest 행 목록을 검사해 오류 메시지 목록을 돌려줍니다 (비어 있으면 정상).
    clip_id 중복, 필수 필드, label과 segments 일치, 구간 start < end, event_group 존재를 확인합니다.
    """
    errors = []
    seen = set()
    for i, r in enumerate(rows):
        where = f"row {i} ({r.get('clip_id') if isinstance(r, dict) else '?'})"
        if not isinstance(r, dict):
            errors.append(f"{where}: not an object")
            continue
        missing = [k for k in FIELDS if k not in r]
        if missing:
            errors.append(f"{where}: missing fields {missing}")
        cid = r.get("clip_id")
        if not cid or not isinstance(cid, str):
            errors.append(f"{where}: empty clip_id")
        elif cid in seen:
            errors.append(f"{where}: duplicate clip_id")
        else:
            seen.add(cid)
            if r.get("dataset") and not cid.startswith(f"{r['dataset']}/"):
                errors.append(f"{where}: clip_id must start with '<dataset>/'")
        if not r.get("video"):
            errors.append(f"{where}: empty video")
        if not r.get("event_group"):
            errors.append(f"{where}: empty event_group")
        if r.get("label") not in LABELS:
            errors.append(f"{where}: label must be one of {LABELS}")
        if r.get("label_source") not in LABEL_SOURCES:
            errors.append(f"{where}: label_source must be one of {LABEL_SOURCES}")
        if "crop_px" in r:
            msg = crop_error(r["crop_px"])
            if msg:
                errors.append(f"{where}: {msg}")
        if "fps" in r:
            f = r["fps"]
            if isinstance(f, bool) or not isinstance(f, (int, float)) or not (1.0 <= float(f) <= 240.0):
                errors.append(f"{where}: fps must be a number in [1, 240]")

        segs = r.get("segments")
        if not isinstance(segs, list):
            errors.append(f"{where}: segments must be a list")
            continue
        bad_seg = False
        for s in segs:
            try:
                start, end = float(s["start_s"]), float(s["end_s"])
                cls = s["cls"]
            except (KeyError, TypeError, ValueError):
                errors.append(f"{where}: malformed segment {s!r}")
                bad_seg = True
                continue
            if cls not in OMNIFALL_CLASSES:
                errors.append(f"{where}: unknown segment class {cls!r}")
            if not start < end:
                errors.append(f"{where}: segment {cls} start {start} >= end {end}")
            if start < 0:
                errors.append(f"{where}: segment {cls} negative start {start}")
        if bad_seg:
            continue
        falls = sorted((s for s in segs if s["cls"] == "fall"),
                       key=lambda s: (float(s["start_s"]), float(s["end_s"])))  # make_row과 같은 정렬
        if segs:
            expected = "fall" if falls else "adl"
            if r.get("label") != expected:
                errors.append(f"{where}: label {r.get('label')!r} but segments say {expected!r}")
        elif r.get("label") == "fall" and r.get("label_source") != "folder":
            errors.append(f"{where}: label 'fall' without fall segments")
        exp_start = float(falls[0]["start_s"]) if falls else None
        exp_end = float(falls[0]["end_s"]) if falls else None
        if r.get("fall_start_s") != exp_start or r.get("fall_end_s") != exp_end:
            errors.append(f"{where}: fall_start_s/fall_end_s {r.get('fall_start_s')}/{r.get('fall_end_s')} "
                          f"do not match first fall segment {exp_start}/{exp_end}")
    return errors


def load_manifest(path) -> list:
    """JSONL manifest를 행 목록으로 읽습니다 (빈 줄은 건너뜀)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e})") from e
    return rows


def write_manifest(path, rows):
    """행 목록을 JSONL로 원자적으로 씁니다 (임시 파일에 쓴 뒤 교체)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
