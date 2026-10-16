"""
URFD(UR Fall Detection) 프레임 단위 상태 라벨 CSV -> 시간 구간 (표준 라이브러리만 사용)

원본: urfall-cam0-falls.csv, urfall-cam0-adls.csv (fetch_public의 "urfd" 소스, 헤더 없음)
  열: 시퀀스 이름(fall-01 / adl-01), 프레임 번호(1부터), 상태, 이후는 깊이 기반 특징값(사용 안 함)
  상태: -1 = 눕지 않음, 0 = 넘어지는 중(ADL에서는 눕는 중), 1 = 누워 있음
CSV에 없는 프레임도 있어 같은 상태가 이어지는 행들을 하나의 구간으로 묶습니다.
시각은 원본 30fps 기준 (frame - 1) / 30초, 구간 끝은 마지막 프레임 다음 시각입니다.

OmniFall 클래스로 바꾸는 규칙:
  낙상 시퀀스: 0 -> "fall", 1 -> "lying"
  ADL 시퀀스:  0 -> "lie_down" (침대/바닥에 스스로 눕기, 하드 네거티브), 1 -> "lying"
  -1은 어떤 동작인지 알 수 없어 구간을 만들지 않습니다.
"""
import csv
from pathlib import Path

URFD_FPS = 30.0
STATE_NOT_LYING, STATE_FALLING, STATE_LYING = -1, 0, 1

CLASS_MAP = {
    "fall": {STATE_FALLING: "fall", STATE_LYING: "lying"},
    "adl": {STATE_FALLING: "lie_down", STATE_LYING: "lying"},
}


def seq_kind(seq: str) -> str:
    """시퀀스 이름에서 종류("fall" / "adl")를 돌려줍니다."""
    kind = seq.split("-", 1)[0].strip().lower()
    if kind not in CLASS_MAP:
        raise ValueError(f"unknown URFD sequence name: {seq!r}")
    return kind


def read_states(path) -> dict:
    """
    CSV를 읽어 {시퀀스 이름: [(프레임, 상태), ...]} (프레임 오름차순)를 돌려줍니다.
    빈 줄은 건너뛰고, 형식이 틀린 줄이나 같은 시퀀스의 중복 프레임은 ValueError를 냅니다.
    """
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.reader(f), 1):
            if not row or not "".join(row).strip():
                continue
            try:
                seq = row[0].strip()
                frame = int(float(row[1]))
                state = int(float(row[2]))
            except (IndexError, ValueError) as e:
                raise ValueError(f"{path}:{n}: malformed URFD row {row[:3]!r}") from e
            if state not in (STATE_NOT_LYING, STATE_FALLING, STATE_LYING):
                raise ValueError(f"{path}:{n}: unknown URFD state {state}")
            if frame < 1:
                raise ValueError(f"{path}:{n}: frame must be >= 1 (got {frame})")
            out.setdefault(seq, []).append((frame, state))
    for seq in out:
        out[seq].sort()
        # 같은 프레임이 두 번 나오면 구간이 겹치거나 상태가 모호해지므로 거부
        frames = [f for f, _ in out[seq]]
        dup = sorted({f for f, g in zip(frames, frames[1:]) if f == g})
        if dup:
            raise ValueError(f"{path}: {seq} has duplicate frame rows {dup[:5]}")
    return out


def states_to_segments(states, kind: str, fps: float = URFD_FPS) -> list:
    """
    (프레임, 상태) 목록을 OmniFall 이름의 구간 목록으로 바꿉니다.
    같은 상태가 연속된 행들이 한 구간이며, -1 구간은 버립니다.
    """
    cmap = CLASS_MAP[kind]
    segs = []
    run = None  # [state, first_frame, last_frame]
    for frame, state in sorted(states):
        if run is not None and run[0] == state:
            run[2] = frame
            continue
        if run is not None:
            segs.append(run)
        run = [state, frame, frame]
    if run is not None:
        segs.append(run)
    out = []
    seen_lying = False
    for st, f0, f1 in segs:
        if st == STATE_LYING:
            seen_lying = True
        if st not in cmap:
            continue
        # 누운 뒤에 나오는 '0'(전이) 구간은 넘어지는/눕는 동작이 아니라 일어나는 동작
        cls = "stand_up" if (st == STATE_FALLING and seen_lying) else cmap[st]
        out.append({"cls": cls, "start_s": round((f0 - 1) / fps, 6), "end_s": round(f1 / fps, 6)})
    return out


def load_urfd_segments(csv_paths, fps: float = URFD_FPS) -> dict:
    """여러 CSV(낙상/ADL)를 읽어 {시퀀스 이름: 구간 목록}을 돌려줍니다."""
    out = {}
    for p in csv_paths:
        for seq, states in read_states(Path(p)).items():
            if seq in out:
                raise ValueError(f"URFD sequence {seq!r} appears in more than one CSV")
            out[seq] = states_to_segments(states, seq_kind(seq), fps)
    return out
