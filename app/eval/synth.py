"""
Level1Engine 회귀 테스트/오프라인 평가용 합성 관측(obs) 생성기 (표준 라이브러리만 사용)

실제 영상을 대체하지는 못하지만, 상태 전이/AI 게이트/쿨다운 같은 제어 흐름을
결정적으로 재현하기 위한 용도입니다.
"""
import math
import random
from typing import Any, Dict, List, Optional, Tuple

FRAME_H, FRAME_W = 1080, 1920
PX_ASPECT = FRAME_H / FRAME_W  # 정규화 x 단위를 y 단위에 맞추는 비율

# COCO-17 직립 자세 템플릿: 골반 중심 기준 (dx, dy), 키 1.0 단위, y는 아래쪽이 +
STANDING_TEMPLATE: Dict[int, Tuple[float, float]] = {
    0: (0.00, -0.45),   # nose
    1: (-0.02, -0.47),  # left eye
    2: (0.02, -0.47),   # right eye
    3: (-0.04, -0.46),  # left ear
    4: (0.04, -0.46),   # right ear
    5: (-0.10, -0.32),  # left shoulder
    6: (0.10, -0.32),   # right shoulder
    7: (-0.12, -0.17),  # left elbow
    8: (0.12, -0.17),   # right elbow
    9: (-0.12, -0.03),  # left wrist
    10: (0.12, -0.03),  # right wrist
    11: (-0.06, 0.00),  # left hip
    12: (0.06, 0.00),   # right hip
    13: (-0.06, 0.24),  # left knee
    14: (0.06, 0.24),   # right knee
    15: (-0.06, 0.48),  # left ankle
    16: (0.06, 0.48),   # right ankle
}

STAND_HIP_Y = 0.55
LIE_HIP_Y = 0.80
BED_HIP_Y = 0.62
HEIGHT = 0.70


def skeleton(hip_x: float, hip_y: float, height: float, angle_rad: float,
             rng: random.Random, jitter: float = 0.0005, conf: float = 0.9) -> List[Dict[str, Any]]:
    """골반 중심/키/기울기로 17개 키포인트를 만듭니다. angle 0은 직립, pi/2는 수평."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    kps = []
    for kid, (dx, dy) in STANDING_TEMPLATE.items():
        rx = dx * c - dy * s
        ry = dx * s + dy * c
        x = hip_x + rx * height * PX_ASPECT + rng.gauss(0.0, jitter)
        y = hip_y + ry * height + rng.gauss(0.0, jitter)
        kps.append({"id": kid, "x": x, "y": y, "conf": conf})
    return kps


def bbox_of(kps: List[Dict[str, Any]], pad: float = 0.02) -> List[float]:
    xs = [k["x"] for k in kps]
    ys = [k["y"] for k in kps]
    return [
        max(0.0, min(xs) - pad), max(0.0, min(ys) - pad),
        min(1.0, max(xs) + pad), min(1.0, max(ys) + pad),
    ]


def make_obs(ts: float, i: int, kps: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    present = bool(kps)
    track = {
        "track_id": 0,
        "has_person": present,
        "bbox": bbox_of(kps) if present else None,
        "conf": 0.9 if present else 0.0,
        "keypoints_raw": kps or [],
        "keypoints_smooth": None,
        "keypoints": kps or [],
        "quality_score": (sum(k["conf"] for k in kps) / len(kps)) if present else 0.0,
        "frame_shape": (FRAME_H, FRAME_W),
    }
    return {"ts": ts, "frame_index": i, "target_switched": False, "tracks": [track]}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def make_scenario(name: str, fps: float = 15.0, seed: int = 0, lie_s: float = 90.0,
                  t0: float = 1000.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    시나리오 이름으로 (obs 목록, 정답 정보)를 만듭니다.

    - stand: 60초 직립 (adl)
    - fast_fall: 3초 직립 -> 0.5초 만에 쓰러짐 -> lie_s초 정지 (fall)
    - lie_on_bed_slow: 3초 직립 -> 4초에 걸쳐 천천히 눕기 -> lie_s초 정지 (adl)
    - fall_then_recover: fast_fall 후 3초 누움 -> 1.5초 기립 -> 10초 직립 (fall, 알람 기대 안 함)
    - fall_lie_then_get_up: fast_fall 후 20초 누움 -> 1.5초 기립 -> 10초 직립
    - fall_writhe: fast_fall 후 lie_s초 동안 바닥에서 계속 움직임 (정지 판정 안 됨 -> 타임아웃 경로)
    - suppressed_leave_and_return: fast_fall 후 12초 누움 -> 기립 -> 0.3초 서 있다가 화면 밖
      -> 120초 부재 -> 서서 재등장 15초
    """
    rng = random.Random(seed)
    dt = 1.0 / fps
    # (지속시간, 함수(u) -> (hip_y, angle) 또는 None(사람 없음)[, jitter]) 구간 목록
    stand = lambda u: (STAND_HIP_Y, 0.0)
    lie = lambda u: (LIE_HIP_Y, math.pi / 2)
    falling = lambda u: (_lerp(STAND_HIP_Y, LIE_HIP_Y, u), _lerp(0.0, math.pi / 2, u))
    getting_up = lambda u: (_lerp(LIE_HIP_Y, STAND_HIP_Y, u), _lerp(math.pi / 2, 0.0, u))
    if name == "stand":
        segs = [(60.0, stand)]
        truth = {"label": "adl", "fall_start_ts": None, "fall_end_ts": None}
    elif name == "fast_fall":
        segs = [
            (3.0, stand),
            (0.5, lambda u: (_lerp(STAND_HIP_Y, LIE_HIP_Y, u), _lerp(0.0, math.pi / 2, u))),
            (lie_s, lambda u: (LIE_HIP_Y, math.pi / 2)),
        ]
        truth = {"label": "fall", "fall_start_ts": t0 + 3.0, "fall_end_ts": t0 + 3.5}
    elif name == "lie_on_bed_slow":
        segs = [
            (3.0, stand),
            (4.0, lambda u: (_lerp(STAND_HIP_Y, BED_HIP_Y, u), _lerp(0.0, math.pi / 2, u))),
            (lie_s, lambda u: (BED_HIP_Y, math.pi / 2)),
        ]
        truth = {"label": "adl", "fall_start_ts": None, "fall_end_ts": None}
    elif name == "fall_then_recover":
        segs = [
            (3.0, stand),
            (0.5, lambda u: (_lerp(STAND_HIP_Y, LIE_HIP_Y, u), _lerp(0.0, math.pi / 2, u))),
            (3.0, lambda u: (LIE_HIP_Y, math.pi / 2)),
            (1.5, lambda u: (_lerp(LIE_HIP_Y, STAND_HIP_Y, u), _lerp(math.pi / 2, 0.0, u))),
            (10.0, stand),
        ]
        truth = {"label": "fall", "fall_start_ts": t0 + 3.0, "fall_end_ts": t0 + 3.5}
    elif name == "fall_lie_then_get_up":
        segs = [
            (3.0, stand),
            (0.5, lambda u: (_lerp(STAND_HIP_Y, LIE_HIP_Y, u), _lerp(0.0, math.pi / 2, u))),
            (20.0, lambda u: (LIE_HIP_Y, math.pi / 2)),
            (1.5, lambda u: (_lerp(LIE_HIP_Y, STAND_HIP_Y, u), _lerp(math.pi / 2, 0.0, u))),
            (10.0, stand),
        ]
        truth = {"label": "fall", "fall_start_ts": t0 + 3.0, "fall_end_ts": t0 + 3.5}
    elif name == "fall_writhe":
        segs = [(3.0, stand), (0.5, falling), (lie_s, lie, 0.015)]
        truth = {"label": "fall", "fall_start_ts": t0 + 3.0, "fall_end_ts": t0 + 3.5}
    elif name == "suppressed_leave_and_return":
        segs = [
            (3.0, stand), (0.5, falling), (12.0, lie), (1.5, getting_up), (0.3, stand),
            (120.0, lambda u: None),
            (15.0, stand),
        ]
        truth = {"label": "adl", "fall_start_ts": None, "fall_end_ts": None}
    else:
        raise ValueError(f"unknown scenario: {name}")

    obs_list = []
    i = 0
    seg_start = 0.0
    for seg in segs:
        dur, fn = seg[0], seg[1]
        jitter = seg[2] if len(seg) > 2 else 0.0005
        n = max(1, int(round(dur * fps)))
        for j in range(n):
            u = j / max(1, n - 1)
            pose = fn(u)
            ts = t0 + seg_start + j * dt
            kps = None if pose is None else skeleton(0.5, pose[0], HEIGHT, pose[1], rng, jitter=jitter)
            obs_list.append(make_obs(ts, i, kps))
            i += 1
        seg_start += n * dt

    truth.update({"scenario": name, "fps": fps})
    return obs_list, truth


# 평가용 합성 세트: (시나리오, 채점 라벨)
# fall_then_recover는 3초 만에 스스로 일어나 알람이 필요 없으므로 adl로 채점
EVAL_SET = (
    ("stand", "adl"),
    ("lie_on_bed_slow", "adl"),
    ("fall_then_recover", "adl"),
    ("fast_fall", "fall"),
    ("fall_writhe", "fall"),
    ("fall_lie_then_get_up", "fall"),
)


def run_engine(obs_list, params=None, verifier=None) -> List[Dict[str, Any]]:
    """obs 목록을 새 엔진에 순서대로 넣고 발생한 이벤트 목록을 반환합니다."""
    from app.core.level1 import Level1Engine, Level1Params

    engine = Level1Engine(params or Level1Params(), verifier=verifier)
    events = []
    for obs in obs_list:
        ev = engine.step(obs)
        if ev is not None:
            events.append(ev)
    return events


class CountingVerifier:
    """고정 p_fall을 반환하고 호출 횟수를 셉니다."""

    model_id = "synthetic"

    def __init__(self, p_fall: Optional[float]):
        self.p_fall = p_fall
        self.calls = 0

    def __call__(self, seq):
        self.calls += 1
        return self.p_fall


class RaisingVerifier:
    """항상 예외를 던지는 검증기."""

    model_id = "raising"

    def __init__(self):
        self.calls = 0

    def __call__(self, seq):
        self.calls += 1
        raise RuntimeError("boom")
