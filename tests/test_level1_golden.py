"""
Level1Engine 이벤트 기록(golden trace) 비교 테스트 (표준 라이브러리만 사용)

동작을 바꾸지 않아야 하는 리팩토링의 안전망입니다. 합성 시나리오 전체를 여러 가짜 AI로 재생해
발생한 이벤트(종류, 시각, 사유, values 전체)가 기록과 완전히 같은지 확인합니다.
recall/FA만 보는 compare_results보다 훨씬 엄격합니다.

판정 로직을 의도적으로 바꿨다면 기록을 다시 만듭니다:
  python -m tests.test_level1_golden --regen
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path

from app.eval.synth import EVAL_SET, CountingVerifier, RaisingVerifier, make_scenario, run_engine

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "level1_events_v0.json"
FPS_LIST = (10.0, 15.0, 30.0)
SEEDS = (0, 1)
LIE_S = 90.0

# 환경변수에 영향받지 않도록 AI 관련 기본값을 모두 고정
BASE_PARAMS = dict(
    sustain_s=10.0,
    abnormal_timeout_s=10.0,
    ai_suppress_pfall_th=0.4,
    ai_escalate_s=60.0,
    ai_escalate_requires_drop=True,
)
VARIANTS = {
    "off": ("off", lambda: CountingVerifier(0.9)),
    "fall": ("veto", lambda: CountingVerifier(0.9)),
    "adl": ("veto", lambda: CountingVerifier(0.05)),
    "none": ("veto", lambda: CountingVerifier(None)),
    "raise": ("veto", lambda: RaisingVerifier()),
}
VOLATILE_KEYS = {"latency_ms"}


def _clean(v):
    """비결정적 필드 제거, 실수는 반올림 (플랫폼 간 표기 차이 방지)."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in sorted(v.items()) if k not in VOLATILE_KEYS}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, float):
        return round(v, 9)
    return v


def generate():
    import logging

    from app.core.level1 import Level1Params

    logging.disable(logging.CRITICAL)
    try:
        out = {}
        for name, _ in EVAL_SET:
            for fps in FPS_LIST:
                for seed in SEEDS:
                    obs, _ = make_scenario(name, fps=fps, seed=seed, lie_s=LIE_S)
                    for vname, (mode, make_verifier) in VARIANTS.items():
                        verifier = make_verifier()
                        events = run_engine(obs, Level1Params(ai_mode=mode, **BASE_PARAMS), verifier)
                        key = f"{name}|fps{int(fps)}|s{seed}|{vname}"
                        cleaned = _clean(events)
                        out[key] = {
                            "verifier_calls": verifier.calls,
                            # 사람이 읽을 수 있는 흐름 + 이벤트 전체(values, signals 포함)의 해시 (파일 크기 절약, 비교는 엄격)
                            "trace": [[e["type"], e["ts"], e.get("reason")] for e in cleaned],
                            "events_sha": hashlib.sha256(
                                json.dumps(cleaned, sort_keys=True).encode()
                            ).hexdigest()[:16],
                        }
        return out
    finally:
        logging.disable(logging.NOTSET)


class Level1GoldenTraceTest(unittest.TestCase):
    def test_event_trace_matches_golden(self):
        self.assertTrue(GOLDEN_PATH.exists(), f"missing {GOLDEN_PATH}; run python -m tests.test_level1_golden --regen")
        expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        actual = json.loads(json.dumps(generate()))
        self.assertEqual(sorted(actual), sorted(expected))
        for key in sorted(expected):
            self.assertEqual(actual[key], expected[key], key)


if __name__ == "__main__":
    if "--regen" in sys.argv:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(generate(), ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                               encoding="utf-8")
        print(f"wrote {GOLDEN_PATH}")
    else:
        unittest.main()
