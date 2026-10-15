"""core module - 핵심 처리 엔진

무거운 의존성(cv2, numpy, torch)을 패키지 import 시점에 끌어오지 않도록
PEP 562 지연 export를 사용합니다. `from app.core import Level1Engine`처럼
기존 사용법은 그대로 동작하며, 실제 모듈은 처음 접근할 때 로드됩니다.
"""
import importlib

_LAZY = {
    "Level1Engine": ".level1",
    "Level1Params": ".level1",
    "PresenceEngine": ".level1",
    "PresenceParams": ".level1",
    "build_observation": ".obs_schema",
}

__all__ = list(_LAZY)


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
