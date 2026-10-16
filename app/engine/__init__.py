"""engine module - 자세 감지 + EMA 평활화

LivePipeline은 cv2/torch/ultralytics를 import하므로 PEP 562 지연 export로 둡니다.
(app.engine.target_select 같은 표준 라이브러리 모듈은 장치 의존성 없이 import 가능)
"""
import importlib

_LAZY = {
    "LivePipeline": ".live",
    "ema_smooth_keypoints_inplace": ".smoothing",
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
