"""
AI 분류기 가중치 경로 해석 (표준 라이브러리만 사용)

탐색 순서:
1. 명시 경로(explicit) - 주어지면 그 경로만 시도하고 다른 후보로 넘어가지 않음
2. 환경변수 AI_MODEL_PATH - 위와 동일하게 그 경로만 시도
3. 패키지 기준 weights/ 아래 기본 파일명 (active_best -> best 순)

상대경로는 현재 작업 디렉터리, 저장소 루트 순으로 찾습니다.
"""
import hashlib
import os
from pathlib import Path
from typing import Mapping, Optional

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
# 런타임이 찾는 기본 가중치 폴더 (학습 스크립트의 기본 저장 위치와 같음)
DEFAULT_WEIGHTS_DIR = PKG_DIR / "weights"

DEFAULT_WEIGHT_NAMES = ("fall_model_active_best.pth", "fall_model_best.pth")


def _resolve_one(path_str: str) -> Optional[str]:
    p = Path(path_str).expanduser()
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, REPO_ROOT / p]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return None


def resolve_weights_path(
    explicit: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    search_dir: Optional[Path] = None,
) -> Optional[str]:
    """사용할 가중치 파일의 절대경로를 반환합니다. 찾지 못하면 None."""
    if explicit:
        return _resolve_one(explicit)

    env_path = (env if env is not None else os.environ).get("AI_MODEL_PATH", "").strip()
    if env_path:
        return _resolve_one(env_path)

    base = Path(search_dir) if search_dir is not None else DEFAULT_WEIGHTS_DIR
    for name in DEFAULT_WEIGHT_NAMES:
        c = base / name
        if c.is_file():
            return str(c.resolve())
    return None


def sha256_8(path: str) -> str:
    """가중치 파일 식별용 sha256 앞 8자리."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]
