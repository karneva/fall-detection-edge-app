"""
가중치 메타데이터 사이드카 "<weights>.meta.json" (표준 라이브러리만 사용)

가중치(.pth)만으로는 어떤 입력 형식(feature_version, 채널 수, 길이, 주파수)으로 학습했는지 알 수 없어서,
다른 형식의 모델을 실수로 런타임에 올리면 조용히 틀린 확률을 내게 됩니다.
학습 스크립트가 가중치 옆에 메타를 남기고, 런타임은 이를 확인해 호환되지 않으면 분류기를 끕니다(fail-open).

메타 파일이 없는 기존 가중치는 legacy v1(Fall1DCNN, 34채널 x 30프레임)로 간주합니다.
"""
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

META_VERSION = 1
META_SUFFIX = ".meta.json"

# 이 런타임(app/ai_classifier/inference.py)이 실행할 수 있는 형식
RUNTIME_ARCH = "Fall1DCNN"
RUNTIME_FEATURE_VERSION = "v1"
RUNTIME_IN_CHANNELS = 34
RUNTIME_SEQ_LEN = 30

_REPO_ROOT = Path(__file__).resolve().parents[2]


def meta_path_for(weights_path) -> Path:
    """가중치 경로에 대응하는 메타 파일 경로 (예: fall_model_best.pth -> fall_model_best.pth.meta.json)."""
    p = Path(weights_path)
    return p.with_name(p.name + META_SUFFIX)


def legacy_v1_meta() -> Dict[str, Any]:
    """메타 파일이 없는 기존 가중치에 적용하는 기본값 (지금까지 학습된 v1 모델)."""
    return {
        "meta_version": META_VERSION,
        "arch": RUNTIME_ARCH,
        "feature_version": RUNTIME_FEATURE_VERSION,
        "in_channels": RUNTIME_IN_CHANNELS,
        "seq_len": RUNTIME_SEQ_LEN,
        "hz": None,
        "temperature": 1.0,
        "suppress_pfall_th": None,
        "trained_on": [],
        "git_sha": None,
        "created_at": None,
    }


def make_meta(trained_on: Optional[Iterable[str]] = None, git_sha: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """학습 스크립트용: v1 기본값에 학습 정보와 생성 시각을 채운 메타를 만듭니다."""
    meta = legacy_v1_meta()
    meta["trained_on"] = [str(x) for x in (trained_on or [])]
    meta["git_sha"] = git_sha
    meta["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta.update(overrides)
    return meta


def load_meta(weights_path) -> Optional[Dict[str, Any]]:
    """
    가중치 옆의 메타를 읽습니다. 파일이 없으면 None (legacy로 처리할지는 호출 측이 결정).
    파일은 있는데 읽을 수 없거나 JSON 객체가 아니면 ValueError
    (의도를 알 수 없는 모델을 legacy로 추측해 쓰지 않도록).
    """
    mp = meta_path_for(weights_path)
    if not mp.is_file():
        return None
    try:
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"unreadable meta {mp.name}: {e}") from e
    if not isinstance(meta, dict):
        raise ValueError(f"meta {mp.name} is not a JSON object")
    return meta


def write_meta(weights_path, meta: Dict[str, Any]) -> str:
    """메타를 가중치 옆에 원자적으로 저장하고 경로를 반환합니다."""
    reason = _check_fields(meta)
    if reason is not None:
        raise ValueError(reason)
    mp = meta_path_for(weights_path)
    tmp = mp.with_name(mp.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, mp)
    except BaseException:
        # 직렬화 실패 등으로 중단되면 반쯤 쓴 임시 파일을 남기지 않음 (기존 메타는 그대로)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(mp)


def _check_fields(meta) -> Optional[str]:
    if not isinstance(meta, dict):
        return f"meta is not a dict ({type(meta).__name__})"
    return None


def check_compatible(meta) -> Optional[str]:
    """이 런타임이 해당 모델을 쓸 수 없으면 그 이유(문자열)를, 쓸 수 있으면 None을 반환합니다."""
    reason = _check_fields(meta)
    if reason is not None:
        return reason
    mv = meta.get("meta_version", META_VERSION)
    if not isinstance(mv, int) or isinstance(mv, bool) or mv < 1 or mv > META_VERSION:
        return f"meta_version={mv!r} unsupported (runtime supports <= {META_VERSION})"
    checks = (
        ("arch", RUNTIME_ARCH),
        ("feature_version", RUNTIME_FEATURE_VERSION),
        ("in_channels", RUNTIME_IN_CHANNELS),
        ("seq_len", RUNTIME_SEQ_LEN),
    )
    for key, expected in checks:
        got = meta.get(key)
        if got != expected or isinstance(got, bool):
            return f"{key}={got!r} (runtime requires {expected!r})"
    return None


def current_git_sha(repo_root=None) -> Optional[str]:
    """저장소의 현재 커밋 해시 (git이 없거나 저장소가 아니면 None). 수정 사항이 있으면 '-dirty'를 붙입니다."""
    root = str(repo_root or _REPO_ROOT)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, timeout=10)
        if sha.returncode != 0:
            return None
        out = sha.stdout.strip() or None
        if out is None:
            return None
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                            capture_output=True, text=True, timeout=10)
        if st.returncode == 0 and st.stdout.strip():
            out += "-dirty"
        return out
    except Exception:
        return None
