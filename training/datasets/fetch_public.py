"""
공개 다운로드 낙상/일상동작 데이터셋 수집기 (가입·신청 없이 받을 수 있는 것만)

  python -m training.datasets.fetch_public list
  python -m training.datasets.fetch_public download gmdcsa24 caucafall
  python -m training.datasets.fetch_public download --all
  python -m training.datasets.fetch_public extract gmdcsa24

- 원본은 dataset/<name>/raw/, 압축 해제본은 dataset/<name>/extracted/ (둘 다 git 제외)
- 이어받기 지원(curl -C -), 공개 체크섬이 있으면 검증
- 받은 결과(URL, 크기, sha256, 라이선스)는 dataset/_downloads.json에 기록
- 데이터 선정 근거와 라이선스 주의사항: docs/DATASETS.md
  (대부분 비상업/연구용이며 재배포 금지인 데이터도 있으므로 원본과 파생 데이터는 git에 올리지 않음)
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.paths import REPO_ROOT

DATASET_ROOT = REPO_ROOT / "dataset"
LOG_NAME = "_downloads.json"

HF_W = "https://huggingface.co/datasets/simplexsigil2/wanfall/resolve/main/"
HF_O = "https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/"

# name -> 출처. files: (저장 이름, URL, 체크섬 알고리즘, 기대값)
SOURCES = {
    "omnifall_labels": {
        "what": "OmniFall/WanFall 통합 구간 라벨과 분할 (영상은 각 원본에서)",
        "license": "CC BY-NC-SA 4.0 (README) / CC BY-NC 4.0 (HF 태그)",
        "page": "https://huggingface.co/datasets/simplexsigil2/wanfall",
        "files": [(p.replace("/", "__"), HF_W + p, None, None) for p in (
            "README.md", "labels/label2id.csv", "labels/le2i.csv", "labels/caucafall.csv", "labels/GMDCSA24.csv",
            "labels/mcfd.csv", "labels/up_fall.csv", "labels/cmdfall.csv", "labels/edf.csv", "labels/occu.csv",
            "labels/OOPS.csv", "labels/of-syn.csv", "data_files/frame_wise_labels.tar.zst",
        )] + [(f"splits__{k}__{d}__{s}.csv", HF_W + f"splits/{k}/{d}/{s}.csv", None, None)
              for k in ("cs", "cv") for d in ("le2i", "caucafall", "gmdcsa24", "mcfd", "up_fall", "cmdfall")
              for s in ("train", "val", "test")]
          + [("omnifall_README.md", HF_O + "README.md", None, None)],
    },
    "gmdcsa24": {
        "what": "실제 침실 3곳, 낙상/ADL(침대 눕기 등 하드 네거티브), 초 단위 구간 라벨",
        "license": "CC BY 4.0 (Zenodo) / MIT (GitHub)",
        "page": "https://zenodo.org/records/13354453",
        "files": [("GMDCSA24-v2.1.zip",
                   "https://zenodo.org/api/records/13354453/files/ekramalam/"
                   "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1.zip/content",
                   "md5", "3d36f2c5c1a666b99639e4e9fd843efb")],
    },
    "caucafall": {
        "what": "벽 모서리 CCTV, 낙상 50 / ADL 50, 조명 변화·IR·가림, 프레임 단위 라벨",
        "license": "CC BY 4.0",
        "page": "https://data.mendeley.com/datasets/7w7fccy7ky/4",
        "files": [("CAUCAFall-v4.zip", "https://data.mendeley.com/public-api/zip/7w7fccy7ky/download/4", None, None)],
    },
    "le2i": {
        "what": "벽 높이 고정 카메라 1대, 4개 장소(Home/Coffee room/Office/Lecture room)",
        "license": "CC BY-NC-SA 3.0",
        "page": "https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html",
        "files": [("FallDataset.zip", "https://search-data.ubfc.fr/imvia/dl_data.php?file=101", None, None)],
    },
    "mcfd": {
        "what": "Multiple Cameras Fall Dataset (Montreal), 카메라 8대, 자세 코드 구간 라벨",
        "license": "명시 없음 (기술보고서 인용 요청), 연구용",
        "page": "https://www.iro.umontreal.ca/~labimage/Dataset/",
        # 원 호스트가 봇 차단(Anubis)으로 막히면 Wayback 미러 사용:
        # https://web.archive.org/web/20211114165908/http://www.iro.umontreal.ca/~labimage/Dataset/dataset.zip
        "files": [
            ("dataset.zip", "https://www.iro.umontreal.ca/~labimage/Dataset/dataset.zip", None, None),
            ("technicalReport.pdf", "https://web.archive.org/web/2025/"
                                    "https://www.iro.umontreal.ca/~labimage/Dataset/technicalReport.pdf", None, None),
        ],
    },
    "figshare_ursul29": {
        "what": "29명 영상 기반 낙상/ADL",
        "license": "CC BY 4.0 (얼굴 비식별 처리 없음)",
        "page": "https://figshare.com/articles/dataset/"
                "Sensor-Based_Fall_Detection_Dataset_with_2017_Activities_from_29_Subjects/28596332",
        "files": [("VideoDataset.zip", "https://ndownloader.figshare.com/files/52990358",
                   "md5", "c784167d08f2fa94e3afd36cec758e1f")],
    },
    "charades": {
        "what": "실내 일상 영상 (침대/소파에 눕기, 앉기 등 하드 네거티브 후보)",
        "license": "AI2 License for Non-Commercial Use (재배포/파생물 공개 금지)",
        "page": "https://arxiv.org/abs/1604.01753",
        "files": [
            ("Charades.zip", "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip", None, None),
            ("Charades_v1_480.zip",
             "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip", None, None),
        ],
    },
    "wanfall_synthetic": {
        "what": "합성 낙상 영상 (사전학습 전용, 평가에 쓰지 않음)",
        "license": "CC BY-NC 4.0",
        "page": "https://huggingface.co/datasets/simplexsigil2/wanfall",
        "files": [
            ("omnifall-synthetic_av1.tar", HF_W + "data_files/omnifall-synthetic_av1.tar", None, None),
            ("syn_frame_wise_labels.tar.zst", HF_W + "data_files/syn_frame_wise_labels.tar.zst", None, None),
            ("videos__metadata.csv", HF_W + "videos/metadata.csv", None, None),
        ],
    },
}


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _curl(url: str, out: Path, attempts: int = 8) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(attempts):
        # 서명 URL(5분 만료)이 섞여 있어 매 시도마다 원래 URL에서 리다이렉트를 다시 받음
        r = subprocess.run(["curl", "-sS", "-L", "-f", "-C", "-", "--retry", "3", "--retry-delay", "5",
                            "--connect-timeout", "30", "-o", str(out), url])
        if r.returncode == 0:
            return True
        if r.returncode == 33:  # 이어받기 미지원 -> 처음부터
            out.unlink(missing_ok=True)
        print(f"  retry {i + 1}/{attempts} (curl exit {r.returncode})", flush=True)
        time.sleep(10)
    return False


def _load_log(root: Path) -> dict:
    p = root / LOG_NAME
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_log(root: Path, log: dict):
    root.mkdir(parents=True, exist_ok=True)
    (root / LOG_NAME).write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


def download(name: str, root: Path = DATASET_ROOT) -> bool:
    src = SOURCES[name]
    log = _load_log(root)
    entry = log.setdefault(name, {})
    entry.update({"license": src["license"], "page": src["page"]})
    files = entry.setdefault("files", {})
    all_ok = True
    for fname, url, algo, expected in src["files"]:
        out = root / name / "raw" / fname
        print(f"[{name}] {fname}", flush=True)
        ok = _curl(url, out)
        rec = {"url": url, "ok": ok, "bytes": out.stat().st_size if out.exists() else 0,
               "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if ok:
            rec["sha256"] = file_hash(out, "sha256")
            if algo:
                rec["checksum_ok"] = file_hash(out, algo) == expected
                ok = ok and rec["checksum_ok"]
        files[fname] = rec
        all_ok = all_ok and ok
        _save_log(root, log)
        print(f"  -> ok={ok} bytes={rec['bytes']}", flush=True)
    return all_ok


def _safe_target(base: Path, member: str) -> Path:
    target = (base / member).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise ValueError(f"unsafe path in archive: {member}")
    return target


def _is_mac_metadata(name: str) -> bool:
    """macOS가 zip에 넣는 메타데이터 (__MACOSX/, AppleDouble ._파일)."""
    parts = name.replace("\\", "/").split("/")
    return "__MACOSX" in parts or parts[-1].startswith("._")


def safe_extract(archive: Path, dest: Path):
    """
    경로 조작(../, 절대경로)과 링크를 거부하며 zip/tar를 풉니다.
    macOS 메타데이터는 건너뛰고, 안에 든 zip은 같은 규칙으로 한 번 더 풉니다.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if _is_mac_metadata(info.filename):
                    continue
                target = _safe_target(dest, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if m.issym() or m.islnk() or m.isdev():
                    raise ValueError(f"link/device in archive: {m.name}")
                _safe_target(dest, m.name)
            try:
                t.extractall(dest, filter="data")
            except TypeError:  # Python < 3.12: 위에서 경로/링크를 이미 검사함
                t.extractall(dest)
    else:
        raise ValueError(f"not an archive: {archive}")
    for inner in list(dest.rglob("*.zip")):
        if not zipfile.is_zipfile(inner):
            continue
        target = inner.with_suffix("")
        if target.exists() and not target.is_dir():
            target = inner.parent / (inner.name + "_extracted")
        safe_extract(inner, target)
        inner.unlink()


def extract(name: str, root: Path = DATASET_ROOT):
    raw = root / name / "raw"
    for f in sorted(raw.iterdir()):
        if f.suffix in (".zip", ".tar") and (zipfile.is_zipfile(f) or tarfile.is_tarfile(f)):
            dest = root / name / "extracted" / f.stem
            print(f"[{name}] extract {f.name} -> {dest.relative_to(root)}", flush=True)
            safe_extract(f, dest)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download public fall/ADL datasets (no registration needed)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for c in ("download", "extract"):
        p = sub.add_parser(c)
        p.add_argument("names", nargs="*")
        p.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for k, v in SOURCES.items():
            print(f"{k:20s} {v['license']}\n{'':20s} {v['what']}")
        return 0
    names = list(SOURCES) if args.all else args.names
    unknown = [n for n in names if n not in SOURCES]
    if not names or unknown:
        ap.error(f"choose from: {', '.join(SOURCES)}" + (f" (unknown: {unknown})" if unknown else ""))
    ok = True
    for n in names:
        if args.cmd == "download":
            ok = download(n) and ok
        else:
            extract(n)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
