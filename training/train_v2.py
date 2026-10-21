"""
배포용 사건 구간 분류기 학습 (Day 8 결정 3): E1 = Fall1DCNN-GAP, v2 사건 구간 창(92채널 x 60스텝, 15Hz 4초),
단계 E와 같은 레시피("improved")와 창 생성(windows_event_v2, 같은 증강 rng 규칙)으로 manifest 전체(평가 5개 데이터셋 +
figshare)를 학습합니다. 판정은 E*max 방식 (ABNORMAL 진입부터 판정 시점까지 0.5초 간격 4초 창들의 최대 p_fall, 0.4 미만이면 억제).

- 앞단: 학습 창을 만드는 규칙 엔진 재생도 운영 기본값과 같게 (lost_grace_s 0.5, 대상 전환 multi, 회전 폴백 켬,
  LEVEL1 이후 재진입 래치 켬. study_params는 래치를 끄므로 명시적으로 켬). 메타 frontend에 래치 값도 기록.
  성능 추정(p8_full 5-fold)은 래치 끈 재생으로 fold 모델을 학습/평가해 학습 창 수가 조금 다름 (전체 데이터 기준
  학습 창 16,126 켬 vs 16,137 끔): 5-fold 값은 거의 같은 레시피의 추정치.
  키포인트는 회전 폴백과 후보 수(n_cand)가 들어 있는 추출본(기본 dataset/_keypoints/v3_fb)이어야 함.
- 검증: 전체 행을 그룹 단위 15% (folds.split_train_val, seed 0)로 떼어 early stopping에만 씀. 검증 지표(창 AUC,
  검증 클립 재생 결과)는 early stopping에 쓴 집합이라 낙관적이며 성능 추정이 아님 (성능은 run_progression 5-fold 결과).
- 출력: <out> (torch.save {"arch": "E1", "state_dict", "mu", "sd"}, run_progression E1_s*.pt와 같은 형식,
  가중치는 CPU 텐서, mu/sd는 채널별 표준화 float32 배열)와 <out>.meta.json (app.ai_classifier.model_meta).

- 추가 manifest (--extra-manifest, 예: 현장 검수 클립 training/review_queue.py export): 기본 manifest 행에 더해 학습.
  키포인트는 --extra-kp-dir(기본 --kp-dir)에서 읽음. clip_id가 기본 manifest와 겹치면 거부.
  추가 manifest를 쓰면 배포 가중치 위치(training.paths.is_runtime_weights_path)에는 저장하지 않음 (후보로 저장 후 평가).

실행 (저장소 루트에서)
  python -m training.train_v2 --kp-dir dataset/_keypoints/v3_fb --out dataset/_models/fall_cnn_gap_v2.pt
  python -m training.train_v2 --extra-manifest dataset/_manifest/field_runs.jsonl
      --extra-kp-dir dataset/_keypoints/field_v3_fb --out dataset/_models/candidates/fall_cnn_gap_v2_field.pt
  (현장 클립 재학습은 위 과정을 묶은 python -m training.retrain_from_runs 를 씀)
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "dataset" / "_manifest" / "manifest_v1.jsonl"
DEFAULT_KP_DIR = REPO_ROOT / "dataset" / "_keypoints" / "v3_fb"
DEFAULT_OUT = REPO_ROOT / "dataset" / "_models" / "fall_cnn_gap_v2.pt"

ARCH_KEY = "E1"                  # run_progression 저장 형식의 arch 값
META_ARCH = "fall_cnn_gap"       # 메타의 arch (런타임이 보는 이름)
FEATURE_VERSION = "v2"
WINDOW_S = 4.0
HZ = 15
MODE = "max"
STEP_S = 0.5
THRESHOLD = 0.4
VAL_FRAC = 0.15
VAL_SPLIT_SEED = 0
# 운영 기본 앞단 (Day 8 결정 1). 알람 뒤 래치는 운영 기본값과 5-fold 평가(p8_full)처럼 끔
# (배포 중인 fall_cnn_gap_v2는 래치를 켠 재생으로 학습했고 meta에 그대로 기록되어 있음)
FRONTEND: Dict[str, Any] = {"lost_grace_s": 0.5, "switch_policy": "multi", "rotate_fallback": True,
                            "post_alarm_latch": False}

__all__ = ["FRONTEND", "train_v2", "build_meta", "merge_extra_rows", "main"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_meta(n_windows: Dict[str, Any], val: Dict[str, Any], trained_on: Sequence[str], keypoints: str,
               frontend: Dict[str, Any], train: Dict[str, Any], git_sha: Optional[str] = None,
               n_params: Optional[int] = None) -> Dict[str, Any]:
    """가중치 옆 메타 (model_meta.make_meta 위에 v2 사건 구간 분류기 필드)"""
    from app.ai_classifier import model_meta as MM
    from training.experiments import windows as W

    return MM.make_meta(
        trained_on=trained_on, git_sha=git_sha,
        arch=META_ARCH, feature_version=FEATURE_VERSION,
        in_channels=int(W.V2_CHANNELS), seq_len=int(W.V2_STEPS), hz=HZ,
        window_s=WINDOW_S, window_pre_s=float(W.V2_PRE_S), window_post_s=float(W.V2_POST_S),
        mode=MODE, step_s=STEP_S, threshold=THRESHOLD, suppress_pfall_th=THRESHOLD,
        keypoints=str(keypoints), frontend=dict(frontend),
        standardize="per-channel mu/sd in checkpoint (mu, sd)",
        n_windows=n_windows, val=val, train=train, n_params=n_params,
    )


def merge_extra_rows(rows: List[Dict[str, Any]], extra_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    기본 manifest 행에 추가 행을 붙입니다 (표준 라이브러리만 사용). 추가 행은 manifest 검증을 통과해야 하고
    clip_id가 기본 행과 겹치면 안 됩니다 (겹치면 키포인트 폴더가 섞임). 문제가 있으면 ValueError.
    """
    from training.datasets.manifest import validate_rows

    errors = validate_rows(extra_rows)
    if errors:
        raise ValueError("extra manifest: " + "; ".join(errors[:5]))
    base_ids = {r["clip_id"] for r in rows}
    dup = sorted(r["clip_id"] for r in extra_rows if r["clip_id"] in base_ids)
    if dup:
        raise ValueError(f"extra manifest clip_id already in base manifest: {dup[:5]}")
    return list(rows) + list(extra_rows)


def train_v2(rows: List[Dict[str, Any]], clips: Dict[str, Dict[str, Any]], out_path, seed: int = 0,
             device=None, epochs: Optional[int] = None, frontend: Optional[Dict[str, Any]] = None,
             val_frac: float = VAL_FRAC, keypoints: str = "v3_fb", event_val: bool = True,
             git_sha: Optional[str] = None) -> Dict[str, Any]:
    """
    rows(manifest 행) 전체로 E1을 학습해 out_path와 메타를 씁니다. 반환: 메타 dict.
    키포인트가 없는 행은 뺌. epochs: epoch 상한 덮어쓰기 (점검용). event_val: 검증 클립을 E*max로 재생해 지표 기록.
    재생 전역 설정(replay.SWITCH_POLICY, USE_FALLBACK)은 학습 동안만 frontend 값으로 바꾸고 되돌림.
    """
    import numpy as np
    import torch

    from app.ai_classifier import model_meta as MM
    from training.experiments import folds as FO
    from training.experiments import models as M
    from training.experiments import replay as R
    from training.experiments import run_progression as RP
    from training.experiments import trainer as T
    from training.experiments import windows as W
    from training.experiments.verifiers import SlidingWindowVerifier

    fe = dict(FRONTEND if frontend is None else frontend)
    out_path = Path(out_path)
    rows = [r for r in rows if r["clip_id"] in clips]
    if not rows:
        raise ValueError("키포인트가 있는 manifest 행이 없습니다")
    dev = T.resolve_device(device)
    saved = (R.SWITCH_POLICY, R.USE_FALLBACK)
    R.SWITCH_POLICY, R.USE_FALLBACK = str(fe["switch_policy"]), bool(fe["rotate_fallback"])
    try:
        latch = bool(fe.get("post_alarm_latch", False))
        p_off = R.study_params("off", lost_grace_s=float(fe["lost_grace_s"]), post_alarm_latch=latch)
        p_veto = R.study_params("veto", lost_grace_s=float(fe["lost_grace_s"]), ai_suppress_pfall_th=THRESHOLD,
                                post_alarm_latch=latch)
        tr_rows, va_rows = FO.split_train_val(rows, frac=val_frac, seed=VAL_SPLIT_SEED)
        _log(f"rows {len(rows)} (fit {len(tr_rows)}, val {len(va_rows)}), device {dev}, frontend {fe}")
        t0 = time.time()
        X_va, y_va, m_va = W.windows_event_v2(va_rows, clips, p_off, augment=False, return_meta=True)
        X_tr, y_tr, m_tr = W.windows_event_v2(tr_rows, clips, p_off, rng=np.random.default_rng(1000 + seed),
                                              augment=True, return_meta=True)
        if len(y_tr) == 0 or len(set(np.asarray(y_tr).tolist())) < 2:
            raise ValueError(f"학습 창에 두 클래스가 모두 있어야 합니다 (n={len(y_tr)})")
        n_windows = {"train": int(len(y_tr)), "val": int(len(y_va)),
                     "train_pos_frac": float(np.mean(y_tr)), "val_pos_frac": float(np.mean(y_va)) if len(y_va) else None,
                     "train_kinds": W.count_windows(m_tr, by=("kind", "y"))}
        _log(f"windows train {X_tr.shape} pos {np.mean(y_tr):.3f}, val {X_va.shape} ({time.time() - t0:.0f}s)")

        std = RP._Standardizer.fit(X_tr)
        T.set_seed(seed)
        model = M.Fall1DCNNGAP(W.V2_CHANNELS)
        n_params = M.count_params(model)
        recipe: Dict[str, Any] = {"name": "improved"}
        if epochs is not None:
            recipe["epochs"] = int(epochs)
        ts = time.time()
        model, hist = T.train_classifier(model, std(X_tr), y_tr, std(X_va), y_va, recipe, seed=seed, device=dev)
        train = RP._train_summary(hist, time.time() - ts, y_tr)
        train.update({"recipe": "improved", "seed": int(seed), "epochs_override": epochs, "device": str(dev)})
        _log(f"trained best ep {hist['best_epoch']}/{hist['stopped_epoch']} val loss {hist['best_val_loss']}")

        val: Dict[str, Any] = {"split": f"split_train_val frac {val_frac} seed {VAL_SPLIT_SEED} (early stopping)",
                               "n_clips": len(va_rows)}
        if len(y_va):
            p_va = T.predict_proba(model, std(X_va), device=dev)
            val["window"] = RP._window_metrics(y_va, p_va, m_va)
        if event_val:
            va_eval = [r for r in va_rows if r.get("dataset") in FO.EVAL_DATASETS]
            if va_eval:
                ver = SlidingWindowVerifier(RP._e_scorer(("nn", model, std), str(dev)), mode=MODE, model_id="v2-val")
                outs = RP._evaluate(va_eval, clips, p_veto, ver)
                s = R.summarize(outs)
                val["event"] = {k: s.get(k) for k in ("n_fall", "n_adl", "recall", "adl_fa_rate", "balanced",
                                                       "rules_ceiling", "adl_rules_fa_rate", "suppressed_true_falls")}
                _log(f"val event recall {s['recall']} adl_fa {s['adl_fa_rate']} balanced {s['balanced']}")
    finally:
        R.SWITCH_POLICY, R.USE_FALLBACK = saved

    trained_on = sorted({str(r.get("dataset")) for r in rows})
    meta = build_meta(n_windows, val, trained_on, keypoints, fe, train, git_sha=git_sha, n_params=n_params)
    # numpy 스칼라/배열이 섞여 있어도 JSON으로 저장되게 (run_progression 결과 파일과 같은 변환)
    meta = json.loads(json.dumps(meta, default=RP._json_default))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tmp = out_path.with_name(out_path.name + ".tmp")
    torch.save({"arch": ARCH_KEY, "state_dict": state, "mu": std.mu, "sd": std.sd}, tmp)
    os.replace(tmp, out_path)
    MM.write_meta(out_path, meta)
    reason = MM.check_compatible(meta)
    if reason is not None:
        _log(f"WARNING: runtime check_compatible rejects this meta: {reason}")
    _log(f"saved {out_path} (+ {MM.meta_path_for(out_path).name})")
    return meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="배포용 v2 사건 구간 분류기(E1, Fall1DCNN-GAP) 학습")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--kp-dir", default=str(DEFAULT_KP_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None, help="epoch 상한 덮어쓰기 (점검용)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU 스레드 수")
    ap.add_argument("--lost-grace", type=float, default=FRONTEND["lost_grace_s"])
    ap.add_argument("--switch-policy", default=FRONTEND["switch_policy"], choices=("all", "multi"))
    ap.add_argument("--no-fallback", action="store_true", help="회전 폴백 프레임을 미검출로 재생")
    ap.add_argument("--no-event-val", action="store_true", help="검증 클립 재생 지표를 건너뜀")
    ap.add_argument("--extra-manifest", default=None, help="기본 manifest에 더할 manifest (예: 현장 검수 클립)")
    ap.add_argument("--extra-kp-dir", default=None, help="추가 manifest의 키포인트 폴더 (기본: --kp-dir)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    if args.extra_manifest:
        from training.paths import is_runtime_weights_path
        if is_runtime_weights_path(out):
            # 현장 데이터를 섞은 모델은 평가 전에 배포 위치를 덮어쓰지 않음
            raise SystemExit(f"--extra-manifest 학습 결과를 배포 가중치 위치에 쓸 수 없습니다: {out} (--out으로 후보 경로 지정)")

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("app").setLevel(logging.WARNING)
    import torch

    from app.ai_classifier.model_meta import current_git_sha
    from training.datasets.manifest import load_manifest
    from training.experiments import windows as W

    if args.threads:
        torch.set_num_threads(int(args.threads))
    kp_dir = Path(args.kp_dir)
    if not kp_dir.is_absolute():
        kp_dir = REPO_ROOT / kp_dir
    rows = load_manifest(args.manifest)
    t0 = time.time()
    clips = W.load_clips(rows, kp_dir=kp_dir)
    _log(f"loaded {len(clips)}/{len(rows)} clips from {kp_dir} in {time.time() - t0:.0f}s")
    keypoints = kp_dir.name
    if args.extra_manifest:
        extra = load_manifest(args.extra_manifest)
        rows = merge_extra_rows(rows, extra)
        ex_kp = Path(args.extra_kp_dir) if args.extra_kp_dir else kp_dir
        if not ex_kp.is_absolute():
            ex_kp = REPO_ROOT / ex_kp
        ex_clips = W.load_clips(extra, kp_dir=ex_kp)
        clips.update(ex_clips)
        if ex_kp != kp_dir:
            keypoints = f"{kp_dir.name}+{ex_kp.name}"
        _log(f"extra {len(ex_clips)}/{len(extra)} clips from {args.extra_manifest} ({ex_kp})")
    fe = {"lost_grace_s": float(args.lost_grace), "switch_policy": args.switch_policy,
          "rotate_fallback": not args.no_fallback, "post_alarm_latch": FRONTEND["post_alarm_latch"]}
    train_v2(rows, clips, out, seed=args.seed, device=None if args.device == "auto" else args.device,
             epochs=args.epochs, frontend=fe, keypoints=keypoints, event_val=not args.no_event_val,
             git_sha=current_git_sha())
    return 0


if __name__ == "__main__":
    sys.exit(main())
