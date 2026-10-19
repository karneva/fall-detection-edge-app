# 다른 PC에서 이어서 작업하기

코드는 git으로 옮겨지지만, 데이터와 실험 결과(`dataset/`, 약 106GB)는 git에서 빠져 있습니다.
실험을 다시 돌리지 않고 이어 가려면 아래 폴더를 따로 복사합니다.

## 1. 환경

- Python 3.10 이상 (작성 PC는 3.13), NVIDIA GPU 권장 (작성 PC는 RTX 4050 Laptop 6GB)
- torch는 [pytorch.org](https://pytorch.org) 안내대로 CUDA 버전에 맞춰 먼저 설치

```bash
python -m venv .venv            # 또는 저장소 밖 경로 (Windows 긴 경로 문제가 있으면 C:\fdv 같은 짧은 경로)
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # CUDA에 맞게
pip install -r requirements.txt -r requirements-train.txt -r requirements-dev.txt
python -m unittest discover -s tests -t .        # numpy 없는 파이썬에서도 돌아가는 테스트
python -m pytest tests -q -p no:cacheprovider     # numpy/torch 테스트까지
```

## 2. 복사할 데이터

| 폴더 | 크기 | 없으면 |
| --- | --- | --- |
| `dataset/_manifest/` | 1MB | `build_manifest`로 다시 만들 수 있지만 원본 영상이 필요 |
| `dataset/_keypoints/` | 약 0.2~0.3GB | 원본 영상으로 다시 추출 (v1 약 80분, v3_fb 약 1.5시간) |
| `dataset/_experiments/` | 약 0.2GB | 단계별 실험 결과와 E/G/L 모델. 없으면 처음부터 다시 학습 (몇 시간) |
| `dataset/_models/` | 약 0.2GB | yolov8s-pose.pt, yolov8x-pose.pt (자동 다운로드 가능), yolov8s-pose-ft_v1.pt (파인튜닝 1차) |
| `dataset/_pose/eval_v1/` | 14MB | 포즈 평가셋 초안 (CVAT 수정용) |
| `dataset/_pose/v1/` | 1.5GB | 의사 라벨 데이터 (파인튜닝 다시 할 때만) |
| `dataset/omnifall_labels/` | 6MB | 라벨 CSV |

위 폴더만 있으면 재생 기반 실험(규칙, R1, R2, 회전 폴백 on/off, E*, 실패 분석)은 원본 영상 없이 돌아갑니다.
키포인트를 다시 뽑거나 포즈 모델을 학습하려면 원본 영상이 필요합니다.
`python -m training.datasets.fetch_public download omnifall_labels le2i caucafall gmdcsa24 mcfd urfd figshare_ursul29`로
받으면 됩니다(공개 데이터, 수십 GB).

## 3. 지금 상태 (2026-10-19)

진행 상황과 다음 할 일은 [ROADMAP.md](ROADMAP.md)의 Day 7에 있습니다.

- 키포인트 추출본
  - `v1`: 처음 추출 (단계 A~G 결과의 입력)
  - `v2_base` / `v2_conf025`: 프레임별 대상 전환 기록 추가 (conf 0.5 / 0.25)
  - `v3_fb`: 회전 폴백과 사람 후보 수(n_cand) 기록 추가. 폴백 on/off, 대상 전환 정책 all/multi를 이 추출본 하나로 비교
- 측정 명령 예 (같은 5-fold, 규칙만과 R1):

```bash
python -m training.experiments.run_progression --stages A,R --folds 0-4 --seeds 1 --jobs 5 \
    --kp-dir dataset/_keypoints/v3_fb --out dataset/_experiments/p_v3_fb --folds-file dataset/_experiments/folds_v1.json
# 폴백 끈 결과:     ... --no-fallback --out dataset/_experiments/p_v3_nofb
# 대상 전환 정책:   ... --switch-policy multi
```

## 4. 작업 방식 (지금까지 지킨 것)

- 커밋 작성자는 본인 git 계정만 씀 (Co-Authored-By 같은 AI 표시 없음)
- 커밋 메시지는 `[feat]: ...`, `[fix]: ...`, `[docs]: ...` 형식, 한국어
- 하루 작업이 끝나면 의미 단위로 커밋. push는 요청할 때만
- 모든 변경은 같은 5-fold, 사건 단위 지표로 측정하고 채택 여부를 정함 ([ROADMAP.md](ROADMAP.md)의 채택 규칙)
- 운영 동작을 바꾸는 개선(R1, 회전 폴백)은 기본값을 기존 동작으로 두고 측정 뒤 결정
- 운영 엔진 판정이 바뀌는 변경은 `tests/test_level1_golden.py`로 확인 (의도된 변경일 때만 `--regen`)
- README에는 "한계" 섹션을 두지 않음
- Jetson 기기는 더 이상 없음: 지연·최적화 비교는 노트북에서 측정
