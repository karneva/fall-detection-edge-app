# 포즈 평가셋 라벨 확인 가이드

포즈 모델을 파인튜닝한 효과를 관절 정확도(OKS AP)로 재려면 사람이 확인한 정답이 필요합니다.
평가 데이터셋(le2i, caucafall, gmdcsa24, mcfd, urfd)의 낙상 직후 프레임 150장에 교사 모델(yolov8x-pose + 회전)로
초안 라벨을 찍어 두었습니다. 초안을 보고 틀린 곳만 고치면 됩니다.

- 이미지: `dataset/_pose/eval_v1/images/val/` (150장, 파일 이름에 클립 id)
- 초안: `dataset/_pose/eval_v1/annotations_draft.json` (COCO Keypoints, 사람 229명)
- 예상 시간: 100~150장 기준 2~3시간

초안을 그대로 쓰면 교사 모델과 비슷한 모델이 유리하게 나오므로, 확인하지 않은 초안으로는 평가하지 않습니다.

## 1. CVAT에 올리기

1. [CVAT](https://app.cvat.ai)에서 Project를 만들고, 라벨 `person`을 Skeleton으로 추가합니다.
   Skeleton 설정에서 COCO 17관절 순서(nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
   left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle,
   right_ankle)로 점 17개를 만듭니다.
2. Task를 만들고 `images/val/`의 150장을 올립니다.
3. Task 메뉴의 Upload annotations에서 형식 `COCO Keypoints 1.0`을 고르고 `annotations_draft.json`을 올립니다.

## 2. 고칠 것

| 경우 | 할 일 |
| --- | --- |
| 사람이 보이는데 라벨이 없음 | 상자와 관절을 새로 찍음 |
| 사람이 아닌 것에 라벨이 있음 (가구, 그림자) | 삭제 |
| 같은 사람에 라벨이 두 개 | 하나 삭제 |
| 관절 위치가 틀림 | 점을 옮김 (관절 중심 기준, 몇 픽셀 차이는 괜찮음) |
| 왼쪽/오른쪽이 바뀜 | 사람 기준 왼쪽(화면 기준 아님)으로 바로잡음 |
| 가려져서 위치를 짐작할 수 없음 | 그 관절을 Absent(없음)로 표시 |
| 가려졌지만 위치를 짐작할 수 있음 | Occluded로 두고 짐작한 위치에 찍음 |

애매하면 고치지 않고 넘어가도 됩니다. 일관성이 정확성보다 중요합니다.

## 3. 내보내고 평가하기

1. CVAT에서 Export dataset, 형식 `COCO Keypoints 1.0`으로 받습니다.
2. 압축 안의 `annotations/person_keypoints_default.json`을 `dataset/_pose/eval_v1/annotations.json`으로 둡니다.
3. 저장소 루트에서:

```bash
python -m training.pose.evalset to-yolo --coco dataset/_pose/eval_v1/annotations.json --out dataset/_pose/eval_v1
python -m training.pose.evalset eval --data dataset/_pose/eval_v1/data.yaml \
    --models dataset/_models/yolov8s-pose.pt dataset/_models/yolov8s-pose-ft_v1.pt
```

결과는 모델별 pose mAP50-95 / mAP50(OKS 기준)입니다. 평가셋은 모델 선택에 쓰지 않았으므로 두 모델을 그대로 비교할 수 있습니다.
