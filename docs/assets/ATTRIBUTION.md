# 데모 이미지 출처와 라이선스

이 폴더의 데모 이미지는 공개 데이터셋 두 개의 영상 프레임으로 만들었습니다. 두 데이터셋 모두 CC BY 4.0이라
출처를 밝히면 복제, 수정, 재배포할 수 있습니다. 다른 데이터셋(Le2i, URFD, MCFD, Figshare 등)의 영상은 쓰지 않았습니다.

## 사용한 데이터셋

**CAUCAFall**
- 저작자: Jose Camilo Eraso, Elena Muñoz, Mariela Muñoz, Jesus Pinto
- 출처: "Dataset CAUCAFall", Mendeley Data, Version 4 (2022), https://doi.org/10.17632/7w7fccy7ky.4
- 라이선스: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**GMDCSA24**
- 저작자: Ekram Alam, Abu Sufian, Paramartha Dutta, Marco Leo, Ibrahim A. Hameed
- 출처: "GMDCSA24: A Dataset for Human Fall Detection in Videos", Data in Brief (2024).
  데이터: Ekram Alam, ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos v2.1, Zenodo (2024),
  https://doi.org/10.5281/zenodo.13354453
- 라이선스: Zenodo 기록은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)입니다.
  같은 내용의 GitHub 저장소는 MIT 라이선스(Copyright (c) 2024 Ekram Alam)입니다.

## 이미지별 원본

| 파일 | 원본 클립 | 내용 |
| --- | --- | --- |
| `demo_fall_strip.jpg` | CAUCAFall Subject 9, Fall backwards (`FallBackwardsS9.avi`) | 낙상 장면 5프레임 |
| `demo_fall_timeline.png` | 같은 클립 (영상 프레임 없음, 키포인트에서 계산한 값만) | 상자 비율과 엔진 상태의 시간축 |
| `demo_adl_veto.jpg` | GMDCSA24 Subject 2, ADL `09.mp4` | 일상 동작(침대에서 일어나 바닥에 앉기) 4프레임 |
| `demo_rotation_fallback.jpg` | GMDCSA24 Subject 2, Fall `06.mp4`, 프레임 번호 177 (0부터 셈, t=5.93s) | 회전 폴백 전후 |

클립 설명 문구는 각 데이터셋의 원본 표기(CAUCAFall 폴더 이름, GMDCSA24 `ADL.csv`/`Fall.csv`의 Description)를 그대로 옮겼습니다.

## 원본에서 바꾼 점

- 프레임 크기를 줄이고 여러 프레임을 한 장에 이어 붙였습니다.
- YOLOv8-Pose로 추출한 관절 골격과 사람 상자, 상태 테두리, 영어 설명 문구를 덧그렸습니다.
- 회전 폴백 그림의 가운데 칸은 원본 프레임을 90도 돌린 것입니다.
- 얼굴을 따로 잘라 내거나 확대하지 않았습니다.
- 영상 프레임이 들어간 세 이미지는 아래쪽에 출처, 라이선스 주소, 바꾼 점을 적어 두어 이미지만 따로 써도 표시가 남습니다.

이미지의 시각, 엔진 상태, `p_fall`은 이 저장소의 재생 코드가 낸 값이며, 데이터셋 저작자의 결과나 의견이 아닙니다.
데이터셋 저작자가 이 프로젝트를 보증하지 않습니다.

## 다시 만들기

```bash
# 저장소 루트, 학습용 venv (numpy, opencv, torch, matplotlib, ultralytics). dataset/에 영상, v3_fb 키포인트,
# p8_full 실험 결과가 있어야 합니다.
python -m training.demo.render_demo --out docs/assets
```

재생 결과가 연구 결과 파일(`dataset/_experiments/p8_full/fold{k}/E1max.json`)과 다르면 이미지를 만들지 않고 멈춥니다.
