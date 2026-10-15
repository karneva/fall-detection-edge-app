# Datasets: URFD 이후 확보할 낙상/일상동작 데이터

현재 1D CNN은 공개 데이터로 **URFD**(낙상 약 30건, ADL 약 40건)만 씁니다.
이 문서는 추가로 수집할 수 있는 데이터를 조사해 이 프로젝트 기준으로 우선순위를 매긴 것입니다.
(조사일 2026-10-15, 각 데이터셋의 공식 페이지/논문/저장소에서 확인. 수치가 출처마다 다른 항목은 받은 뒤 직접 확인 필요)

## 선정 기준

1. **고정된 실내 RGB 카메라**로 찍었는가 (YOLOv8-Pose로 COCO-17 키포인트를 다시 뽑을 수 있어야 함)
2. **낙상 시작/끝 구간 라벨**이 있는가 (낙상 onset 기준 창을 만들기 위해 필요)
3. **하드 네거티브**(침대/소파에 눕기, 털썩 앉기, 줍기)가 있는가
4. **받기 쉬운가**, 라이선스가 포트폴리오/연구 용도로 허용되는가

깊이/IMU 전용, 1인칭 영상, 정지 이미지 데이터는 이 파이프라인에 넣을 수 없어 제외했습니다.

## Tier 1: 지금 확보

| 데이터셋 | 규모 | 특징 | 받는 방법 | 라이선스 | 용도 |
| --- | --- | --- | --- | --- | --- |
| [Le2i / ImViA FDD](https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html) | 낙상 약 143건, 4개 장소 | 벽 높이 고정 카메라 1대, 배포 구도와 가장 유사 | 공개 다운로드 (약 9GB, 브라우저로) | CC BY-NC-SA 3.0 | 양성 + ADL, 장소별 교차 테스트 |
| [CAUCAFall](https://data.mendeley.com/datasets/7w7fccy7ky/4) | 낙상 50, ADL 50, 10명 | 벽 모서리 2.15m CCTV, 조명 변화·IR·가림 포함, 프레임 단위 라벨 | 공개 다운로드 | **CC BY 4.0** (상업 가능) | 양성, 교차 테스트 |
| [GMDCSA24](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos) | 낙상·ADL 각 약 80건, 4명 | 실제 침실, 침대 눕기·요가 등 하드 네거티브, 초 단위 구간 라벨 | GitHub / Zenodo (약 1.1GB) | **MIT / CC BY 4.0** | 양성 + 하드 네거티브 |
| [MCFD (Montreal)](https://www.iro.umontreal.ca/~labimage/Dataset/) | 낙상 시나리오 22, 카메라 8대 | 천장 근처 광각, 소파에 눕기·웅크림 등 자세 코드 라벨 | 공개 다운로드 (약 3.8GB) | 명시 없음 (연구용) | 양성 + 하드 네거티브, 시점 테스트 |
| [OmniFall 라벨](https://huggingface.co/datasets/simplexsigil2/omnifall) | 8개 연출 데이터셋 통합 | Le2i·GMDCSA24·CAUCAFall·MCFD·UP-Fall·CMDFall 등을 16클래스 구간 라벨로 재주석 | HF 공개 (라벨만, 영상은 원본에서) | CC BY-NC-SA 4.0 | **라벨 기준 포맷** |
| [AI-Hub 낙상사고 위험동작 영상-센서 (71641)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71641) | 낙상 클립 약 1.7만(이벤트 약 2천), 비낙상 약 5.7천 | 요양시설·가정·병원, 40대 이상 75%, 보조기구 사용자 포함, fall_start/end 프레임 | 내국인 신청 후 승인 | AI-Hub 약관 (학습 용도, 재배포·국외반출 금지) | 양성 + near-fall 하드 네거티브 |
| [SAFER-Activities](https://safer-activities.github.io/) | 낙상 약 5.4천, 비낙상 약 8만 | CCTV 8대, 휠체어 낙상, bending·sitting·lying 등 하드 네거티브, 2D 포즈 동봉 | HF 로그인 + 동의 (약 173GB) | CC BY-NC-SA 4.0 | 양성 + 하드 네거티브, 가정 OOD 테스트 |

**AI-Hub 71641**은 한국 요양시설/가정 환경과 프레임 단위 낙상 구간이라는 점에서 가장 잘 맞습니다.
신청 화면에서 "안심존(IRB) 전용"인지 "다운로드 가능"인지 꼭 확인하세요. 안심존 전용이면 로컬 학습이 어려워 Tier 3로 내려갑니다.

## Tier 2: 가치 있음 (요청/승인 필요, 대용량, 도메인 차이)

| 데이터셋 | 특징 | 받는 방법 | 라이선스 | 용도 |
| --- | --- | --- | --- | --- |
| [CMDFall](https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html) | 50명, 같은 사람·장면에서 낙상과 대조 동작(침대 눕기, 앉기, 줍기)을 짝으로 촬영 | 저자에게 이메일 요청 (약 350GB) | 연구용 | 양성 + 하드 네거티브 |
| [UP-Fall](https://sites.google.com/up.edu.mx/har-up/) | 17명, 카메라 2대 | 공개 (전체 800GB+, 카메라 ZIP만 선택) | 명시 없음 (연구용) | 양성 + 네거티브 |
| [AI-Hub 매장 이상행동 (71550, 전도)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71550) | 실내 매장 CCTV, 전도 클래스 | 내국인 신청 | AI-Hub 약관 | 양성 |
| [AI-Hub 시니어 이상행동 (167)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=167) | 고령자 행동 | 내국인 신청 | AI-Hub 약관 | 샘플 확인 후 결정 |
| [Toyota Smarthome](https://project.inria.fr/toyotasmarthome/) | 고령자 실제 ADL 장시간 영상 (낙상 없음) | 요청 폼, **기관 이메일 필수** | 학술 연구 전용 | 하드 네거티브, 시간당 오탐 측정 |
| [ETRI-Activity3D LivingLab](https://ai4robot.github.io/etri-activity3d-livinglab/) | 한국 고령자 생활 행동 | nanum.etri.re.kr 가입 + 사용 협약서 | 연구용, ETRI 사사 표기 | 하드 네거티브 |
| [Charades](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip) | 침대/소파에 눕기, 앉기 등 실내 일상 | 공개 (480p 약 16GB) | AI2 비상업, 재배포 금지 | 하드 네거티브 |
| [Fall Dataset 29 Subjects (Figshare)](https://figshare.com/articles/dataset/Sensor-Based_Fall_Detection_Dataset_with_2017_Activities_from_29_Subjects/28596332) | 29명, 영상 포함 | 공개 (약 2.5GB) | **CC BY 4.0** | 양성, 시점 테스트 |
| [EVIORA-HAR](https://ieee-dataport.org/documents/eviora-har-elderly-multi-view-and-wearable-motion-recognition-dataset-daily-activities) | 고령자 다시점 ADL | IEEE DataPort (접근 등급 미확인) | CC BY 4.0 | 하드 네거티브 |

## Tier 3: 교차 테스트 / 사전학습 / 참고

| 데이터셋 | 용도 | 비고 |
| --- | --- | --- |
| [SFU Falls in Long-Term Care](https://nyu.databrary.org/volume/739) | **실제 요양시설 노인 낙상** 재현율 점검 | 공개분 28건, 전체는 연구자 인증/요청 |
| [OOPS-Fall](https://oops.cs.columbia.edu/data/) (OmniFall) | 실제 웹 영상 OOD 강건성 | `pip install omnifall` 후 필요한 부분만 추출 |
| [WanFall](https://huggingface.co/datasets/simplexsigil2/wanfall) (OmniFall OF-Syn) | 합성 낙상 사전학습 | 평가에는 쓰지 않음 |
| [PYSKL NTU RGB+D 2D 포즈](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md) | 사전학습 (A43 falling 등) | NTU ROSE Lab 계약 별도, 재배포 금지 |
| [FallVision](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/75QPKK) | 사전학습 | 라이선스 표기 불일치 → 비상업으로 간주 |
| [AI-Hub 이상행동 CCTV (171)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=171), [사람동작 (62)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=62), [지능형 관제 CCTV (71850)](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71850) | 실신/쓰러짐 보조 양성, OOD 테스트 | 내국인 신청 |
| [KU Leuven HQFSD / MUVIM](https://iiw.kuleuven.be/onderzoek/advise/datasets) | 장시간 ADL 오탐률 | 저자 요청 |

## 제외

| 데이터셋 | 이유 |
| --- | --- |
| EDF/OCCU, TST Fall, FUKinect-Fall | 깊이/Kinect 스켈레톤만 있고 RGB 없음 |
| KFall, FallAllD | 웨어러블 IMU 전용 (낙상 분류표는 자체 촬영 체크리스트로 참고) |
| EGOFALLS | 1인칭 카메라 |
| AI-Hub 71803(독거노인), 170(음향), 648/61/71827/71467 | 영상 없음, 낙상 없음, 또는 도메인 불일치 |
| ElderSim / KIST SynADL | 사용 기간이 정해진 EULA 가능성 (장기 포트폴리오에 부적합) |
| PKU-MMD, Kinetics, HMDB51, ActivityNet 등 | 낙상 클래스 불명확하거나 웹/드론 영상 |
| E-FPDS, Kaggle CCTV 이미지 | 정지 이미지라 시계열 학습 불가 |

## 공개 데이터 받기

가입·신청 없이 받을 수 있는 데이터는 스크립트로 받습니다 (저장소 루트에서 실행, `dataset/`는 git 제외).

```bash
python -m training.datasets.fetch_public list                  # 대상과 라이선스
python -m training.datasets.fetch_public download --all        # 이어받기 지원, 공개 체크섬 검증
python -m training.datasets.fetch_public extract gmdcsa24      # 경로 조작을 막는 안전한 압축 해제
```

| 이름 | 내용 | 원본 크기 | 확인된 구성 (2026-10-15) |
| --- | --- | --- | --- |
| `omnifall_labels` | OmniFall/WanFall 통합 구간 라벨과 분할 | 10MB | 라벨 CSV, cs/cv 분할 |
| `gmdcsa24` | GMDCSA24 | 1.1GB | mp4 160개 (피험자 4명), md5 일치 |
| `caucafall` | CAUCAFall | 8.3GB | avi 100개 (피험자 10명), 프레임 png + 라벨 txt |
| `le2i` | Le2i | 9.6GB | avi 190개, 주석 txt 131개 (일부 장면은 원래 주석 없음) |
| `mcfd` | MCFD | 3.8GB | avi 192개 (시나리오 24 x 카메라 8) |
| `figshare_ursul29` | Fall Dataset 29 Subjects | 2.5GB | mp4 2,018개, md5 일치 |
| `charades` | Charades 480p + 주석 (하드 네거티브 후보) | 16.3GB | mp4 9,848개 |
| `wanfall_synthetic` | 합성 낙상 (사전학습 전용) | 9.7GB | mp4 12,000개, HF sha256 일치 |

원본과 압축 해제본을 합치면 약 104GB입니다. 압축 해제 후 `dataset/*/raw/`를 지우면 약 51GB를 줄일 수 있습니다.

받은 파일의 URL, 크기, sha256은 `dataset/_downloads.json`에 남습니다.
UP-Fall(약 850GB), OOPS(45GB 스트리밍), EDF/OCCU(깊이 전용)는 공개지만 크기나 용도 때문에 넣지 않았습니다.

## 확보 순서

1. **오늘 신청** (승인에 시간이 걸림): AI-Hub 71641, 71550, 167 / CMDFall 이메일 / ETRI 사용 협약서
2. **바로 다운로드**: CAUCAFall → GMDCSA24 → OmniFall 라벨 → MCFD → Le2i (모두 `dataset/<name>/`, git 제외)
3. **어댑터 작성** (Day 4~5): 데이터셋별 원본을 통합 manifest로 변환, 라벨은 OmniFall 구간을 우선 사용,
   키포인트는 운영과 같은 `yolov8s-pose`로 재추출, 15Hz 리샘플링
4. **평가 프로토콜 고정**: 피험자/시나리오/시점 그룹 단위 분할, leave-one-dataset-out
   (예: URFD + Le2i 학습 → CAUCAFall/GMDCSA24 테스트)
5. **하드 네거티브 보강**: SAFER-Activities, Charades 눕기/앉기 구간, UP-Fall 카메라 일부
6. **최종 OOD 리포트**: SFU 실제 노인 낙상, OOPS-Fall은 테스트에만 사용

통합 manifest 필드 (안):

```json
{"dataset": "le2i", "video": "Home_01/video (3).avi", "fps": 25, "subject": "s3", "cam": "c1",
 "fall_start_f": 112, "fall_end_f": 138, "label_source": "omnifall_v3", "license": "CC BY-NC-SA 3.0",
 "sha256": "...", "split_group": "le2i/Home_01"}
```

## 자체 촬영 (공개 데이터로는 시점 차이를 메울 수 없음)

- **카메라를 실제 설치와 똑같이**: 같은 카메라·해상도·fps·높이(약 2.0~2.4m 벽 모서리)로 찍고, 높이/각도/방 크기를 manifest에 기록
- **낙상**: 앞/뒤/옆, 의자에서 미끄러짐(느린 붕괴), 침대에서 떨어짐, 일어나다 균형 상실, 가구 뒤로 가려지는 낙상, 화면 가장자리 낙상
- **하드 네거티브**: 실제 오탐(`runs/clips/not_fall`)을 우선 재현 → 침대/소파에 빠르게 눕기, 털썩 앉기, 바닥에 앉기, 줍기, 스트레칭, 두 사람 장면
- **긴 연속 ADL 세션** 30~60분: 시간당 오탐(FA/h)은 짧은 클립으로는 측정할 수 없음
- **조건 다양화**: 낮/밤(IR), 조명, 옷/담요, 가구 배치, 가능하면 방 2곳 이상
- **안전**: 매트는 러그로 가리고 네거티브 촬영에도 같은 매트를 깔기 ("매트 = 낙상" 학습 방지). 고령자에게는 낙상 연기를 시키지 않음
- **라벨**: `eval/labels.example.jsonl` 형식, 시작 = 처음 균형을 잃는 순간, 끝 = 바닥에서 정지한 순간. 일부는 2명이 독립 라벨링해 일치도 확인
- **동의/개인정보**: 서면 동의(목적, 보관 기간·장소, 파기, 학습 사용, 공개 여부 별도), 공개 자료는 얼굴 블러 또는 스켈레톤만

## 주의사항

- **도메인 차이**: 공개 데이터 대부분은 젊은 배우의 연출 낙상(매트 위, 실험실, 낮은 카메라)입니다. 공개 데이터 점수가 실제 설치 성능을 보장하지 않으므로 최종 판단은 자체 촬영 holdout과 실제 낙상 데이터로 합니다.
- **시점 차이**: 카메라 높이가 64cm부터 천장까지 제각각이라 이미지 좌표 기반 특징이 흔들립니다. features v2의 스케일/위치 정규화와 증강을 적용하고, 데이터셋별 성능을 따로 보고합니다.
- **키포인트 형식**: Kinect 20/25관절, HRNet/ViTPose 키포인트는 YOLOv8 COCO-17과 정의와 노이즈가 다릅니다. RGB가 있으면 반드시 운영과 같은 모델로 재추출합니다.
- **프레임레이트**: 12~60fps로 제각각이고 메타데이터가 틀린 경우도 있습니다. 실측 후 15Hz로 리샘플링합니다.
- **낙상 라벨 정의**: "바닥에 닿은 이후"(CAUCAFall), fall/fallen 구분(OmniFall), 시작/끝 프레임(AI-Hub), 클립 단위(NTU/ETRI)로 다릅니다. manifest에 `label_source`를 남깁니다.
- **다중 시점 누수**: 같은 사건을 여러 카메라로 찍은 데이터(MCFD, AI-Hub 71641, CMDFall)는 모든 시점을 같은 분할에 넣습니다.
- **누운 상태 네거티브**: 바닥에 누운 구간을 네거티브로 넣으면 실제 낙상 후 누운 상태까지 억제하도록 학습될 수 있습니다. 하강 과정이 없는 구간만 네거티브로 씁니다.
- **라이선스**: 대부분 비상업/연구 전용입니다. 상업 이용 가능: CAUCAFall, GMDCSA24, Figshare 29 Subjects, EVIORA-HAR(CC BY 4.0). 어떤 데이터로 학습한 가중치인지 메타데이터에 기록합니다.
- **재배포 금지**: AI-Hub, NTU, Charades, ETRI, Toyota 등은 원본/파생 데이터 공개를 금지합니다. 원본 영상, 추출 키포인트, npy는 git 제외 폴더(`dataset/`, `data/`, `app/ai_classifier/data/`)에만 두고, git에는 manifest(상대 경로, sha256, 라벨 출처, 라이선스, URL)와 어댑터 코드만 커밋합니다. AI-Hub는 NIA 출처, ETRI는 사사 문구를 표기합니다.
