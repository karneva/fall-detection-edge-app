# scripts

장치에서 수동으로 실행하는 보조 도구입니다. 앱(`app/`)은 이 폴더를 import하지 않습니다.
모두 저장소 루트에서 실행합니다.

| 스크립트 | 용도 |
| --- | --- |
| `jetson/export_tensorrt.py` | YOLO Pose `.pt`를 TensorRT `.engine`으로 변환 |
| `jetson/benchmark_models.py` | `.pt`와 `.engine`의 추론 속도/FPS 비교 |
| `jetson/ip_reporter.py` | 장치의 로컬 IP를 주기적으로 서버에 보고 (`runs/device_state.json`의 토큰 사용) |

```bash
python scripts/jetson/export_tensorrt.py
python scripts/jetson/benchmark_models.py
python scripts/jetson/ip_reporter.py
```

`ip_reporter.py`를 systemd나 cron으로 실행하고 있었다면 실행 경로를
`scripts/jetson/ip_reporter.py`로 바꿔야 합니다 (이전 위치: 저장소 루트).
