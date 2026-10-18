"""
포즈 모델(YOLOv8-pose) 개선: 누운 사람 검출을 위한 의사 라벨과 파인튜닝 (로드맵 Day 7)

rotation.py   회전 TTA: 영상을 0/90/180/270도로 돌려 검출하고 결과를 원래 좌표로 되돌림
pseudo.py     교사 모델(yolov8x-pose) + 회전 TTA로 프레임별 의사 라벨 생성, 품질 필터
dataset.py    평가에 쓰지 않는 영상에서 프레임 추출, YOLO pose 학습 형식(images/labels, data.yaml) 작성
"""
