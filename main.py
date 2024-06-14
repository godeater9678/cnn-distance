import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드 (사전 학습된 모델 사용)
model = YOLO("yolov8n.pt")  # "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt" 중 선택

# 비디오 파일 로드
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# 카메라 초점 거리와 객체 실제 너비 (예시)
focal_length = 700  # 카메라 초점 거리 (예시)
real_object_width = 1000  # 객체의 실제 너비 (예시, cm 단위)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지
    results = model(frame)

    # 탐지 결과에서 클래스와 박스 정보 추출
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    names = results[0].names

    # 결과 시각화 및 거리 계산
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        w = x2 - x1
        h = y2 - y1
        label = names[int(class_ids[i])]
        confidence = confidences[i]

        # 거리 계산 meter
        distance = (real_object_width * focal_length) / w / 100

        # 결과 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Distance: {distance:.2f} m", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 프레임 저장
    out.write(frame)

    #결과 프레임 출력 (원하는 경우 활성화)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
