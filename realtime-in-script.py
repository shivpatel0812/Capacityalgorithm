from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np
import boto3
from datetime import datetime, timedelta
import json
import time

CLEMONS_BUCKET_NAME = "onecameratest"
AWS_ACCESS_KEY = "key"
AWS_SECRET_KEY = "key"
camera_id = "onecameratest"

s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
model = YOLO('yolov8n.pt')
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()

frame_width, frame_height = 640, 480
door_line_x = int(frame_width * 0.5)
box_width = int(frame_width * 0.2)
box_height = int(frame_height * 0.8)
box_x = door_line_x - box_width // 2
box_y = (frame_height - box_height) // 2

enter_count = 0
previous_positions = {}
previous_detections = None
skip_interval = 3
frame_count = 0

now = datetime.now()
minute = now.minute
next_minute = ((minute // 5) + 1) * 5
if next_minute == 60:
    next_upload_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
else:
    next_upload_time = now.replace(minute=next_minute, second=0, microsecond=0)

def generate_camera_data(camera_id, enter_count):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"camera_id": camera_id, "time": current_time, "in": enter_count}

def upload_camera_data_to_s3(camera_id, data):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    object_key = f"{camera_id}/{timestamp}.json"
    json_data = json.dumps(data)
    s3_client.put_object(Bucket=CLEMONS_BUCKET_NAME, Key=object_key, Body=json_data, ContentType="application/json")
    print(f"Uploaded data for {camera_id} to s3://{CLEMONS_BUCKET_NAME}/{object_key}")

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_count += 1
        if frame_count % skip_interval == 0:
            results = model(frame)
            persons = [det for det in results[0].boxes if det.cls == 0]
            previous_detections = persons
            current_positions = {}
            for person_id, person in enumerate(persons):
                x1, y1, x2, y2 = map(int, person.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {person_id} ({person.conf.item():.2f})',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                current_positions[person_id] = (center_x, center_y)
                if person_id in previous_positions:
                    prev_center_x, _ = previous_positions[person_id]
                    if box_y <= center_y <= box_y + box_height:
                        if prev_center_x > door_line_x and center_x < door_line_x:
                            enter_count += 1
                            print(f"Frame {frame_count}: Person {person_id} ENTERED. Total Enters: {enter_count}")
            previous_positions = current_positions
        else:
            if previous_detections is not None:
                for person_id, person in enumerate(previous_detections):
                    x1, y1, x2, y2 = map(int, person.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {person_id} ({person.conf.item():.2f})',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.line(frame, (door_line_x, box_y), (door_line_x, box_y + box_height), (255, 0, 0), 2)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), 2)
        cv2.putText(frame, f'Enter: {enter_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 Real-Time Detection", frame)
        if datetime.now() >= next_upload_time:
            data = generate_camera_data(camera_id, enter_count)
            upload_camera_data_to_s3(camera_id, data)
            enter_count = 0
            next_upload_time += timedelta(minutes=5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print("An error occurred:", e)
finally:
    print("Final enter count:", enter_count)
    cv2.destroyAllWindows()
    picam2.stop()
