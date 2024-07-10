import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import torch

class GestureModel:
    def __init__(self, model_path, label_path, resolution=(416, 416), camera_index = 0):
        self.model = YOLO(model_path)
        self.class_list = self.load_labels(label_path)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.frame = None
        self.lock = threading.Lock()
        self.confidence = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.picture_taken = False 

    def load_labels(self, label_path):
        with open(label_path, "r") as file:
            data = file.read().strip()
        return data.split("\n")

    def capture_frame(self):
        success, frame = self.cap.read()
        if success:
            return frame
        else:
            print("Error: Unable to capture frame")
            return None

    def detect_gestures(self, frame):
        frame_copy = frame.copy()
        results = self.model(frame_copy, stream=True)

        for r in results:
            boxes = r.boxes
            if not boxes:
                print("\n*******************No gestures detected.*********************\n")
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 255), 2)
                self.confidence = round(float(box.conf[0].cpu()), 2)
                cls = int(box.cls[0])
                label = f"{self.class_list[cls]}: {self.confidence}"
                cv2.putText(
                    frame_copy, label, (x1, y1 - 10),
                    self.font, 0.5,
                    (255, 0, 0), 1, cv2.LINE_AA
                )
                print(f"Detected gesture: {label} at ({x1}, {y1}, {x2}, {y2})")

        return frame_copy, self.confidence

    def countdown_and_save(self, frame):
        TIMER = 3
        prev = time.time()

        while TIMER >= 0:
            ret, img = self.cap.read()
            if not ret:
                break
            cv2.putText(img, str(TIMER),
                        (200, 250), self.font,
                        4, (255, 0, 0),
                        2, cv2.LINE_AA)
            cv2.imshow('Video Capture', img)
            cv2.waitKey(125)

            cur = time.time()
            if cur - prev >= 1:
                prev = cur
                TIMER -= 1

        ret, img = self.cap.read()
        if ret:
            cv2.imshow('Video Capture', img)
            cv2.waitKey(2000)
            if not os.path.exists('./images'):
                os.makedirs('./images')
            filename = os.path.join('./images', f"{len(os.listdir('./images'))}.jpg")
            cv2.imwrite(filename, img)
            self.picture_taken = True  # Set the flag to indicate that a picture has been taken
        return img


if __name__ == "__main__":
    model_path = "best_ncnn_model"
    label_path = "coco2.txt"
    gesture_model = GestureModel(model_path, label_path)

    while True:
        frame = gesture_model.capture_frame()
        if frame is None:
            break

        processed_frame, confidence = gesture_model.detect_gestures(frame)
        cv2.imshow("Video Capture", processed_frame)

        if confidence >= 0.7 and not gesture_model.picture_taken:
            gesture_model.countdown_and_save(frame)
            gesture_model.picture_taken = True  # Ensure the flag is set

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if confidence < 0.7:
            gesture_model.picture_taken = False  # Reset the flag when the gesture is no longer detected

    gesture_model.cap.release()
    cv2.destroyAllWindows()
