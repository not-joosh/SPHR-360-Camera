import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import torch

class FaceDetection:
    def __init__(self, model_path, label_path, resolution=0.5):
        self.model = YOLO(model_path)
        self.class_list = self.load_labels(label_path)
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.is_running = True
        self.frame = None
        self.lock = threading.Lock()
        self.confidence = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def load_labels(self, label_path):
        with open(label_path, "r") as file:
            data = file.read().strip()
        return data.split("\n")


    def capture_frames(self):
        while self.is_running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.frame = frame
            else:
                print("Error: Unable to capture frame")
                break

    def detect_gestures(self):
        '''
        Detect faces in the frame and display the frame.
        '''
        TIMER = int(3)
        while self.is_running:
            if self.frame is not None:
                with self.lock:
                    frame_copy = self.frame.copy()

                # Perform bounding box detection
                results = self.model(frame_copy, stream=True)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Draw bounding box
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 255), 2)

                        # Convert tensor to float and then round
                        self.confidence = round(float(box.conf[0].cpu()), 2)
                        print("Confidence --->", self.confidence)

                        cls = int(box.cls[0])
                        label = f"{self.class_list[cls]}: {self.confidence}"


                        # Draw class label
                        cv2.putText(
                            frame_copy, label, (x1, y1 - 10),
                            self.font, 0.5,
                            (255, 0, 0), 1, cv2.LINE_AA
                        )

                # Display the frame with a reduced refresh rate
                cv2.imshow("Video Capture", frame_copy)
                if self.confidence >= 0.7:
                    prev = time.time()

                    while TIMER >= 0:
                        ret, img = self.cap.read()

                        # Display countdown on each frame
                        # specify the font and draw the
                        # countdown using puttext
                        cv2.putText(img, str(TIMER),
                                    (200, 250), self.font,
                                    4, (255, 0, 0),
                                    2, cv2.LINE_AA)
                        cv2.imshow('a', img)
                        cv2.waitKey(125)

                        # current time
                        cur = time.time()

                        # Update and keep track of Countdown
                        # if time elapsed is one second
                        # then decrease the counter
                        if cur - prev >= 1:
                            prev = cur
                            TIMER = TIMER - 1
                    else:
                        ret, img = self.cap.read()

                        # Display the clicked frame for 2
                        # sec.You can increase time in
                        # waitKey also
                        cv2.imshow('a', img)

                        # display the captured image for 2 secs
                        # save the frame
                        cv2.waitKey(2000)
                        filename = str(len(next(os.walk('images'))[1])) + ".jpg"
                        cv2.imwrite(filename, img)
                        # timer countdown can be reset to take more pictures
                # Release resources
                elif cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False



    def run(self):
        frame_thread = threading.Thread(target=self.capture_frames)
        frame_thread.start()

        self.detect_gestures()

        frame_thread.join()

if __name__ == "__main__":
    model_path = "best.pt"
    label_path = "coco2.txt"

    face_detection = FaceDetection(model_path, label_path, resolution=(416, 416))
    face_detection.run()