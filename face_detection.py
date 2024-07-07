import cv2
import numpy as np
from ultralytics import YOLO
import threading
import torch

class FaceDetection:
    def __init__(self, model_path, label_path, resolution):
        self.model = YOLO(model_path)
        self.class_list = self.load_labels(label_path)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.is_running = True
        self.frame = None
        self.lock = threading.Lock()  

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

    def detect_faces(self):
        '''
        Detect faces in the frame and display the frame.
        '''
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
                        confidence = round(float(box.conf[0].cpu()), 2)
                        print("Confidence --->", confidence)

                        cls = int(box.cls[0])
                        label = f"{self.class_list[cls]}: {confidence}"

                        # center point of box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        cv2.circle(frame_copy, (center_x, center_y), 1, (255,255, 0), -1)

                        #center point of frame
                        frame_center_x = frame_copy.shape[1] // 2
                        frame_center_y = frame_copy.shape[0] // 2

                        cv2.circle(frame_copy, (frame_center_x, frame_center_y), 1, (0,0,255), 0)

                        # Draw class label
                        cv2.putText(
                            frame_copy, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 1, cv2.LINE_AA
                        )

                # Display the frame with a reduced refresh rate
                cv2.imshow("Video Capture", frame_copy)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        frame_thread = threading.Thread(target=self.capture_frames)
        frame_thread.start()

        self.detect_faces()

        frame_thread.join()


