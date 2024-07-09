import cv2
import numpy as np
from ultralytics import YOLO
import threading

class FaceDetection:
    def __init__(self, model_path, label_path, resolution):
        self.model = YOLO(model_path)
        self.class_list = self.load_labels(label_path)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Ensure higher frame rate
        self.is_running = True
        self.frame = None
        self.lock = threading.Lock()
        self.batch_size = 5
        self.frames = []
        self.current_zoom_factor = 1.0  # Initialize the current zoom factor

    def load_labels(self, label_path):
        with open(label_path, "r") as file:
            data = file.read().strip()
        return data.split("\n")

    def capture_frames(self):
        while self.is_running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.frames.append(frame)
                    if len(self.frames) > self.batch_size:
                        self.frames.pop(0)  # Keep only the latest frames
            else:
                print("Error: Unable to capture frame")
                break

    def detect_faces(self):
        target_face_size = 0.4  # Increased target size for better zoom
        zoom_threshold = 0.01  # More sensitive threshold
        smoothing_factor = 0.05  # Smaller factor for smoother transitions

        while self.is_running:
            if self.frames:
                with self.lock:
                    frame_batch = self.frames[:]

                batch_results = []
                for frame in frame_batch:
                    results = self.model(frame, stream=True)
                    batch_results.append(results)

                if not batch_results:
                    continue

                # Use the last frame in the batch for detection
                latest_frame = frame_batch[-1].copy()
                results = batch_results[-1]

                all_boxes = []
                for r in results:
                    boxes = r.boxes
                    if not boxes:
                        continue

                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        all_boxes.append((x1, y1, x2, y2))

                        cv2.rectangle(latest_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                        cls = int(box.cls[0])
                        confidence = round(float(box.conf[0].cpu()), 2)
                        label = f"{self.class_list[cls]}: {confidence}"
                        cv2.putText(latest_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                if all_boxes:
                    min_x = min(x1 for x1, _, _, _ in all_boxes)
                    min_y = min(y1 for _, y1, _, _ in all_boxes)
                    max_x = max(x2 for _, _, x2, _ in all_boxes)
                    max_y = max(y2 for _, _, _, y2 in all_boxes)

                    face_center_x = (min_x + max_x) // 2
                    face_center_y = (min_y + max_y) // 2

                    box_width = max_x - min_x
                    box_height = max_y - min_y

                    actual_box_size = box_height / latest_frame.shape[0]
                    print(f"Actual Box Size: {actual_box_size}")

                    desired_zoom_factor = self.current_zoom_factor
                    if actual_box_size < target_face_size:
                        desired_zoom_factor = min(2.0, self.current_zoom_factor * (target_face_size / actual_box_size))
                    elif actual_box_size > target_face_size:
                        desired_zoom_factor = max(0.5, self.current_zoom_factor * (target_face_size / actual_box_size))

                    # Smooth transition by gradually adjusting the zoom factor
                    if abs(desired_zoom_factor - self.current_zoom_factor) > zoom_threshold:
                        self.current_zoom_factor += smoothing_factor * (desired_zoom_factor - self.current_zoom_factor)
                        print(f"Desired Zoom Factor: {desired_zoom_factor}, Current Zoom Factor: {self.current_zoom_factor}")

                        latest_frame = self.smooth_zoom(latest_frame, face_center_x, face_center_y, self.current_zoom_factor)

                        red_x1, red_y1, red_x2, red_y2 = self.calculate_zoomed_in_box(
                            latest_frame, face_center_x, face_center_y, self.current_zoom_factor)
                        cv2.rectangle(latest_frame, (red_x1, red_y1), (red_x2, red_y2), (0, 0, 255), 2)

                cv2.line(latest_frame, (face_center_x, 0), (face_center_x, latest_frame.shape[0]), (0, 255, 0), 1)
                cv2.line(latest_frame, (0, face_center_y), (latest_frame.shape[1], face_center_y), (0, 255, 0), 1)

                cv2.imshow("Video Capture", latest_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False

        self.cap.release()
        cv2.destroyAllWindows()

    def smooth_zoom(self, frame, center_x, center_y, zoom_factor):
        h, w, _ = frame.shape
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)

        x1 = max(0, min(center_x - new_w // 2, w - new_w))
        y1 = max(0, min(center_y - new_h // 2, h - new_h))
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)

        cropped_frame = frame[y1:y2, x1:x2]
        resized_frame = cv2.resize(cropped_frame, (w, h))

        return resized_frame

    def calculate_zoomed_in_box(self, frame, center_x, center_y, zoom_factor):
        h, w, _ = frame.shape
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)

        x1 = max(0, min(center_x - new_w // 2, w - new_w))
        y1 = max(0, min(center_y - new_h // 2, h - new_h))
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)

        return x1, y1, x2, y2

    def run(self):
        frame_thread = threading.Thread(target=self.capture_frames)
        frame_thread.start()

        self.detect_faces()

        frame_thread.join()

# Main program
if __name__ == "__main__":
    model_path = "faces_v7_ncnn_model"
    label_path = "coco1.txt"
    face_detection = FaceDetection(model_path, label_path, resolution=(320, 320))
    face_detection.run()
