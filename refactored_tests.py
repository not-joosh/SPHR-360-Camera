import cv2
import numpy as np
from ultralytics import YOLO
import threading

class FaceDetection:
    def __init__(self, model_path, label_path, resolution, os_type="windows"):
        self.model = YOLO(model_path)
        self.class_list = self.load_labels(label_path)
        # Capture for with respect to the OS
        if os_type == "windows":
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        # Video capture for linux
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.is_running = True
        self.frame = None
        self.lock = threading.Lock()
        self.batch_size = 5
        self.frames = []
        self.current_zoom_factor = 1.0

    def load_labels(self, label_path):
        '''
        Load class labels from a file.
        '''
        with open(label_path, "r") as file:
            data = file.read().strip()
        return data.split("\n")

    def capture_frame(self):
        '''
        Capture a single frame from the video feed.
        '''
        success, frame = self.cap.read()
        if success:
            return frame
        else:
            print("Error: Unable to capture frame")
            return None

    def process_frames(self, frames):
        '''
        Process a batch of frames using the YOLO model.
        '''
        batch_results = []
        for frame in frames:
            results = self.model(frame, stream=True)
            batch_results.append(results)
        return batch_results

    def detect_faces(self, frame, results):
        '''
        Detect faces in the frame and display the frame with bounding boxes and labels.
        '''
        all_boxes = []
        for r in results:
            boxes = r.boxes
            if not boxes:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                all_boxes.append((x1, y1, x2, y2))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                cls = int(box.cls[0])
                confidence = round(float(box.conf[0].cpu()), 2)
                label = f"{self.class_list[cls]}: {confidence}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        return frame, all_boxes

    def get_face_centers(self, all_boxes):
        '''
        Get the center coordinates of all detected faces.
        '''
        centers = []
        for (x1, y1, x2, y2) in all_boxes:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            centers.append((center_x, center_y))
        return centers

    def calculate_zoom(self, all_boxes, frame_shape):
        '''
        Calculate the zoom factor based on the size of the detected faces.
        '''
        target_face_size = 0.4
        zoom_threshold = 0.01
        smoothing_factor = 0.05

        min_x = min(x1 for x1, _, _, _ in all_boxes)
        min_y = min(y1 for _, y1, _, _ in all_boxes)
        max_x = max(x2 for _, _, x2, _ in all_boxes)
        max_y = max(y2 for _, _, _, y2 in all_boxes)

        face_center_x = (min_x + max_x) // 2
        face_center_y = (min_y + max_y) // 2

        box_height = max_y - min_y
        actual_box_size = box_height / frame_shape[0]

        desired_zoom_factor = self.current_zoom_factor

        if actual_box_size < target_face_size:
            desired_zoom_factor = min(1.5, self.current_zoom_factor * (target_face_size / actual_box_size))
        elif actual_box_size > target_face_size:
            desired_zoom_factor = max(0.5, self.current_zoom_factor * (target_face_size / actual_box_size))

        if abs(desired_zoom_factor - self.current_zoom_factor) > zoom_threshold:
            self.current_zoom_factor += smoothing_factor * (desired_zoom_factor - self.current_zoom_factor)

        # Ensure zoom is not too close or too far
        self.current_zoom_factor = min(max(self.current_zoom_factor, 0.75), 1.5)

        return face_center_x, face_center_y, self.current_zoom_factor

    def smooth_zoom(self, frame, center_x, center_y, zoom_factor):
        '''
        Smoothly zoom into the center of the frame.
        '''
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
        '''
        Calculate the bounding box for the zoomed-in region.
        '''
        h, w, _ = frame.shape
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)

        x1 = max(0, min(center_x - new_w // 2, w - new_w))
        y1 = max(0, min(center_y - new_h // 2, h - new_h))
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)

        return x1, y1, x2, y2

    def calculate_offsets(self, all_boxes, frame_shape):
        '''
        Calculate the offset of the combined face center from the frame center.
        '''
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2

        if not all_boxes:
            return (0, 0)

        min_x = min(x1 for x1, _, _, _ in all_boxes)
        min_y = min(y1 for _, y1, _, _ in all_boxes)
        max_x = max(x2 for _, _, x2, _ in all_boxes)
        max_y = max(y2 for _, _, _, y2 in all_boxes)

        combined_center_x = (min_x + max_x) // 2
        combined_center_y = (min_y + max_y) // 2

        offset_x = combined_center_x - frame_center_x
        offset_y = combined_center_y - frame_center_y

        return offset_x, offset_y

    def release_resources(self):
        '''
        Release the video capture and destroy all OpenCV windows.
        '''
        self.cap.release()
        cv2.destroyAllWindows()
