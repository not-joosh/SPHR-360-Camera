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
        Detect faces in the frame and display the frame with dynamic, smooth zoom.
        Adjust zoom to fit all detected faces and draw a red box around the zoomed-in region.
        '''
        # Desired face size as a proportion of the frame height
        target_face_size = 0.3  
        zoom_factor = 1.0
        smoothing_factor = 0.1  # Adjust for smoother zooming

        while self.is_running:
            if self.frame is not None:
                with self.lock:
                    frame_copy = self.frame.copy()

                # Perform bounding box detection
                results = self.model(frame_copy, stream=True)

                if not results:
                    continue

                all_boxes = []
                for r in results:
                    boxes = r.boxes
                    if not boxes:
                        continue

                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        all_boxes.append((x1, y1, x2, y2))

                        # Draw original bounding box
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 255), 2)

                        # Draw class label
                        cls = int(box.cls[0])
                        confidence = round(float(box.conf[0].cpu()), 2)
                        label = f"{self.class_list[cls]}: {confidence}"
                        cv2.putText(frame_copy, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                if all_boxes:
                    # Calculate the bounding box that fits all faces
                    min_x = min(x1 for x1, _, _, _ in all_boxes)
                    min_y = min(y1 for _, y1, _, _ in all_boxes)
                    max_x = max(x2 for _, _, x2, _ in all_boxes)
                    max_y = max(y2 for _, _, _, y2 in all_boxes)

                    face_center_x = (min_x + max_x) // 2
                    face_center_y = (min_y + max_y) // 2

                    # Calculate the bounding box width and height
                    box_width = max_x - min_x
                    box_height = max_y - min_y

                    # Calculate the actual size of the bounding box as a proportion of the frame height
                    actual_box_size = box_height / frame_copy.shape[0]

                    # Calculate the desired zoom factor to fit all faces
                    desired_zoom_factor = 1.0
                    if actual_box_size < target_face_size:
                        desired_zoom_factor = min(1.5, zoom_factor * (target_face_size / actual_box_size))
                    elif actual_box_size > target_face_size:
                        desired_zoom_factor = max(0.5, zoom_factor * (target_face_size / actual_box_size))

                    # Smooth transition towards the desired zoom factor
                    zoom_factor += (desired_zoom_factor - zoom_factor) * smoothing_factor

                    # Apply smooth zoom to focus on the bounding box of all faces
                    frame_copy = self.smooth_zoom(frame_copy, face_center_x, face_center_y, zoom_factor)

                    # Draw a new red bounding box around the zoomed-in region
                    red_x1, red_y1, red_x2, red_y2 = self.calculate_zoomed_in_box(frame_copy, face_center_x, face_center_y, zoom_factor)
                    cv2.rectangle(frame_copy, (red_x1, red_y1), (red_x2, red_y2), (0, 0, 255), 2)

                # Display the frame with a reduced refresh rate
                cv2.imshow("Video Capture", frame_copy)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

    def smooth_zoom(self, frame, center_x, center_y, zoom_factor):
        '''
        Smoothly zoom into the region around the face. Temporary feature for 
        focusing on the detected face.
        '''
        h, w, _ = frame.shape
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)

        # Ensure the new dimensions don't exceed the frame boundaries
        x1 = max(0, min(center_x - new_w // 2, w - new_w))
        y1 = max(0, min(center_y - new_h // 2, h - new_h))
        x2 = x1 + new_w
        y2 = y1 + new_h

        # Crop to the zoomed region
        cropped_frame = frame[y1:y2, x1:x2]
        resized_frame = cv2.resize(cropped_frame, (w, h))

        return resized_frame

    def calculate_zoomed_in_box(self, frame, center_x, center_y, zoom_factor):
        '''
        Calculate the coordinates of the zoomed-in region.
        '''
        h, w, _ = frame.shape
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)

        # Ensure the new dimensions don't exceed the frame boundaries
        x1 = max(0, min(center_x - new_w // 2, w - new_w))
        y1 = max(0, min(center_y - new_h // 2, h - new_h))
        x2 = x1 + new_w
        y2 = y1 + new_h

        return x1, y1, x2, y2
    def run(self):
        frame_thread = threading.Thread(target=self.capture_frames)
        frame_thread.start()

        self.detect_faces()

        frame_thread.join()

# Main program
if __name__ == "__main__":
    # Setting up the model and necessary paths
    # model_path = "faces_v7.pt"
    model_path = "faces_v7_ncnn_model"
    label_path = "coco1.txt"
    face_detection = FaceDetection(model_path, label_path, resolution=(320, 320))
    face_detection.run()
