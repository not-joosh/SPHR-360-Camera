import cv2
import time
import threading
import serial
from gestures_model import GestureModel
from face_model import FaceDetection

hardwareFlag = True
# Create a serial object
try:
    ser = serial.Serial('COM3', 9600)  # Replace 'COM1' with the appropriate port and '9600' with the desired baud rate
except:
    hardwareFlag = False
    print("Hardware not connected")

if __name__ == "__main__":
    # Paths and settings
    face_model_path = "faces_ncnn_model"
    face_label_path = "faces_labels.txt"
    gesture_model_path = "gestures_ncnn_model"
    gesture_label_path = "gestures_labels.txt"
    resolution = (1280, 720)
    enable_zoom = False  # Disable or Enable zooming

    # Initialize the models
    face_detection = FaceDetection(face_model_path, face_label_path, resolution)
    gesture_model = GestureModel(gesture_model_path, gesture_label_path)


    while face_detection.is_running:
        # Capture frame
        frame = face_detection.capture_frame()
        if frame is None:
            break

        # Process frame for gesture detection
        processed_frame, gesture_confidence = gesture_model.detect_gestures(frame)

        # If a gesture is detected with high confidence, take a picture
        if gesture_confidence >= 0.7 and not gesture_model.picture_taken:
            gesture_model.countdown_and_save(frame)
            gesture_model.picture_taken = True  # Ensure the flag is set
        if gesture_confidence < 0.7:
            gesture_model.picture_taken = False  # Reset the flag when the gesture is no longer detected

        # Process frame for face detection
        face_detection.frames.append(frame)

        # Keep the batch size fixed
        if len(face_detection.frames) > face_detection.batch_size:
            face_detection.frames.pop(0)

        # Process the batch of frames
        results = face_detection.process_frames(face_detection.frames)

        # Process the latest frame
        if results:
            latest_frame = face_detection.frames[-1].copy()
            results = results[-1]

            latest_frame, all_boxes = face_detection.detect_faces(latest_frame, results)

            # Calculate face centers and offsets
            if all_boxes:
                face_centers = face_detection.get_face_centers(all_boxes)
                offset_x, offset_y = face_detection.calculate_offsets(all_boxes, latest_frame.shape)

                print("Combined Face Center Offset (x, y):", offset_x, offset_y)

                # Drawing a dot at the offset position making it green
                offset_pos_x = latest_frame.shape[1] // 2 + offset_x
                offset_pos_y = latest_frame.shape[0] // 2 + offset_y
                cv2.circle(latest_frame, (offset_pos_x, offset_pos_y), 5, (0, 255, 0), -1)

                # Drawing a red dot at the center of each face
                for (center_x, center_y) in face_centers:
                    cv2.circle(latest_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Drawing a blue dot at the center of the frame
                frame_center_x = latest_frame.shape[1] // 2
                frame_center_y = latest_frame.shape[0] // 2
                cv2.circle(latest_frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)

                # Enable zoom (if required)
                if enable_zoom:
                    face_center_x, face_center_y, current_zoom_factor = face_detection.calculate_zoom(all_boxes, latest_frame.shape)
                    latest_frame = face_detection.smooth_zoom(latest_frame, face_center_x, face_center_y, current_zoom_factor)

                    red_x1, red_y1, red_x2, red_y2 = face_detection.calculate_zoomed_in_box(
                        latest_frame, face_center_x, face_center_y, current_zoom_factor)
                    cv2.rectangle(latest_frame, (red_x1, red_y1), (red_x2, red_y2), (0, 0, 255), 2)

            # Display the combined frame
            cv2.imshow("Video Capture", latest_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                face_detection.is_running = False

        time.sleep(0.03)

    face_detection.cap.release()
    cv2.destroyAllWindows()