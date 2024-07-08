# Main program
import cv2
import time
from refactored_tests import FaceDetection
import serial

# Create a serial object
ser = serial.Serial('COM3', 9600)  # Replace 'COM1' with the appropriate port and '9600' with the desired baud rate

# Use the serial object for communication
# Example: ser.write(b'Hello')  # Write data to the serial port
# Example: data = ser.read()  # Read data from the serial port

# Close the serial connection when done


if __name__ == "__main__":

    # Setting up the model and necessary paths
    model_path = "faces_v7_ncnn_model"
    label_path = "coco1.txt"
    resolution = (1280, 720)
    enable_zoom = True  # Disable or Enable zooming

    # Initialize the FaceDetection class
    face_detection = FaceDetection(model_path, label_path, resolution)

    # Variables
    # None so far cause not needed, but probably for the servo integration

    while face_detection.is_running:
        # TODO: This loop is needed, but we can possibly turn this entire thing to a function 
        # and just return the offset calculations that way we have access to them in the main function
        frame = face_detection.capture_frame()
        if frame is not None:
            face_detection.frames.append(frame)

            # Keep the batch size fixed
            # Remove the oldest frame if the batch size is exceeded to maintain the fixed batch size
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

                    ser.write(str(offset_x).encode() + ",".encode() + str(offset_y).encode() + "\n".encode())
                    
                    '''
                    
                    Drawing the center lines for debugging purposes

                    '''
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

                    # Debugging output
                    # TODO: Remove this section in the final version
                    print(f"Frame center: ({frame_center_x}, {frame_center_y})")
                    print(f"Offset position: ({offset_pos_x}, {offset_pos_y})")

                    '''
                    
                    Debug ends here

                    '''

                    # TODO: Turn zooming into a separate function
                    # Enable zoom (if required)
                    if enable_zoom:
                        face_center_x, face_center_y, current_zoom_factor = face_detection.calculate_zoom(all_boxes, latest_frame.shape)
                        latest_frame = face_detection.smooth_zoom(latest_frame, face_center_x, face_center_y, current_zoom_factor)

                        red_x1, red_y1, red_x2, red_y2 = face_detection.calculate_zoomed_in_box(
                            latest_frame, face_center_x, face_center_y, current_zoom_factor)
                        cv2.rectangle(latest_frame, (red_x1, red_y1), (red_x2, red_y2), (0, 0, 255), 2)

                cv2.imshow("Video Capture", latest_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    face_detection.is_running = False

        time.sleep(0.03) 

    ser.close()
    face_detection.release_resources()
