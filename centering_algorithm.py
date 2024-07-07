import cv2
import RPi.GPIO as GPIO
import time

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 640
FOV_X = 60  # degrees
FOV_Y = 45  # degrees
STEPS_PER_REV = 200

# Setup GPIO pins for stepper motors
X_DIR_PIN = 20
X_STEP_PIN = 21
Y_DIR_PIN = 19
Y_STEP_PIN = 26

GPIO.setmode(GPIO.BCM)
GPIO.setup(X_DIR_PIN, GPIO.OUT)
GPIO.setup(X_STEP_PIN, GPIO.OUT)
GPIO.setup(Y_DIR_PIN, GPIO.OUT)
GPIO.setup(Y_STEP_PIN, GPIO.OUT)

# Helper function to move stepper motor
def move_stepper(dir_pin, step_pin, steps, direction):
    GPIO.output(dir_pin, direction)
    for _ in range(steps):
        GPIO.output(step_pin, GPIO.HIGH)
        time.sleep(0.001)  # Adjust the delay as needed for your stepper motor
        GPIO.output(step_pin, GPIO.LOW)
        time.sleep(0.001)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            offset_x = face_center_x - (FRAME_WIDTH // 2)
            offset_y = face_center_y - (FRAME_HEIGHT // 2)

            angle_x = (offset_x / FRAME_WIDTH) * FOV_X
            angle_y = (offset_y / FRAME_HEIGHT) * FOV_Y

            steps_x = int(angle_x * (STEPS_PER_REV / 360))
            steps_y = int(angle_y * (STEPS_PER_REV / 360))

            # Move stepper motors
            if steps_x != 0:
                move_stepper(X_DIR_PIN, X_STEP_PIN, abs(steps_x), GPIO.HIGH if steps_x > 0 else GPIO.LOW)
            if steps_y != 0:
                move_stepper(Y_DIR_PIN, Y_STEP_PIN, abs(steps_y), GPIO.HIGH if steps_y > 0 else GPIO.LOW)

            # Draw rectangle around the face and show the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Face Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()