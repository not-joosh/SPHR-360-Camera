import cv2
from ultralytics import YOLO
import math

# Load model
model = YOLO("faces.pt")  

# start webcam
cap = cv2.VideoCapture(0)

#setting the capture resolution to 640x480
cap.set(3, 640) # set width
cap.set(4, 480) # set height

#open text file containing labels
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

while True:
    success, frame = cap.read()

    # Perform bounding box
    results = model(frame, stream=True)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0] #returns coordinates with the format [x1, y1, x2, y2]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", class_list[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, class_list[cls], org, font, fontScale, color, thickness)
  
    # Display frame
    cv2.imshow("Video Capture", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()