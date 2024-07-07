import datetime
from threading import Thread
import cv2
from ultralytics import YOLO
from imutils.video import WebcamVideoStream, FPS
import math
from multiprocessing import Process, Queue
import time

class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._numFrames += 1

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._numFrames / self.elapsed()
    
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False


    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Function to perform YOLO inference
def yolo_inference(input_queue, output_queue, class_list, model_path):
    model = YOLO(model_path)
    while True:
        if not input_queue.empty():
            frame = input_queue.get()
            if frame is None:  # Signal to stop the process
                break

            results = model(frame, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame, f"{class_list[cls]} {confidence:.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            output_queue.put(frame)

# Function to display video frames
def display_video(output_queue):
    while True:
        if not output_queue.empty():
            frame = output_queue.get()
            if frame is None:
                break
            cv2.imshow("Video Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                output_queue.put(None)  # Signal to stop the display thread
                break
    cv2.destroyAllWindows()

# Load class names
with open("coco1.txt", "r") as f:
    class_list = f.read().split("\n")

if __name__ == "__main__":
    # Initialize video stream and FPS counter
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()

    input_queue = Queue(maxsize=5)
    output_queue = Queue(maxsize=5)

    process = Process(target=yolo_inference, args=(input_queue, output_queue, class_list, "faces.pt"))
    process.start()

    display_thread = Thread(target=display_video, args=(output_queue,))
    display_thread.start()

    while True:
        frame = vs.read()
        if not input_queue.full():
            input_queue.put(frame)

        fps.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            input_queue.put(None)  # Signal to stop the process
            break

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    vs.stop()
    process.join()
    display_thread.join()
