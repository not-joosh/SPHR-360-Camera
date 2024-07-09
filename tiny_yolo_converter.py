import torch
from ultralytics import YOLO


# model = YOLO("faces_model.pt")
# model.export(format="ncnn")  

model = YOLO("gestures_model.pt")
model.export(format="ncnn")

