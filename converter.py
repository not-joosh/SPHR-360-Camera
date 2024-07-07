import torch
from ultralytics import YOLO


# Load a YOLOv8n PyTorch model
model = YOLO("faces_v7.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'yolov8n_ncnn_model' directory
