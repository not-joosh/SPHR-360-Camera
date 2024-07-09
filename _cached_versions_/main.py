from face_detection import FaceDetection

# Main program
if __name__ == "__main__":
    # Setting up the model and necessary paths
    # model_path = "faces_v7.pt"
    model_path = "faces_v7_ncnn_model"
    label_path = "coco1.txt"
    face_detection = FaceDetection(model_path, label_path, resolution=(320, 320))
    face_detection.run()

