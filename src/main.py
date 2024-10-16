import os
from pathlib import Path
import random
import numpy as np
import cv2
from ultralytics import YOLO
import yaml

def extract_frames(video_path, n_frames, output_dir):
    """
    Extracts n equally spaced frames from the video and allows the user to select ROI for each frame.
    Saves the images with the ROI to the output directory.
    """
    # Create output directory if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine the frame intervals
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    for i, frame_idx in enumerate(frame_indices):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame {frame_idx}")
            continue
        
        # Set window size to match image size
        window_name = f"Select ROI for Frame {i+1}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        height, width, _ = frame.shape
        cv2.resizeWindow(window_name, width, height)
        
        # Allow user to select ROI
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        if roi != (0, 0, 0, 0):  # If ROI is valid, crop and save
            x, y, w, h = roi
            roi_frame = frame[y:y+h, x:x+w]
            output_path = os.path.join(output_dir, f"frame_{i+1}.jpg")
            cv2.imwrite(output_path, roi_frame)
            print(f"Saved: {output_path}")
        else:
            print(f"No ROI selected for frame {i+1}")
        
        # Close the ROI window
        cv2.destroyAllWindows()

    cap.release()

def format_data(raw_dir, train_dir, validate_dir, train_validate_ratio):
    parent_dir = Path(train_dir).parent.absolute()
    
    raw_dir, train_dir, validate_dir = [dir_name+"/" if not dir_name.endswith("/") else dir_name for dir_name in (raw_dir, train_dir, validate_dir)]

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(validate_dir):
        os.makedirs(validate_dir)

    all_image_dir = os.listdir(raw_dir)
    random.shuffle(all_image_dir)

    train_image_dir = all_image_dir[:int(train_validate_ratio*len(all_image_dir))]
    validate_image_dir = all_image_dir[int(train_validate_ratio*len(all_image_dir)):]

    for image_dir in train_image_dir:
        os.rename(raw_dir+image_dir, train_dir+image_dir)
    
    for image_dir in validate_image_dir:
        os.rename(raw_dir+image_dir, validate_dir+image_dir)

    config_dir = parent_dir/"data.yaml"

    config_details = {
        "path": parent_dir.as_posix(),
        "train": train_dir,
        "val": validate_dir,
        "test": None,
        "names": ["roi"]
    }

    with open(config_dir, "w+") as f:
        config_formatted = yaml.dump(config_details)
        f.write(config_formatted)

def train_yolo_model(data_path, model_save_path, epochs=50):
    """
    Trains a YOLOv8 model on the annotated dataset and saves the model.
    """
    # Load a pre-trained YOLOv8 model (YOLOv8n is the small version, you can change to "yolov8s.pt", "yolov8m.pt", etc.)
    model = YOLO("yolov8n.pt")
    
    # Train the model on the annotated data
    model.train(data=data_path, epochs=epochs, imgsz=640)

    # Save the trained model
    model.export(model_save_path)
    print(f"Model saved at {model_save_path}")

def predict_and_play_video(video_path, model_path):
    """
    Runs the trained YOLO model on the video and displays the prediction.
    """
    # Load the trained YOLO model
    model = YOLO(model_path)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make predictions on the frame
        results = model.predict(frame)
        
        # Draw the predicted boxes on the frame
        annotated_frame = results[0].plot()  # .plot() function draws the predictions on the frame
        
        # Set window size to match frame size
        window_name = "YOLOv8 Predictions"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        height, width, _ = annotated_frame.shape
        cv2.resizeWindow(window_name, width, height)
        
        # Show the frame with predictions
        cv2.imshow(window_name, annotated_frame)

        # Press "q" to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
video_path = "./input/test-1.mp4"
output_dir = "./train-test/raw/"
n_frames = 10  # Number of frames to extract
extract_frames(video_path, n_frames, output_dir)

format_data(raw_dir="./train-test/raw/", train_dir="./train-test/train", validate_dir="./train-test/validate", train_validate_ratio=0.8)

# # Example usage:
# data_path = "./train-test/data.yaml"  # The dataset's YAML file (YOLO format)
# model_save_path = "trained_model.pt"
# train_yolo_model(data_path, model_save_path)

# # Example usage:
# model_path = "trained_model.pt"
# predict_and_play_video(video_path, model_path)