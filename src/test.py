import os
import shutil
from pathlib import Path

import random
import numpy as np
import cv2
from ultralytics import YOLO
import yaml

def extract_frames(video_dir, n_frames, output_dir):
    """
    Extracts n equally spaced frames from the video and allows the user to select ROI for each frame.
    Saves the full frames as .jpg and corresponding labels as .txt in YOLO format.
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    # Create output directories if not exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Load the video
    cap = cv2.VideoCapture(video_dir)
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
        
        # Save the full frame
        frame_name = f"frame_{i+1}.jpg"
        frame_dir = os.path.join(images_dir, frame_name)
        cv2.imwrite(frame_dir, frame)
        print(f"Saved: {frame_dir}")
        
        # Set window size to match image size
        window_name = f"Select ROI for Frame {i+1}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        height, width, _ = frame.shape
        cv2.resizeWindow(window_name, width, height)
        
        # Allow user to select ROI
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi != (0, 0, 0, 0):  # If ROI is valid, save the label in YOLO format
            x, y, w, h = roi
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height

            label_name = frame_name.replace(".jpg", ".txt")
            label_dir = os.path.join(labels_dir, label_name)

            with open(label_dir, "w") as f:
                f.write(f"0 {x_center} {y_center} {norm_w} {norm_h}\n")
            print(f"Saved label: {label_dir}")
        else:
            label_name = frame_name.replace(".jpg", ".txt")
            label_dir = os.path.join(labels_dir, label_name)

            with open(label_dir, "w") as f:
                f.write(f"0 0 0 0 0\n")
            print(f"No ROI selected for frame {i+1}")

    cap.release()

def format_data(raw_dir, train_validate_ratio):
    """
    Splits the dataset into training, validation, and test sets.
    The test set will contain the full dataset (i.e., all images and labels).
    """
    raw_dir = os.path.abspath(raw_dir)
    parent_dir = os.path.abspath(os.path.join(raw_dir, os.pardir))

    train_img_dir = os.path.join(parent_dir, "images", "train")
    validate_img_dir = os.path.join(parent_dir, "images", "val")
    test_img_dir = os.path.join(parent_dir, "images", "test")  # Test folder

    train_label_dir = os.path.join(parent_dir, "labels", "train")
    validate_label_dir = os.path.join(parent_dir, "labels", "val")
    test_label_dir = os.path.join(parent_dir, "labels", "test")  # Test folder
    
    raw_img_dir = os.path.join(raw_dir, "images")
    raw_label_dir = os.path.join(raw_dir, "labels")

    # Create necessary directories
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(validate_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(validate_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    all_images = os.listdir(raw_img_dir)
    random.shuffle(all_images)

    # Split the dataset into training and validation
    train_split = all_images[:int(train_validate_ratio * len(all_images))]
    validate_split = all_images[int(train_validate_ratio * len(all_images)):]

    # Copy images and labels to train and validate directories
    for image_file in train_split:
        shutil.copy(os.path.join(raw_img_dir, image_file), os.path.join(train_img_dir, image_file))
        label_file = image_file.replace(".jpg", ".txt")
        label_dir = os.path.join(raw_label_dir, label_file)
        if os.path.exists(label_dir):
            shutil.copy(label_dir, os.path.join(train_label_dir, label_file))
    
    for image_file in validate_split:
        shutil.copy(os.path.join(raw_img_dir, image_file), os.path.join(validate_img_dir, image_file))
        label_file = image_file.replace(".jpg", ".txt")
        label_dir = os.path.join(raw_label_dir, label_file)
        if os.path.exists(label_dir):
            shutil.copy(label_dir, os.path.join(validate_label_dir, label_file))

    # Copy all data to the test set (all images and labels)
    for image_file in all_images:
        shutil.copy(os.path.join(raw_img_dir, image_file), os.path.join(test_img_dir, image_file))
        label_file = image_file.replace(".jpg", ".txt")
        label_dir = os.path.join(raw_label_dir, label_file)
        if os.path.exists(label_dir):
            shutil.copy(label_dir, os.path.join(test_label_dir, label_file))

    # Create the YOLO data config file with the test directory
    config_dir = os.path.join(parent_dir, "data.yaml")
    config_details = {
        "path": parent_dir,
        "train": os.path.join("images", "train"),
        "val": os.path.join("images", "val"),
        "test": os.path.join("images", "test"),
        "names": {
            "0": "roi"
        }
    }

    with open(config_dir, "w") as f:
        yaml.dump(config_details, f)

def train_yolo_model(data_dir, pre_train_dir, post_train_dir, epochs=50):
    """
    Trains a YOLO model using the provided dataset.
    """
    model = YOLO(pre_train_dir)
    
    # Train the model on the annotated data
    model.train(data=data_dir, epochs=epochs, imgsz=640)

    # Save the trained model
    model.save(post_train_dir)
    print(f"Model saved at {post_train_dir}")

def predict_and_play_video(video_dir, model_dir, save_dir=None, max_det=None, conf=0.5):
    """
    Uses a trained YOLO model to predict objects in a video, displays the predictions, and optionally saves them as an MP4 video.
    
    Parameters:
    - video_dir: The path to the video file.
    - model_dir: The path to the trained YOLO model.
    - save_dir: Directory where the output video will be saved (optional). If not provided, the video won't be saved.
    - max_det: Maximum number of detections per frame.
    - conf: Confidence threshold for detections.
    """
    model = YOLO(model_dir)
    cap = cv2.VideoCapture(video_dir)

    # Get video properties for saving
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object if save_dir is provided
    out = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        output_dir = os.path.join(save_dir, "predictions.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
        out = cv2.VideoWriter(output_dir, fourcc, fps, (width, height))
        print(f"Saving output video to: {output_dir}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame, max_det=max_det, conf=conf)
        annotated_frame = results[0].plot()

        # Display the frame with predictions
        window_name = "YOLOv8 Predictions"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.imshow(window_name, annotated_frame)

        # Write the frame to output video if save_dir is provided
        if out:
            out.write(annotated_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Release the video writer if it was created
    if out:
        out.release()
        print(f"Video saved to: {output_dir}")

# Example usage:
video_dir = "./input/test-1.mp4"
output_dir = "./train-test/raw/"
n_frames = 20  # Number of frames to extract
# extract_frames(video_dir=video_dir, n_frames=n_frames, output_dir=output_dir)

# Example usage:
train_validate_ratio = 0.8
# format_data(raw_dir=output_dir, train_validate_ratio=train_validate_ratio)

# Example usage:
data_dir = "./train-test/data.yaml"  # The dataset's YAML file
pre_train_dir = "./models/yolo11n.pt"
post_train_dir = "./train-test/best.pt"
# train_yolo_model(data_dir=data_dir, pre_train_dir=pre_train_dir, post_train_dir=post_train_dir, epochs=50)

# Example usage:
model_dir = post_train_dir
save_dir = "./train-test/output/"
predict_and_play_video(video_dir=video_dir, model_dir=model_dir, save_dir=save_dir, max_det=1, conf=0.01)