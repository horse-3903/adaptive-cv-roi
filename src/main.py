import os
import shutil
from pathlib import Path

import random
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import yaml

class AdaptiveROI:
    def __init__(self, video_dir: str, parent_dir: str = None):
        video_dir = Path(video_dir).absolute().as_posix()
        assert os.path.exists(video_dir)
        self.video_dir = video_dir
        
        all_dir = os.listdir(".")
        prev_roi = [dir for dir in all_dir if "adaptive-roi-" in dir]
        self.index = len(prev_roi)

        if not parent_dir:
            parent_dir = f"./adaptive-roi-{self.index}/"

        parent_dir = Path(parent_dir).absolute().as_posix()
        os.makedirs(parent_dir, exist_ok=True)
        self.parent_dir = parent_dir + "/"
        
    def extract_data(self, n_frames: int = 15, raw_dir: str = None):
        """
        Extracts n equally spaced frames from the video and allows the user to select ROI for each frame.
        Saves the full frames as .jpg and corresponding labels as .txt in YOLO format.
        """
        if not raw_dir:
            raw_dir = os.path.join(self.parent_dir, "raw")

        raw_dir = Path(raw_dir).absolute().as_posix() + "/"

        self.raw_img_dir = os.path.join(raw_dir, "images")
        self.raw_label_dir = os.path.join(raw_dir, "labels")
        
        # Create output directories if not exist
        os.makedirs(self.raw_img_dir, exist_ok=True)
        os.makedirs(self.raw_label_dir, exist_ok=True)
        
        # Load the video
        cap = cv2.VideoCapture(self.video_dir)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine the frame intervals
        frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)

        for i, frame_idx in enumerate(frame_indices):
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame {frame_idx}")
                continue
            
            # Save the full frame
            frame_name = f"frame_{i+1}.jpg"
            frame_dir = os.path.join(self.raw_img_dir, frame_name)
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
                label_dir = os.path.join(self.raw_label_dir, label_name)

                with open(label_dir, "w") as f:
                    f.write(f"0 {x_center} {y_center} {norm_w} {norm_h}\n")
                print(f"Saved label: {label_dir}")
            else:
                label_name = frame_name.replace(".jpg", ".txt")
                label_dir = os.path.join(self.raw_label_dir, label_name)

                with open(label_dir, "w") as f:
                    f.write(f"0 0 0 0 0\n")
                print(f"No ROI selected for frame {i+1}")

        cap.release()

    def format_data(self, train_val_ratio: float):
        """
        Splits the dataset into training, validation, and test sets.
        The test set will contain the full dataset (i.e., all images and labels).
        """
        self.train_img_dir = os.path.join(self.parent_dir, "images", "train")
        self.validate_img_dir = os.path.join(self.parent_dir, "images", "val")
        self.test_img_dir = os.path.join(self.parent_dir, "images", "test")  # Test folder

        self.train_label_dir = os.path.join(self.parent_dir, "labels", "train")
        self.validate_label_dir = os.path.join(self.parent_dir, "labels", "val")
        self.test_label_dir = os.path.join(self.parent_dir, "labels", "test")  # Test folder

        # Create necessary directories
        os.makedirs(self.train_img_dir, exist_ok=True)
        os.makedirs(self.validate_img_dir, exist_ok=True)
        os.makedirs(self.test_img_dir, exist_ok=True)
        
        os.makedirs(self.train_label_dir, exist_ok=True)
        os.makedirs(self.validate_label_dir, exist_ok=True)
        os.makedirs(self.test_label_dir, exist_ok=True)

        # Split the dataset into training and validation
        all_img = os.listdir(self.raw_img_dir)
        random.shuffle(all_img)

        train_split = all_img[:int(train_val_ratio * len(all_img))]
        validate_split = all_img[int(train_val_ratio * len(all_img)):]

        # Copy images and labels to train and validate directories
        for image_file in train_split:
            shutil.copy(os.path.join(self.raw_img_dir, image_file), os.path.join(self.train_img_dir, image_file))
            label_file = image_file.replace(".jpg", ".txt")
            label_dir = os.path.join(self.raw_label_dir, label_file)
            if os.path.exists(label_dir):
                shutil.copy(label_dir, os.path.join(self.train_label_dir, label_file))
        
        for image_file in validate_split:
            shutil.copy(os.path.join(self.raw_img_dir, image_file), os.path.join(self.validate_img_dir, image_file))
            label_file = image_file.replace(".jpg", ".txt")
            label_dir = os.path.join(self.raw_label_dir, label_file)
            if os.path.exists(label_dir):
                shutil.copy(label_dir, os.path.join(self.validate_label_dir, label_file))

        # Copy all data to the test set (all images and labels)
        for image_file in all_img:
            shutil.copy(os.path.join(self.raw_img_dir, image_file), os.path.join(self.test_img_dir, image_file))
            label_file = image_file.replace(".jpg", ".txt")
            label_dir = os.path.join(self.raw_label_dir, label_file)
            if os.path.exists(label_dir):
                shutil.copy(label_dir, os.path.join(self.test_label_dir, label_file))

        # Create the YOLO data config file with the test directory
        self.config_dir = os.path.join(self.parent_dir, "data.yaml")
        config_details = {
            "path": self.parent_dir,
            "train": os.path.join("images", "train"),
            "val": os.path.join("images", "val"),
            "test": os.path.join("images", "test"),
            "names": {
                "0": "roi"
            }
        }

        with open(self.config_dir, "w") as f:
            yaml.dump(config_details, f)

    def config_model_train(self):
        pass

    def train_model(self, pre_train_dir: str = None, model_dir: str = None, epochs=50):
        """
        Trains a YOLO model using the provided dataset.
        """
        if not pre_train_dir:
            pre_train_dir = "./models/yolo11n.pt"

        pre_train_dir = Path(pre_train_dir).absolute().as_posix()
        assert os.path.exists(pre_train_dir)
        self.pre_train_dir = pre_train_dir + "/"

        if not model_dir:
            project_dir = os.path.join(self.parent_dir, "models")
            os.makedirs(project_dir, exist_ok=True)
            all_dir = os.listdir(project_dir)
            prev_roi = [dir for dir in all_dir if "model-" in dir]
            model_index = len(prev_roi)
            model_dir = os.path.join(self.parent_dir, f"models/model-{model_index:02d}/best.pt")

        model_dir = Path(model_dir).absolute().as_posix()
        save_dir = Path(model_dir).parent.absolute().as_posix() + "/"
        self.model_dir = model_dir

        model = YOLO(self.pre_train_dir)
        
        # Train the model on the annotated data
        model.train(data=self.config_dir, epochs=epochs, imgsz=640, save=True, project=project_dir, name=f"model-{model_index:02d}")

        # Save the trained model
        model.save(self.model_dir)
        print(f"Model saved at {self.model_dir}")

    def predict_and_play_video(self, predict_dir: str=None, max_det=None, conf=0.5):
        """
        Uses a trained YOLO model to predict objects in a video, displays the predictions, and optionally saves them as an MP4 video.
        
        Parameters:
        - predict_dir: Directory where the output video will be saved (optional). If not provided, the video won't be saved.
        - max_det: Maximum number of detections per frame.
        - conf: Confidence threshold for detections.
        """
        if not predict_dir:
            predict_dir = os.path.join(self.parent_dir, "predictions")

        predict_dir = Path(predict_dir).absolute().as_posix()
        assert os.path.exists(predict_dir)

        model = YOLO(self.model_dir)
        cap = cv2.VideoCapture(self.video_dir)

        # Get video properties for saving
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object if save_dir is provided
        out = None
        os.makedirs(predict_dir, exist_ok=True)
        predict_dir = os.path.join(predict_dir, "predictions.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
        out = cv2.VideoWriter(predict_dir, fourcc, fps, (width, height))
        print(f"Saving output video to: {predict_dir}")

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
            print(f"Video saved to: {predict_dir}")