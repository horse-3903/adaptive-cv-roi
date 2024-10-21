import os
import cv2
import shutil
import yaml
import numpy as np
import random
from ultralytics import YOLO
from pathlib import Path
from util import create_dir_name

class AdaptiveROI:
    def __init__(self, video_dir, parent_dir = None):
        self.video_dir = Path(video_dir).absolute()
        assert self.video_dir.exists()
        
        self.index, tmp_parent_dir = create_dir_name(parent_dir=".", prefix="adaptive-roi")

        if not parent_dir:
            self.parent_dir = Path(tmp_parent_dir).absolute()
        else:
            self.parent_dir = Path(parent_dir).absolute()

        self.parent_dir.mkdir(parents=True, exist_ok=True)

        self.raw_img_dir = None
        self.raw_label_dir = None

        self.train_img_dir = None
        self.validate_img_dir = None
        self.test_img_dir = None

        self.train_label_dir = None
        self.validate_label_dir = None
        self.test_label_dir = None
        
        self.train_config_dir = None
        self.pre_train_dir = None
        self.model_dir = None

    @classmethod
    def load_from_file(cls, config_dir):
        assert config_dir.endswith(".yaml")

        with open(config_dir, "r") as f:
            config_dict = yaml.safe_load(f)

        instance = cls(video_dir=config_dict["video_dir"], parent_dir=config_dict["parent_dir"])
        instance.raw_img_dir = Path(config_dict["raw_img_dir"]).absolute()
        instance.raw_label_dir = Path(config_dict["raw_label_dir"]).absolute()
        instance.train_img_dir = Path(config_dict["train_img_dir"]).absolute()
        instance.validate_img_dir = Path(config_dict["validate_img_dir"]).absolute()
        instance.test_img_dir = Path(config_dict["test_img_dir"]).absolute()
        instance.train_label_dir = Path(config_dict["train_label_dir"]).absolute()
        instance.validate_label_dir = Path(config_dict["validate_label_dir"]).absolute()
        instance.test_label_dir = Path(config_dict["test_label_dir"]).absolute()
        instance.train_config_dir = Path(config_dict["train_config_dir"]).absolute()
        instance.pre_train_dir = Path(config_dict["pre_train_dir"]).absolute()
        instance.model_dir = Path(config_dict["model_dir"]).absolute()

        return instance

    def save_to_file(self, config_dir = None):
        config_info = {
            "video_dir": self.video_dir.as_posix(),
            "parent_dir": self.parent_dir.as_posix(),
            "raw_img_dir": self.raw_img_dir.as_posix() if self.raw_img_dir else None,
            "raw_label_dir": self.raw_label_dir.as_posix() if self.raw_label_dir else None,
            "train_img_dir": self.train_img_dir.as_posix() if self.train_img_dir else None,
            "validate_img_dir": self.validate_img_dir.as_posix() if self.validate_img_dir else None,
            "test_img_dir": self.test_img_dir.as_posix() if self.test_img_dir else None,
            "train_label_dir": self.train_label_dir.as_posix() if self.train_label_dir else None,
            "validate_label_dir": self.validate_label_dir.as_posix() if self.validate_label_dir else None,
            "test_label_dir": self.test_label_dir.as_posix() if self.test_label_dir else None,
            "train_config_dir": self.train_config_dir.as_posix() if self.train_config_dir else None,
            "pre_train_dir": self.pre_train_dir.as_posix() if self.pre_train_dir else None,
            "model_dir": self.model_dir.as_posix() if self.model_dir else None,
        }

        self.config_dir = config_dir or os.path.join(self.parent_dir, "config.yaml")
        
        with open(self.config_dir, "w+") as f:
            yaml.dump(config_info, f)

        print("Details saved to :", self.config_dir)

    def extract_data(self, n_frames: int = 15, raw_dir = None):
        if not raw_dir:
            raw_dir = self.parent_dir / "raw"

        raw_dir = Path(raw_dir).absolute()
        self.raw_img_dir = raw_dir / "images"
        self.raw_label_dir = raw_dir / "labels"
        
        self.raw_img_dir.mkdir(parents=True, exist_ok=True)
        self.raw_label_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(self.video_dir.as_posix())
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame {frame_idx}")
                continue
            
            frame_name = f"frame_{i+1}.jpg"
            frame_dir = self.raw_img_dir / frame_name
            cv2.imwrite(frame_dir.as_posix(), frame)
            print(f"Saved: {frame_dir}")
            
            window_name = f"Select ROI for Frame {i+1}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            height, width, _ = frame.shape
            cv2.resizeWindow(window_name, width, height)
            
            roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()

            label_name = frame_name.replace(".jpg", ".txt")
            label_dir = self.raw_label_dir / label_name

            if roi != (0, 0, 0, 0):
                x, y, w, h = roi
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_w = w / width
                norm_h = h / height

                with open(label_dir, "w") as f:
                    f.write(f"0 {x_center} {y_center} {norm_w} {norm_h}\n")
                print(f"Saved label: {label_dir}")
            else:
                with open(label_dir, "w") as f:
                    f.write(f"0 0 0 0 0\n")
                print(f"No ROI selected for frame {i+1}")

        cap.release()
        self.save_to_file()

    def format_data(self, train_val_ratio: float):
        self.train_img_dir = self.parent_dir / "images/train"
        self.validate_img_dir = self.parent_dir / "images/val"
        self.test_img_dir = self.parent_dir / "images/test"

        self.train_label_dir = self.parent_dir / "labels/train"
        self.validate_label_dir = self.parent_dir / "labels/val"
        self.test_label_dir = self.parent_dir / "labels/test"

        self.train_img_dir.mkdir(parents=True, exist_ok=True)
        self.validate_img_dir.mkdir(parents=True, exist_ok=True)
        self.test_img_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_label_dir.mkdir(parents=True, exist_ok=True)
        self.validate_label_dir.mkdir(parents=True, exist_ok=True)
        self.test_label_dir.mkdir(parents=True, exist_ok=True)

        all_img = os.listdir(self.raw_img_dir)
        random.shuffle(all_img)

        train_split = all_img[:int(train_val_ratio * len(all_img))]
        validate_split = all_img[int(train_val_ratio * len(all_img)):]

        for image_file in train_split:
            shutil.copy(self.raw_img_dir / image_file, self.train_img_dir / image_file)
            label_file = image_file.replace(".jpg", ".txt")
            if (self.raw_label_dir / label_file).exists():
                shutil.copy(self.raw_label_dir / label_file, self.train_label_dir / label_file)
        
        for image_file in validate_split:
            shutil.copy(self.raw_img_dir / image_file, self.validate_img_dir / image_file)
            label_file = image_file.replace(".jpg", ".txt")
            if (self.raw_label_dir / label_file).exists():
                shutil.copy(self.raw_label_dir / label_file, self.validate_label_dir / label_file)

        for image_file in all_img:
            shutil.copy(self.raw_img_dir / image_file, self.test_img_dir / image_file)
            label_file = image_file.replace(".jpg", ".txt")
            if (self.raw_label_dir / label_file).exists():
                shutil.copy(self.raw_label_dir / label_file, self.test_label_dir / label_file)

        self.train_config_dir = self.parent_dir / "data.yaml"
        train_config_details = {
            "path": self.parent_dir.as_posix(),
            "train": self.train_img_dir.as_posix(),
            "val": self.validate_img_dir.as_posix(),
            "test": self.test_img_dir.as_posix(),
            "names": {
                "0": "roi"
            }
        }

        with open(self.train_config_dir, "w") as f:
            yaml.dump(train_config_details, f)

        self.save_to_file()

    def train_model(self, pre_train_dir = None, model_dir = None, epochs = 50):
        if not pre_train_dir:
            pre_train_dir = Path("./models/yolo11n.pt").absolute()

        self.pre_train_dir = Path(pre_train_dir).absolute()

        if not model_dir:
            project_dir = os.path.join(self.parent_dir, "models")
            os.makedirs(project_dir, exist_ok=True)

            model_index, model_dir = create_dir_name(project_dir, "model")
            model_train_dir = self.parent_dir / model_dir / "run"
            model_dir = self.parent_dir / model_dir / "best.pt"

        self.model_dir = Path(model_dir).absolute()
        self.model_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(self.pre_train_dir)

        model.train(
            model=self.pre_train_dir.as_posix(),
            data=self.train_config_dir.as_posix(),
            epochs=epochs,
            project=self.model_dir.as_posix(),
            name="run",
            exist_ok=True
        )

        model.save(self.model_dir)

        self.save_to_file()

    def predict_and_play_video(self, prediction_dir = None, max_det = None, conf = 0.5):
        if not prediction_dir:
            prediction_dir = self.parent_dir / "predictions"

        prediction_dir = Path(prediction_dir).absolute()
        prediction_dir.mkdir(parents=True, exist_ok=True)

        video_index, video_dir = create_dir_name(prediction_dir, "predict", "avi")

        model = YOLO(self.model_dir)
        cap = cv2.VideoCapture(self.video_dir)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = None

        prediction_dir = os.path.join(prediction_dir, video_dir)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for .mp4 format
        out = cv2.VideoWriter(prediction_dir, fourcc, fps, (width, height))
        print(f"Saving output video to: {prediction_dir}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(frame, max_det=max_det, conf=conf)
            annotated_frame = results[0].plot()

            window_name = "YOLOv11 Predictions"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, width, height)
            cv2.imshow(window_name, annotated_frame)

            if out:
                out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if out:
            out.release()
            print(f"Video saved to: {prediction_dir}")

        self.save_to_file()