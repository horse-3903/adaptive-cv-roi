# Adaptive ROI Detection with YOLOv11

This repository contains a Python-based system for automatic Region of Interest (ROI) detection in videos using the YOLOv11 model. It extracts specific frames from video, allows manual selection of ROI, formats the data, trains a model, and finally generates predictions on the video. The system is designed to be adaptive, allowing dynamic adjustments to the ROI based on custom inputs.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Extract Data](#extract-data)
  - [Format Data](#format-data)
  - [Train Model](#train-model)
  - [Predict and Play Video](#predict-and-play-video)
- [Sample Output](#sample-output)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Extract specific frames from a video for ROI selection.
- Manually label ROIs by selecting bounding boxes on frames.
- Split labeled data into training, validation, and test sets.
- Train the YOLOv11 model on the labeled data.
- Predict and display ROIs on the original video.
- Save annotated predictions as a video file (`predictions.mp4`).

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/adaptive-roi-detection.git
   cd adaptive-roi-detection
   ```

2. Install dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** This system requires OpenCV, PyYAML, NumPy, and Ultralytics YOLO.

---

## Usage

### Extract Data

The `extract_data` function allows you to extract and manually label ROI in a specified number of frames.

```python
adaptive_roi = AdaptiveROI(video_dir="path/to/video.mp4")
adaptive_roi.extract_data(n_frames=15)
```

This will open the video in a frame-by-frame manner and let you select ROIs interactively.

### Format Data

Once you have the labeled data, you can format it for training and validation:

```python
adaptive_roi.format_data(train_val_ratio=0.8)
```

This splits the data into training and validation sets based on the provided ratio.

### Train Model

Train a YOLOv11 model using the pre-configured dataset:

```python
adaptive_roi.train_model(pre_train_dir="path/to/yolo11n.pt", epochs=50)
```

The model will be trained on the selected ROI frames, and the best weights will be saved.

### Predict and Play Video

Use the trained model to predict ROI in the original video and save the output:

```python
adaptive_roi.predict_and_play_video(prediction_dir="path/to/save/predictions")
```

The predicted bounding boxes will be visualized in real-time and saved as `predictions.mp4`.

---

## Sample Output

A sample prediction output can be seen in the video file [predictions.mp4](predictions.mp4).

![Model Predictions](predictions.gif)

---

## Configuration

The system saves its configuration to a YAML file, which contains details about the directories, data, and trained models.

You can also load an existing configuration using:

```python
adaptive_roi = AdaptiveROI.load_from_file("path/to/config.yaml")
```

And save the current configuration:

```python
adaptive_roi.save_to_file()
```

---

## Contributing

Contributions to this project are welcome. Feel free to submit pull requests or report issues.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

