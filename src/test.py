from main import AdaptiveROI

def test_adaptive_roi(video_dir: str):
    """
    This function tests the AdaptiveROI class by executing its methods in the correct order.
    
    Parameters:
    - video_dir: Path to the input video file.
    - parent_dir: Directory where output (frames, annotations, trained models) will be saved.
    - pre_trained_model: Path to a pre-trained YOLO model.
    - trained_model_output: Directory where the trained YOLO model will be saved.
    """
    
    # Initialize the AdaptiveROI class
    adaptive_roi = AdaptiveROI(video_dir=video_dir)
    
    # Step 1: Extract frames and select ROIs for each frame
    print("\n--- Extracting Frames and Selecting ROIs ---")
    adaptive_roi.extract_data(n_frames=10)  # Example: extract 5 frames from the video
    
    # Step 2: Format data for YOLO (split into train, val, test)
    print("\n--- Formatting Data into Train, Validation, Test Sets ---")
    adaptive_roi.format_data(train_val_ratio=0.8)  # 80% train, 20% val
    
    # Step 3: Train the YOLO model using the extracted data
    print("\n--- Training the YOLO Model ---")
    adaptive_roi.train_model()
    
    # Step 4: Perform predictions on the video using the trained model and play/save video
    print("\n--- Running YOLO Predictions on Video ---")
    adaptive_roi.predict_and_play_video(conf=0.01)

if __name__ == "__main__":
    # Define dirs
    video_dir = "./input/test-1.mp4"

    # Run the test function
    test_adaptive_roi(video_dir)