from main import AdaptiveROI

a = AdaptiveROI(video_dir="./input/test-1.mp4", parent_dir="./adaptive-roi-0")

a.model_dir = "./adaptive-roi-0/models/model-00/best.pt"

a.predict_and_play_video(conf=0.01, max_det=1)