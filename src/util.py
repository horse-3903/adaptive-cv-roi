import os
import subprocess
from pathlib import Path

def create_dir_name(parent_dir: str, prefix: str, suffix: str = None):
    all_dir = os.listdir(parent_dir)
    prev_roi = [dir for dir in all_dir if prefix in dir]
    index = len(prev_roi)

    if suffix:
        suffix = "." + suffix

    final_dir = Path(parent_dir).joinpath(f"{prefix}-{index:02d}{suffix or ''}")

    return index, final_dir.absolute().as_posix() + ("/" if not suffix else "")

# def convert_format