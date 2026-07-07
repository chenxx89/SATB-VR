import io
import pathlib

import torch
import decord
import numpy as np
from typing import Tuple
from PIL import Image

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', 'bmp')
VALID_VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.webm', '.mkv', '.y4m')

decord.bridge.set_bridge("torch")


def load_image(path):
    img = Image.open(path)
    return np.array(img, dtype=np.float32)


def load_video(video_path) -> Tuple[torch.Tensor, float]:
    video_obj = video_path
    is_dir = False

    if isinstance(video_path, (str, pathlib.Path)):
        video_path = pathlib.Path(video_path)
        is_dir = video_path.is_dir()
        if is_dir:
            frame_files = sorted([f.name for f in video_path.iterdir()
                                  if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS])
            video_num_frames = len(frame_files)
        else:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            video_obj = io.BytesIO(video_bytes)

    if not is_dir:
        video_reader = decord.VideoReader(uri=video_obj, num_threads=2)
        video_num_frames = len(video_reader)
        fps = video_reader.get_avg_fps()
    else:
        fps = 25

    indices = list(range(0, video_num_frames))

    video_path = pathlib.Path(video_path) if not isinstance(video_path, pathlib.Path) else video_path
    if video_path.is_dir():
        selected_frame_files = list(map(frame_files.__getitem__, indices))
        frames = torch.stack(
            [torch.from_numpy(load_image(video_path.joinpath(f))) for f in selected_frame_files])
    else:
        frames = video_reader.get_batch(indices)

    frames = frames.float().div(255.0).clip(0, 1).permute(0, 3, 1, 2).contiguous()

    return frames, fps
