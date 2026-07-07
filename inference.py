
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers.training_utils import free_memory
from diffusers.utils import export_to_video

from SATB_VR.colorfix import wavelet_reconstruction
from SATB_VR.data import load_video
from SATB_VR.pipeline import CogVideoXVRPipeline
from SATB_VR.utils import collect_video_paths

def main():
    parser = argparse.ArgumentParser(description="SATB-VR")
    parser.add_argument("--ckpt_path", type=str, default='./ckpts', help="Path to checkpoints directory.")
    parser.add_argument("--input_dir", type=str, default="./test_samples/inputs", help="Path to input videos directory.")
    parser.add_argument("--output_dir", type=str, default="./test_samples/outputs", help="Path to output videos directory.")
    parser.add_argument("--upscale", type=float, default=0., help='The upsample scale. Default upscale=0, short-size resized to 1024.')
    parser.add_argument("--enable_text_encoder", action="store_true", help="Whether to use text encoder.")
    parser.add_argument("--enable_captioner", action="store_true", help="Whether to use cogvlm2 captioner.")
    parser.add_argument('--enable_spatial_tiling', action="store_true", help="Whether to enable spatial tiling.")
    parser.add_argument('--enable_temporal_tiling', action="store_true", help="Whether to enable temporal tiling.")
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_images", action='store_true')
    args = parser.parse_args()

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    pipe = CogVideoXVRPipeline.from_args(args)

    # enable cpu offload
    pipe.enable_model_cpu_offload(device=args.device)  # faster, but use more GPU memory
    # pipe.enable_sequential_cpu_offload(device=args.device)  # slower, but use less GPU memory

    video_paths = collect_video_paths(args.input_dir)
    print(f"Found {len(video_paths)} videos in {args.input_dir}.")

    os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)

    for video_path in tqdm(video_paths, 'Restoration'):
        basename = os.path.splitext(video_path.split('/')[-1])[0]
        save_filepath = os.path.join(args.output_dir, "videos", f"{basename}.mp4")

        if os.path.isfile(save_filepath):
            print(f"{video_path} has already been processed, skipping...")
            continue

        # [F, C, H, W]
        input_video, fps = load_video(video_path)
        if args.upscale == 0.:
            scale_factor = 1024. / min(input_video.size()[2], input_video.size()[3])
        elif args.upscale != 1.0:
            scale_factor = args.upscale
        if args.upscale != 1.0:
            input_video = F.interpolate(input_video, scale_factor=scale_factor, mode='bicubic').clip(0, 1)
        print(f"Processing {video_path} with shape {input_video.shape}.")

        video = pipe(args, input_video, fps, generator)

        # colorfix
        samples = wavelet_reconstruction(video[0], input_video.to(args.device))

        # save output video and image
        samples = samples.cpu().clip(0, 1).permute(0, 2, 3, 1).float().numpy()
        export_to_video(samples, save_filepath, fps=fps, quality=8)
        if args.save_images:
            image_dir = os.path.join(args.output_dir, "images", f"{basename}")
            os.makedirs(image_dir, exist_ok=True)
            for i in range(len(samples)):
                Image.fromarray((samples[i] * 255).clip(0, 255).astype(np.uint8)
                                ).save(os.path.join(image_dir, f"{i:06d}.png"))

        # print GPU memory usage
        print(torch.cuda.memory_summary(abbreviated=False))

        del video, input_video, samples
        free_memory()


if __name__ == "__main__":
    main()
