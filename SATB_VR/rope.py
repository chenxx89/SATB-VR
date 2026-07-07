from typing import Optional, Tuple

import torch

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid


def prepare_rotary_positional_embeddings(
    latent_height: int,
    latent_width: int,
    num_frames: int,
    patch_size: int = 2,
    patch_size_t: Optional[int] = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    sample_height: int = 60,
    sample_width: int = 90,
) -> Tuple[torch.Tensor, torch.Tensor]:

    grid_height = latent_height // patch_size
    grid_width = latent_width // patch_size

    if patch_size_t is None:
        # CogVideoX 1.0 I2V
        base_size_width = sample_width // patch_size
        base_size_height = sample_height // patch_size
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        # CogVideoX 1.5 I2V
        # config from https://github.com/THUDM/CogVideo/blob/2fdc59c3ce48aee1ba7572a1c241e5b3090abffa/sat/configs/cogvideox1.5_5b_i2v.yaml#L33
        max_size_width = 300 // patch_size
        max_size_height = 300 // patch_size
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(max_size_height, max_size_width),
        )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin
