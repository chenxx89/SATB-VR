import itertools
import math

import torch


def prepare_tiling_infos_generator(enable_spatial_tiling, enable_temporal_tiling, latents, tile_size, tile_stride, temporal_tile_size, temporal_tile_stride):
    if not enable_spatial_tiling and not enable_temporal_tiling:
        yield [slice(None), torch.ones_like(latents)]
        return

    batch_size, num_frames, num_channels, height, width = latents.shape

    if not enable_spatial_tiling:
        tile_size = max(height, width)
    if not enable_temporal_tiling:
        temporal_tile_size = num_frames

    def create_start_indices(size, tile_size, tile_stride):
        if size <= tile_size:
            tile_stride = tile_size
        else:
            num_tiles = (size - tile_size) // tile_stride + 1
            if (size - tile_size) % tile_stride != 0:
                num_tiles += 1
            tile_stride = math.ceil((size - tile_size) / (num_tiles - 1))
        i_list = list(range(0, max(1, size - tile_size + 1), tile_stride))
        if size >= tile_size and (size - tile_size) % tile_stride != 0:
            i_list.append(size - tile_size)
        return i_list, tile_size, tile_stride

    ti_list, t_tile_size, t_tile_stride = create_start_indices(num_frames, temporal_tile_size, temporal_tile_stride)
    hi_list, h_tile_size, h_tile_stride = create_start_indices(height, tile_size, tile_stride)
    wi_list, w_tile_size, w_tile_stride = create_start_indices(width, tile_size, tile_stride)

    def compute_valid_weights_range(i, i_end, size, tile_size, tile_stride):
        float_padding = (tile_size - tile_stride) / 2
        end = tile_size - math.floor(float_padding) if i_end < size else tile_size
        start = math.ceil(float_padding) if i > 0 else 0
        remainder = i % tile_stride
        if remainder > 0:
            start = tile_size - (math.floor(float_padding) + remainder)
        return slice(start, end)

    for ti, hi, wi in itertools.product(ti_list, hi_list, wi_list):
        ti_end = min(ti + t_tile_size, num_frames)
        hi_end = min(hi + h_tile_size, height)
        wi_end = min(wi + w_tile_size, width)
        tile_slice = [slice(None), slice(ti, ti_end), slice(None), slice(hi, hi_end), slice(wi, wi_end)]

        t_valid_slice = compute_valid_weights_range(ti, ti_end, num_frames, t_tile_size, t_tile_stride)
        h_valid_slice = compute_valid_weights_range(hi, hi_end, height, h_tile_size, h_tile_stride)
        w_valid_slice = compute_valid_weights_range(wi, wi_end, width, w_tile_size, w_tile_stride)
        weights = torch.zeros((1, ti_end - ti, 1, hi_end - hi, wi_end - wi))
        weights[:, t_valid_slice, :, h_valid_slice, w_valid_slice] = 1

        yield tile_slice, weights.repeat(batch_size, 1, num_channels, 1, 1).to(latents.device)
