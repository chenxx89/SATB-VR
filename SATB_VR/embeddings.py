from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .resnet import SpatioTemporalResBlock

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class ControlPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        in_channels: int = 16,
        embed_dim: int = 3072,
        text_embed_dim: int = 4096,
        time_embed_dim: int = 512,
        **kwargs
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t

        self.text_proj = zero_module(nn.Linear(text_embed_dim, embed_dim))
        self.res_block = nn.ModuleList([
            SpatioTemporalResBlock(in_channels, 320, time_embed_dim,merge_strategy="learned", groups=16),
            SpatioTemporalResBlock(320, 320, time_embed_dim, merge_strategy="learned", groups=32),
            SpatioTemporalResBlock(320, in_channels, time_embed_dim, merge_strategy="learned", groups=16)
        ])
        self.proj = zero_module(nn.Linear(in_channels * patch_size * patch_size * patch_size_t, embed_dim))


    def forward(self, control_states: torch.Tensor, 
                encoder_hidden_states: torch.Tensor,
                hidden_states: torch.Tensor, 
                emb: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        batch_size, num_frames, channels, height, width = control_states.shape

        text_embeds = self.text_proj(encoder_hidden_states)

        control_states = rearrange(control_states, "B F C H W -> (B F) C H W")
        res_emb = repeat(emb, 'b d -> (b f) d', f=num_frames)
        for module in self.res_block:
            control_states = module(control_states,res_emb,torch.ones((num_frames),device=control_states.device))
        control_states = rearrange(control_states, "(B F) C H W -> B F C H W", B=batch_size, F=num_frames)

        p = self.patch_size
        p_t = self.patch_size_t

        control_states = control_states.permute(0, 1, 3, 4, 2)
        control_states = control_states.reshape(
            batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
        )
        control_states = control_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
        control_states = self.proj(control_states)

        control_states = torch.cat([text_embeds, control_states], dim=1).contiguous()  

        embeds = hidden_states + control_states

        return embeds
