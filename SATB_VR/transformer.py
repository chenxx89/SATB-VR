import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.cogvideox_transformer_3d import (
    CogVideoXTransformer3DModel,
)
from diffusers.utils import logging

from .embeddings import ControlPatchEmbed

logger = logging.get_logger(__name__)

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class Connector(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.num_attention_heads = 48
        self.attention_head_dim = hidden_size // self.num_attention_heads
        self.h_to_q = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.SiLU(),
            nn.Linear(512, hidden_size)
        )
        self.h_to_k = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.SiLU(),
            nn.Linear(512, hidden_size)
        )
        self.norm_q = nn.LayerNorm(self.attention_head_dim, eps=1e-6)
        self.norm_k = nn.LayerNorm(self.attention_head_dim, eps=1e-6)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.SiLU(),
            zero_module(nn.Linear(512, hidden_size))
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.SiLU(),
            zero_module(nn.Linear(512, hidden_size))
        )

    def forward(self, c, h):
        q, k, v = self.h_to_q(h), self.h_to_k(c), c
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_attention_heads), (q, k, v))
        q, k = self.norm_q(q), self.norm_k(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=self.num_attention_heads)
        h = h + self.out_layer(out) + self.c_mlp(c)

        return h


class CogVideoXVRTransformer(CogVideoXTransformer3DModel):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        enable_connector: bool = False,
        enable_control_patchemb: bool = False,
    ):
        super().__init__(
            num_attention_heads, attention_head_dim, in_channels, out_channels,
            flip_sin_to_cos, freq_shift, time_embed_dim, ofs_embed_dim, text_embed_dim, num_layers,
            dropout, attention_bias, sample_width, sample_height, sample_frames,
            patch_size, patch_size_t, temporal_compression_ratio, max_text_seq_length,
            activation_fn, timestep_activation_fn, norm_elementwise_affine, norm_eps,
            spatial_interpolation_scale, temporal_interpolation_scale, use_rotary_positional_embeddings,
            use_learned_positional_embeddings, patch_bias
        )
        inner_dim = num_attention_heads * attention_head_dim

        self.connectors = None
        if enable_connector:
            self.connectors = nn.ModuleList(
                [Connector(inner_dim) for _ in range(num_layers)]
            )

        self.control_patch_embed = None
        if enable_control_patchemb:
            self.control_patch_embed = ControlPatchEmbed(
                    patch_size=patch_size,
                    patch_size_t=patch_size_t,
                    in_channels=in_channels,
                    embed_dim=inner_dim,
                    text_embed_dim=text_embed_dim,
                    time_embed_dim=time_embed_dim
                )

        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        control_hidden_states: List[torch.Tensor] = None,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)


        if self.ofs_embedding is not None and ofs is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        if self.control_patch_embed is not None:
            hidden_states, control_states = hidden_states.split(channels // 2, dim=2)
            hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
            hidden_states = self.embedding_dropout(hidden_states)
            hidden_states = self.control_patch_embed(control_states, 
                                                     encoder_hidden_states, 
                                                     hidden_states, 
                                                     emb)
        else:
            hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
            hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )
            if control_hidden_states is not None and self.connectors is not None:
                control_interval = math.ceil(len(self.transformer_blocks) / len(control_hidden_states))
                hidden_states = self.connectors[i](control_hidden_states[i // control_interval], 
                                                   hidden_states).to(hidden_states.dtype)
                

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t
        output = hidden_states.reshape(
            batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
        )
        output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
