import torch


class CogVideoXUnifyAttnProcessor2_0:

    # attn_func should processes in [B, H, N, D]
    def __init__(self, attn_func=None):
        if attn_func is None:
            attn_func = torch.nn.functional.scaled_dot_product_attention
        self.attn_func = attn_func

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = self.attn_func(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


# [TODO] support sage with parallel ring attention simultaneously
try:
    from sageattention import sageattn
    q = k = v = torch.randn(2, 48, 16, 64).to('cuda', dtype=torch.bfloat16)
    sageattn(q, k, v, is_causal=False)
    def sage_attn_wrapper(q, k, v, **kwargs):
        return sageattn(q, k, v, **kwargs)
    SAGE_AVAILABLE = True
except Exception as e:
    print(f"sageattention is not available with error: {e}")
    sage_attn_wrapper = None
    SAGE_AVAILABLE = False

try:
    from flash_attn_interface import flash_attn_func
    FA3_AVAILABLE = True
except Exception as e:
    print(f"fa3 is not available with error: {e}")
    FA3_AVAILABLE = False


if not FA3_AVAILABLE:
    flash_attn_v3_wrapper = None
else:

    def flash_attn_v3_wrapper(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    ):
        assert attn_mask is None
        assert dropout_p == 0.0
        # FA processes in [B, N, H, D]
        original_dtype = q.dtype
        hidden_states, *_ = flash_attn_func(
            q.transpose(1, 2).to(dtype=torch.bfloat16),
            k.transpose(1, 2).to(dtype=torch.bfloat16),
            v.transpose(1, 2).to(dtype=torch.bfloat16),
            causal=is_causal,
        )
        hidden_states = hidden_states.transpose(1, 2).to(dtype=original_dtype)
        return hidden_states


def optimize_transformer(transformer, attn_type='fa3'):
    if attn_type == 'fa3' and not FA3_AVAILABLE:
        print("fa3 is not available, skip optimizing")
        return transformer
    if attn_type == 'sage' and not SAGE_AVAILABLE:
        print("sageattention is not available, skip optimizing")
        return transformer
    attn_fn = {
        'fa3': flash_attn_v3_wrapper,
        'sage': sage_attn_wrapper
    }[attn_type]
    print(f"optimizing transformer with {attn_type}")

    for block in transformer.transformer_blocks:
        if not hasattr(block.attn1.processor, 'attn_func'):
            block.attn1.processor = CogVideoXUnifyAttnProcessor2_0()

        block.attn1.processor.attn_func = attn_fn

    return transformer
