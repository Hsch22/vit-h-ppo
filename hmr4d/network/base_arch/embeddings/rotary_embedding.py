import torch
import torch.nn as nn
from einops import repeat


def rotate_half(x):
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2, cos=None, sin=None):
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]
        if cos is not None and sin is not None:
            cos = cos[-seq_len:]
            sin = sin[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    if cos is None or sin is None:
        freqs_f32 = freqs.to(device=t.device, dtype=torch.float32)
        cos = freqs_f32.cos()
        sin = freqs_f32.sin()
    cos = cos.to(device=t.device, dtype=t.dtype)
    sin = sin.to(device=t.device, dtype=t.dtype)
    if scale != 1.0:
        cos = cos * scale
        sin = sin * scale

    if start_index == 0 and end_index == t.shape[-1]:
        return (t * cos) + (rotate_half(t) * sin)
    t_left, t_mid, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t_mid = (t_mid * cos) + (rotate_half(t_mid) * sin)
    return torch.cat((t_left, t_mid, t_right), dim=-1)


def get_encoding(d_model, max_seq_len=4096):
    """Return: (L, D)"""
    t = torch.arange(max_seq_len).float()
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    freqs = torch.einsum("i, j -> i j", t, freqs)
    freqs = repeat(freqs, "i j -> i (j r)", r=2)
    return freqs


class ROPE(nn.Module):
    """Minimal impl of a lang-style positional encoding."""

    def __init__(self, d_model, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Pre-cache freqs, cos, and sin as non-persistent buffers (preserves state_dict).
        encoding = get_encoding(d_model, max_seq_len)
        self.register_buffer("encoding", encoding, False)
        self.register_buffer("cos_cached", encoding.cos(), False)
        self.register_buffer("sin_cached", encoding.sin(), False)

    def rotate_queries_or_keys(self, x):
        """
        Args:
            x : (B, H, L, D)
        Returns:
            rotated_x: (B, H, L, D)
        """

        seq_len, d_model = x.shape[-2:]
        assert d_model == self.d_model

        if seq_len > self.max_seq_len:
            encoding = get_encoding(d_model, seq_len).to(device=x.device, dtype=torch.float32)
            cos = encoding.cos()
            sin = encoding.sin()
        else:
            encoding = self.encoding[:seq_len]
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]

        return apply_rotary_emb(encoding, x, seq_dim=-2, cos=cos, sin=sin)
