import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def img2windows(img, H_sp, W_sp):
    """
    Splits image feature map into stripes/windows with automatic padding.
    Input:
        img: (B, C, H, W)
        H_sp: stripe height
        W_sp: stripe width
    Returns:
        img_perm: (-1, H_sp * W_sp, C)
        padding: (pad_h, pad_w)
    """
    B, C, H, W = img.shape
    pad_h = (H_sp - H % H_sp) % H_sp
    pad_w = (W_sp - W % W_sp) % W_sp
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h))
    _, _, H_pad, W_pad = img.shape
    img_reshape = img.view(B, C, H_pad // H_sp, H_sp, W_pad // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm, (pad_h, pad_w)

def windows2img(img_splits_hw, H_sp, W_sp, H, W, padding=(0, 0)):
    """
    Reconstructs image feature map from stripes/windows and removes padding.
    Input:
        img_splits_hw: (-1, H_sp * W_sp, C)
        H_sp: stripe height
        W_sp: stripe width
        H, W: original feature map dimensions before padding
        padding: (pad_h, pad_w)
    Returns:
        img: (B, H, W, C)
    """
    pad_h, pad_w = padding
    H_pad = H + pad_h
    W_pad = W + pad_w
    B = int(img_splits_hw.shape[0] / (H_pad * W_pad / H_sp / W_sp))
    img = img_splits_hw.view(B, H_pad // H_sp, W_pad // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)
    if pad_h > 0 or pad_w > 0:
        img = img[:, :H, :W, :].contiguous()
    return img

class LePEAttention(nn.Module):
    """
    Cross-Shaped Window Self-Attention (CSWSA) with Locally-enhanced Positional Encoding (LePE).
    Computes self-attention along horizontal or vertical stripes, lowering time complexity from
    O(N^2) to O(N*sqrt(N)).
    """
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == -1:
            # Full window attention (last stage)
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            # Horizontal stripe attention
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            # Vertical stripe attention
            W_sp, H_sp = self.resolution, self.split_size
        else:
            raise ValueError(f"Unknown attention stripe index mode: {idx}")

        self.H_sp = H_sp
        self.W_sp = W_sp

        # LePE: depthwise convolution operating on V
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x, pad = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x, pad

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        H_sp, W_sp = self.H_sp, self.W_sp

        pad_h = (H_sp - H % H_sp) % H_sp
        pad_w = (W_sp - W % W_sp) % W_sp
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        _, _, H_pad, W_pad = x.shape
        x = x.view(B, C, H_pad // H_sp, H_sp, W_pad // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)

        lepe = func(x)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        Input qkv: (3, B, L, C)
        Returns: (B, L, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, f"flatten img_tokens length {L} != resolution^2 ({H}x{W})"

        q, pad = self.im2cswin(q)
        k, _ = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        dropout_p = self.attn_drop.p if self.training else 0.0
        attn_out = nn.functional.scaled_dot_product_attention(
            q, k, v, scale=self.scale, dropout_p=dropout_p
        )
        x = attn_out + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)

        # Window to Image with unpadding
        x = windows2img(x, self.H_sp, self.W_sp, H, W, pad).view(B, -1, C)
        return x
