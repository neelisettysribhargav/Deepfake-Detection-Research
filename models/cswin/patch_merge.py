import torch
import torch.nn as nn
import numpy as np

class Merge_Block(nn.Module):
    """
    Down-sampling block (Conv S=2):
    Doubles channel count (dim -> dim_out = 2 * dim) and reduces spatial resolution
    by half (H -> H / 2, W -> W / 2) using strided convolution (stride=2, kernel_size=3, padding=1),
    followed by Layer Normalization as described in the paper Section 2.2.1 (b).
    """
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        return x
