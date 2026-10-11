import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Image Patching and Token Embedding:
    Divides input image (B, C, H, W) into non-overlapping patch embeddings of dimension embed_dim
    using a 7x7 convolution with stride 4 and padding 2, followed by LayerNorm.
    For 256x256 images, spatial dimensions become 64x64 token embeddings.
    """
    def __init__(self, in_chans=3, embed_dim=64, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=patch_size, padding=2)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)
        # flatten to (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
