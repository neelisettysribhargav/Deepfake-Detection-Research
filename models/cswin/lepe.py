import torch
import torch.nn as nn

class LocallyEnhancedPositionalEncoding(nn.Module):
    """
    Locally-enhanced Positional Encoding (LePE):
    Implemented by a depthwise convolution operating on the Value (V) tokens.
    """
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return self.conv(x)
