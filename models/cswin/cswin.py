import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
import torch.utils.checkpoint as checkpoint

from models.cswin.attention import LePEAttention, img2windows, windows2img
from models.cswin.block import Mlp, CSWinBlock, DropPath
from models.cswin.patch_merge import Merge_Block
from models.cswin.patch_embed import PatchEmbed

# Native ImageNet constants and helpers (eliminates hard external timm dependency for IDE linters)
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization built into PyTorch."""
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

def to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)

_model_registry = {}

def register_model(fn):
    """Model registration decorator compatible with timm."""
    _model_registry[fn.__name__] = fn
    return fn

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stage1_conv_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'cswin_224': _cfg(input_size=(3, 224, 224)),
    'cswin_256': _cfg(input_size=(3, 256, 256)),
    'cswin_384': _cfg(input_size=(3, 384, 384), crop_pct=1.0),
}

class CSWinTransformer(nn.Module):
    """
    CSWin Transformer Backbone for Deepfake Detection.
    Matches the architecture specified in:
    "Building an efficient Deep Fake detection system using the recognition
    capabilities of convolutional neural networks and transformers" (Discover Computing 2025).

    Key architectural components:
    1. Patch Embedding: Conv 7x7 stride 4 -> (H/4, W/4, C)
    2. 4 Stages of Cross-Shaped Window Self-Attention (CSWSA) blocks
    3. Token Down-sampling Blocks: Conv 3x3 stride 2 (Conv S=2) doubling channels and halving resolution
    4. Global Average Pooling (GAP) Layer
    5. Regularization Dropout Layer (0.1 to 0.5) to prevent overfitting
    6. Classification Head (single logit for Binary Cross-Entropy)
    """
    def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=1, embed_dim=64,
                 depth=[2, 4, 32, 2], split_size=[1, 2, 8, 8], num_heads=[2, 4, 8, 16],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, use_chk=False,
                 act='gelu', block_drop_rate=None, head_drop_rate=None, use_pos_embed=False):
        super().__init__()
        # Variant options (defaults reproduce the current model exactly):
        #   act: 'gelu' (official CSWin) or 'relu' (paper text, Section 2.2.1 b)
        #   block_drop_rate: dropout inside every block's MLP (None -> drop_rate)
        #   head_drop_rate:  dropout before the classifier   (None -> drop_rate)
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[str(act).lower()]
        block_drop_rate = drop_rate if block_drop_rate is None else block_drop_rate
        head_drop_rate = drop_rate if head_drop_rate is None else head_drop_rate
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.img_size = img_size
        self.use_pos_embed = use_pos_embed
        heads = num_heads

        # Stage 1: Conv embedding (H/4, W/4)
        self.stage1_conv_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            norm_layer=norm_layer
        )

        if self.use_pos_embed:
            num_patches = (img_size // patch_size) * (img_size // patch_size)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]

        # Stage 1 CSWin blocks
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=block_drop_rate, attn_drop=attn_drop_rate, act_layer=act_layer,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        # Stage 2: Merge 1 (Conv S=2) + CSWin blocks
        self.merge1 = Merge_Block(curr_dim, curr_dim * 2, norm_layer=norm_layer)
        curr_dim = curr_dim * 2
        self.stage2 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=block_drop_rate, attn_drop=attn_drop_rate, act_layer=act_layer,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
            for i in range(depth[1])])

        # Stage 3: Merge 2 (Conv S=2) + CSWin blocks
        self.merge2 = Merge_Block(curr_dim, curr_dim * 2, norm_layer=norm_layer)
        curr_dim = curr_dim * 2
        self.stage3 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=block_drop_rate, attn_drop=attn_drop_rate, act_layer=act_layer,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
            for i in range(depth[2])])

        # Stage 4: Merge 3 (Conv S=2) + CSWin blocks
        self.merge3 = Merge_Block(curr_dim, curr_dim * 2, norm_layer=norm_layer)
        curr_dim = curr_dim * 2
        self.stage4 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=block_drop_rate, attn_drop=attn_drop_rate, act_layer=act_layer,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
            for i in range(depth[-1])])

        # Normalization layer
        self.norm = norm_layer(curr_dim)

        # Global Average Pooling Layer (Section 2.2.1 d)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Dropout layer to prevent overfitting (Section 3.1: dropout rate between 0.5 and 0.1)
        self.head_drop = nn.Dropout(head_drop_rate) if head_drop_rate > 0. else nn.Identity()

        # Classification Head (Section 2.2.1 d)
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.head.weight, std=0.02) if isinstance(self.head, nn.Linear) else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        curr_dim = self.head.in_features if isinstance(self.head, nn.Linear) else self.num_features
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        """
        Extracts multi-stage features and applies Global Average Pooling.
        Input x: (B, 3, H, W)
        Output: (B, C)
        """
        x = self.stage1_conv_embed(x)
        if self.use_pos_embed and self.pos_embed is not None:
            B, H, W, C = x.shape
            x = (x.view(B, H * W, C) + self.pos_embed).view(B, H, W, C)

        for blk in self.stage1:
            x = checkpoint.checkpoint(blk, x) if self.use_chk else blk(x)

        for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
                               [self.stage2, self.stage3, self.stage4]):
            x = pre(x)
            for blk in blocks:
                x = checkpoint.checkpoint(blk, x) if self.use_chk else blk(x)

        x = self.norm(x)
        # Global Average Pooling: average across all spatial tokens (B, L, C) -> (B, C)
        x = torch.mean(x, dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x


# Model Registration Functions

@register_model
def CSWin_64_24322_small_256(pretrained=False, **kwargs):
    """
    CSWin-Small configured for 256x256 resolution as described in the paper:
    - 35 million parameters (34.13M)
    - 256x256 image input resolution (Section 2.1)
    - Depth: [2, 4, 32, 2]
    - Embed Dim: 64
    - Split Size: [1, 2, 8, 8] (exact mathematical alignment for 256x256 stages: 64, 32, 16, 8)
    - Heads: [2, 4, 8, 16]
    Supports:
    - act: 'gelu' or 'relu'
    - block_drop_rate: float
    - head_drop_rate: float
    """
    model = CSWinTransformer(
        img_size=256,
        patch_size=4,
        embed_dim=64,
        depth=[2, 4, 32, 2],
        split_size=kwargs.pop('split_size', [1, 2, 8, 8]),
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.,
        **kwargs
    )
    model.default_cfg = default_cfgs['cswin_256']
    return model

@register_model
def cswin_paper_faithful_256(pretrained=False, **kwargs):
    """
    Paper-faithful variant strictly matching Section 2.2.1(b) / Page 6 of Discover Computing (2025):
    - ReLU activation in Feed-Forward MLP layers.
    - Additive 2D positional encodings on patch token embeddings.
    - 256x256 native input resolution.
    - Depth: [2, 4, 32, 2], Heads: [2, 4, 8, 16], Embed Dim: 64.
    """
    kwargs.setdefault('act', 'relu')
    kwargs.setdefault('use_pos_embed', True)
    model = CSWinTransformer(
        img_size=256,
        patch_size=4,
        embed_dim=64,
        depth=[2, 4, 32, 2],
        split_size=kwargs.pop('split_size', [1, 2, 8, 8]),
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.,
        **kwargs
    )
    model.default_cfg = default_cfgs['cswin_256']
    return model

@register_model
def CSWin_64_24322_small_224(pretrained=False, **kwargs):
    """
    CSWin-Small standard 224x224 baseline variant.
    """
    model = CSWinTransformer(
        img_size=224,
        patch_size=4,
        embed_dim=64,
        depth=[2, 4, 32, 2],
        split_size=[1, 2, 7, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.,
        **kwargs
    )
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_64_12211_tiny_224(pretrained=False, **kwargs):
    model = CSWinTransformer(
        img_size=224, patch_size=4, embed_dim=64, depth=[1, 2, 21, 1],
        split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_96_24322_base_224(pretrained=False, **kwargs):
    model = CSWinTransformer(
        img_size=224, patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
        split_size=[1, 2, 7, 7], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_144_24322_large_224(pretrained=False, **kwargs):
    model = CSWinTransformer(
        img_size=224, patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
        split_size=[1, 2, 7, 7], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_96_24322_base_384(pretrained=False, **kwargs):
    model = CSWinTransformer(
        img_size=384, patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
        split_size=[1, 2, 12, 12], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model

@register_model
def CSWin_144_24322_large_384(pretrained=False, **kwargs):
    model = CSWinTransformer(
        img_size=384, patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
        split_size=[1, 2, 12, 12], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model
