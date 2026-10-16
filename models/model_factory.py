import os
import sys
import torch
import torch.nn as nn

try:
    from models.cswin import (
        CSWin_64_24322_small_256,
        cswin_paper_faithful_256,
        CSWin_64_24322_small_224,
    )
except ImportError:
    from .cswin import (
        CSWin_64_24322_small_256,
        cswin_paper_faithful_256,
        CSWin_64_24322_small_224,
    )

def get_model(
    name,
    num_classes=1,
    pretrained=False,
    checkpoint=None,
    drop_rate=0.1,
    **kwargs
):
    """
    Model factory matching the paper's CSWin Transformer architecture and comparison baselines.
    Supports:
      - CSWin_64_24322_small_256: Official CSWin-Small 256 (supports act='gelu'|'relu', block_drop_rate, head_drop_rate)
      - cswin_paper_faithful_256: Paper-faithful CSWin-Small 256 (Page 6: ReLU + Additive Pos Embed)
      - convnext_tiny: Modern CNN baseline (Torchvision)
      - inception_v3: Paper Table 6 comparison baseline (Torchvision)
      - xception: Paper Table 6 comparison baseline (timm)
    """
    if name == "CSWin_64_24322_small_256":
        model = CSWin_64_24322_small_256(
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            **kwargs
        )
    elif name in ["cswin_paper_faithful", "cswin_paper_faithful_256"]:
        model = cswin_paper_faithful_256(
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            **kwargs
        )
    elif name == "CSWin_64_24322_small_224":
        model = CSWin_64_24322_small_224(
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            **kwargs
        )
    elif name == "convnext_tiny":
        import torchvision.models as tv
        weights = tv.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = tv.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif name == "inception_v3":
        import torchvision.models as tv
        if pretrained:
            model = tv.inception_v3(weights=tv.Inception_V3_Weights.DEFAULT)
            model.aux_logits = False
            model.AuxLogits = None
        else:
            model = tv.inception_v3(weights=None, aux_logits=False, init_weights=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "xception":
        import importlib
        try:
            timm_mod = importlib.import_module("timm")
            model = timm_mod.create_model("xception", pretrained=pretrained, num_classes=num_classes)
        except ImportError:
            raise ImportError(
                "The 'timm' package is required for Xception. "
                "Please run: pip install timm"
            )
    else:
        raise ValueError(f"Unknown model architecture: {name}")

    if pretrained and checkpoint:
        print(f"Loading pretrained weights from {checkpoint}")
        checkpoint_data = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False
        )

        state_dict = checkpoint_data.get("state_dict_ema", checkpoint_data.get("model_state_dict", checkpoint_data))

        # Safely remove classification head if classes count mismatch
        head_weight = state_dict.get("head.weight")
        if head_weight is not None and head_weight.shape[0] != num_classes:
            print(f"Adapting head weights: checkpoint has {head_weight.shape[0]} classes, target has {num_classes}")
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Pretrained weights loaded with status: {msg}")

    return model