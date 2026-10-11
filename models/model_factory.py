import torch
from models.cswin import (
    CSWin_64_24322_small_256,
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
    Model factory matching the paper's CSWin Transformer architecture.
    Default model is CSWin_64_24322_small_256 for 256x256 resolution.
    """
    if name == "CSWin_64_24322_small_256":
        model = CSWin_64_24322_small_256(
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
    else:
        raise ValueError(f"Unknown model architecture: {name}")

    if pretrained and checkpoint:
        print(f"Loading pretrained weights from {checkpoint}")
        checkpoint_data = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True
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