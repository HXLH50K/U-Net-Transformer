# from .unet import Unet
from .ddrnet.DDRNet import DualResNet
from .utransformer.U_Transformer import U_Transformer

from typing import Optional
import torch


def create_model(
    arch: str,
    in_channels: int = 3,
    classes: int = 1,
    **kwargs,
) -> torch.nn.Module:
    """Models wrapper. Allows to create any model just with parametes

    """

    archs = [DualResNet, U_Transformer]
    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError(
            "Wrong architecture type `{}`. Avalibale options are: {}".format(
                arch,
                list(archs_dict.keys()),
            ))
    return model_class(
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )
