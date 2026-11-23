"""2D U-Net (Ronneberger et al., 2015) built with MONAI."""
from __future__ import annotations

from monai.networks.nets import UNet


def build_unet(channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units: int = 2) -> UNet:
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,          # one logit per pixel: tumor vs background
        channels=tuple(channels),
        strides=tuple(strides),
        num_res_units=num_res_units,
    )
