"""SRCNN (Dong et al., 2014) -- the classic 3-layer SR baseline.

Note: SRCNN operates on an image already upsampled to the target size, so it
expects the LR input to be pre-upscaled (bicubic) to HR resolution. The
DataPipeline handles this when ``pre_upscale: true`` is set in the dataset cfg.
"""

import torch.nn as nn

from . import MODELS


@MODELS.register("SRCNN")
class SRCNN(nn.Module):
    def __init__(self, num_channels: int = 3, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.features(x)
