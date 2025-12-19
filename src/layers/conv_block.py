import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=True):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2) if use_pool else None

    def forward(self, x):
        x = self.conv(x)
        if self.pool is not None:
            return x, self.pool(x)   # skip + downsampled
        return x, x
