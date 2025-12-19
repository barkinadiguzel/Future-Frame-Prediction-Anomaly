import torch.nn as nn
from layers.conv_block import ConvBlock


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 64, use_pool=True)
        self.enc2 = ConvBlock(64, 128, use_pool=True)
        self.enc3 = ConvBlock(128, 256, use_pool=True)
        self.enc4 = ConvBlock(256, 512, use_pool=True)

        # bottleneck (no pooling)
        self.enc5 = ConvBlock(512, 512, use_pool=False)

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        s5, x = self.enc5(x)

        skips = [s1, s2, s3, s4]
        return x, skips
