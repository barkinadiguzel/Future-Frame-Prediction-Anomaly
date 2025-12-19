import torch.nn as nn
from layers.deconv_block import DeconvBlock


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dec4 = DeconvBlock(512, 512, 256)
        self.dec3 = DeconvBlock(256, 256, 128)
        self.dec2 = DeconvBlock(128, 128, 64)
        self.dec1 = DeconvBlock(64, 64, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, skips):
        s1, s2, s3, s4 = skips

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        x = self.final(x)
        return self.tanh(x)
