import torch.nn as nn
from blocks.encoder_block import Encoder
from blocks.decoder_block import Decoder


class Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        out = self.decoder(bottleneck, skips)
        return out
