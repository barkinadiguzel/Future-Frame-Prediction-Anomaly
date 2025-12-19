import torch
import torch.nn as nn
from src.blocks.encoder_block import EncoderBlock
from src.blocks.decoder_block import DecoderBlock

class FutureFramePredictor(nn.Module):
    def __init__(self, in_channels=3, feature_channels=[64, 128, 256, 512]):
        super().__init__()

        # Encoder
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for out_channels in feature_channels:
            self.encoders.append(EncoderBlock(prev_channels, out_channels))
            prev_channels = out_channels

        # Decoder
        self.decoders = nn.ModuleList()
        reversed_channels = feature_channels[::-1]
        for i in range(len(reversed_channels)-1):
            self.decoders.append(
                DecoderBlock(reversed_channels[i], reversed_channels[i+1])
            )
        # Final layer to map to RGB
        self.final_conv = nn.Conv2d(reversed_channels[-1], in_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward
        skips = []
        out = x
        for enc in self.encoders:
            out, skip = enc(out)
            skips.append(skip)

        # Decoder forward with skip connections
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i+2)]  # reverse order
            out = dec(out, skip)

        # Final predicted frame
        pred = self.final_conv(out)
        return pred
