import torch
import torch.nn as nn

from src.model.generator import Generator
from src.model.discriminator import Discriminator


class FutureFrameModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        past_frames=4,
        feature_channels=[64, 128, 256, 512],
        disc_base_channels=64
    ):
        super().__init__()

        # Generator (U-Net)
        self.generator = Generator(
            in_channels=in_channels * past_frames,
            feature_channels=feature_channels,
            out_channels=in_channels
        )

        # Discriminator (PatchGAN)
        self.discriminator = Discriminator(
            in_channels=in_channels,
            base_channels=disc_base_channels
        )

    def forward(self, past_frames, future_frame=None):
        # Predict future frame
        pred_frame = self.generator(past_frames)

        # Test mode (no discriminator)
        if future_frame is None:
            return pred_frame

        # Training mode
        pred_score = self.discriminator(pred_frame)
        real_score = self.discriminator(future_frame)

        return pred_frame, pred_score, real_score
