import torch.nn as nn
from blocks.discriminator_block import Discriminator as DBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = DBlock(in_channels)

    def forward(self, x):
        return self.model(x)
