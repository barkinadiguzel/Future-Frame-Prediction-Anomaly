import torch
import torch.nn as nn

class IntensityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        return self.criterion(pred, target)
