import torch
import torch.nn as nn

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # vertical gradients
        pred_dx = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_dx = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        # horizontal gradients
        pred_dy = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        loss_x = torch.mean(torch.abs(pred_dx - target_dx))
        loss_y = torch.mean(torch.abs(pred_dy - target_dy))

        return loss_x + loss_y
