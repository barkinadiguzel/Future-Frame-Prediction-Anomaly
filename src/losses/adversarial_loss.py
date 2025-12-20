import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def d_loss(self, pred_real, pred_fake):
        real_loss = self.mse(pred_real, torch.ones_like(pred_real))
        fake_loss = self.mse(pred_fake, torch.zeros_like(pred_fake))

        return 0.5 * (real_loss + fake_loss)

    def g_loss(self, pred_fake):
        return 0.5 * self.mse(pred_fake, torch.ones_like(pred_fake))
