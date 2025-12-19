import torch
import torch.nn as nn

class OpticalFlowLoss(nn.Module):
    def __init__(self, flownet):
        super().__init__()
        self.flownet = flownet
        for p in self.flownet.parameters():
            p.requires_grad = False

        self.l1 = nn.L1Loss()

    def forward(self, pred_next, true_next, current_frame):
        # predicted flow
        flow_pred = self.flownet(pred_next, current_frame)

        # ground truth flow
        flow_true = self.flownet(true_next, current_frame)

        return self.l1(flow_pred, flow_true)
