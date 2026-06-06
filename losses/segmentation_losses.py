import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1,2,3))
        denom = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - dice.mean()

def boundary_loss(logits, target):
    prob = torch.sigmoid(logits)
    dx_p = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1]).mean()
    dy_p = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :]).mean()
    dx_t = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1]).mean()
    dy_t = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :]).mean()
    return torch.abs(dx_p - dx_t) + torch.abs(dy_p - dy_t)

def heteroscedastic_seg_loss(logits, log_var, target):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    precision = torch.exp(-log_var)
    return (precision * bce + log_var).mean()
