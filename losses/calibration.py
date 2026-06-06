import torch
import torch.nn.functional as F

def differentiable_brier_loss(logits, labels):
    probs = F.softmax(logits, dim=1)
    y = F.one_hot(labels.long(), probs.shape[1]).float()
    return torch.mean(torch.sum((probs - y) ** 2, dim=1))

def segmentation_brier_loss(logits, target):
    prob = torch.sigmoid(logits)
    return torch.mean((prob - target) ** 2)
