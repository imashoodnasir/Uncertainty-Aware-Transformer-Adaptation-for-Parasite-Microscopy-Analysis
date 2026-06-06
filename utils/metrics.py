import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def classification_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:
                metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except Exception:
            metrics["auc"] = float("nan")
    return metrics

def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2 * inter + eps) / (denom + eps)).mean().item()

def iou_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()

def expected_calibration_error(probs, labels, n_bins=15):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
    if probs.ndim == 2:
        conf, pred = probs.max(dim=1)
        correct = pred.eq(labels.long())
    else:
        conf = probs.reshape(-1)
        pred = (conf > 0.5).long()
        labels = labels.reshape(-1).long()
        correct = pred.eq(labels)
    ece = torch.zeros(1)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = conf.gt(lo.item()) * conf.le(hi.item())
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc = correct[in_bin].float().mean()
            avg_conf = conf[in_bin].mean()
            ece += torch.abs(avg_conf - acc) * prop
    return ece.item()

def brier_score(probs, labels):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
    if probs.ndim == 2:
        y = torch.nn.functional.one_hot(labels.long(), probs.shape[1]).float()
        return torch.mean(torch.sum((probs - y) ** 2, dim=1)).item()
    return torch.mean((probs.reshape(-1) - labels.reshape(-1).float()) ** 2).item()
