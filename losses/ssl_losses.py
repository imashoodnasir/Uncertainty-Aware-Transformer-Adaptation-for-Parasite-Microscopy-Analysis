import torch
import torch.nn.functional as F

def info_nce_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

def domain_alignment_loss(features, domain_labels=None):
    if domain_labels is None:
        return features.new_tensor(0.0)
    unique = torch.unique(domain_labels)
    if len(unique) < 2:
        return features.new_tensor(0.0)
    means = []
    for u in unique:
        means.append(features[domain_labels == u].mean(dim=0))
    loss = 0.0
    count = 0
    for i in range(len(means)):
        for j in range(i+1, len(means)):
            loss = loss + F.mse_loss(means[i], means[j])
            count += 1
    return loss / max(count, 1)
