import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear requires nn.Linear.")
        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.zeros(rank, base_layer.in_features))
        self.B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.A), self.B) * self.scaling
        return base_out + lora_out

def apply_lora_to_linear_layers(module, rank=8, alpha=16, dropout=0.0, target_keywords=("qkv", "proj", "fc1", "fc2")):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(k in name.lower() for k in target_keywords):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        else:
            apply_lora_to_linear_layers(child, rank, alpha, dropout, target_keywords)

def mark_only_lora_trainable(model):
    for n, p in model.named_parameters():
        p.requires_grad = ("A" in n or "B" in n or "head" in n or "decoder" in n or "classifier" in n or "uncertainty" in n)
