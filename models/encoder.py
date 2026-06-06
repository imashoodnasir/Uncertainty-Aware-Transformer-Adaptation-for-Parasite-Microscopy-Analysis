import torch
import torch.nn as nn
import timm
from .lora import apply_lora_to_linear_layers, mark_only_lora_trainable

class ViTEncoder(nn.Module):
    def __init__(self, backbone="vit_small_patch16_224", pretrained=True, lora=False, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        if lora:
            for p in self.backbone.parameters():
                p.requires_grad = False
            apply_lora_to_linear_layers(self.backbone, rank=rank, alpha=alpha, dropout=dropout)
            mark_only_lora_trainable(self)

    def forward(self, x):
        return self.backbone(x)

    def forward_tokens(self, x):
        if hasattr(self.backbone, "forward_features"):
            z = self.backbone.forward_features(x)
            if z.ndim == 3:
                return z
        feat = self.forward(x)
        return feat.unsqueeze(1)
