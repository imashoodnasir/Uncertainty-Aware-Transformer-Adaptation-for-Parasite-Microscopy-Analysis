import torch
import torch.nn as nn
from .encoder import ViTEncoder
from .heads import ClassificationHead, SimpleSegmentationDecoder

class ParasiteFramework(nn.Module):
    def __init__(self, cfg, task="malaria_cls", pretrained=True, use_lora=True):
        super().__init__()
        self.task = task
        self.encoder = ViTEncoder(
            backbone=cfg["backbone"],
            pretrained=pretrained,
            lora=use_lora,
            rank=cfg["lora_rank"],
            alpha=cfg["lora_alpha"],
            dropout=cfg["dropout"]
        )
        dim = self.encoder.feature_dim
        if task == "malaria_cls":
            self.head = ClassificationHead(dim, cfg["num_classes_malaria"], cfg["dropout"])
        elif task == "chula_cls":
            self.head = ClassificationHead(dim, cfg["num_classes_chula"], cfg["dropout"])
        elif task == "bbbc_seg":
            self.head = SimpleSegmentationDecoder(dim, cfg["image_size"], cfg["patch_size"])
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x):
        if self.task in ["malaria_cls", "chula_cls"]:
            feat = self.encoder(x)
            return self.head(feat)
        tokens = self.encoder.forward_tokens(x)
        return self.head(tokens)
