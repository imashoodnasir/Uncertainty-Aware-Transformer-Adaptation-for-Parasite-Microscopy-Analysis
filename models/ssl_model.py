import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ViTEncoder

class SSLMicroscopyModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.encoder = ViTEncoder(cfg["backbone"], pretrained=pretrained, lora=False)
        dim = self.encoder.feature_dim
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.reconstructor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3 * cfg["image_size"] * cfg["image_size"])
        )
        self.image_size = cfg["image_size"]

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x1, x2=None):
        z1 = self.encode(x1)
        p1 = F.normalize(self.projector(z1), dim=1)
        recon = self.reconstructor(z1).reshape(x1.shape[0], 3, self.image_size, self.image_size)
        if x2 is None:
            return z1, p1, recon
        z2 = self.encode(x2)
        p2 = F.normalize(self.projector(z2), dim=1)
        return z1, z2, p1, p2, recon
