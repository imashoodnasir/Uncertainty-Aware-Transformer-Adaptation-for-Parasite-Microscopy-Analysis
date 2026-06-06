import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class SimpleSegmentationDecoder(nn.Module):
    def __init__(self, in_dim, image_size=224, patch_size=16, hidden=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        grid = image_size // patch_size
        self.grid = grid
        self.proj = nn.Linear(in_dim, hidden)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.GELU()
        )
        self.mask_head = nn.Conv2d(16, 1, 1)
        self.var_head = nn.Conv2d(16, 1, 1)

    def forward(self, tokens):
        if tokens.ndim == 2:
            b, d = tokens.shape
            tokens = tokens.unsqueeze(1).repeat(1, self.grid * self.grid, 1)
        if tokens.shape[1] == self.grid * self.grid + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != self.grid * self.grid:
            tokens = tokens[:, -self.grid * self.grid:, :]
        x = self.proj(tokens)
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, self.grid, self.grid)
        x = self.decoder(x)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mask_logits = self.mask_head(x)
        log_var = self.var_head(x).clamp(-6, 6)
        return mask_logits, log_var
