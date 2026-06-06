from pathlib import Path
import torch
from torch.utils.data import Dataset
from .common import read_rgb, read_mask, list_images, split_paths

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, split="train", seed=42):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        all_images = list_images(self.image_dir)
        paired = []
        for img_path in all_images:
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                mask_path = self.mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                paired.append((img_path, mask_path))
        if not paired:
            raise RuntimeError("No image-mask pairs found.")
        tr, va, te = split_paths([p[0] for p in paired], seed=seed)
        index_map = {"train": tr, "val": va, "test": te}
        self.pairs = [paired[i] for i in index_map[split]]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = read_rgb(img_path)
        mask = read_mask(mask_path)
        if self.transform:
            out = self.transform(image=img, mask=mask)
            img = out["image"]
            mask = out["mask"]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return img.float(), mask.float()
