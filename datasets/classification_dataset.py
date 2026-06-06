from pathlib import Path
import torch
from torch.utils.data import Dataset
from .common import read_rgb, list_images, split_paths

class FolderClassificationDataset(Dataset):
    def __init__(self, root, transform=None, split="train", seed=42, selected_indices=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not self.classes:
            raise RuntimeError(f"No class folders found in {root}")
        class_to_idx = {c:i for i,c in enumerate(self.classes)}
        paths, labels = [], []
        for c in self.classes:
            for p in list_images(self.root / c):
                paths.append(p)
                labels.append(class_to_idx[c])
        if selected_indices is None:
            tr, va, te = split_paths(paths, labels, seed=seed)
            index_map = {"train": tr, "val": va, "test": te}
            selected_indices = index_map[split]
        self.paths = [paths[i] for i in selected_indices]
        self.labels = [labels[i] for i in selected_indices]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = read_rgb(self.paths[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img.float(), torch.tensor(label).long()
