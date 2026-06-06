from pathlib import Path
from torch.utils.data import Dataset
from .common import read_rgb, list_images

class SSLMicroscopyDataset(Dataset):
    def __init__(self, roots, transform_a=None, transform_b=None):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.paths = []
        for r in roots:
            self.paths.extend(list_images(r))
        if not self.paths:
            raise RuntimeError("No SSL images found.")
        self.transform_a = transform_a
        self.transform_b = transform_b if transform_b is not None else transform_a

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = read_rgb(self.paths[idx])
        a = self.transform_a(image=img)["image"]
        b = self.transform_b(image=img)["image"]
        return a.float(), b.float()
