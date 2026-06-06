from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (m > 0).astype("float32")

def list_images(folder):
    folder = Path(folder)
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

def split_paths(paths, labels=None, val_size=0.15, test_size=0.15, seed=42):
    idx = list(range(len(paths)))
    strat = labels if labels is not None else None
    train_idx, temp_idx = train_test_split(idx, test_size=val_size+test_size, random_state=seed, stratify=strat)
    temp_labels = [labels[i] for i in temp_idx] if labels is not None else None
    rel_test = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(temp_idx, test_size=rel_test, random_state=seed, stratify=temp_labels)
    return train_idx, val_idx, test_idx
