import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import cv2

from utils.config import load_config
from datasets.transforms import get_val_transforms
from datasets.segmentation_dataset import SegmentationDataset
from models.framework import ParasiteFramework
from uncertainty import mc_segmentation_predict
from utils.metrics import dice_score, expected_calibration_error

def corrupt(x, kind):
    x = x.clone()
    if kind == "gaussian_noise":
        return (x + 0.1 * torch.randn_like(x)).clamp(-3, 3)
    if kind == "brightness_down":
        return (x - 0.3).clamp(-3, 3)
    if kind == "brightness_up":
        return (x + 0.3).clamp(-3, 3)
    if kind == "low_contrast":
        mean = x.mean(dim=(2,3), keepdim=True)
        return (mean + 0.5 * (x - mean)).clamp(-3, 3)
    if kind == "blur":
        return torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x

def run(cfg, checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg["data_root"])
    ds = SegmentationDataset(root / "BBBC041" / "images", root / "BBBC041" / "masks", get_val_transforms(cfg["image_size"]), "test", cfg["seed"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)
    model = ParasiteFramework(cfg, task="bbbc_seg", pretrained=False, use_lora=True).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    rows = []
    for kind in ["clean", "gaussian_noise", "blur", "low_contrast", "brightness_down", "brightness_up"]:
        dices, eces = [], []
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            x = corrupt(x, kind)
            mean, epi, ale, total = mc_segmentation_predict(model, x, cfg["mc_samples"])
            dices.append(dice_score(mean.cpu(), y.cpu()))
            eces.append(expected_calibration_error(mean.cpu(), y.cpu(), cfg["ece_bins"]))
        rows.append({"corruption": kind, "dice": np.mean(dices), "ece": np.mean(eces)})
    out = Path(cfg["output_dir"]) / "robustness_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows))
    print("Saved:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default="bbbc_seg")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.checkpoint)
