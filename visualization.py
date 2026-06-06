import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.config import load_config
from datasets.transforms import get_val_transforms
from datasets.segmentation_dataset import SegmentationDataset
from models.framework import ParasiteFramework
from uncertainty import mc_segmentation_predict

def denorm(x):
    x = x.detach().cpu().permute(1,2,0).numpy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

def overlay_mask(img, mask):
    img = img.copy()
    m = mask.squeeze()
    contour = (m > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 2)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0

def visualize(cfg, checkpoint, max_samples=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg["data_root"])
    ds = SegmentationDataset(root / "BBBC041" / "images", root / "BBBC041" / "masks", get_val_transforms(cfg["image_size"]), "test", cfg["seed"])
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    model = ParasiteFramework(cfg, task="bbbc_seg", pretrained=False, use_lora=True).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    out_dir = Path(cfg["output_dir"]) / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (x, y) in enumerate(dl):
        if i >= max_samples:
            break
        x = x.to(device)
        mean, epi, ale, total = mc_segmentation_predict(model, x, cfg["mc_samples"])
        img = denorm(x[0])
        pred = mean[0].cpu().numpy()
        gt = y[0].numpy()
        ov = overlay_mask(img, pred)

        fig, ax = plt.subplots(1, 5, figsize=(16, 4))
        ax[0].imshow(img); ax[0].set_title("Input")
        ax[1].imshow(gt.squeeze(), cmap="gray"); ax[1].set_title("Ground Truth")
        ax[2].imshow(pred.squeeze(), cmap="gray"); ax[2].set_title("Prediction")
        ax[3].imshow(ov); ax[3].set_title("Overlay")
        ax[4].imshow(total[0,0].cpu().numpy(), cmap="jet"); ax[4].set_title("Uncertainty")
        for a in ax: a.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i:03d}.png", dpi=200)
        plt.close(fig)
    print("Saved visualizations to:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    visualize(load_config(args.config), args.checkpoint)
