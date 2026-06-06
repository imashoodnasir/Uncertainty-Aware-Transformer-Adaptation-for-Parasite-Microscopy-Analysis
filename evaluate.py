import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd

from utils.config import load_config
from datasets.transforms import get_val_transforms
from datasets.classification_dataset import FolderClassificationDataset
from datasets.segmentation_dataset import SegmentationDataset
from models.framework import ParasiteFramework
from utils.metrics import classification_metrics, dice_score, iou_score, expected_calibration_error, brier_score
from uncertainty import mc_classification_predict, mc_segmentation_predict

def make_test_loader(cfg, task):
    root = Path(cfg["data_root"])
    if task == "malaria_cls":
        ds = FolderClassificationDataset(root / "NIH_Malaria", get_val_transforms(cfg["image_size"]), "test", cfg["seed"])
    elif task == "chula_cls":
        ds = FolderClassificationDataset(root / "Chula_ParasiteEgg_11", get_val_transforms(cfg["image_size"]), "test", cfg["seed"])
    elif task == "bbbc_seg":
        ds = SegmentationDataset(root / "BBBC041" / "images", root / "BBBC041" / "masks", get_val_transforms(cfg["image_size"]), "test", cfg["seed"])
    else:
        raise ValueError(task)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

def evaluate(cfg, task, checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl = make_test_loader(cfg, task)
    model = ParasiteFramework(cfg, task=task, pretrained=False, use_lora=True).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    if task in ["malaria_cls", "chula_cls"]:
        y_true, y_pred, y_prob = [], [], []
        for x, y in dl:
            x = x.to(device)
            mean, var = mc_classification_predict(model, x, cfg["mc_samples"])
            pred = mean.argmax(dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(pred.cpu().tolist())
            y_prob.append(mean.cpu())
        y_prob = torch.cat(y_prob, dim=0)
        metrics = classification_metrics(y_true, y_pred, y_prob.numpy())
        metrics["ece"] = expected_calibration_error(y_prob, torch.tensor(y_true), cfg["ece_bins"])
        metrics["brier"] = brier_score(y_prob, torch.tensor(y_true))
    else:
        dice_vals, iou_vals, eces, briers = [], [], [], []
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            mean, epi, ale, total = mc_segmentation_predict(model, x, cfg["mc_samples"])
            dice_vals.append(dice_score(mean.cpu(), y.cpu()))
            iou_vals.append(iou_score(mean.cpu(), y.cpu()))
            eces.append(expected_calibration_error(mean.cpu(), y.cpu(), cfg["ece_bins"]))
            briers.append(brier_score(mean.cpu(), y.cpu()))
        metrics = {
            "dice": sum(dice_vals) / len(dice_vals),
            "iou": sum(iou_vals) / len(iou_vals),
            "ece": sum(eces) / len(eces),
            "brier": sum(briers) / len(briers)
        }
    print(metrics)
    out = Path(cfg["output_dir"]) / f"{task}_evaluation.csv"
    pd.DataFrame([metrics]).to_csv(out, index=False)
    print("Saved:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", required=True, choices=["malaria_cls", "chula_cls", "bbbc_seg"])
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.task, args.checkpoint)
