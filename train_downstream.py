import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import load_config
from utils.seed import set_seed
from datasets.transforms import get_train_transforms, get_val_transforms
from datasets.classification_dataset import FolderClassificationDataset
from datasets.segmentation_dataset import SegmentationDataset
from models.framework import ParasiteFramework
from losses.segmentation_losses import DiceLoss, boundary_loss, heteroscedastic_seg_loss
from losses.calibration import differentiable_brier_loss, segmentation_brier_loss
from utils.metrics import dice_score, expected_calibration_error

def make_loaders(cfg, task):
    root = Path(cfg["data_root"])
    if task == "malaria_cls":
        train_ds = FolderClassificationDataset(root / "NIH_Malaria", get_train_transforms(cfg["image_size"]), "train", cfg["seed"])
        val_ds = FolderClassificationDataset(root / "NIH_Malaria", get_val_transforms(cfg["image_size"]), "val", cfg["seed"])
    elif task == "chula_cls":
        train_ds = FolderClassificationDataset(root / "Chula_ParasiteEgg_11", get_train_transforms(cfg["image_size"]), "train", cfg["seed"])
        val_ds = FolderClassificationDataset(root / "Chula_ParasiteEgg_11", get_val_transforms(cfg["image_size"]), "val", cfg["seed"])
    elif task == "bbbc_seg":
        train_ds = SegmentationDataset(root / "BBBC041" / "images", root / "BBBC041" / "masks", get_train_transforms(cfg["image_size"]), "train", cfg["seed"])
        val_ds = SegmentationDataset(root / "BBBC041" / "images", root / "BBBC041" / "masks", get_val_transforms(cfg["image_size"]), "val", cfg["seed"])
    else:
        raise ValueError(task)
    return (
        DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"]),
        DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    )

def load_ssl_encoder_if_exists(model, cfg):
    p = Path(cfg["output_dir"]) / "ssl_encoder.pt"
    if p.exists():
        state = torch.load(p, map_location="cpu")
        missing, unexpected = model.encoder.load_state_dict(state, strict=False)
        print("Loaded SSL encoder:", p)
        print("Missing:", len(missing), "Unexpected:", len(unexpected))

def train(cfg, task):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl = make_loaders(cfg, task)
    model = ParasiteFramework(cfg, task=task, pretrained=True, use_lora=True).to(device)
    load_ssl_encoder_if_exists(model, cfg)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    dice_loss = DiceLoss()
    best_metric = -1.0

    for epoch in range(cfg["downstream_epochs"]):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"{task} epoch {epoch+1}/{cfg['downstream_epochs']}"):
            x, y = x.to(device), y.to(device)
            if task in ["malaria_cls", "chula_cls"]:
                logits = model(x)
                ce = F.cross_entropy(logits, y)
                cal = differentiable_brier_loss(logits, y)
                loss = ce + 0.05 * cal
            else:
                logits, log_var = model(x)
                d = dice_loss(logits, y)
                b = F.binary_cross_entropy_with_logits(logits, y)
                bd = boundary_loss(logits, y)
                unc = heteroscedastic_seg_loss(logits, log_var, y)
                cal = segmentation_brier_loss(logits, y)
                loss = cfg["lambda_dice"] * d + cfg["lambda_bce"] * b + cfg["lambda_boundary"] * bd + cfg["lambda_uncertainty"] * unc + 0.05 * cal
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        metric = validate(model, val_dl, task, device, cfg)
        print(f"Epoch {epoch+1}: loss={train_loss/len(train_dl):.4f}, val_metric={metric:.4f}")
        if metric > best_metric:
            best_metric = metric
            ckpt = out / f"{task}_best.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg, "task": task}, ckpt)
            print("Saved:", ckpt)

def validate(model, dl, task, device, cfg):
    model.eval()
    ys, ps, probs = [], [], []
    dices = []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            if task in ["malaria_cls", "chula_cls"]:
                logits = model(x)
                prob = torch.softmax(logits, dim=1)
                pred = prob.argmax(dim=1)
                ys.extend(y.cpu().tolist())
                ps.extend(pred.cpu().tolist())
                probs.append(prob.cpu())
            else:
                logits, _ = model(x)
                pred = torch.sigmoid(logits)
                dices.append(dice_score(pred.cpu(), y.cpu()))
    if task in ["malaria_cls", "chula_cls"]:
        return sum(int(a == b) for a, b in zip(ys, ps)) / len(ys)
    return sum(dices) / len(dices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", required=True, choices=["malaria_cls", "chula_cls", "bbbc_seg"])
    args = parser.parse_args()
    train(load_config(args.config), args.task)
