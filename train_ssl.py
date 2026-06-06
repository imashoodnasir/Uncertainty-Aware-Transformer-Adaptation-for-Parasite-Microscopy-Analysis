import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import load_config
from utils.seed import set_seed
from datasets.ssl_dataset import SSLMicroscopyDataset
from datasets.transforms import get_ssl_transforms
from models.ssl_model import SSLMicroscopyModel
from losses.ssl_losses import info_nce_loss, domain_alignment_loss

def train(cfg):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    roots = [
        Path(cfg["data_root"]) / "NIH_Malaria",
        Path(cfg["data_root"]) / "BBBC041" / "images",
        Path(cfg["data_root"]) / "Chula_ParasiteEgg_11",
    ]
    roots = [r for r in roots if r.exists()]
    ds = SSLMicroscopyDataset(roots, get_ssl_transforms(cfg["image_size"]), get_ssl_transforms(cfg["image_size"]))
    dl = DataLoader(ds, batch_size=cfg["ssl_batch_size"], shuffle=True, num_workers=cfg["num_workers"], drop_last=True)

    model = SSLMicroscopyModel(cfg, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best = 1e9
    for epoch in range(cfg["ssl_epochs"]):
        model.train()
        total = 0.0
        for x1, x2 in tqdm(dl, desc=f"SSL Epoch {epoch+1}/{cfg['ssl_epochs']}"):
            x1, x2 = x1.to(device), x2.to(device)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                z1, z2, p1, p2, recon = model(x1, x2)
                l_mask = F.mse_loss(recon, x1)
                l_contrast = info_nce_loss(p1, p2, cfg["temperature"])
                l_domain = domain_alignment_loss(z1, None)
                loss = cfg["lambda_mask"] * l_mask + cfg["lambda_contrast"] * l_contrast + cfg["lambda_domain"] * l_domain
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        avg = total / len(dl)
        print(f"SSL epoch {epoch+1}: loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save(model.encoder.state_dict(), out / "ssl_encoder.pt")
    print("Saved:", out / "ssl_encoder.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
