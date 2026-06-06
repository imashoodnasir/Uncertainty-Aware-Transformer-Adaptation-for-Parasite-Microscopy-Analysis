from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd

def main(root):
    root = Path(root)
    rows = []
    for cls in sorted([p for p in root.iterdir() if p.is_dir()]):
        for img in cls.rglob("*"):
            if img.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                rows.append({"path": str(img), "label": cls.name})
    df = pd.DataFrame(rows)
    tr, tmp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=42)
    out = root / "splits"
    out.mkdir(exist_ok=True)
    tr.to_csv(out / "train.csv", index=False)
    va.to_csv(out / "val.csv", index=False)
    te.to_csv(out / "test.csv", index=False)
    print("Saved splits to", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    main(args.root)
