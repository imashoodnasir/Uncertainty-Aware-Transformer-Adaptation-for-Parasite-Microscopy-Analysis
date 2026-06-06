import yaml
from pathlib import Path

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data_root"] = str(Path(cfg["data_root"]))
    cfg["output_dir"] = str(Path(cfg["output_dir"]))
    return cfg
