# Uncertainty-Aware Self-Supervised Microscopy Foundation Framework

This repository provides a complete PyTorch implementation template for the paper:

**Self-Supervised Foundation Learning with Uncertainty-Aware Transformer Adaptation for Parasite Microscopy Analysis**

The framework supports:

- NIH Malaria classification
- BBBC041 parasite localization/segmentation
- Chula-ParasiteEgg-11 multi-class parasite classification
- self-supervised masked image modeling
- contrastive learning
- LoRA-based transformer adaptation
- multi-scale segmentation
- uncertainty estimation using Monte Carlo dropout
- calibration evaluation
- robustness testing
- qualitative visualization

## 1. Installation

```bash
conda create -n parasite_ssl python=3.10 -y
conda activate parasite_ssl
pip install -r requirements.txt
```

## 2. Dataset Structure

Place datasets as follows:

```text
data/
├── NIH_Malaria/
│   ├── Parasitized/
│   └── Uninfected/
├── BBBC041/
│   ├── images/
│   └── masks/
└── Chula_ParasiteEgg_11/
    ├── class_1/
    ├── class_2/
    └── ...
```

For BBBC041, masks must have the same basename as images.

Example:

```text
images/sample_001.png
masks/sample_001.png
```

## 3. Configuration

Edit:

```text
configs/default.yaml
```

Important options:

```yaml
data_root: ./data
image_size: 224
batch_size: 16
ssl_epochs: 50
downstream_epochs: 50
num_workers: 4
```

## 4. Self-Supervised Pretraining

```bash
python train_ssl.py --config configs/default.yaml
```

This trains the transformer encoder using:

```text
L_ssl = L_mask + L_contrast + L_domain
```

The checkpoint is saved to:

```text
outputs/ssl_encoder.pt
```

## 5. Downstream Training

### NIH Malaria Classification

```bash
python train_downstream.py --config configs/default.yaml --task malaria_cls
```

### Chula-ParasiteEgg-11 Classification

```bash
python train_downstream.py --config configs/default.yaml --task chula_cls
```

### BBBC041 Segmentation

```bash
python train_downstream.py --config configs/default.yaml --task bbbc_seg
```

## 6. Evaluation

```bash
python evaluate.py --config configs/default.yaml --task malaria_cls --checkpoint outputs/malaria_cls_best.pt
python evaluate.py --config configs/default.yaml --task chula_cls --checkpoint outputs/chula_cls_best.pt
python evaluate.py --config configs/default.yaml --task bbbc_seg --checkpoint outputs/bbbc_seg_best.pt
```

## 7. Uncertainty Estimation

Monte Carlo dropout is enabled during evaluation using:

```yaml
mc_samples: 10
```

The framework computes:

- mean prediction
- predictive variance
- epistemic uncertainty
- Expected Calibration Error
- Brier score
- uncertainty-error correlation

## 8. Robustness Testing

```bash
python scripts/run_robustness.py --config configs/default.yaml --task bbbc_seg --checkpoint outputs/bbbc_seg_best.pt
```

Supported corruptions:

- Gaussian noise
- blur
- low contrast
- brightness shift
- JPEG-like compression
- synthetic stain perturbation

## 9. Visualization

```bash
python visualization.py --config configs/default.yaml --task bbbc_seg --checkpoint outputs/bbbc_seg_best.pt
```

Generated figures include:

- segmentation overlays
- uncertainty heatmaps
- attention-like activation maps
- failure cases
- calibration plots

## 10. Notes

This is a complete research implementation scaffold. Dataset formats vary across public sources, so minor path adjustments may be required depending on how the downloaded datasets are organized.

## 11. Citation

Use this implementation to reproduce the proposed uncertainty-aware self-supervised microscopy foundation framework.
