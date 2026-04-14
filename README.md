# 🫁 Heterogeneous Multi-Architecture Ensemble for Multi-Label Chest X-Ray Classification

[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](link-to-paper)
[![Dataset](https://img.shields.io/badge/Dataset-NIH%20ChestX--ray14-green)](https://nihcc.app.box.com/v/ChestXray-NIHCC)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Heterogeneous Multi-Architecture Ensemble with Test-Time Augmentation for Multi-Label Chest X-Ray Classification on NIH ChestX-ray14**
>
> [Author 1 Name]\*, [Author 2 Name]\*, [Guide/Advisor Name]†
>
> \*Equal Contribution  †Advisor
>
> [Department, Institution]

## 📋 Abstract

We propose a heterogeneous five-model ensemble framework combining CNN-based and transformer-based architectures for multi-label thoracic disease classification on the NIH ChestX-ray14 dataset. Our ensemble integrates **EfficientNet-B0**, **EfficientNet-B3**, **Vision Transformer (ViT-Base-16)**, **DINOv2** (with multi-modal anatomical features), and **ConvNeXt V2 Tiny** — achieving a **macro-averaged AUC of 0.8642** with test-time augmentation (TTA), surpassing the established CheXNet benchmark (0.841) by **+2.32%**.

---

## 🏆 Key Results

| Metric | Score |
|--------|-------|
| **Final Ensemble AUC (5-Way + TTA)** | **0.8642** |
| CheXNet Benchmark | 0.8410 |
| Improvement over CheXNet | **+2.32%** |
| Minority Class Avg AUC | **0.9458** |
| Best Per-Class (Hernia) | **0.9832** |

### AUC Progression

```
Single Model (ViT)     ████████████████████████████  0.8379
2-Way (B0+B3)          █████████████████████████████  0.8076
3-Way (+ViT)           █████████████████████████████  0.8166
4-Way (+DINO)          ██████████████████████████████ 0.8592
4-Way + TTA            ██████████████████████████████ 0.8607
5-Way (+CNX)           ██████████████████████████████ 0.8623
5-Way + Full TTA       ██████████████████████████████ 0.8642  ← Final
```

---

## 🏗️ Architecture Overview

```
                    ┌──────────────────┐
                    │  Input Chest     │
                    │  X-ray (224×224) │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
     ┌──────┴──────┐  ┌─────┴─────┐  ┌──────┴──────┐
     │  CNN Branch │  │ Transformer│  │  Modern CNN │
     │             │  │   Branch   │  │   Branch    │
     ├─────────────┤  ├───────────┤  ├─────────────┤
     │EfficientNet │  │ViT-Base-16│  │ ConvNeXt V2 │
     │   B0 (20%)  │  │   (30%)   │  │  Tiny (22%) │
     ├─────────────┤  ├───────────┤  └─────────────┘
     │EfficientNet │  │  DINOv2   │
     │   B3 (18%)  │  │   (10%)   │
     └──────┬──────┘  └─────┬─────┘
            │                │
            └────────┬───────┘
                     │
            ┌────────┴─────────┐
            │ Weighted Average │
            │  Ensemble + TTA  │
            └────────┬─────────┘
                     │
            ┌────────┴─────────┐
            │   14 Pathology   │
            │   Predictions    │
            └──────────────────┘
```

---

## 📊 Individual Model Performance

| Model | Architecture | Test AUC (Macro) | Parameters |
|-------|-------------|-----------------|------------|
| EfficientNet-B0 | CNN | 0.8293 | ~5.3M |
| EfficientNet-B3 | CNN | 0.8350 | ~12M |
| ViT-Base-16 | Transformer | 0.8379 | ~86M |
| DINOv2 | Self-supervised ViT | 0.7570 | ~86M |
| ConvNeXt V2 Tiny | Modern CNN | 0.7973 | ~28M |

## 📊 Per-Class AUC (Final 5-Way Ensemble + TTA)

| Pathology | AUC | Category |
|-----------|------|----------|
| Hernia | **0.9832** | Minority |
| Emphysema | **0.9593** | Minority |
| Fibrosis | **0.9465** | Minority |
| Cardiomegaly | **0.9392** | Majority |
| Edema | **0.9293** | Minority |
| Pneumonia | **0.9107** | Minority |
| Pneumothorax | **0.9012** | Majority |
| Effusion | 0.8243 | Majority |
| Mass | 0.8234 | Majority |
| Consolidation | 0.7731 | Majority |
| Atelectasis | 0.7616 | Majority |
| Nodule | 0.7607 | Majority |
| Infiltration | 0.7216 | Majority |

---

## 📁 Repository Structure

```
chest-xray-ensemble/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── notebooks/                         # Training notebooks (Jupyter)
│   ├── 01_EfficientNet_B0_Training.ipynb
│   ├── 02_EfficientNet_B3_Training.ipynb
│   ├── 03_ViT_Base16_Training.ipynb
│   ├── 04_DINOv2_MultiModal_Training.ipynb
│   ├── 05_ConvNeXt_V2_Training.ipynb
│   └── Accuracy.ipynb
│
├── logs/                              # Training logs
│   ├── training_log.txt               # EfficientNet-B0 training log
│   ├── phase5_effnet_b3_training.log
│   ├── phase6_vit_training.log
│   └── phase7_dinov2_ultimate_training.log
│
├── results/                           # Evaluation results (JSON)
│   ├── balanced_training_results.json
│   ├── ensemble_b0_b3_results.json
│   ├── ensemble_tta_results.json
│   ├── final_3way_ensemble_results.json
│   ├── FINAL_4WAY_ENSEMBLE_RESULTS.json
│   ├── final_test_results.json
│   ├── FINAL_RESULTS.json
│   ├── optimized_thresholds.json
│   ├── phase5_effnet_b3_results.json
│   ├── phase5_test_results.json
│   ├── phase6_vit_results.json
│   ├── phase7_dinov2_results.json
│   ├── phase7_ultimate_dinov2_test_results.json
│   ├── phase8_convnext_results.json
│   └── phase8_final_results.json
│
├── figures/                           # Figures for paper
│   ├── convnext_training_curve.png
│   ├── dinov2_training_curve.png
│   ├── balanced_vs_baseline.png
│   ├── final_ensemble_dashboard.png
│   ├── balanced_model_training.png
│   ├── 3way_ensemble_results.png
│   ├── 3way_perclass_auc.png
│   └── balanced_perclass_auc.png
│
├── paper/                             # IEEE Research Paper
│   └── IEEE_Paper_Chest_Xray_Ensemble.pdf
│
└── evaluation/                        # Evaluation scripts and reports
    └── evaluation_report.txt
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.12
CUDA >= 11.3 (for GPU training)
```

### Installation

```bash
git clone https://github.com/[username]/chest-xray-ensemble.git
cd chest-xray-ensemble
pip install -r requirements.txt
```

### Dataset Setup

1. Download the NIH ChestX-ray14 dataset from [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
2. Extract images to `data/images/`
3. Place `Data_Entry_2017.csv` in `data/`

```bash
mkdir -p data/images
# Download and extract NIH ChestX-ray14 dataset
# Total size: ~45 GB (112,120 frontal-view chest X-rays)
```

### Training Individual Models

```bash
# Train each model using the corresponding notebook
jupyter notebook notebooks/01_EfficientNet_B0_Training.ipynb
jupyter notebook notebooks/02_EfficientNet_B3_Training.ipynb
jupyter notebook notebooks/03_ViT_Base16_Training.ipynb
jupyter notebook notebooks/04_DINOv2_MultiModal_Training.ipynb
jupyter notebook notebooks/05_ConvNeXt_V2_Training.ipynb
```

### Running Ensemble

```bash
# Run ensemble experiments
jupyter notebook notebooks/06_Bagging_Experiments.ipynb
```

---

## ⚙️ Ensemble Weights

Final optimized weights for the 5-Way ensemble with TTA:

```python
ensemble_weights = {
    "EfficientNet-B0": 0.20,
    "EfficientNet-B3": 0.18,
    "ViT-Base-16":     0.30,
    "DINOv2":          0.10,
    "ConvNeXt-V2":     0.22
}

# Test-Time Augmentation modes
tta_modes = ["original", "horizontal_flip", "center_crop_zoom", "brightness"]
```

---

## 📈 Training Curves

### ConvNeXt V2 Tiny
- 19 epochs, best validation AUC **0.8245** at epoch 13
- See `figures/convnext_training_curve.png`

### DINOv2 (Ultimate)
- 10 epochs with focal loss, best validation AUC **0.7835** at epoch 5
- See `figures/dinov2_training_curve.png`

### Class-Balanced EfficientNet-B0
- 10 epochs, test AUC **0.7945** with significant minority class improvement
- Hernia: 0.55 → 0.883 (+60.5%)
- Edema: 0.70 → 0.878 (+25.4%)

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{author2026chest,
  title={Heterogeneous Multi-Architecture Ensemble with Test-Time Augmentation 
         for Multi-Label Chest X-Ray Classification on {NIH} {ChestX}-ray14},
  author={[Author 1 Last Name], [First Name] and [Author 2 Last Name], [First Name]},
  booktitle={Proceedings of [Conference/Journal Name]},
  year={2026},
  note={Under review}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NIH Clinical Center for the ChestX-ray14 dataset
- Stanford ML Group for the CheXNet benchmark
- PyTorch and Hugging Face communities
- [Guide/Advisor Name] for guidance and supervision

---

## 📬 Contact

- **[Author 1 Name]** — [email1@institution.edu]
- **[Author 2 Name]** — [email2@institution.edu]
- **[Guide/Advisor Name]** (Advisor) — [guide@institution.edu]

---

*This repository accompanies the IEEE research paper: "Heterogeneous Multi-Architecture Ensemble with Test-Time Augmentation for Multi-Label Chest X-Ray Classification on NIH ChestX-ray14"*
