# 🫁 Multimodal Disease Prediction from Chest X-Rays & Radiology Reports

> A late-fusion deep learning system that combines **DenseNet-121** (image) and **ClinicalBERT** (text) to predict 5 thoracic diseases from chest X-rays and free-text radiology reports.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Diseases Predicted](#diseases-predicted)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Requirements](#setup--requirements)
- [How to Run](#how-to-run)
- [Notebook Cell Guide](#notebook-cell-guide)
- [Results](#results)
- [Key Design Decisions](#key-design-decisions)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Overview

This project implements the multimodal disease prediction pipeline described in the research paper **"Multimodal Disease Prediction from Chest X-Rays and Radiology Reports"**. The core idea is that chest X-ray images alone and radiology reports alone both carry complementary diagnostic information — combining them with a **late-fusion architecture** produces better predictions than either modality alone.

The model takes two inputs for each patient study:
- A **chest X-ray image** (JPG format)
- The corresponding **free-text radiology report** (Findings + Impression sections)

And outputs a **multi-label binary prediction** across 5 thoracic disease classes.

---

## Architecture

```
Chest X-Ray Image                    Radiology Report Text
      │                                       │
      ▼                                       ▼
 DenseNet-121                          ClinicalBERT
 (ImageNet pretrained)              (Bio_ClinicalBERT)
      │                                       │
  GAP Pooling                          [CLS] Token
      │                                       │
  [1024-dim]                            [768-dim]
      │                                       │
      └──────────── Concatenate ─────────────┘
                         │
                    [1792-dim]
                         │
                  FC Layer (512)
                         │
                      ReLU
                         │
                    Dropout (0.3)
                         │
                   FC Layer (5)
                         │
                      Sigmoid
                         │
              5 Disease Probabilities
```

### Image Branch — DenseNet-121
- Pretrained on ImageNet
- Global Average Pooling (GAP) replaces the original classifier head
- Outputs a **1,024-dimensional** feature vector
- Input images preprocessed with **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance low-contrast lung structures

### Text Branch — ClinicalBERT
- Uses `emilyalsentzer/Bio_ClinicalBERT` from Hugging Face
- Only the `[CLS]` token embedding is used as the sentence-level representation
- Outputs a **768-dimensional** contextual embedding
- Only Findings and Impression sections of each report are extracted (not the full report)

### Fusion Head
- Late fusion: both feature vectors are concatenated **after** independent encoding
- FC(1792→512) → ReLU → Dropout(0.3) → FC(512→5) → Sigmoid
- Loss: **Weighted Binary Cross-Entropy** to handle class imbalance

---

## Diseases Predicted

The model performs **multi-label classification** — a single X-ray can have multiple conditions simultaneously.

| # | Disease | Type |
|---|---------|------|
| 1 | Cardiomegaly | Cardiac |
| 2 | Pleural Effusion | Pleural |
| 3 | Edema | Pulmonary |
| 4 | Pneumothorax | Pleural |
| 5 | Pneumonia | Infectious |

---

## Dataset

This project uses **two PhysioNet datasets** together. Both require signing a Data Use Agreement at [physionet.org](https://physionet.org).

### Dataset 1 — MIMIC-CXR-JPG v2.0.0
**URL:** https://physionet.org/content/mimic-cxr-jpg/2.0.0

Provides:
- Chest X-ray images in JPG format (`files/p*/p*/s*/*.jpg`)
- Disease labels via CheXpert labeler (`mimic-cxr-2.0.0-chexpert.csv.gz`)
- Official train/validation/test split (`mimic-cxr-2.0.0-split.csv.gz`)
- Full image path list (`IMAGE_FILENAMES`) — used for selective downloading

### Dataset 2 — MIMIC-CXR v2.0.0 (base)
**URL:** https://physionet.org/content/mimic-cxr/2.0.0

Provides:
- Free-text radiology reports (`mimic-cxr-reports.tar.gz`)
- Reports follow structure: `files/p{prefix}/p{subject_id}/s{study_id}.txt`

### Label Convention
| Value | Meaning |
|-------|---------|
| `1.0` | Finding is present |
| `0.0` | Finding is absent |
| `-1.0` | Uncertain — treated as `0` in this project |
| `NaN` | Not mentioned — treated as `0` |

### Dataset Size
| Component | Size | Downloaded to |
|-----------|------|---------------|
| CSV labels + split | ~50 MB | Colab disk |
| Radiology reports | ~1 GB | Colab disk |
| Images (subset 5,000) | ~750 MB | Colab disk |
| **Total needed** | **~2 GB** | Colab disk (~70 GB free) |

> ⚠️ The full image dataset is ~17 GB. This project downloads only a subset using the `IMAGE_FILENAMES` trick for practical training in Colab.

---

## Project Structure

```
multimodal-disease-prediction/
│
├── multimodal_disease_prediction_colab.py   # Main notebook code (all cells)
├── README.md                                # This file
│
└── outputs/                                 # Generated after running
    ├── best_multimodal_model.pth            # Best model weights (saved to Drive)
    ├── final_multimodal_model.pth           # Final model weights
    ├── test_results.csv                     # Per-class AUROC, F1, P, R
    ├── model_comparison.csv                 # Comparison table (Table 3)
    ├── training_curves.png                  # Loss & AUROC over epochs (Figure 5)
    ├── per_class_auroc.png                  # Per-class AUROC bar chart (Figure 3)
    ├── confusion_matrices.png               # 5 confusion matrices (Figure 6)
    ├── model_comparison_chart.png           # Model comparison chart (Figure 4)
    └── class_distribution.png              # Label distribution (Figure 1)
```

---

## Setup & Requirements

### Environment
- **Platform:** Google Colab (recommended — free GPU)
- **Runtime:** GPU (T4 or better) — set via `Runtime → Change runtime type → T4 GPU`
- **Python:** 3.8+

### Dependencies
All installed automatically in Cell 1:

```
transformers        # ClinicalBERT (Hugging Face)
torchvision         # DenseNet-121, image transforms
scikit-learn        # AUROC, F1, confusion matrix
opencv-python       # CLAHE image preprocessing
pandas              # Data manipulation
numpy               # Numerical operations
matplotlib          # Plotting
seaborn             # Heatmaps
tqdm                # Progress bars
```

### PhysioNet Access
1. Create an account at [physionet.org](https://physionet.org)
2. Complete the CITI training course (required for MIMIC datasets)
3. Sign the Data Use Agreement for:
   - [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0)
   - [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0)
4. Approval typically takes 24–48 hours

---

## How to Run

### Step 1 — Open in Google Colab
Upload `multimodal_disease_prediction_colab.py` to Colab, or copy each cell manually into a new notebook.

### Step 2 — Set GPU Runtime
```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

### Step 3 — Run Cells in Order

The notebook is divided into clearly labelled cells. Run them **top to bottom**, one at a time:

```
Cell 1    →  Install libraries
Cell 2A   →  Test PhysioNet credentials         ← Must show HTTP 200 OK
Cell 2B   →  Download CSV labels & split files
Cell 2C   →  Download radiology reports
Cell 2D   →  Extract reports archive
Cell 2E   →  Download image subset (~750 MB)
Cell 2F   →  Set all file paths
Cell 3    →  Imports & hyperparameters
Cell 4    →  Load & parse labels
Cell 5    →  Load & preprocess reports
Cell 6    →  Image preprocessing & augmentation
Cell 7    →  Dataset class definition
Cell 8    →  Tokenizer & DataLoaders
Cell 9    →  Model architecture
Cell 10   →  Loss, optimiser, scheduler
Cell 11   →  Training & validation loop        ← Longest step (~hours on GPU)
Cell 12   →  Plot training curves
Cell 13   →  Test set evaluation
Cell 14   →  Per-class AUROC chart
Cell 15   →  Confusion matrices
Cell 16   →  Model comparison table & chart
Cell 17   →  Class distribution plot
Cell 18   →  Save model to Google Drive
```

> ⚠️ **Important:** Colab resets its disk on session end. Cell 18 saves your trained model weights to Google Drive so they are not lost.

---

## Notebook Cell Guide

| Cell | Purpose | Expected Output |
|------|---------|-----------------|
| 1 | Install packages | All packages install without errors |
| 2A | Verify credentials | `HTTP/1.1 200 OK` |
| 2B | Download CSVs | Three `✔` checkmarks with file sizes |
| 2C | Download reports | `.tar.gz` file ~1 GB |
| 2D | Extract reports | Thousands of `.txt` files found |
| 2E | Download images | JPG count printed at end |
| 2F | Set paths | Four `✔` path confirmations |
| 3 | Config | `Running on: cuda` |
| 4 | Labels | Row count + label prevalence table |
| 5 | Reports | Cache size + missing report count |
| 6 | Transforms | No output (definitions only) |
| 7 | Dataset class | No output (definition only) |
| 8 | DataLoaders | Train/Val/Test sizes printed |
| 9 | Model | Architecture summary + parameter count |
| 10 | Optimiser | Positive class weights printed |
| 11 | Training | Per-epoch loss & AUROC for 25 epochs |
| 12 | Curves | `training_curves.png` saved |
| 13 | Evaluation | Per-class AUROC, Precision, Recall, F1 |
| 14–17 | Plots | 4 chart files saved |
| 18 | Save | Model saved to Google Drive |

---

## Results

Expected results from the paper (your run may vary slightly based on image subset size):

### Per-Class Test AUROC (Table 2)

| Disease | Image Only | Text Only | **Late Fusion** |
|---------|-----------|-----------|-----------------|
| Cardiomegaly | 0.87 | 0.88 | **0.91** |
| Pleural Effusion | 0.89 | 0.90 | **0.93** |
| Edema | 0.86 | 0.87 | **0.90** |
| Pneumothorax | 0.85 | 0.85 | **0.88** |
| Pneumonia | 0.77 | 0.78 | **0.82** |
| **Mean** | 0.85 | 0.86 | **0.89** |

### Model Comparison (Table 3)

| Model | Modality | AUROC | F1 |
|-------|---------|-------|----|
| ResNet-50 | Image | 0.81 | 0.70 |
| DenseNet-121 (CheXNet) | Image | 0.85 | 0.74 |
| ClinicalBERT | Text | 0.86 | 0.76 |
| Early Fusion | Image + Text | 0.87 | 0.79 |
| **Late Fusion (Proposed)** | **Image + Text** | **0.89** | **0.82** |

### Hyperparameters Used

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Learning rate | 1e-4 |
| Epochs | 25 (early stopping patience: 5) |
| Optimiser | Adam |
| LR Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Dropout | 0.3 |
| Image size | 224 × 224 |
| Max text tokens | 256 |
| Fusion FC hidden | 512 |

---

## Key Design Decisions

**Why Late Fusion instead of Early Fusion?**
Early fusion (concatenating raw features before any encoding) loses the modality-specific representations. Late fusion lets each encoder specialise independently before combining, which the paper shows produces higher AUROC.

**Why CLAHE preprocessing?**
Chest X-rays often have low-contrast regions in the lung fields. CLAHE enhances local contrast without over-amplifying noise, making subtle pathologies like early pneumonia more visible to DenseNet-121.

**Why only Findings and Impression sections?**
Radiology reports also contain Indication and Technique sections which describe the patient history and scan method — not the actual findings. Feeding these to ClinicalBERT adds noise. Extracting only Findings + Impression improves text quality.

**Why Weighted BCE Loss?**
MIMIC-CXR has severe class imbalance — Pneumonia and Pneumothorax are much rarer than Pleural Effusion. Weighted loss penalises false negatives on rare classes more heavily, preventing the model from ignoring minority classes.

---

## Limitations

- Training on a subset (~5,000 studies) will produce lower AUROC than the paper's full-dataset results
- Colab session resets mean you must re-download data each session (model weights survive via Drive)
- ClinicalBERT is truncated to 256 tokens — very long reports get cut off
- Uncertain labels (`-1.0`) are treated as negative, which may introduce label noise
- The model is not clinically validated and should not be used for real diagnosis

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{multimodal_cxr_2024,
  title   = {Multimodal Disease Prediction from Chest X-Rays and Radiology Reports},
  journal = {[Journal Name]},
  year    = {2024},
  note    = {Registration No: 23-CS-77}
}
```

Also cite the dataset:

```bibtex
@article{mimic_cxr_jpg,
  title   = {MIMIC-CXR-JPG — chest radiographs with structured labels},
  author  = {Johnson, Alistair and others},
  journal = {arXiv preprint arXiv:1901.07042},
  year    = {2019}
}
```

---

## Acknowledgements

- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0) — Johnson et al., PhysioNet
- [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) — Alsentzer et al., 2019
- [CheXNet / DenseNet-121](https://arxiv.org/abs/1711.05225) — Rajpurkar et al., Stanford ML Group

---

<p align="center">
  Made for academic research purposes only · Not for clinical use
</p>
