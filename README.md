# 🛸 UAV-CrossDomain-Traffic-Detection
> **Cross-Domain Vehicle Detection from UAV Imagery using YOLO11 with High-Resolution P2 Head & Normalized Wasserstein Distance (NWD) Loss.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-green.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 Overview

Vehicle detection from Unmanned Aerial Vehicles (UAVs) faces two severe challenges:
1. **Extreme Scale Variance & Tiny Objects:** Vehicles (especially motorcycles) occupy only a few pixels ($<16\times16$ px) when viewed from high altitude, causing severe feature loss in deep convolutional strides.
2. **Domain Shift & Generalization Collapse:** Models trained on single-country datasets suffer catastrophic performance drops when tested on unseen UAV environments due to variations in camera pitch, lighting, traffic density, and vehicle morphologies.

This repository provides an end-to-end framework and rigorous benchmark covering **3 distinct geographic domains** (VisDrone2019 - China, Songdo Vision - South Korea, JATAYU - India), along with novel architectural enhancements (**P2 Head** and **NWD Loss**) to overcome small-object bottlenecks.

---

## 🚀 Key Contributions

* **Cross-Domain Benchmark Protocol:** Unified taxonomy for 4 classes (Car, Motorcycle, Bus, Truck) across 3 international UAV datasets (totaling >20,000 images). Evaluated In-domain vs. Cross-domain generalization across 12 distinct scenarios.
* **P2 High-Resolution Head (Stride 4):** Redesigned the YOLO11 architecture by tapping into low-level feature maps (P2/4), preserving spatial details essential for ultra-small aerial objects.
* **Gaussian Bounding Box Regression (NWD Loss):** Integrated Normalized Wasserstein Distance (NWD) loss to replace traditional CIoU/GIoU, resolving gradient vanishing and extreme sensitivity to minute pixel offsets on small targets.

---

## 📊 Benchmark Results

### 1. Cross-Domain Generalization Matrix (mAP@0.5)

| Training Model \ Test Set | VisDrone (🇨🇳) | Songdo (🇰🇷) | JATAYU (🇮🇳) |
| :--- | :---: | :---: | :---: |
| **Model A (VisDrone)** | **63.65%** | 57.11% | 10.92% |
| **Model B (Songdo)** | 3.59% | **90.94%** | 10.02% |
| **Model C (JATAYU)** | 2.44% | 33.13% | **70.78%** |
| **Model D (Triple-Domain Fusion)** | **62.07%** | **89.42%** | **68.04%** |

> **Key Finding:** Single-domain models collapse under cross-domain evaluation (e.g., Songdo model drops from 90.9% to 3.5% on VisDrone). Triple-Domain fusion maintains high robustness across all three domains without catastrophic forgetting.

### 2. Strict Localization Matrix (mAP@0.5:0.95)

| Training Model \ Test Set | VisDrone (🇨🇳) | Songdo (🇰🇷) | JATAYU (🇮🇳) |
| :--- | :---: | :---: | :---: |
| **Model A (VisDrone)** | **42.03%** | 36.09% | 6.89% |
| **Model B (Songdo)** | 1.73% | **65.67%** | 5.64% |
| **Model C (JATAYU)** | 1.24% | 20.99% | **50.04%** |
| **Model D (Triple-Domain Fusion)** | **40.50%** | **63.77%** | **47.30%** |

### 3. Balanced Performance (F1-Score Matrix)

| Training Model \ Test Set | VisDrone (🇨🇳) | Songdo (🇰🇷) | JATAYU (🇮🇳) |
| :--- | :---: | :---: | :---: |
| **Model A (VisDrone)** | **65.15%** | 59.94% | 20.08% |
| **Model B (Songdo)** | 9.35% | **88.81%** | 20.78% |
| **Model C (JATAYU)** | 7.31% | 38.21% | **65.48%** |
| **Model D (Triple-Domain Fusion)** | **63.91%** | **87.06%** | **65.65%** |

---

## 🔬 Model D (Triple-Domain) Per-Class Performance

| Dataset | Class | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **VisDrone (🇨🇳)** | Car | 79.72% | 80.05% | 79.89% | 82.44% | 58.17% |
| | **Motorcycle** | 62.32% | **49.14%** | 54.95% | **52.01%** | 23.38% |
| | Bus | 75.64% | 55.38% | 63.94% | 60.28% | 43.75% |
| | Truck | 67.47% | 47.08% | 55.46% | 53.54% | 36.71% |
| **Songdo (🇰🇷)** | Car | 97.36% | 98.95% | 98.15% | 99.36% | 79.42% |
| | **Motorcycle** | 75.49% | **60.49%** | 67.16% | **67.36%** | 25.91% |
| | Bus | 93.93% | 97.28% | 95.58% | 98.99% | 81.82% |
| | Truck | 83.69% | 89.29% | 86.40% | 91.97% | 67.94% |
| **JATAYU (🇮🇳)** | Car | 47.82% | 78.12% | 59.32% | 67.32% | 43.59% |
| | **Motorcycle** | 59.42% | **42.25%** | 49.39% | **47.90%** | 16.57% |
| | Bus | 56.35% | 67.58% | 61.45% | 62.57% | 51.84% |
| | Truck | 87.24% | 87.55% | 87.39% | 94.35% | 77.20% |

> **Diagnostic Insight:** Across all domains, Motorcycle is the critical bottleneck, suffering up to 20-30% lower recall than cars due to tiny bounding box dimensions and rapid motion blur. This motivated the development of **P2 Head** and **NWD Loss**.

---

## 🛠️ Repository Structure

`	ext
├── configs/
│   ├── models/
│   │   └── yolo11m-p2.yaml          # Custom YOLO11m architecture with P2 Head (Stride 4)
│   ├── data_visdrone_only.yaml      # VisDrone dataset config
│   ├── data_songdo_only.yaml        # Songdo Vision dataset config
│   ├── data_jatayu_only.yaml        # JATAYU dataset config
│   └── data_triple_domain.yaml      # Merged Triple-Domain dataset config
├── data_prep/
│   ├── setup_visdrone.py            # Converts VisDrone CSV to YOLO 4-class format
│   ├── remap_songdo.py              # Remaps Songdo Vision class IDs
│   ├── remap_jatayu.py              # Unifies JATAYU 10-class labels to standard 4 classes
│   └── merge_datasets.py            # Fuses 3 datasets into Triple-Domain
├── training/
│   ├── train_benchmark.py           # Master training & cross-domain evaluation pipeline
│   ├── train_p2_visdrone.py         # Trains custom YOLO11m-P2 on VisDrone
│   ├── patch_nwd_loss.py            # Injects NWD Loss into Ultralytics engine
│   └── train_nwd.py                 # Training script for P2 + NWD Loss
├── evaluation/
│   ├── eval_model_d_detailed.py     # Detailed per-class validation & confusion matrices
│   ├── analyze_motorcycle.py        # Small-object bbox distribution and error diagnosis
│   ├── fair_compare_p2_nwd.py       # Fair ablation comparison (P2-CIoU vs P2-NWD)
│   └── check_p2_val.py              # Standalone P2 validation checker
├── inference/
│   ├── track_model_d.py             # Video inference & vehicle tracking (BoT-SORT / ByteTrack)
│   ├── count_vehicles.py            # Directional traffic flow counting
│   └── predict_video.py             # Bounding box prediction on UAV video streams
├── results/
│   ├── cross_domain_matrix.json     # Raw 12 cross-domain evaluation metrics
│   ├── cross_domain_matrix.csv      # Tabular CSV benchmark data
│   ├── model_d_detailed_eval.json   # Per-class evaluation outputs and confusion matrices
│   ├── motorcycle_analysis.json     # Spatial distribution of tiny objects
│   ├── fair_test_P2_CIoU.txt        # Benchmark logs for P2 + CIoU
│   └── fair_test_P2_NWD.txt         # Benchmark logs for P2 + NWD
├── docs/
│   ├── Bao_Cao_Benchmark_CrossDomain.md # Comprehensive Vietnamese research report
│   └── paper/mixstyle_domaingenarate.pdf # MixStyle reference literature
├── requirements.txt
└── README.md
`

---

## ⚡ Quick Start

### 1. Installation
`ash
git clone https://github.com/Thanhhuy1809/UAV-CrossDomain-Traffic-Detection.git
cd UAV-CrossDomain-Traffic-Detection
pip install -r requirements.txt
`

### 2. Preprocess Datasets
`ash
# Normalize and remap labels to standard 4 classes
python data_prep/setup_visdrone.py
python data_prep/remap_songdo.py
python data_prep/remap_jatayu.py
python data_prep/merge_datasets.py
`

### 3. Run Benchmark Training (Models A, B, C, D)
`ash
python training/train_benchmark.py
`

### 4. Train YOLO11m with P2 Head
`ash
python training/train_p2_visdrone.py
`

### 5. Train with Normalized Wasserstein Distance (NWD) Loss
`ash
# Step 1: Patch loss function
python training/patch_nwd_loss.py

# Step 2: Run training
python training/train_nwd.py
`

### 6. UAV Traffic Tracking & Counting
`ash
python inference/track_model_d.py --source /path/to/uav_video.mp4 --conf 0.25
`

---

## 📄 Citation & License
This project is licensed under the MIT License. If you use this benchmark or architecture in your research, please cite this repository.
