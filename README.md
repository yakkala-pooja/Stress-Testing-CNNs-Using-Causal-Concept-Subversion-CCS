# Stress-Testing CNNs Using Causal Concept Subversion (CCS)

This repository implements **Causal Concept Subversion (CCS)**, a novel framework for probing the **robustness and causal reasoning** capabilities of convolutional neural networks (CNNs) by injecting **semantically dissimilar out-of-distribution (OOD) patches** into **causally significant regions** of input images. The goal is to analyze how fragile a model's predictions are when faced with perceptually plausible but conceptually misleading perturbations.

---

## Overview

Conventional adversarial attacks often rely on imperceptible noise or pixel-level perturbations. In contrast, **CCS** introduces **semantic and causal-level perturbations** by:

- Identifying **causally important regions** using **Grad-CAM**.
- Selecting **semantically dissimilar patches** using **LPIPS** (Low Perceptual Similarity) and **CLIP** (Contrastive Language-Image Pretraining).
- Replacing critical regions with these patches while preserving visual coherence.
- Quantifying prediction instability with the **Conceptual Fragility Index (CFI)**.

This framework is useful for:
- Evaluating CNN robustness to causal inconsistencies.
- Understanding overconfidence in models.
- Exploring generalization failure under semantic shifts.

---

## Project Structure

```
Stress-Testing-CNNs-Using-Causal-Concept-Subversion-CCS/
├── Intel_Resnet/
│   └── model                 # ResNet-18
├── EfficientNet_Intel/
│   └── model                 # EfficientNet-B0
├── notebooks/
│   └── casual-subversion-attack.ipynb     # Attack as well as Measurement
├── Results/
│   └── Resnet and EfficientNet Images
├── Report/
│   └── Report.pdf            # Report of CCS Attack and it's results
├── requirements.txt          # Dependencies
└── README.md                 # You are here!
```

---

## Conceptual Fragility Index (CFI)

The **Conceptual Fragility Index (CFI)** is a novel metric proposed in this work. It quantifies the **volatility in predictions** when a model is presented with causally inconsistent versions of the same image.

CFI is computed as:

\[
CFI = \alpha \cdot \text{Prediction Entropy Change} + \beta \cdot \text{Semantic Drift (CLIP distance)}
\]

Where:
- **Prediction Entropy Change** reflects uncertainty introduced by subversion.
- **Semantic Drift** measures how far the modified image embedding moves from the original in CLIP space.

---

## Datasets

The framework is evaluated on two datasets:

### 1. First In-Distribution Dataset (ID)
Used for clean baseline evaluation and Grad-CAM region extraction.

🔗 [**Kaggle Link** to Intel Image Dataset](https://www.kaggle.com/datasets/rahmasleam/intel-image-dataset) 

### 2. Seconf In-Distribution Dataset (ID)
Used for clean baseline evaluation and Grad-CAM region extraction.

🔗 [**Kaggle Link** to ImageNet Dataset](https://www.kaggle.com/datasets/dimensi0n/imagenet-256)

Each patch is chosen using LPIPS and CLIP to ensure **visual plausibility** and **semantic conflict**.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/CCS-Robustness.git
cd CCS-Robustness
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:
- PyTorch
- OpenCV
- torchvision
- lpips
- CLIP (via `openai/CLIP`)
- scikit-learn
- matplotlib

### 3. Download datasets

Place the downloaded datasets in the `data/` directory following the structure below:

```
data/
├── id_dataset/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ood_patches/
    ├── patch1.jpg
    ├── patch2.jpg
    └── ...
```

---

## Running the Experiment

```bash
python main.py --model resnet18 --config config.yaml
```

Arguments:
- `--model`: Choose between `resnet18` or `efficientnet_b0`.
- `--config`: Path to the YAML file specifying parameters such as alpha/beta for CFI, LPIPS/CLIP thresholds, Grad-CAM layers, etc.

---

## Results & Visualizations

- Heatmaps showing Grad-CAM before and after subversion.
- CLIP similarity graphs for patch selection.
- CFI trends across multiple images.
- Confidence scores and entropy visualizations.

---

## Key Findings

- ResNet-18 is **overconfident** and highly vulnerable to causal subversion.
- EfficientNet-B0, while slightly less accurate on clean inputs, shows **better calibration** under perturbations.
- Models depend more on **texture and local features** than meaningful global semantics.
- CCS reveals a **hidden brittleness** not exposed by conventional pixel-level attacks.

---

## Citation

If you find this project useful in your research, please consider citing:

```
@misc{causalconceptsubversion2025,
  title={Stress-Testing CNNs Using Causal Concept Subversion (CCS)},
  author={Pooja Yakkala},
  year={2025},
  note={https://github.com/your-username/CCS-Robustness},
}
```

---

## Contributing

Feel free to fork this repo, open issues, or submit PRs. Contributions are welcome!

---
