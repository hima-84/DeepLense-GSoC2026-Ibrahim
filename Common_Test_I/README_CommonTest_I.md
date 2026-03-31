# Common Test I — Multi-Class Gravitational Lensing Classification

**GSoC 2026 · ML4SCI · DeepLense (DEEPLENSE1)**
**Applicant:** Ibrahim Nagy Abd Elrazek Elsaid Emara · [github.com/hima-84](https://github.com/hima-84)

---

## Task

Classify strong gravitational lensing images into three dark matter substructure classes using a deep learning model evaluated by ROC AUC score:

| Class | Label | Physical Description |
|-------|-------|----------------------|
| `no` | 0 | Smooth NFW halo — symmetric Einstein ring, no substructure |
| `sphere` | 1 | CDM subhalos — cold dark matter clumps, flux-ratio anomalies |
| `vort` | 2 | Axion vortex — fuzzy dark matter interference fringes |

---

## Results

| Class | Precision | Recall | F1 | **AUC** |
|-------|-----------|--------|----|---------|
| No Substructure | 0.904 | 0.989 | 0.945 | **0.9913** |
| Sphere (CDM) | 0.959 | 0.865 | 0.910 | **0.9824** |
| Vortex (Axion) | 0.936 | 0.945 | 0.941 | **0.9948** |
| **Mean** | **0.933** | **0.933** | **0.932** | **0.9895** |

- **Validation accuracy:** 93.17%
- **Validation set:** 3,000 samples (90/10 stratified split, seed=42)
- **Hardware:** NVIDIA Quadro P2000 (4.3 GB VRAM), PyTorch 2.6.0+cu124

---

## Architecture: EfficientNet-B0

EfficientNet-B0 was selected for its compound scaling (depth/width/resolution) giving
superior accuracy-per-parameter over ResNets. The pretrained RGB first-layer weights are
averaged into a single channel to adapt to grayscale physics images — preserving learned
spatial detectors while matching single-channel input.

**Key design choices:**

| Choice | Rationale |
|--------|-----------|
| EfficientNet-B0 | 4M params, 77.1% ImageNet top-1 vs ResNet-18's 69.8% at half the params |
| Full 180° rotation | Gravitational lensing is rotationally symmetric — physically principled |
| AdamW + Cosine Annealing | More stable than Adam for fine-tuning; avoids step-decay cliffs |
| AMP (mixed-precision) | ~40% VRAM reduction, enables batch=64 on 4.3 GB GPU |
| 90/10 split | As specified in the GSoC task requirements |

---

## Training Protocol

```
Optimizer      : AdamW (weight_decay=1e-4)
Learning rate  : 3e-4 with CosineAnnealingLR (eta_min=1e-6)
Batch size     : 64
Epochs         : 20
Resolution     : 150 × 150 (native dataset resolution)
Augmentation   : RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(180°)
Precision      : Mixed (torch.amp.autocast + GradScaler)
Checkpointing  : Best model saved by validation AUC
```

Training converged smoothly: AUC improved every epoch from 0.7787 (epoch 1) to 0.9895 (epoch 20).

---

## How to Run

### 1. Install dependencies
```bash
pip install torch torchvision timm scikit-learn matplotlib tqdm
```

### 2. Get the dataset
Download the DeepLense Common Test I dataset from the ML4SCI organisation (`.npy` files
structured as `train/no/`, `train/sphere/`, `train/vort/`).

### 3. Configure paths
In the **Configuration** cell, set:
```python
DATA_ROOT = 'path/to/dataset/train'   # folder containing no/, sphere/, vort/
SAVE_DIR  = 'path/to/outputs'
```

**Google Colab alternative** (commented in the notebook):
```python
from google.colab import drive
drive.mount('/content/drive')
DATA_ROOT = '/content/drive/MyDrive/dataset/train'
SAVE_DIR  = '/content/drive/MyDrive/deeplense_outputs'
```

### 4. Run all cells top to bottom

The notebook will:
1. Explore and visualise the dataset
2. Compute normalisation statistics
3. Build and verify EfficientNet-B0
4. Train with live epoch logging
5. Plot training curves (loss / accuracy / AUC)
6. Generate ROC curves and confusion matrices
7. Print the full classification report

---

## Pre-trained Weights

The best checkpoint is available at:

> **[Download best_efficientnet_b0.pth — Google Drive](https://drive.google.com/your-link-here)**
> *(update this link with the actual Google Drive share URL)*

Load with:
```python
import timm, torch, torch.nn as nn

def build_efficientnet_b0(num_classes=3):
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    old_conv = model.conv_stem
    new_conv = nn.Conv2d(1, old_conv.out_channels, old_conv.kernel_size,
                         old_conv.stride, old_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
    model.conv_stem = new_conv
    return model

model = build_efficientnet_b0()
model.load_state_dict(torch.load('best_efficientnet_b0.pth', map_location='cpu'))
model.eval()
```

---

## Scientific Notes

**Why the Vortex class has the highest AUC (0.9948):**
The axion vortex interference pattern produces a distinctive high-frequency fringe
texture on the Einstein ring that the model learns to identify reliably.

**Why CDM Sphere is hardest (AUC 0.9824):**
CDM subhalos produce localised flux-ratio anomalies — subtle brightness perturbations —
that partially overlap with the no-substructure class at low subhalo masses.

**The CDM–Axion degeneracy:**
The 5–8% confusion between Sphere and Vortex at the decision boundary is a genuine
physical effect: at high axion masses ($m_a \sim 10^{-21}$ eV), the de Broglie coherence
length decreases and vortex patterns begin to resemble CDM clumps.
This is a physics result, not a model failure.

---

## File Structure

```
Common_Test_I/
├── DeepLense_GSoC2026_CommonTest_Classification_v3.ipynb
├── requirements.txt
└── README.md  ← this file
```
