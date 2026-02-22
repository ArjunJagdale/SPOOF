# SpoofNet – Depthwise Face Anti-Spoofing using PyTorch

SpoofNet is a CNN-based face anti-spoofing system built using PyTorch.
The model detects whether a face image is **real** or a **spoof attack** (e.g., replay/photo attack) using a hybrid deep learning architecture that combines semantic understanding with explicit texture modeling.

---

# Results
* **Train Acc** : 99.03%
* **Test  Acc** : 92.86%
* **AUC** : 0.9792 | at epoch 13/20
✓ Best model saved (acc=92.86%)

---

## Overview

This project implements:

* A pretrained **MobileNetV3 (Small)** backbone for global feature extraction
* A custom **depthwise-separable CNN texture branch** for spoof artifact detection
* Feature fusion of semantic + texture representations
* Binary classification using `BCEWithLogitsLoss`
* ONNX export for lightweight deployment

---

# Model Architecture

### Input → [64, 3, 224, 224]
64 normalized RGB images, ready for the backbone.

### MobileNetV3-Small Backbone (features_only)
The backbone runs the full forward pass internally through its inverted residual / hard-swish blocks, but only returns intermediate feature maps. You get two taps:

- features[2] — stage 3 output → [64, 40, 28, 28] — richer spatial resolution, mid-level textures
- features[-1] — final stage output → [64, 96, 7, 7] — deep semantic features, small spatial grid

From here the network splits into two parallel paths.

### Texture Branch (left, from features[2])
- Conv #1 + BN + ReLU [64,40,28,28] → [64,64,28,28]
Projects 40 backbone channels up to 64. 3×3 conv captures local spatial patterns.
- Conv #2 + BN + ReLU → stays [64,64,28,28]
Deepens representation without changing spatial size.
- Conv #3 + BN + ReLU → stays [64,64,28,28]

Third layer in the texture stack — extracts richer texture cues (liveness artifacts, moiré, print texture).

### DepthwiseBlock(64) — Residual → stays [64,64,28,28]
This is the key block. Internally: DW Conv 3×3 (groups=64) → BN → ReLU → PW Conv 1×1 → BN → ReLU. Then adds the original x back (skip connection). Efficient: DW handles spatial mixing, PW handles channel mixing. Residual prevents degradation.

### AdaptiveAvgPool2d(1) [64,64,28,28] → [64,64,1,1]
Collapses the 28×28 spatial grid into a single number per channel — position-invariant texture summary.

### Flatten → [64, 64] — texture_feat

### Main Path (right, from features[-1])
- AdaptiveAvgPool2d(1) [64,96,7,7] → [64,96,1,1]
- Standard global average pool on the deep semantic features.
- view/Flatten → [64, 96] — main_feat

### Concatenate [64,64] + [64,96] → [64, 160]
Joins texture-specific features with global semantic features along dim=1.
- Linear(160→128) + ReLU + Dropout(0.3) → [64, 128]
Maps the combined 160-d vector to 128 with nonlinearity. Dropout regularizes during training.
- Linear(128→1) → [64, 1]
Raw logit per image. Sigmoid gives P(real). During training this feeds into BCEWithLogitsLoss.

---

![SpoofNet](https://github.com/user-attachments/assets/824735fa-cb29-401c-b81e-2fffcb137362)

---

## 1️⃣ MobileNetV3 Backbone (Semantic Branch)

* Pretrained `mobilenetv3_small_100`
* Extracts hierarchical feature maps
* Final global pooled embedding represents high-level semantic structure (face shape, alignment, structure consistency)

This branch learns **who/what the face looks like structurally**, but not necessarily the spoof artifacts.

---

## 2️⃣ Custom Depthwise Texture CNN Branch (Self-Designed)

This branch is explicitly designed to capture **micro-texture artifacts**, such as:

* Moiré patterns
* Pixelation
* Reflection artifacts
* Screen refresh noise
* Print texture irregularities

### Why Depthwise Separable Convolutions?

Instead of standard convolutions, this branch uses **depthwise separable convolutions**, which split convolution into two operations:

### Step 1 – Depthwise Convolution

* One spatial filter per input channel
* Learns spatial patterns independently per channel
* Efficient at capturing localized texture features

### Step 2 – Pointwise Convolution (1×1)

* Mixes information across channels
* Learns inter-channel relationships

This design:

* Reduces parameters and FLOPs significantly
* Preserves spatial sensitivity
* Makes the model lightweight and deployment-friendly
* Enhances focus on spoof-related micro-patterns

### Texture Branch Structure

* Depthwise Conv → BatchNorm → ReLU
* Pointwise Conv (1×1) → BatchNorm → ReLU
* Repeated block
* Adaptive Average Pooling
* Feature vector output

This branch operates on **mid-level backbone features**, where spoof artifacts are most visible.

---

## 3️⃣ Feature Fusion

The final classifier receives:

* Global semantic features (from backbone)
* Texture-sensitive features (from depthwise branch)

These vectors are concatenated and passed through:

* Fully connected layer
* Dropout (0.3)
* Final binary output layer

This fusion allows the model to reason both structurally and texturally before making a decision.

---

# Dataset Structure

```
FULL_DATASET_FRAMES/
│
├── train/
│   ├── real/
│   └── attack/
│
└── test/
    ├── real/
    └── attack/
```

Each identity folder contains extracted video frames.

---

# 🔧 Training Details

* Framework: PyTorch
* Loss: `BCEWithLogitsLoss`
* Optimizer: `AdamW` (lr = 1e-4, weight_decay = 1e-4)
* Scheduler: `CosineAnnealingLR`
* Batch size: 64
* Epochs: 20
* Image size: 224×224

### Data Augmentation

* Random horizontal flip
* Random resized crop
* Color jitter
* Normalization (ImageNet statistics)

Augmentation ensures robustness against lighting variation, pose shifts, and camera differences.

---

The strong AUC score indicates effective separation between real and spoof samples, while the train–test gap reflects realistic generalization behavior.

---

# 📦 Model Export

The trained PyTorch model is exported to ONNX:

* Opset version: 18
* Dynamic batch size support
* Verified using ONNX Runtime

This enables:

* Browser-based inference
* Edge deployment
* Real-time webcam spoof detection

---

# 📌 Key Highlights

* Hybrid architecture (pretrained backbone + depthwise texture branch)
* Explicit texture modeling for spoof detection
* Computationally efficient design
* Deployment-ready via ONNX
* Reproducible training pipeline

---

SpoofNet is designed not merely to classify faces, but to detect deception embedded in texture.

Use it wisely.
