# SpoofNet – Depthwise Face Anti-Spoofing using PyTorch

SpoofNet is a CNN-based face anti-spoofing system built using PyTorch.
The model detects whether a face image is **real** or a **spoof attack** (e.g., replay/photo attack) using a hybrid deep learning architecture that combines semantic understanding with explicit texture modeling.

---

## Demo - [Click here](https://arjunjagdale.github.io/SPOOF/)

**Note:** Keep your face close to your webcam or phone camera (front or rear). The model performs best when the face-to-camera distance is approximately **15–25 cm**.

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

The network is intentionally hybrid: one branch captures **global semantic structure**, while the other focuses on **fine-grained texture inconsistencies**, which are critical in spoof detection.

Input size: **224 × 224**

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

# Results

* **Train Accuracy:** 98.86%
* **Test Accuracy:** 91.94%
* **AUC:** 0.9539

The strong AUC score indicates effective separation between real and spoof samples, while the train–test gap reflects realistic generalization behavior.

---

# 📦 Model Export

The trained PyTorch model is exported to ONNX:

* Opset version: 13
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
