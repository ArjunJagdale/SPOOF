# SpoofNet – Face Anti-Spoofing using PyTorch

SpoofNet is a CNN-based face anti-spoofing system built using PyTorch.
The model detects whether a face image is **real** or a **spoof attack** (e.g., replay/photo attack) using a hybrid deep learning architecture.

---

## Demo - [Click here](https://arjunjagdale.github.io/SPOOF/)
Note - make sure to have your face near your webcam or phone(front or rear cam). The inference works well if the distance between face and the device is between 15-25 cms.

## Overview

This project implements:

* A pretrained **MobileNetV3 (Small)** backbone for global feature extraction
* A **custom CNN texture branch** to capture mid-level spoof patterns
* Feature fusion of semantic + texture representations
* Binary classification using BCEWithLogitsLoss
* ONNX export for deployment

---

## Model Architecture

The architecture consists of two main components:

### 1️MobileNetV3 Backbone

* Pretrained `mobilenetv3_small_100`
* Extracts multi-level feature maps
* Final global pooled features used for classification

### Custom Texture CNN Branch (Self-Designed)

* Takes mid-level backbone features
* Two Conv–BatchNorm–ReLU blocks
* Adaptive average pooling
* Focuses on texture artifacts commonly present in spoof attacks

### Feature Fusion

* Concatenation of:

  * Global semantic features
  * Mid-level texture features
* Followed by:

  * Fully connected layer
  * Dropout (0.3)
  * Final binary output layer

Input size: **224 × 224**

---

## Dataset Structure

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

## 🔧 Training Details

* Framework: PyTorch
* Loss: `BCEWithLogitsLoss`
* Optimizer: `AdamW` (lr = 1e-4, weight_decay = 1e-4)
* Scheduler: `CosineAnnealingLR`
* Batch size: 64
* Epochs: 20
* Image size: 224×224
* Data augmentation:

  * Random horizontal flip
  * Random resized crop
  * Color jitter
  * Normalization (ImageNet stats)

---

## Results

* **Training Accuracy:** 99.96%
* **Test Accuracy:** 88.71%
* **ROC-AUC:** 0.9603

The high AUC indicates strong separability between real and spoof samples.

---

## 📦 Model Export

The trained PyTorch model is exported to ONNX:

* Opset version: 13
* Dynamic batch size support
* Verified numerical consistency using ONNX Runtime

This allows deployment in lightweight or browser-based environments.

---

## 📌 Key Highlights

* Hybrid CNN design (pretrained backbone + custom convolutional branch)
* Explicit texture modeling for spoof detection
* Lightweight and deployment-ready
* Reproducible training with fixed seeds

---

Your move now is to utilize this model, thank!!.
