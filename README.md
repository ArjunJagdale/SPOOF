# SpoofNet - Depthwise Face Anti-Spoofing using PyTorch

SpoofNet is a CNN-based face anti-spoofing system built using PyTorch.
The model detects whether a face image is **real** or a **spoof attack** (e.g., replay/photo attack) using a hybrid deep learning architecture that combines semantic understanding with explicit texture modeling.

### live demo - 
[click here](https://arjunjagdale.github.io/SPOOF/) for the live demo, Make sure the face is visible to the webcam with a good lightning!

---

# Results
* **Train Acc** : 99.03%
* **Test  Acc** : 92.86%
* **AUC** : 0.9792 | at epoch 13/20
✓ Best model saved (acc=92.86%)

## Website Demo - 
> below photo was shown from redmi 9 prime and it is classified as spoof, as expected
<img width="1260" height="558" alt="Screenshot 2026-03-20 212632" src="https://github.com/user-attachments/assets/420dd03a-5d4e-4541-83cf-4f2b7e2b7cc5" />
<img width="1260" height="589" alt="image" src="https://github.com/user-attachments/assets/1ef5bff2-1962-44cd-a947-dbfd171b5f7a" />


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
<img width="1524" height="853" alt="image" src="https://github.com/user-attachments/assets/d40c165c-1070-480b-8d28-8008e5dbe4d8" />

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

# Model Export

The trained PyTorch model is exported to ONNX:

* Opset version: 18
* Dynamic batch size support
* Verified using ONNX Runtime

