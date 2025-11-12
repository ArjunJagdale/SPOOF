# SpoofNet: Deep Learning Model for Face Anti-Spoofing

A custom deep learning architecture designed to detect presentation attacks (spoofing) in facial recognition systems. SpoofNet achieves state-of-the-art performance in distinguishing between genuine faces and spoofed presentations (printed photos, digital screens, etc.).

## 🎯 Project Overview

SpoofNet is a convolutional neural network specifically designed to identify subtle visual cues that differentiate real faces from spoofed presentations, including:
- Screen glare and pixelation from digital displays
- Paper texture and print artifacts
- Unnatural lighting and color reproduction
- Flat appearance lacking natural depth
- Moiré patterns and aliasing effects

---
## Demo Video
https://github.com/user-attachments/assets/57018240-5341-4244-99ba-c6310b78f405

---

## 📊 Dataset Statistics

| Split | Images | Percentage |
|-------|--------|------------|
| **Training** | 14,571 | ~70% |
| **Validation** | 3,122 | ~15% |
| **Test** | 3,126 | ~15% |
| **Total** | 20,819 | 100% |

**Classes:**
- `fake` (class 0): Spoofed presentations
- `real` (class 1): Genuine faces

**Model Specifications:**
- **Total Parameters:** 4,881,442
- **Trainable Parameters:** 4,881,442
- **Model Size:** 18.62 MB (FP32)
- **Input Resolution:** 224×224 RGB images
- **Output:** Binary classification (real/fake)

### 🧠 Model Architecture
<img width="1807" height="717" alt="image" src="https://github.com/user-attachments/assets/81685908-ef93-46e8-91ab-45321926c7aa" />

The custom CNN model, **SpoofNet**, consists of 5 convolutional blocks followed by global average pooling and dense classification layers with dropout and batch normalization.
Here is a **simple and clear explanation** of the architecture shown in your diagram—no technical overload, just the core idea:

---

### **SpoofNet Architecture (Simple Explanation)**

The model is made up of **five main convolutional blocks**, each one learning deeper and more detailed features from the input face image. As we move from left to right in the diagram, the number of feature channels increases, meaning the network learns more complex visual patterns at each stage.

**1. Convolutional Blocks (Feature Extraction)**

Each block has the same pattern:

1. **Two Conv2D layers (3×3 filters)**

   * These learn visual patterns like edges, textures, and surface details.
2. **Batch Normalization**

   * Keeps training stable and reduces sensitivity to lighting conditions.
3. **ReLU Activation**

   * Helps the model learn non-linear relationships.
4. **Max Pooling (2×2)**

   * Shrinks the image size, keeping only the most important features.
5. **Dropout2D**

   * Randomly "drops" parts of the feature maps to prevent overfitting.

| Block   | Input → Output Channels | Purpose                                                   |
| ------- | ----------------------- | --------------------------------------------------------- |
| Block 1 | 3 → 32                  | Detect basic edges & colors.                              |
| Block 2 | 32 → 64                 | Learn slightly more complex textures.                     |
| Block 3 | 64 → 128                | Capture facial pore & skin depth cues.                    |
| Block 4 | 128 → 256               | Learn detailed texture differences between real vs spoof. |
| Block 5 | 256 → 512               | High-level deep feature representation.                   |

---

**2. Global Average Pooling**

After the convolution blocks, the feature map is reduced to a **1×1 feature vector** representing the entire image.
This reduces parameters and prevents overfitting.

---

**3. Fully Connected Classifier (Decision Making)**

The flattened feature vector is passed through three linear layers:

| Layer                                           | Transformation                  | Purpose                                  |
| ----------------------------------------------- | ------------------------------- | ---------------------------------------- |
| Linear (512 → 256) + BatchNorm + ReLU + Dropout | First level abstraction         | Avoids overfitting & stabilizes training |
| Linear (256 → 128) + BatchNorm + ReLU + Dropout | Refines class-specific features | Helps learn more discriminative differences|
| Linear (128 → 2)                                | Final output                    | Predicts **Real (0)** or **Spoof (1)**   |

---

**Final Output**

The model outputs **two class scores**, which are converted into:

* **Real Face**
* **Spoof Face**

---


| Attribute                | Value                               |
| ------------------------ | ----------------------------------- |
| **Total Parameters**     | 4,881,442                           |
| **Trainable Parameters** | 4,881,442                           |
| **Activation**           | ReLU                                |
| **Loss Function**        | Label Smoothing Cross Entropy       |
| **Optimizer**            | AdamW + CosineAnnealingWarmRestarts |
| **Input Size**           | 224×224×3                           |

---
## 🎨 Data Augmentation Strategy

The training pipeline incorporates sophisticated augmentation techniques specifically targeting spoof detection:

**Geometric Augmentations:**
- Random horizontal flips (50% probability)
- Random rotation (±10°)
- Random affine transformations
- Random perspective distortion (20% probability)

**Color Augmentations:**
- Brightness adjustment (±20%)
- Contrast variation (±20%)
- Saturation changes (±15%)
- Hue shifts (±3%)

**Quality Augmentations:**
- Gaussian blur (30% probability) - simulates camera focus and screen blur
- Random grayscale (5% probability) - sensor simulation
- Random erasing (15% probability) - occlusion simulation

## 🚀 Training Configuration

**Hyperparameters:**
- **Batch Size:** 16
- **Epochs:** 5 (with early stopping)
- **Learning Rate:** 0.0001
- **Optimizer:** AdamW with weight decay (1e-4)
- **Scheduler:** Cosine Annealing Warm Restarts
- **Loss Function:** Label Smoothing Cross-Entropy (smoothing=0.1)
- **Regularization:** Dropout (0.1-0.5), Gradient Clipping (max_norm=1.0)

## 📈 Performance Metrics

### Validation Performance
- **Best Validation Accuracy:** 99.97%

### Test Set Performance
- **Test Accuracy:** 99.94%

### Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Fake** | 1.0000 | 0.9987 | 0.9994 | 1,546 |
| **Real** | 0.9987 | 1.0000 | 0.9994 | 1,580 |
| **Overall** | **0.9994** | **0.9994** | **0.9994** | **3,126** |

### Confusion Matrix

|  | Predicted Fake | Predicted Real |
|---|----------------|----------------|
| **Actual Fake** | 1,544 | 2 |
| **Actual Real** | 0 | 1,580 |

**Error Analysis:**
- **True Negatives:** 1,544 (correctly identified fake images)
- **False Positives:** 2 (real images incorrectly classified as fake)
- **False Negatives:** 0 (fake images incorrectly classified as real)
- **True Positives:** 1,580 (correctly identified real images)

**Key Insights:**
- Zero false negatives indicate perfect detection of genuine faces
- Only 2 false positives out of 1,546 fake samples (0.13% false alarm rate)
- Near-perfect balance between precision and recall

## 🛠️ Technical Implementation

**Framework:** PyTorch

**Key Features:**
- Batch normalization for training stability
- Dropout layers for regularization
- Global average pooling to reduce parameters
- Gradient clipping to prevent exploding gradients
- Mixed precision training support
- Early stopping with patience=7

## 📋 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
tqdm>=4.62.0
```

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{spoofnet2024,
  title={SpoofNet: A Deep Learning Approach to Face Anti-Spoofing},
  author={Arjun Jagdale},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgments

- Dataset: SPOOFNET_DATASET_SPLIT
- Built with PyTorch and standard computer vision libraries
- Inspired by state-of-the-art face anti-spoofing research

---

**Note:** This model is designed for research and development purposes. For production deployment, additional testing across diverse scenarios and demographic groups is recommended.
