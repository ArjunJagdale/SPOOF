## 🛡️ SpoofNet - Anti-Spoofing Face Authentication Model

This project focuses on detecting **real vs spoof (fake)** face presentations using deep learning. The model is trained to identify spoof attacks such as printed photos, digital displays. The network achieves **high robustness and strong generalization** through custom CNN architecture and extensive data augmentation.


---

### 📂 Dataset

* **Total real images:** ~19,000
* **Total spoof images:** ~19,000
* Dataset split into:

  * **Train**
  * **Validation**
  * **Test**

Images were augmented heavily to simulate real-world variations like brightness, texture, motion blur, grayscale conditions, and occlusions.

---

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

**In Short**

* **Early layers** learn simple visual edges.
* **Middle layers** learn facial texture and surface depth cues.
* **Later layers** learn complex high-level differences between real skin and spoof artifacts.
* **Classifier** decides if the face is **real** or **fake**.


| Attribute                | Value                               |
| ------------------------ | ----------------------------------- |
| **Total Parameters**     | 4,881,442                           |
| **Trainable Parameters** | 4,881,442                           |
| **Activation**           | ReLU                                |
| **Loss Function**        | Label Smoothing Cross Entropy       |
| **Optimizer**            | AdamW + CosineAnnealingWarmRestarts |
| **Input Size**           | 224×224×3                           |

---

```
/project-root
│
├── assets/
│   ├── model_architecture.png   ← place your diagram here
│
├── training.ipynb
├── README.md
└── ...
```
---

### 🚀 Training Summary

| Metric                       | Value      |
| ---------------------------- | ---------- |
| **Best Validation Accuracy** | **0.9939** |
| **Test Accuracy**            | **0.9929** |

#### Training Progress (Key Epochs)

| Epoch | Train Acc | Val Acc    | LR       | Note                       |
| ----- | --------- | ---------- | -------- | -------------------------- |
| 1     | 0.8428    | 0.9679     | 0.000090 | Best model saved           |
| 2     | 0.9427    | 0.9875     | 0.000065 | Best model saved           |
| 3     | 0.9606    | 0.9899     | 0.000035 | Best model saved           |
| 4     | 0.9703    | **0.9939** | 0.000010 | **Final best model saved** |

---

### ✅ Model Evaluation

#### Classification Report

```
              precision    recall  f1-score   support
real            0.99       1.00      0.99      2928
spoof           1.00       0.99      0.99      2990
accuracy                            0.99      5918
```

#### Confusion Matrix

|                  | Predicted Real | Predicted Spoof |
| ---------------- | -------------- | --------------- |
| **Actual Real**  | 2926           | 2               |
| **Actual Spoof** | 40             | 2950            |

#### Confidence Scores

| Class     | Avg. Confidence |
| --------- | --------------- |
| **Real**  | 0.8865          |
| **Spoof** | 0.8926          |

---

### 🏃‍♂️ How to Run

```bash
pip install -r requirements.txt
python train.py
```

To test a single image:

```python
python predict.py --image path_to_image.jpg
```

---

### 🛠 Future Improvements

* Deployment as **real-time anti-spoofing** using ONNX or TFLite
* Add **depth estimation** to detect printed surfaces
* Implement **patch-level attention** for higher robustness

---

### 🤝 Contributions

Pull requests and feature improvements are welcome!

---

### 📜 License

This project is released under the **MIT License**.

---
