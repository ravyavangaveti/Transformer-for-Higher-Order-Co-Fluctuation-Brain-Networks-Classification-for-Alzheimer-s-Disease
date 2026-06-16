# 🌿 Plant Disease Prediction with Deep Neural Networks

A deep learning project that automatically detects plant diseases from leaf images using Convolutional Neural Networks (CNN) and Transfer Learning (MobileNetV2).

---

## 📌 Project Overview

Plant diseases spread fast, reduce crop yield, and often require expert diagnosis. This project builds an AI model that:
- Analyzes leaf images
- Recognizes disease patterns
- Predicts the correct disease category
- Helps enable early detection and prevention

The goal is to help **farmers and researchers** quickly identify plant diseases without needing an agriculture expert on hand.

---

## 📂 Dataset

- **Name:** PlantVillage Dataset
- **Total Images:** 43,000+
- **Classes:** 38 (plant disease + healthy categories)

---

## 🧠 Models Used

### 1. Custom CNN (Built from Scratch)
| Layer | Output Shape |
|-------|-------------|
| Conv2D | (None, 222, 222, 32) |
| MaxPooling2D | (None, 111, 111, 32) |
| Conv2D | (None, 109, 109, 64) |
| MaxPooling2D | (None, 54, 54, 64) |
| Flatten | (None, 186624) |
| Dense(256) | (None, 256) |
| Dense(38) | (None, 38) |

- **Total Parameters:** 47,805,158
- **Training Accuracy:** ~98%
- **Validation Accuracy:** ~87%

---

### 2. Transfer Learning — MobileNetV2
- Pretrained on ImageNet
- Frozen base layers
- Added: GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(38)
- Much faster convergence
- **Final Validation Accuracy: ~95.3%**

---

## 📊 Results

| Model | Validation Accuracy |
|-------|-------------------|
| Custom CNN | ~87% |
| MobileNetV2 (Transfer Learning) | ~95% |

✅ MobileNetV2 outperformed the custom CNN due to pretrained ImageNet features, lightweight architecture, and faster convergence.

---

## 🔍 Sample Predictions

The model correctly identified the following diseases from unseen test images:

| Plant | Disease |
|-------|---------|
| Corn | Common Rust |
| Orange | Citrus Greening (HLB) |
| Apple | Cedar Apple Rust |
| Apple | Black Rot |
| Blueberry | Healthy |
| Cherry | Healthy |
| Grape | Black Measles (Esca) |

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- Matplotlib
- NumPy
- PlantVillage Dataset

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ravyavangaveti/Plant_Disease_Prediction_Dnn.git
   cd Plant_Disease_Prediction_Dnn
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

3. Open the notebook:
   ```bash
   jupyter notebook plant_disease_prediction.ipynb
   ```

---

## 🔮 Future Work

- Expand with more diverse datasets
- Implement real-time disease detection via mobile app
- Fine-tune MobileNetV2 with unfrozen layers for higher accuracy

---

## 👩‍💻 Author

**Ravya Vangaveti**  
M.S. Computer Science (Data Science & Machine Learning)  
University of North Carolina at Greensboro  
[GitHub](https://github.com/ravyavangaveti)
