# 🩺 ADLAH - Skin Cancer Prediction: A Deep Learning Approach on the HAM10000 Dataset

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

## 📖 Overview

ADLAH (**A Deep Learning Approach on HAM10000**) is a deep learning project developed to classify skin lesions into **seven different skin cancer categories** using dermoscopic images from the **HAM10000 dataset**.

The project investigates the effectiveness of several state-of-the-art pretrained Convolutional Neural Networks (CNNs) and compares them with a lightweight custom CNN architecture specifically designed for the dataset.

Despite using significantly fewer parameters and a much smaller image resolution, the proposed custom CNN achieved **98.89% classification accuracy**, outperforming all pretrained models evaluated in this study.

---

# 🎯 Objectives

- Develop an automated skin lesion classification system.
- Evaluate the effectiveness of multiple pretrained CNN architectures.
- Design a lightweight custom CNN optimized for the HAM10000 dataset.
- Compare model performance across multiple train-test splits.
- Analyze the impact of preprocessing and dataset balancing on classification accuracy.

---

# 📂 Dataset

**Dataset Used**

HAM10000 (Human Against Machine with 10000 Training Images)

The dataset contains dermoscopic images of seven common pigmented skin lesions.

### Original Dataset

- Total Images: **10,015**
- Metadata CSV containing:
  - Image ID
  - Diagnosis
  - Age
  - Gender
  - Lesion Location
  - Additional patient information

### Classes

- Melanocytic Nevi (nv)
- Melanoma (mel)
- Benign Keratosis-like Lesions (bkl)
- Basal Cell Carcinoma (bcc)
- Actinic Keratoses (akiec)
- Vascular Lesions (vasc)
- Dermatofibroma (df)

### Dataset Challenge

The original HAM10000 dataset is highly imbalanced.

Example:

- Melanocytic Nevi contains more than **6000 images**
- Remaining six classes together contain only **4015 images**

---

# 🖼 Dataset Preprocessing

The following preprocessing pipeline was implemented.

### 1. Image Resizing

Images were resized to fixed dimensions.

- 96 × 96 × 3
  - Used for pretrained CNN models

- 28 × 28 × 3
  - Used for the custom CNN model

---

### 2. Image Normalization

Pixel values were normalized from

```
0–255
```

to

```
0–1
```

to improve convergence during training.

---

### 3. Dataset Balancing

To overcome severe class imbalance, extensive image augmentation was performed.

Augmentation techniques include:

- Rotation
- Horizontal Flip
- Vertical Flip
- Zoom
- Translation

After balancing,

**Dataset Size**

```
Original Dataset
10,015 Images

↓

Balanced Dataset
46,936 Images
```

All seven classes contain nearly equal numbers of samples.

---

### 4. Color Space Standardization

Images were standardized into RGB format to maintain consistency across the dataset.

---

# 🧠 Models Evaluated

## Pretrained Models

The following transfer learning models were evaluated:

- XceptionNet
- ShuffleNet V2
- ResNet50
- MobileNetV2
- EfficientNetB0
- DenseNet121

All pretrained models were trained using identical hyperparameters for fair comparison.

### Hyperparameters

| Parameter | Value |
|------------|---------|
| Image Size | 96×96 |
| Batch Size | 32 |
| Learning Rate | 0.0001 |
| Epochs | 20 |
| Dropout | 0.5 |

---

# ⚙ Custom CNN Architecture

The proposed custom CNN consists of:

```
Input (28×28×3)

↓

Conv2D (16 Filters)
↓

MaxPooling

↓

Conv2D (32 Filters)
↓

MaxPooling

↓

Conv2D (64 Filters)
↓

MaxPooling

↓

Conv2D (128 Filters)
↓

MaxPooling

↓

Flatten

↓

Dense (64)

↓

Dense (32)

↓

Softmax (7 Classes)
```

### Model Configuration

| Parameter | Value |
|------------|---------|
| Input Size | 28×28×3 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Epochs | 25 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |

---

# 📊 Results

## Test Accuracy

| Model | 80-20 | 70-30 | 60-40 |
|--------|-------|-------|-------|
| XceptionNet | 46.66% | 46.70% | 45.82% |
| ShuffleNet V2 | 58.24% | 55.44% | 57.98% |
| ResNet50 | 25.31% | 24.82% | 24.69% |
| MobileNetV2 | 49.37% | 47.59% | 46.41% |
| EfficientNetB0 | 14.29% | 14.29% | 14.29% |
| DenseNet121 | 52.14% | 51.67% | 50.82% |
| **Custom CNN** | **98.89%** | **98.60%** | **98.24%** |

---

# 🏆 Best Model Performance

### Custom CNN

**80–20 Split**

- Accuracy: **98.89%**

Classification Metrics

| Metric | Score |
|---------|---------|
| Precision | 99% |
| Recall | 99% |
| F1 Score | 99% |

The custom CNN consistently maintained over **98% accuracy** across all train-test splits, demonstrating strong robustness and generalization.

---

# 💻 Technologies Used

## Programming Language

- Python

## Deep Learning Frameworks

- TensorFlow
- Keras

## Libraries

- NumPy
- Pandas
- OpenCV
- Pillow
- Matplotlib
- Scikit-Learn

## Development Environment

- Kaggle Notebook
- Jupyter Notebook

---

# 📈 Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

---

# 📌 Project Highlights

✅ Built a complete end-to-end deep learning pipeline.

✅ Preprocessed and balanced the HAM10000 dataset from **10,015** to **46,936** images.

✅ Benchmarked **six pretrained CNN architectures**.

✅ Designed a lightweight custom CNN from scratch.

✅ Achieved **98.89% classification accuracy**.

✅ Demonstrated that a task-specific CNN can outperform larger pretrained networks on this dataset.

---

# 🚀 Future Work

Possible future improvements include:

- Continual Learning for incremental model updates
- Explainable AI using Grad-CAM
- Mobile deployment using TensorFlow Lite
- Clinical decision support integration
- Ensemble learning with multiple CNN architectures
- Hyperparameter optimization using Bayesian Optimization

---

# 👨‍💻 Author

**Project:** ADLAH – Skin Cancer Prediction: A Deep Learning Approach on the HAM10000 Dataset

Developed as a deep learning research project focusing on efficient and accurate skin lesion classification using convolutional neural networks.

---
