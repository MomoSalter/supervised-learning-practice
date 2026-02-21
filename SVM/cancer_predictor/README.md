# Breast Cancer Classifier

A machine learning project that uses a **Support Vector Machine (SVM)** to classify breast tumors as **malignant or benign** based on digitized cell nucleus features.

---

## Overview

Breast cancer is one of the most common cancers worldwide. Early and accurate classification of tumors can be critical for treatment outcomes. This project applies a supervised ML pipeline to the well-known **Breast Cancer Wisconsin (Diagnostic)** dataset to demonstrate how SVMs can solve binary medical classification problems effectively.

---

## Dataset

- **Source:** `sklearn.datasets.load_breast_cancer()` (UCI ML Repository)
- **Samples:** 569
- **Features:** 30 numeric attributes derived from digitized images of fine needle aspirates (FNA) of breast masses
- **Target Classes:** Malignant (212 samples) | Benign (357 samples)
- **Missing Values:** None

Features capture properties of cell nuclei such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension — each measured as the mean, standard error, and worst value across the image.

---

## Tech Stack

- **Python**
- **pandas** — data loading and exploration
- **scikit-learn** — model training, scaling, evaluation

---

## Project Pipeline

1. **Load & Explore Data** — inspect shape, feature types, class distribution, and descriptive statistics
2. **Train/Test Split** — split dataset into training and test sets
3. **Preprocessing** — normalize features using `StandardScaler` to prepare data for the SVM kernel
4. **Model Training** — fit an `SVC` (Support Vector Classifier) on the training data
5. **Evaluation** — assess model performance using a confusion matrix and classification report

---

## Results

| Metric | Malignant | Benign | Overall |
|---|---|---|---|
| Precision | 0.95 | 0.96 | 0.96 |
| Recall | 0.94 | 0.97 | — |
| F1-Score | 0.94 | 0.97 | 0.96 |
| **Accuracy** | — | — | **96%** |

**Confusion Matrix:**
```
          Predicted
           M    B
Actual M  [60   4]
       B  [ 3  104]
```

---

## Getting Started

### Prerequisites
```bash
pip install scikit-learn pandas
```

### Run the Notebook
```bash
jupyter notebook cancer_predictor.ipynb
```

---

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MomoSalter)