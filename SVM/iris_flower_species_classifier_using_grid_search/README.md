# Iris Species Classifier

A machine learning project that classifies iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — using a Support Vector Machine (SVM) with hyperparameter tuning via Grid Search.

---

## Overview

The Iris dataset is one of the most well-known datasets in the pattern recognition literature, first introduced by statistician R.A. Fisher in 1936. This project walks through the full ML pipeline: data loading, exploration, model training, hyperparameter optimization, and evaluation.

---

## Technologies Used

- **Python**
- **scikit-learn** — dataset, SVM, GridSearchCV, metrics
- **pandas** — data exploration and manipulation

---

## Dataset

- **Source:** `sklearn.datasets.load_iris`
- **Samples:** 150 (50 per class)
- **Features:** 4 numeric attributes
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target Classes:** Setosa, Versicolor, Virginica
- **Missing Values:** None

Petal length and petal width show the highest correlation with the target class (0.95 and 0.96 respectively), making them the most predictive features.

---

## Methodology

1. **Data Loading & Exploration** — Loaded the dataset into a pandas DataFrame and reviewed summary statistics.
2. **Train/Test Split** — Divided the data into training and testing sets.
3. **Model Training** — Trained a Support Vector Classifier (SVC).
4. **Hyperparameter Tuning** — Used `GridSearchCV` to search over:
   - `C`: [0.1, 1, 10, 100]
   - `gamma`: [1, 0.1, 0.01, 0.001]
5. **Best Parameters Found:** `C=1`, `gamma=0.1`

---

## Results

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Setosa      | 1.00      | 1.00   | 1.00     | 10      |
| Versicolor  | 0.90      | 0.90   | 0.90     | 10      |
| Virginica   | 0.90      | 0.90   | 0.90     | 10      |
| **Overall** | **0.93**  | **0.93** | **0.93** | **30** |

The tuned SVM achieved **93% accuracy** on the test set. Setosa was classified perfectly, while Versicolor and Virginica had one misclassification each — consistent with their known overlap in feature space.

---

## Install dependencies

```bash
pip install scikit-learn pandas jupyter
```

---

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MomoSalter)