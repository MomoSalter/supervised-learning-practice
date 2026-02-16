# Diabetes Diagnostic Model

A machine learning classification model that predicts diabetes diagnosis based on medical diagnostic measurements.

## Overview

This project builds a binary classification model to predict whether a patient has diabetes based on various health indicators. The model uses logistic regression with class balancing to handle the inherent imbalance in medical diagnostic datasets.

## Dataset

The dataset contains 768 patient records with the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic influence)
- **Age**: Age in years
- **Outcome**: Class variable (0 or 1) indicating diabetes diagnosis

## Methodology

### 1. Exploratory Data Analysis
- Loaded and inspected the diabetes dataset
- Examined data types, missing values, and statistical distributions
- Analyzed feature correlations and patterns

### 2. Data Preprocessing
- Split data into training (80%) and testing (20%) sets
- Standardized features using StandardScaler for improved model performance
- Applied stratified sampling to maintain class distribution

### 3. Model Training
- Algorithm: Logistic Regression
- Hyperparameters:
  - `class_weight='balanced'`: Addresses class imbalance
  - `max_iter=1000`: Ensures convergence
- Training set used for model fitting

### 4. Model Evaluation
The model was evaluated on the test set using multiple metrics:

| Metric | Score |
|--------|-------|
| **Accuracy** | 73.38% |
| **Recall** | 70.37% |
| **Precision** | 60.32% |

## Key Findings

- The model achieves reasonable accuracy for diabetes prediction
- High recall (70.37%) indicates the model is effective at identifying actual diabetes cases, minimizing false negatives
- The precision-recall tradeoff reflects the model's tendency to err on the side of caution, which is appropriate for medical diagnostics

## Technologies Used

- **Python**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning implementation
- **StandardScaler**: Feature scaling
- **LogisticRegression**: Classification algorithm

## required dependencies:

```bash
pip install pandas scikit-learn jupyter
```

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MomoSalter)
