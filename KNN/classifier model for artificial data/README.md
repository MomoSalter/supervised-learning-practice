# KNN Classifier

A K-Nearest Neighbors (KNN) binary classification project built with Python and scikit-learn. The model is trained on an artificial dataset for educational purposes and achieves **86% accuracy** with an optimally selected `k` value.


---

## Dataset

- **File:** `KNN_Project_Data` (CSV)
- **Rows:** 1,000
- **Features:** 10 continuous numerical columns — `XVPM`, `GWYH`, `TRAT`, `TLLZ`, `IGGA`, `HYKR`, `EDFS`, `GUUB`, `MGJM`, `JHZC`
- **Target:** `TARGET CLASS` — binary (0 or 1), perfectly balanced (500 each)
- **Note:** Data is artificially generated for educational purposes only

---

## Workflow

**1. Load & Explore Data**
- Load CSV with pandas
- Inspect shape, dtypes, and summary statistics (`head`, `info`, `describe`)
- Check class balance with `value_counts()`

**2. Exploratory Data Analysis**
- Pairplot with `seaborn` (colored by `TARGET CLASS`) to visualize feature relationships

**3. Preprocessing**
- Split features (`X`) and target (`y`)
- Train/test split: 80/20, stratified, `random_state=42`
- Feature scaling using `StandardScaler` (fit on train, transform both)

**4. Hyperparameter Tuning — Finding Optimal K**
- Loop over `k = 1` to `39`, train a KNN model for each
- Record the error rate (1 - accuracy) for each `k`
- Plot error rates using seaborn line plot to visually identify the elbow

**5. Train Final Model**
- Select the `k` with the lowest error rate → `k = 29`
- Retrain `KNeighborsClassifier(n_neighbors=29)` on the full training set

**6. Evaluate**
- Confusion matrix and classification report on the test set

---

## Results

| Metric       | Class 0 | Class 1 | Overall |
|--------------|---------|---------|---------|
| Precision    | 0.87    | 0.85    | —       |
| Recall       | 0.85    | 0.87    | —       |
| F1-Score     | 0.86    | 0.86    | —       |
| **Accuracy** | —       | —       | **0.86**|

**Confusion Matrix:**
```
[[85  15]
 [13  87]]
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MomoSalter)