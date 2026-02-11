# House Price Prediction

A machine learning project that predicts house prices using linear regression on the Ames Housing dataset. This project demonstrates data preprocessing, feature engineering, and model evaluation techniques.

## Project Overview

This project implements a linear regression model to predict residential property sale prices based on 79 explanatory variables from the Ames Housing dataset. The model analyzes various property features including:

- Property characteristics (lot size, square footage, year built)
- Quality and condition metrics
- Location attributes (zoning, neighborhood)
- Structural features (rooms, garage, basement)
- Sale conditions and timing

## Technologies Used

- **Python**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib** - Data visualization
- **scikit-learn** - Machine learning implementation
  - LinearRegression - Model algorithm
  - StandardScaler - Feature scaling
  - OneHotEncoder - Categorical encoding
  - train_test_split - Data splitting
  - Metrics: MSE, RMSE, R²

## Workflow

1. **Data Loading** - Import training and test datasets
2. **Data Preprocessing** - Handle missing values, encode categorical variables, scale numerical features
3. **Model Training** - Train Linear Regression on preprocessed data
4. **Validation** - Evaluate performance on validation set
5. **Prediction** - Generate predictions for test set and export to CSV

## Results

### Model Performance

**Training Set:**
- RMSE: 0.48
- R²: 0.9999

**Validation Set:**
- RMSE: 41,710
- R²: 0.6955

### Analysis

The model exhibits **overfitting**, achieving near-perfect accuracy on training data but reduced performance on validation data. This is expected given:
- Small training dataset relative to feature count
- 79 features creating high dimensionality
- Linear model limitations with complex feature interactions

## Getting Started

### Installation

```bash
# Install dependencies
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Usage

1. Ensure your data files (`train.csv` and `test.csv`) are in the `data/` directory
2. Launch Jupyter Notebook:
```bash
jupyter notebook House_price.ipynb
```
3. Run all cells sequentially to train the model and generate predictions
4. Find predictions in `output.csv`

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MomoSalter)
