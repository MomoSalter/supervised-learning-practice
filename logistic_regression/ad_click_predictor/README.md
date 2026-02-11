# Ad Click Prediction Model

A machine learning project that predicts whether an internet user will click on an advertisement based on user behavioral features using Logistic Regression.

## Project Overview

This project implements a binary classification model to predict ad click-through rates. By analyzing user demographics and online behavior patterns, the model achieves high accuracy in predicting user engagement with advertisements.

## Objective

Predict whether a user will click on an advertisement based on the following features:
- Daily time spent on site
- Age
- Area income
- Daily internet usage
- Gender

## Dataset

The project uses a simulated advertising dataset containing 1,000 records with the following attributes:

| Feature | Description |
|---------|-------------|
| Daily Time Spent on Site | Time spent by user on the website (minutes) |
| Age | User's age |
| Area Income | Average income of user's geographical area |
| Daily Internet Usage | Average daily internet consumption (minutes) |
| Ad Topic Line | Headline of the advertisement |
| City | User's city |
| Male | Gender (1 = Male, 0 = Female) |
| Country | User's country |
| Timestamp | Time when user clicked or didn't click |
| **Clicked on Ad** | Target variable (1 = Clicked, 0 = Not Clicked) |

## Technologies Used

- **Python**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning model and evaluation metrics

## Methodology

1. **Data Loading & Exploration**
   - Import and examine the advertising dataset
   - Check data types and missing values
   - Generate descriptive statistics

2. **Exploratory Data Analysis (EDA)**
   - Visualize distributions of numerical features
   - Analyze relationships between variables using joint plots and pair plots
   - Identify patterns and correlations

3. **Data Preprocessing**
   - Select relevant features for modeling
   - Split data into training (70%) and testing (30%) sets

4. **Model Training**
   - Implement Logistic Regression classifier
   - Train the model with maximum 1000 iterations

5. **Model Evaluation**
   - Assess performance using multiple metrics
   - Generate predictions on test data

## Results

The Logistic Regression model achieved excellent performance:

| Metric | Score |
|--------|-------|
| **Accuracy** | 97% |
| **Precision** | 97% |
| **Recall** | 97% |

## Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MomoSalter)
