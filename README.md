# Predicting Automobile Prices Using Multi-model Machine Learning

## Author

**Md. Faisal Iftekhar**  
Department of Computer Science and Engineering  
BRAC University, Dhaka, Bangladesh  
Email: [md.faisal.iftekhar@g.bracu.ac.bd](mailto:md.faisal.iftekhar@g.bracu.ac.bd)

---

## Abstract

This project presents a comparative analysis of four machine learning regression models for predicting automobile prices using the **UCI Automobile Dataset**. The models evaluated are:

- Decision Tree Regressor
- Random Forest Regressor
- K-Neighbors Regressor
- Support Vector Regressor (SVR)

Each model was trained and tested after proper data preprocessing and feature engineering. Evaluation was performed using **Root Mean Squared Error (RMSE)**, **R-squared (R²)**, and **Mean Absolute Percentage Error (MAPE)**. The **Random Forest Regressor** achieved the best performance.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing](#preprocessing)
- [Methodology](#methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [References](#references)

---

## Problem Statement

The objective is to build a regression model capable of accurately predicting the **price of automobiles** based on features such as engine size, horsepower, fuel efficiency, and brand. This model can help in:

- Used car price valuation
- Market trend analysis
- Automated pricing tools

---

## Dataset Description

- **Source**: UCI Machine Learning Repository
- **Dataset**: Automobile Data Set ([Link](https://archive.ics.uci.edu/dataset/10/automobile))
- **Instances**: 205
- **Attributes**: 25 (including both categorical and numerical features)
- **Target Variable**: `price`

---

## Exploratory Data Analysis

Key findings from EDA:

- Several features had missing values, primarily in numeric fields.
- Target variable `price` showed a right-skewed distribution.
- Strong correlations were observed between `price` and features like `engine-size`, `horsepower`, and `curb-weight`.
- Visualizations included:
  - Missing data heatmap
  - Correlation matrix
  - Histogram and boxplot of price
  - Density plots of numeric features

---

## Preprocessing

Steps taken:

- **Missing values**:
  - Filled numerical columns using mean/mode.
  - Dropped rows missing the target variable (`price`).
- **Categorical encoding**:
  - Applied One-Hot Encoding.
- **Feature scaling**:
  - Used `StandardScaler` from Scikit-learn.

---

## Methodology

### A. Models Implemented

1. **Decision Tree Regressor**  
   Simple and interpretable model, prone to overfitting.

2. **Random Forest Regressor**  
   Ensemble model using multiple trees with better generalization.

3. **K-Nearest Neighbors Regressor**  
   Non-parametric, relies on feature proximity.

4. **Support Vector Regressor (SVR)**  
   Effective in high-dimensional spaces, sensitive to scaling.

### B. Training Setup

- Data split: 80% training, 20% testing
- Tools used: Python, Pandas, Scikit-learn, Matplotlib, Seaborn

---

## Evaluation Metrics

Models were evaluated using:

1. **Root Mean Squared Error (RMSE)**  
   \[
   RMSE = \sqrt{\frac{1}{n} \sum\_{i=1}^{n} (y_i - \hat{y}\_i)^2}
   \]

2. **R-squared (R²)**  
   \[
   R^2 = 1 - \frac{\sum (y_i - \hat{y}\_i)^2}{\sum (y_i - \bar{y})^2}
   \]

3. **Mean Absolute Percentage Error (MAPE)**  
   \[
   MAPE = \frac{100\%}{n} \sum\_{i=1}^{n} \left| \frac{y_i - \hat{y}\_i}{y_i} \right|
   \]

---

## Results

| **Model**                | **RMSE** | **R²**  | **MAPE** |
| ------------------------ | -------- | ------- | -------- |
| Decision Tree Regressor  | 2844.65  | 0.9339  | 10.99%   |
| Random Forest Regressor  | 2720.77  | 0.9395  | 9.81%    |
| K-Neighbors Regressor    | 6533.22  | 0.6511  | 18.29%   |
| Support Vector Regressor | 12216.49 | -0.2198 | 41.16%   |

- **Random Forest Regressor** performed best across all metrics.
- **SVR** significantly underperformed, likely due to poor handling of non-linear relationships and outliers.

---

## Discussion

- Random Forest's ensemble strategy helps reduce overfitting and captures complex relationships.
- KNN performed moderately but is sensitive to feature scaling and irrelevant features.
- SVR’s poor performance may be due to inadequate tuning and sensitivity to the data distribution.
- Data size (only 205 instances) may limit model generalization.

---

## Conclusion

This project shows that **Random Forest Regression** is highly effective for predicting automobile prices from tabular data. It achieved:

- **Lowest RMSE**: 2720.77
- **Highest R²**: 0.9395
- **Lowest MAPE**: 9.81%

### Future Work

- Collect more data for better model generalization.
- Perform hyperparameter optimization.
- Explore other models like:
  - Gradient Boosting Regressor
  - XGBoost
  - Neural Networks

---

## References

1. Dua, D. and Graff, C. "UCI Machine Learning Repository." University of California, Irvine, School of Information and Computer Sciences.  
   [https://archive.ics.uci.edu/dataset/10/automobile](https://archive.ics.uci.edu/dataset/10/automobile)

2. Pedregosa et al., "Scikit-learn: Machine Learning in Python", _Journal of Machine Learning Research_, 12, 2825–2830, 2011.

3. Breiman, L., et al., _Classification and Regression Trees_, Wadsworth, 1984.

4. Breiman, L., "Random Forests", _Machine Learning_, 45(1), 5–32, 2001.

5. Cover, T., and Hart, P., "Nearest Neighbor Pattern Classification", _IEEE Transactions on Information Theory_, 13(1), 21–27, 1967.

6. Smola, A. J., and Schölkopf, B., "A Tutorial on Support Vector Regression", _Statistics and Computing_, 14(3), 199–222, 2004.

7. James, G., Witten, D., Hastie, T., Tibshirani, R., _An Introduction to Statistical Learning_, Springer, 2nd Edition, 2021.

8. Wikipedia contributors. "Coefficient of determination." [https://en.wikipedia.org/wiki/Coefficient_of_determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)

---
