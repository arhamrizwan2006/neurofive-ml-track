# Week 2 Task 2: House Price Prediction with Linear Regression

## 📌 Overview
This project is part of the Machine Learning Fundamentals internship at **Neurofive Solutions**. 
The goal was to predict house sale prices using **Linear Regression**, based on the Kaggle 
"House Prices - Advanced Regression Techniques" dataset.

Unlike the previous task (Titanic survival classification), this task involves predicting a 
**continuous numerical value** (price) rather than a category — making it a **regression** 
problem instead of a classification one.

## 📂 Dataset
- Source: Kaggle - House Prices: Advanced Regression Techniques
- File used: `train.csv`
- 1460 rows, 81 columns (features describing house characteristics + `SalePrice` as target)

## 🎯 Features Selected
| Feature | Description |
|---|---|
| `GrLivArea` | Above-ground living area (sq ft) |
| `OverallQual` | Overall material and finish quality (1-10) |
| `Neighborhood` | Location of the house (categorical, one-hot encoded) |
| `TotalBsmtSF` | Total basement area (sq ft) |
| `GarageCars` | Garage capacity in number of cars |

**Target:** `SalePrice`

## 🛠️ Workflow
1. Loaded and explored the dataset (`head`, `info`, `isnull`, `duplicated`)
2. Selected 5 features believed to most affect price
3. One-hot encoded `Neighborhood` (expanded to 28 total feature columns)
4. Split data 80/20 into train and test sets
5. Trained a `LinearRegression` model from scikit-learn
6. Fixed a shape mismatch on the target (`y_train`/`y_test`) using `np.ravel()`
7. Generated predictions and evaluated performance
8. Plotted predicted vs. actual prices

## 📊 Results
| Metric | Value |
|---|---|
| **RMSE** | $36,325.60 |
| **R² Score** | 0.828 |

## 📈 R² — Explained Simply
R² tells us how good our model is at guessing house prices based on the features we gave it, 
like size, quality, and neighborhood. Our score of 0.83 means these features explain about 83% 
of why prices go up or down between houses. The remaining 17% comes from factors we didn't 
include, like exact condition, age, or unique features. In short, our model understands the 
overall pattern of house prices well, but it may still be off when predicting individual houses.

## 📉 Predicted vs Actual Prices
A scatter plot compared predicted prices against actual prices, with a red dashed line 
representing perfect prediction. Most points cluster closely around the line for typical-priced 
homes, but the model tends to **underpredict** for higher-end, luxury houses — likely due to 
fewer expensive homes present in the training data.

## 🧰 Tools Used
- Python, pandas, numpy
- scikit-learn (`LinearRegression`, `train_test_split`, `mean_squared_error`, `r2_score`)
- matplotlib

## ✅ Status
Code-complete — notebook: `week_2_regression.ipynb`
