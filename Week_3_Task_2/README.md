# Week 3 Task 2: Customer Churn Prediction (Telco Dataset)

## 📌 Overview
This project is part of the Machine Learning Fundamentals internship at **Neurofive Solutions**.
The goal was to predict customer churn — which customers are likely to leave — using the Telco
Customer Churn dataset, and compare a Decision Tree Classifier against Logistic Regression.

## 📂 Dataset
- Source: Kaggle - Telco Customer Churn
- File used: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- 7043 rows, 21 columns (customer demographics, account info, services subscribed, and `Churn` as target)

## 🛠️ Workflow
1. Loaded the dataset and checked for missing values
2. Found `TotalCharges` stored as text with 11 hidden blank values (customers with 0 tenure) — converted to numeric and filled with 0
3. Dropped `customerID` (no predictive value)
4. Checked class balance of `Churn` — found a 73.5% / 26.5% imbalance (mentioned below, not corrected)
5. Performed EDA on Contract type, tenure, and MonthlyCharges vs churn
6. One-hot encoded 15 categorical columns (`drop_first=True`), expanding to 31 total columns
7. Split data 80/20 into train and test sets
8. Trained a Decision Tree Classifier and a Logistic Regression model
9. Compared both models using `classification_report`
10. Extracted top 3 features driving churn using `.feature_importances_` from the Decision Tree

## 📊 EDA Findings
| Feature | Finding |
|---|---|
| Contract type | Month-to-month: 42.7% churn — One year: 11.3% — Two year: 2.8% |
| Tenure (avg) | Stayed: 37.6 months — Churned: 18.0 months |
| MonthlyCharges (avg) | Stayed: $61.27 — Churned: $74.44 |

## ⚖️ Class Imbalance
The dataset is imbalanced: about 73.5% of customers did not churn, while only 26.5% churned.
This means a model that always predicts "no churn" would still score ~73.5% accuracy without
learning anything. This imbalance was not corrected in this task (no oversampling or class
weighting applied), but precision, recall, and F1-score for the churn class matter more than
overall accuracy here.

## 📈 Model Comparison
| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| Accuracy | 0.82 | 0.71 |
| Precision (Churn) | 0.69 | 0.46 |
| Recall (Churn) | 0.60 | 0.46 |
| F1-score (Churn) | 0.64 | 0.46 |

Logistic Regression outperformed the Decision Tree across every metric. The untuned Decision
Tree likely overfit the training data, hurting its performance on unseen test data.

## 🌲 Top 3 Features Driving Churn
The top 3 are MonthlyCharges, tenure and TotalCharges with 20.44%, 19.96% and 19.80%
importances respectively.

Together these 3 account for roughly 60% of the model's decisions, outweighing other features
like InternetService or Contract type etc.

## Business Summary
About 1 of every 4 customers leave us and the biggest reasons are billing amount, how new the
customer is and the total money spent by them. Customers with month-to-month plans (43%) leave
much more than customers with two-year plans (~3%). So pushing longer contracts could help a lot.
Between our two models, Logistic Regression worked better (82% accuracy) than Decision Tree
(71%), so we recommend using this.

## 🧰 Tools Used
- Python, pandas, numpy
- scikit-learn (`DecisionTreeClassifier`, `LogisticRegression`, `train_test_split`, `classification_report`)

## ✅ Status
Code-complete — notebook: `week_3_churn.ipynb`
