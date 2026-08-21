# 🎯 Week 5 · Task 1 — Handling Imbalanced & Messy Real-World Data

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-red)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

> Machine Learning Fundamentals Internship — Neurofive Solutions

---

## 📌 Overview

Real-world data is rarely balanced — fraud, disease, and churn cases are the exception, not the rule. This project tackles that head-on using the **Credit Card Fraud Detection** dataset, one of the most extreme class-imbalance problems in ML: just **0.167%** of transactions are fraudulent.

The goal isn't just to build a model — it's to prove *why* the obvious metric (accuracy) lies to you on data like this, and to fix it properly.

---

## 📂 Dataset

| | |
|---|---|
| **Source** | [Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Rows** | 284,807 raw → 283,726 after cleaning |
| **Features** | `V1`–`V28` (PCA-anonymized), `Time`, `Amount` |
| **Target** | `Class` → `0` = Non-Fraud, `1` = Fraud |

> ⚠️ **Note:** The raw CSV (~150 MB) is not included in this repo due to GitHub's file size limits. Download it directly from the Kaggle link above and place `creditcard.csv` in this folder before running the notebook.

---

## 🧹 Data Cleaning

- ✅ No missing values
- ⚠️ **1,081 duplicate rows found and removed** (284,807 → 283,726) — critical step, since duplicates can leak into train/test splits and quietly inflate reported fraud rates

---

## ⚖️ The Imbalance Problem

| Class | Count | % of Data |
|---|---|---|
| 🟢 Non-Fraud (`0`) | 283,253 | 99.833% |
| 🔴 Fraud (`1`) | 473 | 0.167% |

A model could predict *"not fraud"* for every single transaction and still be **99.8% accurate** — while catching zero fraud. That's the trap this task is built around.

*(Bar chart visualizing this distribution is in the notebook.)*

---

## 🧪 Methodology

1. **Stratified 80/20 train-test split** — preserves the fraud ratio in both sets
2. **Baseline model** — Logistic Regression trained on raw, imbalanced data
3. **SMOTE** applied to the *training set only* — synthetically balances fraud vs non-fraud to 226,602 / 226,602
4. **Retrained** Logistic Regression on the balanced data
5. **Compared** Precision / Recall / F1 (fraud class) — before vs. after

---

## 📊 Results — Fraud Class Only

| Metric | 🔹 Baseline | 🔸 After SMOTE |
|---|:---:|:---:|
| Precision | 0.85 | 0.12 |
| Recall | 0.58 | **0.86** |
| F1-score | 0.69 | 0.20 |

**Confusion matrix shift:**

| | Missed Fraud (FN) | Caught Fraud (TP) | False Alarms (FP) |
|---|:---:|:---:|:---:|
| Before | 40 / 95 | 55 / 95 | 10 |
| After | 13 / 95 | **82 / 95** | 626 |

SMOTE nearly **1.5x'd recall** — the model went from missing 42% of fraud to missing just 14% — at the cost of many more false alarms. A real trade-off, not a free win.

---

## 🧠 Why Accuracy Is a Misleading Metric Here

> Fraud makes up only 0.167% of this data, so a model can score near-perfect accuracy just by predicting "non-fraud" almost every time. The baseline model hit **1.00 accuracy** while missing 42% of real fraud cases. After SMOTE, recall jumped significantly — but accuracy actually *dropped slightly* to 0.99. A model that got measurably better at its actual job scored *lower* on accuracy. That's proof accuracy can't be trusted on imbalanced data — **precision and recall on the minority class** are what actually matter.

---

## 📁 Repository Contents
