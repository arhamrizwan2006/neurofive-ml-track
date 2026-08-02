# Week 3 Task 1: Model Evaluation & Tuning (Titanic Classification)

## 📌 Overview
This project is part of the Machine Learning Fundamentals internship at **Neurofive Solutions**.
The goal was to revisit the Titanic survival classification model built in Week 2 Task 1, evaluate
it properly beyond accuracy, and improve it using hyperparameter tuning.

Unlike Week 2 (where the focus was just building the model), this task focuses on **evaluating**
the model with precision/recall/F1-score and **tuning** it using GridSearchCV to see if performance
can be improved.

## 📂 Dataset
- Source: Kaggle - Titanic: Machine Learning from Disaster
- File used: `train.csv`
- Same preprocessing and features as Week 2 Task 1 (8 features after encoding)

## 🎯 Features Used
`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex_male`, `Embarked_Q`, `Embarked_S`

**Target:** `Survived`

## 🛠️ Workflow
1. Reloaded the cleaned Titanic dataset and the Logistic Regression model from Week 2 Task 1
2. Evaluated the baseline model using `classification_report` (precision, recall, F1-score)
3. Explained why accuracy alone can be misleading on this imbalanced dataset (~59% did not survive)
4. Used `GridSearchCV` (5-fold cross-validation) to tune two hyperparameters: `C` and `solver`
5. Evaluated the best tuned model on the test set using `classification_report`
6. Compared baseline vs tuned model using both cross-validation scores and test set metrics

## 📊 Results

### Before vs After Tuning
| Metric | Before Tuning | After Tuning |
|---|---|---|
| Accuracy | 0.81 | 0.78 |
| Precision (Survived) | 0.79 | 0.76 |
| Recall (Survived) | 0.74 | 0.69 |
| F1-score (Survived) | 0.76 | 0.72 |
| CV Mean Accuracy | 0.791 | 0.796 |

**Best hyperparameters found (GridSearchCV):** `C = 1`, `solver = liblinear`

## 📈 Why Accuracy Alone Can Be Misleading — Explained Simply
About 59% of passengers in this dataset did not survive. This means a model that always predicts
"did not survive," without learning anything at all, would still get around 59% accuracy just by
guessing the majority class every time. Our model's 81% accuracy shows it's genuinely learning
something useful, but this is exactly why accuracy alone isn't enough — precision, recall, and
F1-score are needed to see how well the model performs on each class separately, not just overall.

## 🔍 Tuning Outcome — Explained Simply
The tuned model scored slightly lower on the test set (0.78) compared to baseline (0.81), but
scored slightly higher on cross-validation (0.796 vs 0.791) — a more reliable measure since it
averages across 5 different data splits instead of relying on just one. This suggests the test-set
drop is most likely random variance from this particular split, not real evidence that tuning made
the model worse.

## 🧰 Tools Used
- Python, pandas, numpy
- scikit-learn (`LogisticRegression`, `GridSearchCV`, `cross_val_score`, `classification_report`)

## ✅ Status
Code-complete — notebook: `week_3_tuning.ipynb`
