# Week 4 - Task 2: Ensemble Learning — Random Forest vs XGBoost

## Objective
Train and compare Random Forest and XGBoost ensemble models against an earlier single model, and understand how each ensemble method combines trees differently.

## Dataset
Titanic dataset (same as Task 1), reusing the same ColumnTransformer preprocessing pipeline (StandardScaler for numeric, OneHotEncoder for categorical) with FamilySize and Alone engineered features.

## Model Comparison

| Model | Metric | Score |
|---|---|---|
| Logistic Regression (Week 2, manual preprocessing) | Accuracy | 81.01% |
| Logistic Regression (Task 1 Pipeline) | Accuracy | 79.33% |
| Random Forest | Accuracy | 82.68% |
| XGBoost | Accuracy | 77.65% |

Random Forest performed best overall, even beating the original manual Logistic Regression model.

## Feature Importance

Random Forest spread its importance more evenly, relying most on Age and Fare, with Sex as a secondary factor. XGBoost concentrated almost all its importance on Sex and Pclass, barely using Age or Fare at all. This heavy reliance on just two features likely explains XGBoost's lower accuracy compared to Random Forest's more balanced approach.

## Random Forest vs XGBoost — How They Combine Models

Random Forest builds a bunch of decision trees separately using random parts of the data and then just averages their votes to make a final prediction. This makes it pretty stable and less likely to overfit, and it usually spreads importance across a few different features.

XGBoost works differently — it builds trees one after another, and each new tree tries to fix the mistakes the last one made. Because it improves step by step, it can fit patterns more aggressively, but on this data it seems to have leaned way too hard on just Sex and Pclass instead of spreading things out like Random Forest did.

## Files
- `week_4_ensemble.ipynb` - full notebook
- `train.csv` - dataset used
