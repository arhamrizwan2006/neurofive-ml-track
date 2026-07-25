# Week 2 Task 1 - Titanic Survival Classification

## Objective
Build a Logistic Regression model to predict whether a Titanic passenger survived, using the cleaned dataset from Week 1.

## Approach

1. **Data**: Started with the cleaned Titanic dataset from Week 1 (no missing values in Sex/Embarked).

2. **Encoding categorical columns**:
   - `Sex` was mapped to numeric values (male = 1, female = 0), since it only has 2 categories.
   - `Embarked` was one-hot encoded using `pd.get_dummies()` with `drop_first=True`, creating `Embarked_Q` and `Embarked_S` columns (the third category, C, is implied when both are 0). This avoids falsely implying an order between ports, which a simple 0/1/2 mapping would do.

3. **Feature selection**: Dropped `PassengerId`, `Name`, and `Ticket` (no predictive value / not usable as numeric input). Final features used:
   `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S`

4. **Train/test split**: Used `train_test_split` (80% train, 20% test, `random_state=42` for reproducibility).

5. **Model**: Trained a `LogisticRegression` model on the training set.

6. **Evaluation**: Predicted on the test set and evaluated using accuracy, precision, recall, F1 score, and a confusion matrix.

## Results

- **Accuracy**: 0.8101 (81%)
- **Precision**: 0.7857
- **Recall**: 0.7432
- **F1 Score**: 0.7639

### Confusion Matrix

|                    | Predicted: Died | Predicted: Survived |
|--------------------|------------------|----------------------|
| **Actual: Died**    | 90 (TN)          | 15 (FP)              |
| **Actual: Survived**| 19 (FN)          | 55 (TP)               |

**Interpretation**:
- 90 passengers who died were correctly predicted as died (True Negatives).
- 55 passengers who survived were correctly predicted as survived (True Positives).
- 15 passengers who died were incorrectly predicted as survived (False Positives) — false alarms.
- 19 passengers who survived were incorrectly predicted as died (False Negatives) — missed survivors.

The model makes slightly more False Negative errors than False Positive errors, meaning it is a bit more likely to miss an actual survivor than to wrongly flag someone as a survivor. This is reflected in recall (0.7432) being lower than precision (0.7857).

## Tools Used
- Python, pandas, scikit-learn, seaborn, matplotlib