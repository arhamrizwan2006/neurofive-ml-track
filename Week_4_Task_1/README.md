# Week 4 - Task 1: Build a Proper ML Pipeline with Feature Engineering

## Objective
Build a clean, reusable scikit-learn Pipeline that chains preprocessing and modeling into a single object, avoiding data leakage and inconsistent preprocessing between train/test sets.

## Dataset
Titanic dataset (raw, uncleaned) - same dataset used in Week 1 and Week 2.

## Approach
- Dropped low-signal / unusable columns: `PassengerId`, `Ticket`, `Cabin`, `Name`
- Built a `ColumnTransformer` with two branches:
  - **Numeric** (`Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`): `SimpleImputer(median)` → `StandardScaler`
  - **Categorical** (`Pclass`, `Sex`, `Embarked`, `Alone`): `SimpleImputer(most_frequent)` → `OneHotEncoder`
- Chained the preprocessor with `LogisticRegression` into a single `Pipeline`
- Engineered 2 new features:
  - `FamilySize` = SibSp + Parch + 1
  - `Alone` = 1 if FamilySize == 1, else 0
- Saved the final trained pipeline with `joblib`

## Results

| Approach | Accuracy |
|---|---|
| Manual preprocessing (Week 2 style: fillna + get_dummies, no pipeline) | 81.01% |
| Pipeline with engineered features (FamilySize, Alone) | 79.33% |
| Pipeline without engineered features | 79.89% |

The manual approach slightly outperformed the pipeline on this split. Adding `FamilySize` and `Alone` also slightly **decreased** accuracy (79.33% vs 79.89%), suggesting the information they carry is likely already captured by `SibSp` and `Parch` individually - added redundancy rather than new signal, at least for Logistic Regression on this dataset.

Despite the small accuracy dip, the pipeline is preferred in practice for its consistency, reusability, and protection against data leakage between train/test sets.

## Files
- `week_4_pipeline.ipynb` - full notebook
- `titanic_pipeline.pkl` - saved trained pipeline (joblib)
- `train.csv` - dataset used

## Model Saving
The final pipeline (with engineered features) was saved using:
```python
joblib.dump(model, 'titanic_pipeline.pkl')
```
