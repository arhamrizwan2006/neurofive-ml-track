# Week 5 Task 2 — Deploy Model as Live Web App

## Overview
This task takes the best-performing saved model pipeline from a previous week and deploys it as a live, interactive web app using Streamlit. Users can input passenger details and get a real-time survival prediction.

## Live App
🔗 **[Try it here](https://titanic-survival-arham.streamlit.app/)**

## Model Used
The deployed model is the Logistic Regression pipeline from **Week 4 Task 1** (`titanic_pipeline.pkl`), which includes:
- Preprocessing: median imputation + scaling for numeric features, most-frequent imputation + one-hot encoding for categorical features
- Classifier: Logistic Regression
- Features used: Age, Fare, SibSp, Parch, FamilySize, Pclass, Sex, Embarked, Alone

## How It Works
1. User inputs passenger details (class, sex, age, siblings/spouses aboard, parents/children aboard, fare, port of embarkation) via the web form
2. The app derives `FamilySize` and `Alone` from the raw inputs (matching the feature engineering done during training)
3. The saved pipeline preprocesses the input and returns a survival prediction with confidence percentage

## Tech Stack
- `streamlit` — web app framework
- `pandas` — data handling
- `scikit-learn` — model pipeline
- `joblib` — model serialization/loading

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — Streamlit app source code
- `titanic_pipeline.pkl` — trained model pipeline (from Week 4 Task 1)
- `requirements.txt` — Python dependencies
