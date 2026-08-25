# 🚢 Week 5 Task 2 — Deploy Model as Live Web App

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Python](https://img.shields.io/badge/python-3.14-blue)
![Streamlit](https://img.shields.io/badge/streamlit-live-ff4b4b)

> Deploying a trained ML pipeline as a live, interactive prediction tool — turning a notebook model into a real product anyone can use.

---

## 🔗 Live Demo

**[👉 Try the app here](https://titanic-survival-arham.streamlit.app/)**

Enter passenger details and get an instant survival prediction with confidence score.

---

## 📸 What It Does

A simple web interface sits on top of a trained Logistic Regression pipeline. Users fill in passenger details — class, sex, age, family aboard, fare, and port of embarkation — and the app returns:
- ✅ / ❌ Survival prediction
- 📊 Model confidence percentage

---

## 🧠 Model Details

| Component | Detail |
|---|---|
| Source | Week 4 Task 1 pipeline (titanic_pipeline.pkl) |
| Algorithm | Logistic Regression |
| Preprocessing | Median imputation + scaling (numeric), most-frequent imputation + one-hot encoding (categorical) |
| Features | Age, Fare, SibSp, Parch, FamilySize, Pclass, Sex, Embarked, Alone |
| Engineered features | FamilySize = SibSp + Parch + 1, Alone = 1 if FamilySize == 1 |

---

## ⚙️ How It Works

1. User submits passenger details via the form
2. App derives FamilySize and Alone from raw inputs — replicating the exact feature engineering used at training time
3. The saved pipeline (preprocessing + model bundled together) transforms the input and generates a prediction in one step
4. Result is displayed with a confidence score

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| streamlit | Web app framework |
| pandas | Data handling |
| scikit-learn | Model + preprocessing pipeline |
| joblib | Model serialization |

---

## ▶️ Run It Locally

pip install -r requirements.txt
streamlit run app.py

---

## 📁 Files

Week_5_Task_2/
- app.py — Streamlit app source
- titanic_pipeline.pkl — Trained model pipeline (Week 4 Task 1)
- requirements.txt — Python dependencies
- README.md

---

*Part of the Neurofive Solutions ML Fundamentals internship — Week 5.*
