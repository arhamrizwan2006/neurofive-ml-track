<div align="center">

# 🤖 Neurofive Solutions — ML Fundamentals Internship

### Machine learning fundamentals to production-ready models in 6 weeks

**Neurofive Solutions** · Intern ID `NFS-2607-0177` · Jul 2026 → Aug 2026

![Badge](https://img.shields.io/badge/Internship-Neurofive%20Solutions-blueviolet?style=for-the-badge)
![Badge](https://img.shields.io/badge/Track-ML%20Fundamentals-blue?style=for-the-badge)
![Badge](https://img.shields.io/badge/Progress-Week%205-orange?style=for-the-badge)
![Badge](https://img.shields.io/badge/Language-Python-brightgreen?style=for-the-badge)

</div>

---

## 📑 Table of Contents

- [📊 Internship Progress](#-internship-progress)
- [🎯 Week-by-Week Breakdown](#-week-by-week-breakdown)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Repository Structure](#-repository-structure)
- [🚀 How to Use This Repository](#-how-to-use-this-repository)
- [💡 Key Takeaways](#-key-takeaways)
- [📈 Performance Highlights](#-performance-highlights)
- [🔗 Resources & References](#-resources--references)
- [📧 Get Started](#-get-started)

---

## 📊 Internship Progress

<div align="center">

```
Overall Progress:  ████████████████████████░░░░░░  83%  (5 / 6 Weeks)
```

</div>

| Week | Task | Topic | Status |
|:---:|:---:|---|:---:|
| **1** | 1 | Titanic EDA | ✅ Approved |
| **1** | 2 | Data Cleaning & Visualization | ✅ Approved |
| **2** | 1 | Titanic Classification (Logistic Regression) | ✅ Approved |
| **2** | 2 | Housing Price Prediction (Linear Regression) | ✅ Approved |
| **3** | 1 | Model Evaluation & Hyperparameter Tuning | ✅ Approved |
| **3** | 2 | Customer Churn Prediction (Business Problem) | ✅ Approved |
| **4** | 1 | ML Pipeline & Feature Engineering | ✅ Approved |
| **4** | 2 | Ensemble Learning (Random Forest vs XGBoost) | ✅ Approved |
| **5** | 1 | Handling Imbalanced & Messy Real-World Data | 🟡 Submitted |
| **5** | 2 | Deploy Model as Live Web App | 🟡 Built — Pending Submission |
| **6** | — | Capstone: End-to-End ML Project | ⚪ Pending |

---

## 🎯 Week-by-Week Breakdown

<details>
<summary><strong>📌 Week 1 — Foundation & Data Exploration</strong></summary>

#### Task 1️⃣ — Titanic EDA
*Understanding data through exploration and visualization*

```
Raw Dataset → Load & Inspect → Identify Patterns → Visualize Trends
    (Titanic)    (1309 rows)   (Survival rates)   (Age, Class, Sex)
```

| What I Learned |
|---|
| Loading datasets with Pandas |
| Handling missing values (`isnull()`, `dropna()`) |
| Statistical summaries and distributions |
| Correlation analysis |
| Creating meaningful visualizations with Seaborn |

**Notebook:** `Week_1_Task_1/week_1_setup.ipynb`

---

#### Task 2️⃣ — Data Cleaning & Visualization
*From messy data to insights*

| What I Learned |
|---|
| Feature engineering (creating new columns) |
| Handling categorical variables |
| Data type conversions |
| Creating publication-quality visualizations (histograms, boxplots, heatmaps) |
| Interpreting distribution patterns |

**Notebook:** `Week_1_Task_2/week_1_setup.ipynb`

</details>

<details>
<summary><strong>📌 Week 2 — Predictive Modeling</strong></summary>

#### Task 1️⃣ — Titanic Classification
*Predicting survival with Logistic Regression*

```
Features (Age, Sex, Class, ...) → Train/Test Split → Logistic Regression → Predict (Survived?)
              ↓
           81.0% Accuracy ✅
```

| What I Learned |
|---|
| Classification vs Regression |
| Train-test split strategy |
| One-hot encoding categorical features |
| Model training with scikit-learn |
| Evaluating classification metrics |
| Making binary predictions |

**Key Challenge:** Handling imbalanced survival classes
**Notebook:** `Week_2_Task_1/week_2_classification.ipynb`

---

#### Task 2️⃣ — Housing Price Prediction
*Predicting continuous values with Linear Regression*

```
House Features (Size, Quality, Location, ...) → Linear Regression → Predict Price
         (5 engineered features)                    ↓
                                          RMSE: $36,325.60
                                          R²: 0.828 (83% explained)
```

| Metric | Value |
|--------|-------|
| **RMSE** | $36,325.60 |
| **R² Score** | 0.828 |
| **Interpretation** | 83% of price variation explained by features |

| What I Learned |
|---|
| Regression problem formulation |
| Feature selection for impact |
| Handling numerical features at scale |
| Evaluating regression models (RMSE, R²) |
| Interpreting model predictions |
| Visualizing predicted vs actual values |

**Key Insight:** Model underpredicts luxury homes (sparse training data)
**Notebook:** `Week_2_Task_2/week_2_regression.ipynb`

</details>

<details>
<summary><strong>📌 Week 3 — Advanced Techniques & Business Applications</strong></summary>

#### Task 1️⃣ — Model Evaluation & Hyperparameter Tuning
*Beyond accuracy: realistic model assessment*

```
Base Model → Calculate Metrics → GridSearchCV Tuning → Compare Performance
   ↓             ↓                    ↓                    ↓
Logistic      Precision,        C & solver         Before/After
Regression    Recall, F1         optimization      comparison table
```

| Metric | Baseline | Tuned |
|--------|----------|-------|
| **Test Accuracy** | 0.81 | 0.78 |
| **CV Mean Accuracy** | 0.791 | 0.796 |
| **Best Params** | — | C=1, solver=liblinear |

| Techniques Covered |
|---|
| Precision, Recall, F1-score |
| Why accuracy alone can mislead on imbalanced data |
| GridSearchCV for hyperparameter optimization |
| Cross-validation strategies |
| Performance comparison (tuned vs baseline) |

**Key Insight:** Test-set accuracy dropped after tuning, but cross-validation (a more reliable signal) actually showed a slight improvement — the drop was likely random split variance, not a real regression.

**Notebook:** `Week_3_Task_1/week_3_tuning.ipynb`

---

#### Task 2️⃣ — Customer Churn Prediction
*Real-world ML: solving a business problem*

```
Telco Dataset (7043 customers) → EDA → Model Training → Business Insights
        ↓                          ↓          ↓
  Contract Type,              Which features   Logistic Regression:
  Tenure,                      drive churn?    82% accuracy, best
  Monthly Charges              ↓               performer overall
                          Feature Importance
```

**Problem:** Predict which customers will leave (churn)
**Challenge:** Imbalanced dataset — 73.5% stayed, 26.5% churned
**Approach:** Decision Tree vs Logistic Regression comparison

| Metric | Logistic Regression | Decision Tree |
|--------|:---:|:---:|
| **Accuracy** | 0.82 | 0.71 |
| **Precision (Churn)** | 0.69 | 0.46 |
| **Recall (Churn)** | 0.60 | 0.46 |

**Top 3 Churn Drivers:** MonthlyCharges (20.44%), tenure (19.96%), TotalCharges (19.80%) — together accounting for ~60% of the Decision Tree's decisions.

| What I Learned |
|---|
| Handling class imbalance in real datasets |
| Feature importance interpretation |
| Decision Tree classifier (explainability) |
| Comparing model interpretability vs accuracy |
| Writing business summaries for non-technical audiences |
| Presenting ML findings like pitching to a client |

**Notebook:** `Week_3_Task_2/week_3_churn.ipynb`

</details>

<details>
<summary><strong>📌 Week 4 — Pipelines & Ensemble Learning</strong></summary>

#### Task 1️⃣ — ML Pipeline with Feature Engineering
*Building clean, reusable, leak-proof preprocessing*

```
Raw Titanic Data → ColumnTransformer (Scale + Encode) → Logistic Regression → Predict
                          ↓
              + FamilySize, Alone (engineered features)
```

| Approach | Accuracy |
|--------|----------|
| Manual preprocessing (Week 2 style) | 81.01% |
| Pipeline (with engineered features) | 79.33% |
| Pipeline (without engineered features) | 79.89% |

| What I Learned |
|---|
| Building a `ColumnTransformer` to handle numeric and categorical columns differently |
| Chaining preprocessing + model into a single `Pipeline` |
| Feature engineering (FamilySize, Alone) and testing their actual impact |
| Saving trained pipelines with `joblib` for reuse without retraining |

**Key Insight:** Engineered features don't always help — FamilySize and Alone slightly decreased accuracy here, reinforcing the importance of testing rather than assuming.

**Notebook:** `Week_4_Task_1/week_4_pipeline.ipynb`

---

#### Task 2️⃣ — Ensemble Learning: Random Forest vs XGBoost
*Comparing how ensemble methods combine models*

```
Same Pipeline Preprocessing → Random Forest vs XGBoost → Compare Accuracy + Feature Importance
```

| Model | Metric | Score |
|--------|--------|:---:|
| Logistic Regression (Week 2) | Accuracy | 81.01% |
| Logistic Regression (Pipeline) | Accuracy | 79.33% |
| Random Forest | Accuracy | 82.68% ✅ |
| XGBoost | Accuracy | 77.65% |

| What I Learned |
|---|
| Training and comparing Random Forest and XGBoost classifiers |
| Plotting and interpreting feature importances |
| Understanding how bagging (Random Forest) differs from boosting (XGBoost) |

**Key Insight:** Random Forest outperformed XGBoost here, spreading importance across features like Age and Fare, while XGBoost over-relied heavily on Sex and Pclass alone — likely contributing to its lower accuracy.

**Notebook:** `Week_4_Task_2/week_4_ensemble.ipynb`

</details>

<details open>
<summary><strong>📌 Week 5 — Real-World Data & Deployment</strong></summary>

#### Task 1️⃣ — Handling Imbalanced & Messy Real-World Data
*Fraud detection: when 99.8% accuracy means nothing*

```
Credit Card Fraud Dataset → Clean Duplicates → Check Class Balance → SMOTE → Compare Metrics
   (284,807 rows)          (1,081 removed)     (99.83% / 0.17%)   (oversample minority)
```

| Metric (Fraud Class) | Before SMOTE | After SMOTE |
|--------|:---:|:---:|
| **Precision** | 0.85 | 0.12 |
| **Recall** | 0.58 | 0.86 |
| **F1-score** | 0.69 | 0.20 |

| What I Learned |
|---|
| Why accuracy is a meaningless metric on severely imbalanced data |
| Detecting and removing duplicate records before modeling |
| Applying SMOTE to oversample the minority class — training data only, never the test set |
| The real precision/recall trade-off: SMOTE traded precision for a big recall gain |

**Key Insight:** A model with 99.8% accuracy can still fail at its actual job if it never catches the minority class. Precision/recall — not accuracy — tell the real story here.

**Notebook:** `Week_5_Task_1/week_5_imbalance.ipynb` · **Status:** 🟡 Submitted, awaiting review

---

#### Task 2️⃣ — Deploy Model as Live Web App
*From notebook to a real, usable product*

```
Saved Pipeline (joblib) → Streamlit Form (7 inputs) → Live Prediction → Deployed on Streamlit Cloud
```

**🔗 [Live App](https://titanic-survival-arham.streamlit.app/)**

| What I Learned |
|---|
| Serializing a full preprocessing + model pipeline with `joblib` for reuse outside the training notebook |
| Building an interactive UI with Streamlit (`selectbox`, `slider`, `number_input`, `button`) |
| Re-deriving engineered features (FamilySize, Alone) at prediction time to match training exactly |
| Handling deployment-specific issues (relative file paths breaking on cloud servers, subfolder `requirements.txt` discovery) |
| Deploying a public app via Streamlit Community Cloud, connected directly to a GitHub subfolder |

**Key Insight:** A model isn't done just because it's accurate — turning it into something a non-technical person can actually use is its own skill, with its own gotchas (like the app running from a different working directory in the cloud than it does locally).

**Notebook:** `Week_5_Task_2/week_5_deployement.ipynb` · **Status:** 🟡 Built & deployed — portal submission pending

</details>

<details>
<summary><strong>📌 Week 6 — Capstone (Upcoming)</strong></summary>

#### End-to-End Machine Learning Project
*Portfolio centerpiece: a self-chosen problem, not a guided dataset*

Instead of a provided dataset, this task is a self-selected real-world problem taken from raw data through to a deployed, working product — presented the way it would be pitched in an actual interview.

**Status:** ⚪ Pending — not yet started

</details>

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Core Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Data Manipulation** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square) ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square) |
| **Machine Learning** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-006ACC?style=flat-square) ![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-red?style=flat-square) |
| **Persistence & Deployment** | ![joblib](https://img.shields.io/badge/joblib-4B8BBE?style=flat-square) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| **Environment & Tooling** | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white) |

---

## 📁 Repository Structure

```
neurofive-ml-track/
│
├── Week_1_Task_1/                  Titanic EDA
├── Week_1_Task_2/                  Data Cleaning & Visualization
├── Week_2_Task_1/                  Titanic Classification
├── Week_2_Task_2/                  Housing Price Regression
├── Week_3_Task_1/                  Model Evaluation & Tuning
├── Week_3_Task_2/                  Customer Churn Prediction
├── Week_4_Task_1/                  ML Pipeline + Feature Engineering
│   └── titanic_pipeline.pkl        (Saved trained pipeline — reused in Week 5 Task 2)
├── Week_4_Task_2/                  Ensemble Learning (RF vs XGBoost)
├── Week_5_Task_1/                  Imbalanced & Messy Data (SMOTE)
├── Week_5_Task_2/                  Live Deployment
│   ├── app.py                      (Streamlit app source)
│   ├── titanic_pipeline.pkl        (Deployed model)
│   └── requirements.txt            (Deployment dependencies)
│
└── README.md (this file)
```

*Every task folder contains its own notebook and README with task-specific documentation.*

---

## 🚀 How to Use This Repository

| Step | Command |
|---|---|
| 1. Clone | `git clone https://github.com/arhamrizwan2006/neurofive-ml-track.git` |
| 2. Enter | `cd neurofive-ml-track` |
| 3. Install deps | `pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib streamlit matplotlib seaborn jupyter` |
| 4. Launch Jupyter | `jupyter notebook` |
| 5. Explore | Open any `Week_X_Task_Y/*.ipynb` and run cells sequentially |
| 6. Try the live app | `cd Week_5_Task_2 && streamlit run app.py` — or just visit the [deployed link](https://titanic-survival-arham.streamlit.app/) |

---

## 💡 Key Takeaways

### Concepts Mastered

| # | Concept | What It Covers |
|:---:|---|---|
| 1 | **EDA** | Understanding data before modeling |
| 2 | **Data Cleaning & Preprocessing** | Missing values, outliers, categorical encoding |
| 3 | **Classification** | Categorical outcomes — survival, churn, fraud |
| 4 | **Regression** | Continuous outcomes — house prices |
| 5 | **Model Evaluation** | Choosing metrics that fit the problem |
| 6 | **Hyperparameter Tuning** | Systematic optimization via GridSearchCV |
| 7 | **Cross-Validation** | Robust, split-independent evaluation |
| 8 | **Feature Importance** | Understanding what drives predictions |
| 9 | **Business Context** | Translating ML output into decisions |
| 10 | **ML Pipelines** | Leak-proof, reusable preprocessing + modeling |
| 11 | **Ensemble Learning** | Bagging (Random Forest) vs boosting (XGBoost) |
| 12 | **Model Persistence** | Saving/reloading trained models with joblib |
| 13 | **Imbalanced Data Handling** | SMOTE, class weighting, metric selection |
| 14 | **Model Deployment** | Serving a model through a live web app |

### Real-World Skills

| Skill | Applied In |
|---|---|
| 🔧 End-to-end ML workflow (data → model → eval → deployment) | All weeks |
| 🔧 Debugging & troubleshooting model issues | Weeks 4–5 |
| 🔧 Presenting findings to non-technical stakeholders | Week 3 |
| 🔧 Handling imbalanced datasets | Week 5 |
| 🔧 Working with real Kaggle datasets | Weeks 3, 5 |
| 🔧 Production-style pipelines over notebook-only workflows | Week 4 |
| 🔧 Shipping a working app to a public URL | Week 5 |

---

## 📈 Performance Highlights

| Task | Model | Metric | Result |
|------|-------|--------|:---:|
| Titanic Survival | Logistic Regression | Accuracy | 81.0% ✅ |
| House Price | Linear Regression | R² Score | 0.828 ✅ |
| House Price | Linear Regression | RMSE | $36,325.60 |
| Titanic (Tuned) | Logistic Regression + GridSearchCV | CV Accuracy | 79.6% |
| Customer Churn | Logistic Regression | Accuracy | 82.0% ✅ |
| Customer Churn | Decision Tree | Accuracy | 71.0% |
| Titanic (Pipeline) | Logistic Regression | Accuracy | 79.3% |
| Titanic (Ensemble) | Random Forest | Accuracy | 82.7% ✅ |
| Titanic (Ensemble) | XGBoost | Accuracy | 77.7% |
| Fraud Detection | Logistic Regression | Recall (Before → After SMOTE) | 58% → 86% ✅ |
| Deployment | Logistic Regression Pipeline | Live App | ✅ Deployed |

---

## 🔗 Resources & References

| Resource | Link |
|---|---|
| scikit-learn Docs | [scikit-learn.org](https://scikit-learn.org/) |
| Pandas API Reference | [pandas.pydata.org/docs](https://pandas.pydata.org/docs/) |
| Matplotlib/Seaborn Tutorials | [seaborn.pydata.org](https://seaborn.pydata.org/) |
| Kaggle Datasets | [kaggle.com/datasets](https://www.kaggle.com/datasets) |
| ML Mastery Blog | [machinelearningmastery.com](https://machinelearningmastery.com/) |
| XGBoost Docs | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
| imbalanced-learn Docs | [imbalanced-learn.org](https://imbalanced-learn.org/) |
| Streamlit Docs | [docs.streamlit.io](https://docs.streamlit.io/) |

---

## 📧 Get Started

1. Pick any task folder
2. Read the README.md inside
3. Open the Jupyter notebook and run cells sequentially
4. Modify code to experiment and learn
5. Or skip straight to the [live deployed app](https://titanic-survival-arham.streamlit.app/) — no setup needed

---

<div align="center">

**Internship:** Neurofive Solutions — ML Fundamentals
**Duration:** 6 weeks · Jul 16, 2026 – Aug 28, 2026
**Intern ID:** `NFS-2607-0177`
**Status:** 🟡 Week 5 in progress · Week 6 Capstone remaining

*From data to insights. From questions to answers.* 📊🚀

</div>
