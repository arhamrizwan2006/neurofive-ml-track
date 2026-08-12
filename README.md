<div align="center">

# 🤖 Neurofive Solutions — ML Fundamentals Internship

### Machine learning fundamentals to production-ready models in 6 weeks

**Neurofive Solutions** · Intern ID `NFS-2607-0177` · Jul 2026 → Present

![Badge](https://img.shields.io/badge/Internship-Neurofive%20Solutions-blueviolet?style=for-the-badge)
![Badge](https://img.shields.io/badge/Track-ML%20Fundamentals-blue?style=for-the-badge)
![Badge](https://img.shields.io/badge/Progress-Week%204-orange?style=for-the-badge)
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
Overall Progress:  ████████████████████░░░░░░░░░░  66.7%  (4 / 6 Weeks)
```

</div>

| Week | Task | Topic | Status |
|:---:|:---:|---|:---:|
| **1** | 1 | Titanic EDA | ✅ |
| **1** | 2 | Data Cleaning & Visualization | ✅ |
| **2** | 1 | Titanic Classification (Logistic Regression) | ✅ |
| **2** | 2 | Housing Price Prediction (Linear Regression) | ✅ |
| **3** | 1 | Model Evaluation & Hyperparameter Tuning | ✅ |
| **3** | 2 | Customer Churn Prediction (Business Problem) | ✅ |
| **4** | 1 | ML Pipeline & Feature Engineering | ✅ |
| **4** | 2 | Ensemble Learning (Random Forest vs XGBoost) | ✅ |

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

**What I learned:**
- Loading datasets with Pandas
- Handling missing values (`isnull()`, `dropna()`)
- Statistical summaries and distributions
- Correlation analysis
- Creating meaningful visualizations with Seaborn

**Notebook:** `Week_1_Task_1/week_1_setup.ipynb`

---

#### Task 2️⃣ — Data Cleaning & Visualization
*From messy data to insights*

**What I learned:**
- Feature engineering (creating new columns)
- Handling categorical variables
- Data type conversions
- Creating publication-quality visualizations (histograms, boxplots, heatmaps)
- Interpreting distribution patterns

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

**What I learned:**
- Classification vs Regression
- Train-test split strategy
- One-hot encoding categorical features
- Model training with scikit-learn
- Evaluating classification metrics
- Making binary predictions

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

**Model Performance:**

| Metric | Value |
|--------|-------|
| **RMSE** | $36,325.60 |
| **R² Score** | 0.828 |
| **Interpretation** | 83% of price variation explained by features |

**What I learned:**
- Regression problem formulation
- Feature selection for impact
- Handling numerical features at scale
- Evaluating regression models (RMSE, R²)
- Interpreting model predictions
- Visualizing predicted vs actual values

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

**Model Performance:**

| Metric | Baseline | Tuned |
|--------|----------|-------|
| **Test Accuracy** | 0.81 | 0.78 |
| **CV Mean Accuracy** | 0.791 | 0.796 |
| **Best Params** | — | C=1, solver=liblinear |

**Techniques Covered:**
- ✅ Precision, Recall, F1-score
- ✅ Why accuracy alone can mislead on imbalanced data
- ✅ GridSearchCV for hyperparameter optimization
- ✅ Cross-validation strategies
- ✅ Performance comparison (tuned vs baseline)

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

**Model Performance:**

| Metric | Logistic Regression | Decision Tree |
|--------|----------------------|----------------|
| **Accuracy** | 0.82 | 0.71 |
| **Precision (Churn)** | 0.69 | 0.46 |
| **Recall (Churn)** | 0.60 | 0.46 |

**Top 3 Churn Drivers:** MonthlyCharges (20.44%), tenure (19.96%), TotalCharges (19.80%) — together accounting for ~60% of the Decision Tree's decisions.

**What I learned:**
- ✅ Handling class imbalance in real datasets
- ✅ Feature importance interpretation
- ✅ Decision Tree classifier (explainability)
- ✅ Comparing model interpretability vs accuracy
- ✅ Writing business summaries for non-technical audiences
- ✅ Presenting ML findings like pitching to a client

**Notebook:** `Week_3_Task_2/week_3_churn.ipynb`

</details>

<details open>
<summary><strong>📌 Week 4 — Pipelines & Ensemble Learning</strong></summary>

#### Task 1️⃣ — ML Pipeline with Feature Engineering
*Building clean, reusable, leak-proof preprocessing*

```
Raw Titanic Data → ColumnTransformer (Scale + Encode) → Logistic Regression → Predict
                          ↓
              + FamilySize, Alone (engineered features)
```

**Model Performance:**

| Approach | Accuracy |
|--------|----------|
| Manual preprocessing (Week 2 style) | 81.01% |
| Pipeline (with engineered features) | 79.33% |
| Pipeline (without engineered features) | 79.89% |

**What I learned:**
- ✅ Building a `ColumnTransformer` to handle numeric and categorical columns differently
- ✅ Chaining preprocessing + model into a single `Pipeline`
- ✅ Feature engineering (FamilySize, Alone) and testing their actual impact
- ✅ Saving trained pipelines with `joblib` for reuse without retraining

**Key Insight:** Engineered features don't always help — FamilySize and Alone slightly decreased accuracy here, reinforcing the importance of testing rather than assuming.

**Notebook:** `Week_4_Task_1/week_4_pipeline.ipynb`

---

#### Task 2️⃣ — Ensemble Learning: Random Forest vs XGBoost
*Comparing how ensemble methods combine models*

```
Same Pipeline Preprocessing → Random Forest vs XGBoost → Compare Accuracy + Feature Importance
```

**Model Comparison:**

| Model | Metric | Score |
|--------|--------|-------|
| Logistic Regression (Week 2) | Accuracy | 81.01% |
| Logistic Regression (Pipeline) | Accuracy | 79.33% |
| Random Forest | Accuracy | 82.68% ✅ |
| XGBoost | Accuracy | 77.65% |

**What I learned:**
- ✅ Training and comparing Random Forest and XGBoost classifiers
- ✅ Plotting and interpreting feature importances
- ✅ Understanding how bagging (Random Forest) differs from boosting (XGBoost)

**Key Insight:** Random Forest outperformed XGBoost here, spreading importance across features like Age and Fare, while XGBoost over-relied heavily on Sex and Pclass alone — likely contributing to its lower accuracy.

**Notebook:** `Week_4_Task_2/week_4_ensemble.ipynb`

</details>

---

## 🛠️ Tech Stack

**Core Language**
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**Data Manipulation**
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

**Visualization**
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)

**Machine Learning**
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006ACC?style=for-the-badge)

**Model Persistence**
![joblib](https://img.shields.io/badge/joblib-4B8BBE?style=for-the-badge)

**Environment & Tooling**
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

---

## 📁 Repository Structure

```
neurofive-ml-track/
│
├── Week_1_Task_1/
│   ├── week_1_setup.ipynb          (Titanic EDA notebook)
│   ├── train.csv                   (Titanic dataset)
│   └── README.md                   (Task documentation)
│
├── Week_1_Task_2/
│   ├── week_1_setup.ipynb          (Data cleaning notebook)
│   ├── train.csv                   (Raw dataset)
│   ├── cleaned_dataset_train.csv   (Processed output)
│   └── README.md                   (Task documentation)
│
├── Week_2_Task_1/
│   ├── week_2_classification.ipynb (Logistic Regression notebook)
│   ├── train.csv                   (Titanic dataset)
│   ├── cleaned_dataset_train.csv   (Preprocessed data)
│   └── README.md                   (Task documentation)
│
├── Week_2_Task_2/
│   ├── week_2_regression.ipynb     (Linear Regression notebook)
│   ├── train.csv                   (House prices dataset)
│   └── README.md                   (Task documentation)
│
├── Week_3_Task_1/
│   ├── week_3_tuning.ipynb         (Model Evaluation & Tuning notebook)
│   └── README.md
│
├── Week_3_Task_2/
│   ├── week_3_churn.ipynb          (Customer Churn Prediction notebook)
│   └── README.md
│
├── Week_4_Task_1/
│   ├── week_4_pipeline.ipynb       (ML Pipeline notebook)
│   ├── titanic_pipeline.pkl        (Saved trained pipeline)
│   ├── train.csv
│   └── README.md
│
├── Week_4_Task_2/
│   ├── week_4_ensemble.ipynb       (Random Forest vs XGBoost notebook)
│   ├── train.csv
│   └── README.md
│
└── README.md (this file)
```

---

## 🚀 How to Use This Repository

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- pip (package manager)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/arhamrizwan2006/neurofive-ml-track.git
cd neurofive-ml-track
```

2. **Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn jupyter
```

3. **Start Jupyter**
```bash
jupyter notebook
```

4. **Navigate to any `Week_X_Task_Y` folder and open the `.ipynb` notebook**

5. **Run cells sequentially** to reproduce results and see the workflow

---

## 💡 Key Takeaways

### Concepts Mastered
✅ **EDA (Exploratory Data Analysis)** — Understanding data before modeling
✅ **Data Cleaning & Preprocessing** — Handling missing values, outliers, categorical encoding
✅ **Classification** — Predicting categorical outcomes (survived/not survived, churn/no churn)
✅ **Regression** — Predicting continuous values (house prices)
✅ **Model Evaluation** — Choosing the right metrics for your problem
✅ **Hyperparameter Tuning** — Optimizing model performance systematically
✅ **Cross-Validation** — Robust evaluation strategies
✅ **Feature Importance** — Understanding what drives predictions
✅ **Business Context** — Translating ML results into actionable insights
✅ **ML Pipelines** — Chaining preprocessing and modeling into reusable, leak-proof objects
✅ **Ensemble Learning** — Comparing bagging (Random Forest) vs boosting (XGBoost)
✅ **Model Persistence** — Saving and reloading trained models with joblib

### Real-World Skills
🔧 End-to-end ML workflow (data → model → evaluation → deployment)
🔧 Debugging & troubleshooting model issues
🔧 Presenting findings to non-technical stakeholders
🔧 Handling imbalanced datasets
🔧 Working with real Kaggle datasets
🔧 Building production-style pipelines instead of notebook-only workflows

---

## 📈 Performance Highlights

| Task | Model | Metric | Result |
|------|-------|--------|--------|
| Titanic Survival | Logistic Regression | Accuracy | 81.0% ✅ |
| House Price | Linear Regression | R² Score | 0.828 ✅ |
| House Price | Linear Regression | RMSE | $36,325.60 |
| Titanic (Tuned) | Logistic Regression + GridSearchCV | CV Accuracy | 79.6% |
| Customer Churn | Logistic Regression | Accuracy | 82.0% ✅ |
| Customer Churn | Decision Tree | Accuracy | 71.0% |
| Titanic (Pipeline) | Logistic Regression | Accuracy | 79.3% |
| Titanic (Ensemble) | Random Forest | Accuracy | 82.7% ✅ |
| Titanic (Ensemble) | XGBoost | Accuracy | 77.7% |

---

## 🔗 Resources & References

- 📚 [scikit-learn Documentation](https://scikit-learn.org/)
- 📖 [Pandas API Reference](https://pandas.pydata.org/docs/)
- 🎨 [Matplotlib/Seaborn Tutorials](https://seaborn.pydata.org/)
- 🏠 [Kaggle Datasets](https://www.kaggle.com/datasets)
- 🤖 [ML Mastery Blog](https://machinelearningmastery.com/)
- 🌲 [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 📧 Get Started

1. Pick any task folder
2. Read the README.md inside
3. Open the Jupyter notebook
4. Run the cells and follow along
5. Modify code to experiment and learn

**Each notebook is fully commented and beginner-friendly!**

---

<div align="center">

**Internship:** Neurofive Solutions ML Fundamentals
**Duration:** 6 weeks (Jul 2026 – Present)
**Intern ID:** `NFS-2607-0177`
**Status:** ✅ Week 4 — Complete

*From data to insights. From questions to answers.* 📊🚀

</div>
