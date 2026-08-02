# 🤖 Neurofive Solutions — ML Fundamentals Internship

> Machine learning fundamentals to production-ready models in 6 weeks  
> **Neurofive Solutions** | Intern ID: `NFS-2607-0177` | Jul 2026 → Present

![Badge](https://img.shields.io/badge/Internship-Neurofive%20Solutions-blueviolet?style=for-the-badge)
![Badge](https://img.shields.io/badge/Track-ML%20Fundamentals-blue?style=for-the-badge)
![Badge](https://img.shields.io/badge/Progress-Week%203-orange?style=for-the-badge)
![Badge](https://img.shields.io/badge/Language-Python-brightgreen?style=for-the-badge)

---

## 📊 Internship Progress

```
Week 1  ████████░░░░░░░░░░░░░░░░░░░░  Week 2  ████████░░░░░░░░░░░░░░░░░░░░  Week 3  ████████░░░░░░░░░░░░░░░░░░░░
  ✅ COMPLETE                        ✅ COMPLETE                      ✅ COMPLETE
```

| Week | Task | Topic | Status |
|------|------|-------|--------|
| **1** | Task 1 | Titanic EDA | ✅ Complete |
| **1** | Task 2 | Data Cleaning & Visualization | ✅ Complete |
| **2** | Task 1 | Titanic Classification (Logistic Regression) | ✅ Complete |
| **2** | Task 2 | Housing Price Prediction (Linear Regression) | ✅ Complete |
| **3** | Task 1 | Model Evaluation & Hyperparameter Tuning | ✅ Complete |
| **3** | Task 2 | Customer Churn Prediction (Business Problem) | ✅ Complete |

---

## 🎯 Week-by-Week Breakdown

### 📌 **Week 1: Foundation & Data Exploration**

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

---

### 📌 **Week 2: Predictive Modeling**

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

---

### 📌 **Week 3: Advanced Techniques & Business Applications**

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

---

## 🛠️ Tech Stack

<table>
  <tr>
    <td><strong>Language</strong></td>
    <td>Python 3.x</td>
  </tr>
  <tr>
    <td><strong>Data Manipulation</strong></td>
    <td>Pandas, NumPy</td>
  </tr>
  <tr>
    <td><strong>Visualization</strong></td>
    <td>Matplotlib, Seaborn</td>
  </tr>
  <tr>
    <td><strong>Machine Learning</strong></td>
    <td>scikit-learn</td>
  </tr>
  <tr>
    <td><strong>Notebook Environment</strong></td>
    <td>Jupyter</td>
  </tr>
  <tr>
    <td><strong>Version Control</strong></td>
    <td>Git, GitHub</td>
  </tr>
</table>

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
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. **Start Jupyter**
```bash
   jupyter notebook
```

4. **Navigate to any Week_X_Task_Y folder and open the `.ipynb` notebook**

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

### Real-World Skills
🔧 End-to-end ML workflow (data → model → evaluation → deployment)  
🔧 Debugging & troubleshooting model issues  
🔧 Presenting findings to non-technical stakeholders  
🔧 Handling imbalanced datasets  
🔧 Working with real Kaggle datasets  

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

---

## 🔗 Resources & References

- 📚 [scikit-learn Documentation](https://scikit-learn.org/)
- 📖 [Pandas API Reference](https://pandas.pydata.org/docs/)
- 🎨 [Matplotlib/Seaborn Tutorials](https://seaborn.pydata.org/)
- 🏠 [Kaggle Datasets](https://www.kaggle.com/datasets)
- 🤖 [ML Mastery Blog](https://machinelearningmastery.com/)

---

## 📧 Get Started

1. Pick any task folder
2. Read the README.md inside
3. Open the Jupyter notebook
4. Run the cells and follow along
5. Modify code to experiment and learn

**Each notebook is fully commented and beginner-friendly!**

---

**Internship:** Neurofive Solutions ML Fundamentals  
**Duration:** 6 weeks (Jul 2026 – Present)  
**Intern ID:** NFS-2607-0177  
**Status:** ✅ Week 3 – Complete

*From data to insights. From questions to answers.* 📊🚀
