# Neurofive ML Track — Week 1, Task 2

Cleaning and visualizing the Titanic dataset, done as part of the Neurofive Solutions Machine Learning Fundamentals internship track.

## Files
- `week_1_setup.ipynb` — notebook with data cleaning and visualizations
- `train.csv` — raw dataset from Kaggle
- `cleaned_dataset_train.csv` — dataset after handling missing values

## What I did
- Handled missing values: filled `Age` with median, filled `Embarked` with mode, dropped `Cabin` (too many missing values)
- Justified each missing value decision in a markdown note
- Detected outliers in `Fare` using a boxplot
- Created 4 visualizations using matplotlib and seaborn:
  - Histogram of Age distribution
  - Boxplot of Fare (outlier detection)
  - Bar chart of survival count by sex
  - Correlation heatmap of numerical features
- Answered which feature most affects survival, based on the visualizations

## Dataset
Titanic - Machine Learning from Disaster from Kaggle.

## Tools used
Python, pandas, NumPy, matplotlib, seaborn, Jupyter Notebook
