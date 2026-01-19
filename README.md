# 🌸 Iris Dataset – Exploratory Data Analysis (EDA)
## 📌 Project Overview

This project performs Exploratory Data Analysis (EDA) on the classic Iris Flower Dataset to understand feature distributions, relationships, and class separability before applying machine learning models.

The analysis is conducted in a Kaggle Notebook environment, using both manual visualizations and automated profiling tools.

# 📊 Dataset

## Name: Iris Flower Dataset

## Records: 150

## Features:
Sepal Length
Sepal Width
Petal Length
Petal Width

## Species (Target)

# 🛠️ Tools & Technologies

Python

Pandas & NumPy – Data handling

Matplotlib & Seaborn – Visualizations

YData Profiling – Automated EDA (Kaggle compatible)


# 🔍 EDA Steps Covered

## Data loading & cleaning
## Shape, structure & summary statistics
## Missing value analysis
## Feature distributions
## Correlation analysis
## Pairwise feature relationships
## Automated EDA report generation

# 📈 Key Insights

Petal features show strong correlation with species classification

Iris-setosa is linearly separable, while versicolor & virginica overlap

Dataset is clean with no missing values

# 📄 Automated EDA Report

## An interactive HTML report is generated using YData Profiling, providing:

Feature statistics
Correlations
Distribution plots
Warnings & data quality checks

You can download the report directly from the notebook outputs.

# 🚀 How to Run
pip install ydata-profiling
from ydata_profiling import ProfileReport
profile = ProfileReport(df, explorative=True)
profile.to_notebook_iframe()

📂 Repository Structure
├── main.ipynb
├── iris_eda_report.html
└── README.md

# 🎯 Future Work

Feature scaling & preprocessing
Machine learning models (KNN, SVM, Logistic Regression)
Model evaluation & comparison

# 👤 Author

Ali Hamza 
AI & Data Science Student
🔗 Kaggle: https://www.kaggle.com/ruthlessali
⭐ Acknowledgements
