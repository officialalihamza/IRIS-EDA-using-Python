🌸 Iris Dataset – Exploratory Data Analysis (Kaggle Compatible)
🔹 1. Install Required Libraries
!pip install -q ydata-profiling

🔹 2. Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import warnings

warnings.filterwarnings("ignore")

🔹 3. Load the Dataset
df = pd.read_csv('/kaggle/input/d/ruthlessali/iris-classification/IRIS.csv')

# Remove unwanted index columns (if any)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df.head()

🔹 4. Basic Data Understanding
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nInfo:")
df.info()

🔹 5. Statistical Summary
df.describe()

🔹 6. Check Missing Values
df.isnull().sum()

🔹 7. Basic Visualizations
📊 Pairplot
sns.pairplot(df, hue="species")
plt.show()

📊 Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.drop("species", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

🔹 8. Automated EDA using YData Profiling (✅ Kaggle Safe)
profile = ProfileReport(
    df,
    title="Iris Dataset - Exploratory Data Analysis",
    explorative=True
)

profile.to_notebook_iframe()

🔹 9. (Optional) Export EDA Report
profile.to_file("iris_eda_report.html")


You can download the HTML report from Kaggle outputs 📥

✅ FINAL NOTES (Important)

❌ Sweetviz is NOT compatible with Kaggle (NumPy 2.x issue)

✅ YData Profiling is official, stable, and production-ready

This notebook is portfolio-ready

If you want next:

📌 Machine Learning model on Iris

📌 GitHub README

📌 Improved LinkedIn post

📌 End-to-end EDA → ML notebook

Just say the word 🚀

main.ipynb
File
Read me github file

Here is a clean, professional GitHub README.md you can directly copy-paste for your Iris EDA Kaggle notebook.
This is portfolio-ready, recruiter-friendly, and honest about tools used ✅

🌸 Iris Dataset – Exploratory Data Analysis (EDA)
📌 Project Overview

This project performs Exploratory Data Analysis (EDA) on the classic Iris Flower Dataset to understand feature distributions, relationships, and class separability before applying machine learning models.

The analysis is conducted in a Kaggle Notebook environment, using both manual visualizations and automated profiling tools.

📊 Dataset

Name: Iris Flower Dataset

Records: 150

Features:

Sepal Length

Sepal Width

Petal Length

Petal Width

Species (Target)

🛠️ Tools & Technologies

Python

Pandas & NumPy – Data handling

Matplotlib & Seaborn – Visualizations

YData Profiling – Automated EDA (Kaggle compatible)

⚠️ Note: Sweetviz and D-Tale were avoided due to compatibility issues with NumPy 2.x in Kaggle.

🔍 EDA Steps Covered

Data loading & cleaning

Shape, structure & summary statistics

Missing value analysis

Feature distributions

Correlation analysis

Pairwise feature relationships

Automated EDA report generation

📈 Key Insights

Petal features show strong correlation with species classification

Iris-setosa is linearly separable, while versicolor & virginica overlap

Dataset is clean with no missing values

📄 Automated EDA Report

An interactive HTML report is generated using YData Profiling, providing:

Feature statistics

Correlations

Distribution plots

Warnings & data quality checks

You can download the report directly from the notebook outputs.

🚀 How to Run
pip install ydata-profiling

from ydata_profiling import ProfileReport
profile = ProfileReport(df, explorative=True)
profile.to_notebook_iframe()

📂 Repository Structure
├── main.ipynb
├── iris_eda_report.html
└── README.md

🎯 Future Work

Feature scaling & preprocessing

Machine learning models (KNN, SVM, Logistic Regression)

Model evaluation & comparison

👤 Author

Ali Hamza
AI & Data Science Student

🔗 Kaggle: https://www.kaggle.com/ruthlessali

⭐ Acknowledgements

Kaggle

UCI Machine Learning Repository
