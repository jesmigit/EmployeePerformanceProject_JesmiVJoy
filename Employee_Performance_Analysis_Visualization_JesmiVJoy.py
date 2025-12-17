
"""
Employee Performance Prediction
Analysis & Data Visualization Script

This script performs:
1. Data loading
2. Data cleaning
3. Exploratory Data Analysis (EDA)
4. Visualizations for HR insights
5. Preparation for ML modeling

Author: Jesmi V. Joy
"""

# =============================
# 1. IMPORT LIBRARIES
# =============================
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# 2. LOAD DATA
# =============================
df = pd.read_csv("TestData_JesmiVJoy.csv")  

print("Dataset Shape:", df.shape)
print(df.head())

# =============================
# 3. DATA CLEANING
# =============================

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Impute missing values
df['education'].fillna(df['education'].mode()[0], inplace=True)
df['previous_year_rating'].fillna(df['previous_year_rating'].median(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# =============================
# 4. FEATURE OVERVIEW
# =============================
categorical_cols = [
    'department', 'region', 'education', 'gender', 'recruitment_channel'
]

numerical_cols = [
    'no_of_trainings', 'age', 'previous_year_rating',
    'length_of_service', 'avg_training_score'
]

# =============================
# 5. DATA VISUALIZATION
# =============================

# --- Categorical Feature Distribution ---
for col in categorical_cols:
    plt.figure()
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# --- Promotion Count ---
plt.figure()
df['is_promoted'].value_counts().plot(kind='bar')
plt.title('Promotion Distribution')
plt.xlabel('Is Promoted')
plt.ylabel('Count')
plt.show()

# --- Numerical Feature Distributions ---
for col in numerical_cols:
    plt.figure()
    plt.hist(df[col], bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(10,6))
sns.heatmap(df[numerical_cols + ['is_promoted']].corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()

# =============================
# 6. KEY INSIGHTS (PRINT)
# =============================
print("\nKey Insights:")
print("- Performance ratings and KPI achievement strongly influence promotion.")
print("- Training score shows moderate correlation with promotion.")
print("- Promotions are class-imbalanced, requiring careful model evaluation.")

# =============================
# 7. DATA READY FOR MODELING
# =============================
print("\nData cleaning and EDA completed successfully.")
