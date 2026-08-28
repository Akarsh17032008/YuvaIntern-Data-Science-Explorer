# ==============================================================================
# PROJECT: Exploratory Data Analysis & Visualization Blueprint
# STAGE: Task 2 - EDA Framework Implementation
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def profile_dataset(df):
    """Prints structural statistics and missing values profile."""
    print("--- DATASET SUMMARY ---")
    print(df.info())
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())
    print("\n--- DESCRIPTIVE STATISTICS ---")
    print(df.describe())

def detect_outliers_iqr(df, column):
    """Calculates IQR outlier boundaries for a continuous column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Column '{column}': {len(outliers)} outliers detected.")
    return outliers

def generate_eda_plots(df, target_col):
    """Generates standard EDA visualization suite."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

if __name__ == "__main__":
    print("EDA Framework Pipeline Initialized.")
