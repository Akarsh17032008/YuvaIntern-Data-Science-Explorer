# ==============================================================================
# PROJECT: Machine Learning Model Development & Evaluation Blueprint
# STAGE: Task 3 - ML Pipeline Implementation
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def build_preprocessing_pipeline(num_cols, cat_cols):
    """Creates a ColumnTransformer for continuous and categorical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    return preprocessor

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Applies SMOTE, trains an XGBoost classifier, and evaluates performance."""
    # Handle Class Imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Train Model
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train_res, y_train_res)
    
    # Predict and Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("--- MODEL EVALUATION REPORT ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Recall Score: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    print("Machine Learning Pipeline Architecture Initialized.")
