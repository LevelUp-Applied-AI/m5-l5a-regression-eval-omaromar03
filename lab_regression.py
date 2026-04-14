import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    r2_score
)

import matplotlib.pyplot as plt


# =========================
# Task 1: Load + EDA
# =========================
def load_data(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nChurn distribution:\n", df["churned"].value_counts(normalize=True))

    return df


# =========================
# Task 2: Split
# =========================
def split_data(df, target="churned"):
    X = df.drop(columns=[target])
    y = df[target]

    if target == "churned":
        stratify = y
    else:
        stratify = None

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )


# =========================
# Helper: Preprocessing
# =========================
def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    # drop id if exists
    if "customer_id" in numeric_cols:
        numeric_cols = numeric_cols.drop("customer_id")

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    return preprocessor


# =========================
# Task 3: Logistic Regression
# =========================
def logistic_pipeline(X_train, X_test, y_train, y_test):

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }


# =========================
# Task 4: Ridge Regression
# =========================
def ridge_pipeline(df):

    X = df.drop(columns=["monthly_charges"])
    y = df["monthly_charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", Ridge(alpha=1.0))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nRidge Regression:")
    print("MAE:", mae)
    print("R2:", r2)

    return mae, r2


# =========================
# Task 5: Lasso Comparison
# =========================
def lasso_pipeline(df):

    X = df.drop(columns=["monthly_charges"])
    y = df["monthly_charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    ridge = Pipeline([
        ("preprocess", preprocessor),
        ("model", Ridge(alpha=1.0))
    ])

    lasso = Pipeline([
        ("preprocess", preprocessor),
        ("model", Lasso(alpha=0.1))
    ])

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    ridge_coef = ridge.named_steps["model"].coef_
    lasso_coef = lasso.named_steps["model"].coef_

    print("\nRidge vs Lasso coefficients:")
    print("Ridge:", ridge_coef[:10])
    print("Lasso:", lasso_coef[:10])

    # comment:
    # Lasso sets some coefficients to zero → removes weak/unimportant features


# =========================
# Task 6: Cross Validation
# =========================
def cross_validation(X_train, y_train):

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print("\nCV Scores:", scores)
    print("Mean:", scores.mean())
    print("Std:", scores.std())


# =========================
# Task 7: Summary
# =========================
"""
Summary:
- Features like contract type, tenure, and number of support calls appear important.
- Model performs reasonably but recall is more critical due to churn imbalance.
- Improving recall is important to catch more churn customers.
- Future improvements: feature engineering, hyperparameter tuning, and threshold tuning.
"""


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    metrics = logistic_pipeline(X_train, X_test, y_train, y_test)

    ridge_pipeline(df)

    lasso_pipeline(df)

    cross_validation(X_train, y_train)