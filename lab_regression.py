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
# Load Data
# =========================
def load_data(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nChurn distribution:\n", df["churned"].value_counts(normalize=True))

    return df


# =========================
# Split Data
# =========================
def split_data(df, target="churned"):
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if target == "churned" else None

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )


# =========================
# Preprocessor
# =========================
def build_preprocessor(X):
    X_temp = X.copy()

    if "customer_id" in X_temp.columns:
        X_temp = X_temp.drop(columns=["customer_id"])

    categorical_cols = X_temp.select_dtypes(include=["object", "string"]).columns
    numeric_cols = X_temp.select_dtypes(exclude=["object", "string"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    return preprocessor


# =========================
# REQUIRED FUNCTIONS
# =========================
def evaluate_classifier(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_regressor(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# =========================
# Logistic Regression
# =========================
def build_logistic_pipeline(X_train, X_test, y_train, y_test):

    X_train_model = X_train.drop(columns=["customer_id"], errors="ignore")
    X_test_model = X_test.drop(columns=["customer_id"], errors="ignore")

    preprocessor = build_preprocessor(X_train_model)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train_model, y_train)
    y_pred = pipeline.predict(X_test_model)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

    return evaluate_classifier(y_test, y_pred)


# =========================
# Ridge Regression
# =========================
def build_ridge_pipeline(df):

    X = df.drop(columns=["monthly_charges"])
    y = df["monthly_charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    X_train_model = X_train.drop(columns=["customer_id"], errors="ignore")
    X_test_model = X_test.drop(columns=["customer_id"], errors="ignore")

    preprocessor = build_preprocessor(X_train_model)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", Ridge(alpha=1.0))
    ])

    pipeline.fit(X_train_model, y_train)
    y_pred = pipeline.predict(X_test_model)

    metrics = evaluate_regressor(y_test, y_pred)

    print("\nRidge Regression:")
    print("MAE:", metrics["mae"])
    print("R2:", metrics["r2"])

    return metrics


# =========================
# Lasso Comparison
# =========================
def build_lasso_pipeline(df):

    X = df.drop(columns=["monthly_charges"])
    y = df["monthly_charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    X_train_model = X_train.drop(columns=["customer_id"], errors="ignore")

    preprocessor = build_preprocessor(X_train_model)

    ridge = Pipeline([
        ("preprocess", preprocessor),
        ("model", Ridge(alpha=1.0))
    ])

    lasso = Pipeline([
        ("preprocess", preprocessor),
        ("model", Lasso(alpha=0.1))
    ])

    ridge.fit(X_train_model, y_train)
    lasso.fit(X_train_model, y_train)

    ridge_coef = ridge.named_steps["model"].coef_
    lasso_coef = lasso.named_steps["model"].coef_

    print("\nRidge vs Lasso coefficients:")
    print("Ridge:", ridge_coef[:10])
    print("Lasso:", lasso_coef[:10])

    return {"ridge_coef": ridge_coef, "lasso_coef": lasso_coef}


# =========================
# Cross Validation
# =========================
def run_cross_validation(X_train, y_train):

    X_train_model = X_train.drop(columns=["customer_id"], errors="ignore")

    preprocessor = build_preprocessor(X_train_model)

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
        X_train_model,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print("\nCV Scores:", scores)
    print("Mean:", scores.mean())
    print("Std:", scores.std())

    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std())
    }


"""
Summary:
- Contract type, tenure, and support calls are important features.
- Recall is more important due to class imbalance.
- Ridge regression performs well for monthly charges.
- Improvements: feature engineering and tuning.
"""


if __name__ == "__main__":
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    build_logistic_pipeline(X_train, X_test, y_train, y_test)

    build_ridge_pipeline(df)

    build_lasso_pipeline(df)

    run_cross_validation(X_train, y_train)