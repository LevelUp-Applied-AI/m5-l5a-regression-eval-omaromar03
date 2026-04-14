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


def load_data(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nChurn distribution:\n", df["churned"].value_counts(normalize=True))

    return df


def split_data(df, target="churned"):
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if target == "churned" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    if target == "churned":
        print("Train churn rate:", y_train.mean())
        print("Test churn rate:", y_test.mean())

    return X_train, X_test, y_train, y_test


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

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def build_ridge_pipeline(df):
    X = df.drop(columns=["monthly_charges"])
    y = df["monthly_charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
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

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nRidge Regression:")
    print("MAE:", mae)
    print("R2:", r2)

    return {"mae": float(mae), "r2": float(r2)}


def build_lasso_pipeline(df):
    X = df.drop(columns=["monthly_charges"])
    y = df["monthly_charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
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

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipeline,
        X_train_model,
        y_train,
        cv=cv_splitter,
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
- Features like contract type, tenure, and number of support calls appear important for predicting churn.
- Logistic regression achieved moderate performance, and recall is more important here because missing churners is costly.
- Ridge regression performed reasonably well for predicting monthly charges.
- Future improvements could include feature engineering, threshold tuning, and hyperparameter optimization.
"""


if __name__ == "__main__":
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    metrics = build_logistic_pipeline(X_train, X_test, y_train, y_test)

    build_ridge_pipeline(df)

    build_lasso_pipeline(df)

    run_cross_validation(X_train, y_train)