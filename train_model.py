# ==========================================================
# SUPPLY CHAIN RISK CLASSIFICATION PROJECT
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import time
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 70)
print("SUPPLY CHAIN RISK CLASSIFICATION")
print("=" * 70)

df = pd.read_csv("supply_chain.csv")

print("\nDataset Shape:", df.shape)

# ==========================================================
# PREPROCESSING
# ==========================================================

print("\nPREPROCESSING REPORT")
print("-" * 70)

print("\nMissing Values:")
print(df.isnull().sum())

before_rows = len(df)

df.drop_duplicates(inplace=True)

after_rows = len(df)

print("\nRows Before Duplicate Removal :", before_rows)
print("Rows After Duplicate Removal  :", after_rows)

# Fill Missing Values

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values Handled Successfully")

# ==========================================================
# ENCODING
# ==========================================================

country_encoder = LabelEncoder()

df["supplier_country"] = country_encoder.fit_transform(
    df["supplier_country"]
)

target_encoder = LabelEncoder()

df["risk_classification"] = target_encoder.fit_transform(
    df["risk_classification"]
)

print("\nTarget Classes:")
print(target_encoder.classes_)

# ==========================================================
# BALANCE CLASSES
# ==========================================================

print("\nBalancing Dataset...")

class_counts = df["risk_classification"].value_counts()

print("\nBefore Balancing:")
print(class_counts)

min_samples = class_counts.min()

balanced_data = []

for cls in df["risk_classification"].unique():

    class_df = df[df["risk_classification"] == cls]

    class_df = resample(
        class_df,
        replace=False,
        n_samples=min_samples,
        random_state=42
    )

    balanced_data.append(class_df)

df = pd.concat(balanced_data)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nAfter Balancing:")
print(df["risk_classification"].value_counts())

# ==========================================================
# REMOVE TARGET LEAKAGE FEATURES
# ==========================================================

print("\nRemoving Leakage Features...")

X = df.drop(
    columns=[
        "risk_classification",
        "delay_probability",
        "disruption_likelihood_score",
        "delivery_time_deviation",
        "route_risk_level",
        "customs_clearance_time",
        "lead_time_days"
    ]
)

y = df["risk_classification"]

print("\nFeatures Used:")
print(X.columns.tolist())

# ==========================================================
# SCALING
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

results = []

def evaluate_model(model_name, model):

    print("\n")
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    start = time.time()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    end = time.time()

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Training Time : {end-start:.2f} seconds")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=target_encoder.classes_
        )
    )

    results.append([
        model_name,
        accuracy,
        precision,
        recall,
        f1
    ])

# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

evaluate_model(
    "Logistic Regression",
    lr
)

# ==========================================================
# KNN
# ==========================================================

knn = KNeighborsClassifier(
    n_neighbors=15
)

evaluate_model(
    "KNN",
    knn
)

# ==========================================================
# RANDOM FOREST
# ==========================================================

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

evaluate_model(
    "Random Forest",
    rf
)

# ==========================================================
# XGBOOST HYPERPARAMETER TUNING
# ==========================================================

print("\n")
print("=" * 70)
print("XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)

param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1]
}

xgb = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

evaluate_model(
    "Hyper Tuned XGBoost",
    best_xgb
)

# ==========================================================
# MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df)

# ==========================================================
# BEST MODEL
# ==========================================================

best_model_row = results_df.iloc[0]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model     :", best_model_row["Model"])
print("Accuracy  :", round(best_model_row["Accuracy"], 4))
print("Precision :", round(best_model_row["Precision"], 4))
print("Recall    :", round(best_model_row["Recall"], 4))
print("F1 Score  :", round(best_model_row["F1 Score"], 4))

# ==========================================================
# SAVE FILES
# ==========================================================

best_model_name = results_df.iloc[0]["Model"]

if best_model_name == "Random Forest":
    best_model = rf
elif best_model_name == "Hyper Tuned XGBoost":
    best_model = best_xgb
elif best_model_name == "KNN":
    best_model = knn
else:
    best_model = lr

joblib.dump(best_model, "model.pkl")

print("\nSaved Files:")
print("model.pkl")
print("scaler.pkl")
print("country_encoder.pkl")
print("target_encoder.pkl")
print("model_comparison.csv")

print("\nTraining Completed Successfully")