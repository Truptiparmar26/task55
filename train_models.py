import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

DATASET_PATH = "Womens Clothing E-Commerce Reviews.csv"

df = pd.read_csv(DATASET_PATH)

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

target = "recommended_ind"

required_columns = [
    "age",
    "rating",
    "positive_feedback_count",
    "division_name",
    "department_name",
    "class_name",
    target
]

df = df[required_columns].copy()

df[target] = pd.to_numeric(df[target], errors="coerce")

df = df.dropna(subset=[target])

df[target] = df[target].astype(int)

df = df[df[target].isin([0, 1])]

X = df.drop(columns=[target])
y = df[target]

numeric_features = [
    "age",
    "rating",
    "positive_feedback_count"
]

categorical_features = [
    "division_name",
    "department_name",
    "class_name"
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

os.makedirs("models", exist_ok=True)

results = []

print("\nTraining models...")

for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        pipeline,
        os.path.join("models", filename),
        compress=3
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2)
    })

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df)

joblib.dump(results_df, "models/model_info.pkl")

print("\nAll models trained and saved successfully!")