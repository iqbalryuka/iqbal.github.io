"""Melatih dan mengevaluasi model prediksi churn.

Jalankan dari folder project:
    python src/train.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PROJECT_DIR, TARGET, load_clean

RANDOM_STATE = 42
FIG_DIR = PROJECT_DIR / "reports" / "figures"
REPORT_DIR = PROJECT_DIR / "reports"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES),
    ])


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
    }


def split(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def cross_validate_models(X_train, y_train) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in build_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=["roc_auc", "f1", "recall", "precision"])
        rows.append({
            "Model": name,
            "ROC-AUC": scores["test_roc_auc"].mean(),
            "ROC-AUC (std)": scores["test_roc_auc"].std(),
            "F1": scores["test_f1"].mean(),
            "Recall": scores["test_recall"].mean(),
            "Precision": scores["test_precision"].mean(),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def fit_all(X_train, y_train) -> dict:
    fitted = {}
    for name, model in build_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        fitted[name] = pipe.fit(X_train, y_train)
    return fitted


def test_metrics(pipe, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def plot_roc(fitted: dict, X_test, y_test, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_title("Kurva ROC pada Data Uji")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(pipe, X_test, y_test, name: str, path: Path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_test, y_test, display_labels=["Tidak Churn", "Churn"],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_permutation_importance(pipe, X_test, y_test, path: Path, top_n: int = 12) -> pd.Series:
    result = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=imp.head(top_n).values, y=imp.head(top_n).index, color="#4C72B0", ax=ax)
    ax.set_xlabel("Penurunan ROC-AUC saat fitur diacak")
    ax.set_ylabel("")
    ax.set_title("Fitur Paling Berpengaruh (Permutation Importance)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return imp


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)

    print("Cross-validation 5-fold pada data latih...")
    cv_table = cross_validate_models(X_train, y_train)
    print(cv_table.round(4).to_string(index=False))

    fitted = fit_all(X_train, y_train)
    test_table = pd.DataFrame(
        {name: test_metrics(pipe, X_test, y_test) for name, pipe in fitted.items()}
    ).T
    print("\nPerforma pada data uji:")
    print(test_table.round(4).to_string())

    best_name = cv_table.loc[0, "Model"]
    best = fitted[best_name]
    print(f"\nModel terbaik (berdasarkan ROC-AUC CV): {best_name}")
    print(classification_report(y_test, best.predict(X_test),
                                target_names=["Tidak Churn", "Churn"]))

    plot_roc(fitted, X_test, y_test, FIG_DIR / "roc_curve.png")
    plot_confusion(best, X_test, y_test, best_name, FIG_DIR / "confusion_matrix.png")
    importance = plot_permutation_importance(best, X_test, y_test,
                                             FIG_DIR / "feature_importance.png")

    summary = {
        "n_rows": int(len(df)),
        "churn_rate": float(df[TARGET].mean()),
        "best_model": best_name,
        "cross_validation": cv_table.round(4).to_dict(orient="records"),
        "test_metrics": test_table.round(4).to_dict(orient="index"),
        "top_features": importance.head(10).round(4).to_dict(),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Hasil disimpan ke {REPORT_DIR}")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
