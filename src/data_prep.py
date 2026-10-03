"""Memuat dan membersihkan dataset Telco Customer Churn."""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "telco_customer_churn.csv"

TARGET = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "AvgChargePerMonth"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "TenureGroup",
]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data mentah dan menambahkan fitur turunan."""
    df = df.copy()

    # TotalCharges tersimpan sebagai teks; pelanggan dengan tenure 0 bernilai " ".
    # Pelanggan baru tersebut belum pernah ditagih, jadi nilainya diisi 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    # Fitur turunan
    df["AvgChargePerMonth"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 bln", "13-24 bln", "25-48 bln", "49-72 bln"],
    ).astype(str)

    return df.drop(columns=["customerID"])


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))
