"""
Pipeline FEATURE ENGINEERING (bước 3) — chế độ BASIC.

Dataset BRFSS đã ở dạng số sẵn (binary 0/1 hoặc ordinal), không có cột dạng
chữ cần one-hot encode. Bước này chỉ cần scale các cột numeric/ordinal.

Input : data/basic/02-preprocessed/heart_preprocessed.csv
Output: data/basic/03-features/heart_features.csv
        models/basic/scaler.pkl
        models/basic/feature_columns.json
"""

import json

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.pipelines.utils import get_logger

logger = get_logger(__name__)

# Cột nhị phân (0/1) -> giữ nguyên, không cần encode/scale
BINARY_COLS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "DiffWalk", "Sex",
]

# Cột numeric/ordinal -> scale bằng StandardScaler
SCALE_COLS = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income", "Diabetes"]


def load_preprocessed_data(preprocessed_path: str) -> pd.DataFrame:
    logger.info(f"Đang đọc dữ liệu đã preprocessed từ {preprocessed_path}")
    df = pd.read_csv(preprocessed_path)
    logger.info(f"Dữ liệu có shape: {df.shape}")
    return df


def fit_scale_numeric(df: pd.DataFrame):
    existing_cols = [c for c in SCALE_COLS if c in df.columns]
    scaler = StandardScaler()
    df[existing_cols] = scaler.fit_transform(df[existing_cols])
    logger.info(f"Đã fit + scale: {existing_cols}")
    return df, scaler


def build_features(config: dict) -> pd.DataFrame:
    df = load_preprocessed_data(config["data"]["preprocessed_path"])
    df, scaler = fit_scale_numeric(df)

    output_path = config["data"]["features_path"]
    df.to_csv(output_path, index=False)
    logger.info(f"Đã lưu dữ liệu features vào {output_path}, shape={df.shape}")

    scaler_path = config["model"]["scaler_path"]
    joblib.dump(scaler, scaler_path)
    logger.info(f"Đã lưu scaler vào {scaler_path}")

    target_col = config["data"]["target_column"]
    feature_columns = [c for c in df.columns if c != target_col]
    feature_columns_path = config["model"]["feature_columns_path"]
    with open(feature_columns_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu danh sách {len(feature_columns)} feature columns vào {feature_columns_path}")

    return df


if __name__ == "__main__":
    from src.pipelines.utils import load_config

    cfg = load_config("config/basic/local.yaml")
    build_features(cfg)
