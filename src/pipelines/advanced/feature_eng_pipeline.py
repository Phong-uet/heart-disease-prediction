"""
Pipeline FEATURE ENGINEERING (bước 3): encode categorical + scale numeric.

Input : data/02-preprocessed/heart_preprocessed.csv
Output: data/03-features/heart_features.csv
        models/scaler.pkl            (StandardScaler đã fit, dùng lại lúc serving API)
        models/feature_columns.json  (danh sách + thứ tự cột feature, dùng để align input mới)
"""

import json

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from src.pipelines.utils import get_logger

logger = get_logger(__name__)

# Các cột loại bỏ vì permutation importance <= 0 (xem reports/feature_selection_comparison.csv)
# — kết luận từ thí nghiệm feature_selection_experiment.py
DROP_LOW_IMPORTANCE_COLS = ["trestbps", "restecg", "slope"]

# Numeric: bao gồm cả 'ca' (đã impute -1 cho missing ở bước preprocess)
NUMERIC_COLS = ["age", "chol", "thalch", "oldpeak", "ca"]

# Categorical cần one-hot encode
# (fbs, exang đã là 0/1 từ bước preprocess nên không cần encode lại)
CATEGORICAL_COLS = ["sex", "cp", "thal"]


def load_preprocessed_data(preprocessed_path: str) -> pd.DataFrame:
    logger.info(f"Đang đọc dữ liệu đã preprocessed từ {preprocessed_path}")
    df = pd.read_csv(preprocessed_path)
    logger.info(f"Dữ liệu có shape: {df.shape}")
    return df


def drop_low_importance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ các cột có permutation importance <= 0 ở thí nghiệm trước đó."""
    cols_to_drop = [c for c in DROP_LOW_IMPORTANCE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Đã bỏ các cột ít quan trọng: {cols_to_drop}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode các biến categorical."""
    existing_cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)
    logger.info(f"Đã one-hot encode: {existing_cat_cols}")
    return df


def fit_scale_numeric(df: pd.DataFrame):
    """Fit StandardScaler trên tập training và transform. Trả về (df, scaler)."""
    existing_num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    scaler = StandardScaler()
    df[existing_num_cols] = scaler.fit_transform(df[existing_num_cols])
    logger.info(f"Đã fit + scale: {existing_num_cols}")
    return df, scaler


def build_features(config: dict) -> pd.DataFrame:
    """
    Chạy toàn bộ pipeline feature engineering trên tập training.
    Lưu lại scaler và danh sách cột feature để dùng nhất quán lúc serving (API).
    """
    df = load_preprocessed_data(config["data"]["preprocessed_path"])
    df = drop_low_importance_columns(df)
    df = encode_categorical(df)
    df, scaler = fit_scale_numeric(df)

    output_path = config["data"]["features_path"]
    df.to_csv(output_path, index=False)
    logger.info(f"Đã lưu dữ liệu features vào {output_path}, shape={df.shape}")

    # --- Lưu scaler để dùng lại lúc inference/API (KHÔNG fit lại) ---
    scaler_path = config["model"]["scaler_path"]
    joblib.dump(scaler, scaler_path)
    logger.info(f"Đã lưu scaler vào {scaler_path}")

    # --- Lưu danh sách + thứ tự cột feature (trừ target) để align input mới ---
    target_col = config["data"]["target_column"]
    feature_columns = [c for c in df.columns if c != target_col]
    feature_columns_path = config["model"]["feature_columns_path"]
    with open(feature_columns_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    logger.info(
        f"Đã lưu danh sách {len(feature_columns)} feature columns vào {feature_columns_path}"
    )

    return df


if __name__ == "__main__":
    from src.pipelines.utils import load_config

    cfg = load_config("config/advanced/local.yaml")
    build_features(cfg)
