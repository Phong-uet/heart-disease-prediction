"""
Pipeline TIỀN XỬ LÝ (bước 2): làm sạch dữ liệu thô.

Input : data/01-raw/heart.csv
Output: data/02-preprocessed/heart_preprocessed.csv

Dataset: UCI Heart Disease (gộp Cleveland, Hungary, Switzerland, VA Long Beach)
Cột gốc: id, age, sex, dataset, cp, trestbps, chol, fbs, restecg, thalch,
         exang, oldpeak, slope, ca, thal, num
"""

import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.pipelines.utils import get_logger

logger = get_logger(__name__)

# Cột không dùng làm feature
DROP_COLS = ["id", "dataset"]

# Cột boolean dạng chữ (TRUE/FALSE) -> convert sang 1/0
BOOLEAN_COLS = ["fbs", "exang"]

# Numeric bình thường, missing không quá nhiều -> impute median
NUMERIC_IMPUTE_MEDIAN = ["trestbps", "chol", "thalch", "oldpeak"]

# Categorical missing ít -> impute mode
# (fbs, exang đã convert sang 1/0, missing ở mức vừa phải nên cũng impute mode)
CATEGORICAL_IMPUTE_MODE = ["restecg", "fbs", "exang"]

# Cột missing RẤT NHIỀU (>30%) -> tạo giá trị "unknown"/-1 riêng
# thay vì impute mode/median để tránh làm sai lệch phân bố
HIGH_MISSING_CATEGORICAL = ["slope", "thal"]
HIGH_MISSING_NUMERIC = ["ca"]

TARGET_COL_RAW = "num"
TARGET_COL = "target"


def load_raw_data(raw_path: str) -> pd.DataFrame:
    logger.info(f"Đang đọc dữ liệu thô từ {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Dữ liệu có shape: {df.shape}")
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Đã bỏ các cột không dùng: {cols_to_drop}")
    return df


def convert_target_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """num: 0 = không bệnh, 1-4 = có bệnh (các mức độ) -> chuyển về nhị phân."""
    df[TARGET_COL] = (df[TARGET_COL_RAW] > 0).astype(int)
    df = df.drop(columns=[TARGET_COL_RAW])
    logger.info(f"Phân bố target: \n{df[TARGET_COL].value_counts(normalize=True)}")
    return df


def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TRUE/FALSE (string/object) -> 1/0, giữ nguyên NaN để impute sau."""
    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chiến lược xử lý missing:
    - Numeric thường -> median
    - Categorical missing ít -> mode
    - Cột missing quá nhiều (>30%) -> giá trị "unknown" / -1 riêng biệt,
      giữ nguyên tín hiệu "dữ liệu bị thiếu" thay vì áp đặt giá trị phổ biến nhất
    """
    for col in NUMERIC_IMPUTE_MEDIAN:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: fillna median={median_val:.2f}")

    for col in CATEGORICAL_IMPUTE_MODE:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  {col}: fillna mode='{mode_val}'")

    for col in HIGH_MISSING_CATEGORICAL:
        if col in df.columns and df[col].isnull().any():
            missing_pct = df[col].isnull().mean() * 100
            df[col] = df[col].fillna("unknown")
            logger.info(f"  {col}: {missing_pct:.1f}% missing -> fillna 'unknown'")

    for col in HIGH_MISSING_NUMERIC:
        if col in df.columns and df[col].isnull().any():
            missing_pct = df[col].isnull().mean() * 100
            df[col] = df[col].fillna(-1)
            logger.info(f"  {col}: {missing_pct:.1f}% missing -> fillna -1")

    return df


def preprocess(config: dict) -> pd.DataFrame:
    """Chạy toàn bộ bước tiền xử lý, lưu ra data/02-preprocessed/."""
    df = load_raw_data(config["data"]["raw_path"])

    df = drop_unused_columns(df)
    df = convert_target_to_binary(df)
    df = convert_boolean_columns(df)

    logger.info("Đang xử lý missing values...")
    df = handle_missing_values(df)

    assert df.isnull().sum().sum() == 0, "Vẫn còn missing value sau khi xử lý!"

    output_path = config["data"]["preprocessed_path"]
    df.to_csv(output_path, index=False)
    logger.info(f"Đã lưu dữ liệu preprocessed vào {output_path}, shape={df.shape}")

    return df


if __name__ == "__main__":
    from src.pipelines.utils import load_config

    cfg = load_config("config/advanced/local.yaml")
    preprocess(cfg)
