"""
Pipeline TIỀN XỬ LÝ (bước 2) — chế độ BASIC (BRFSS Heart Disease Health Indicators).

Toàn bộ 21 câu hỏi trong dataset này người dùng đều tự trả lời được tại nhà,
không cần xét nghiệm máu hay đo điện tâm đồ.

Input : data/basic/01-raw/heart.csv
Output: data/basic/02-preprocessed/heart_preprocessed.csv

Nguồn: CDC BRFSS 2015 (Behavioral Risk Factor Surveillance System)
Cột gốc: HeartDiseaseorAttack, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
         Diabetes, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
         NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
"""

import pandas as pd

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from src.pipelines.utils import get_logger

logger = get_logger(__name__)

TARGET_COL_RAW = "HeartDiseaseorAttack"
TARGET_COL = "target"

# Toàn bộ cột trong dataset này vốn đã sạch (không missing, đã ở dạng số)
# nên bước preprocess chủ yếu là: đổi tên cột dễ hiểu + ép kiểu int cho binary cols
BINARY_COLS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]
ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income", "Diabetes"]
NUMERIC_COLS = ["BMI", "MentHlth", "PhysHlth"]


def load_raw_data(raw_path: str) -> pd.DataFrame:
    logger.info(f"Đang đọc dữ liệu thô từ {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Dữ liệu có shape: {df.shape}")
    return df


def convert_target(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET_COL] = df[TARGET_COL_RAW].astype(int)
    df = df.drop(columns=[TARGET_COL_RAW])
    logger.info(f"Phân bố target:\n{df[TARGET_COL].value_counts(normalize=True)}")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset gốc toàn bộ là float dù bản chất là int (0/1 hoặc mã thứ bậc)."""
    for col in BINARY_COLS + ORDINAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset BRFSS đã được làm sạch sẵn, nhưng vẫn kiểm tra để chắc chắn."""
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        logger.warning(
            f"Phát hiện {n_missing} missing values, đang điền median/mode..."
        )
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
    else:
        logger.info("Không có missing values.")
    return df


def preprocess(config: dict) -> pd.DataFrame:
    df = load_raw_data(config["data"]["raw_path"])

    df = convert_target(df)
    df = cast_types(df)
    df = check_missing(df)

    assert df.isnull().sum().sum() == 0, "Vẫn còn missing value sau khi xử lý!"

    output_path = config["data"]["preprocessed_path"]
    df.to_csv(output_path, index=False)
    logger.info(f"Đã lưu dữ liệu preprocessed vào {output_path}, shape={df.shape}")

    return df


if __name__ == "__main__":
    from src.pipelines.utils import load_config

    cfg = load_config("config/basic/local.yaml")
    preprocess(cfg)
