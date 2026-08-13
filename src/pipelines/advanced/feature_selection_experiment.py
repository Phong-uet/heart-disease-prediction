"""
Thí nghiệm: so sánh model dùng ĐẦY ĐỦ feature vs model đã LOẠI BỎ
các feature có permutation importance <= 0 (trestbps, restecg, slope).

Chạy: python src/pipelines/feature_selection_experiment.py
"""

import sys
import os

sys.path.append(os.getcwd())

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.pipelines.utils import load_config, get_logger

logger = get_logger(__name__)

# Các cột gốc (trước encode) có permutation importance <= 0 ở thí nghiệm trước
LOW_IMPORTANCE_RAW_COLS = ["trestbps", "restecg", "slope"]

NUMERIC_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_COLS = ["sex", "cp", "restecg", "slope", "thal"]


def build_features_variant(df: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
    """Encode + scale, có thể loại bỏ 1 số cột gốc trước khi build feature."""
    df = df.copy()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def evaluate_variant(df: pd.DataFrame, target_col: str, config: dict, label: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    model = RandomForestClassifier(**config["model"]["params"])

    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=config["training"]["cv_folds"],
        scoring=config["training"]["scoring"],
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    result = {
        "variant": label,
        "n_features": X.shape[1],
        "cv_roc_auc_mean": cv_scores.mean(),
        "cv_roc_auc_std": cv_scores.std(),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }
    return result


def run_experiment(config: dict):
    df_raw = pd.read_csv(config["data"]["preprocessed_path"])
    target_col = config["data"]["target_column"]

    logger.info("=== Variant A: FULL features ===")
    df_full = build_features_variant(df_raw, drop_cols=None)
    result_full = evaluate_variant(df_full, target_col, config, "Full features")

    logger.info("=== Variant B: Đã loại bỏ trestbps, restecg, slope ===")
    df_reduced = build_features_variant(df_raw, drop_cols=LOW_IMPORTANCE_RAW_COLS)
    result_reduced = evaluate_variant(df_reduced, target_col, config, "Reduced features")

    comparison = pd.DataFrame([result_full, result_reduced]).set_index("variant")
    logger.info("\n=== SO SÁNH KẾT QUẢ ===\n" + comparison.to_string())

    comparison.to_csv("reports/advanced/feature_selection_comparison.csv")
    logger.info("Đã lưu bảng so sánh vào reports/advanced/feature_selection_comparison.csv")

    return comparison


if __name__ == "__main__":
    cfg = load_config("config/advanced/local.yaml")
    run_experiment(cfg)
