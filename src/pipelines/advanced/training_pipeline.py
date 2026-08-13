"""
Pipeline huấn luyện model dự đoán bệnh tim.

Input : data/03-features/heart_features.csv
Output: models/best_model.pkl
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.pipelines.utils import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def load_features(features_path: str) -> pd.DataFrame:
    logger.info(f"Đang đọc dữ liệu features từ {features_path}")
    return pd.read_csv(features_path)


def split_data(df: pd.DataFrame, target_col: str, test_size: float, random_state: int):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_model(model_type: str, params: dict):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_type}' chưa được hỗ trợ.")
    return MODEL_REGISTRY[model_type](**params)


def train(config: dict):
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    df = load_features(data_cfg["features_path"])
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col=data_cfg["target_column"],
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
    )

    model = build_model(model_cfg["type"], model_cfg["params"])

    logger.info("Đang chạy cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=train_cfg["cv_folds"],
        scoring=train_cfg["scoring"],
    )
    logger.info(f"CV {train_cfg['scoring']} scores: {cv_scores}, mean={cv_scores.mean():.4f}")

    logger.info("Đang huấn luyện model trên toàn bộ tập train...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test ROC-AUC : {auc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    joblib.dump(model, model_cfg["save_path"])
    logger.info(f"Đã lưu model vào {model_cfg['save_path']}")

    return model, {"accuracy": acc, "roc_auc": auc}


if __name__ == "__main__":
    from src.pipelines.utils import load_config

    cfg = load_config("config/advanced/local.yaml")
    train(cfg)
