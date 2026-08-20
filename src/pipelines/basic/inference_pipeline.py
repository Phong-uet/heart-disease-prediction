"""
Pipeline dự đoán (inference) — chế độ BASIC.
"""

import joblib
import pandas as pd

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from src.pipelines.utils import get_logger

logger = get_logger(__name__)


def load_model(model_path: str):
    logger.info(f"Đang load model từ {model_path}")
    return joblib.load(model_path)


def predict(model, X: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    result = X.copy()
    result["prediction"] = preds
    result["probability"] = probas
    return result


def run_batch_inference(config: dict):
    model = load_model(config["model"]["save_path"])
    X = pd.read_csv(config["data"]["features_path"])

    if config["data"]["target_column"] in X.columns:
        X = X.drop(columns=[config["data"]["target_column"]])

    result = predict(model, X)

    output_path = config["data"]["predictions_path"]
    result.to_csv(output_path, index=False)
    logger.info(f"Đã lưu kết quả dự đoán vào {output_path}")

    return result


if __name__ == "__main__":
    from src.pipelines.utils import load_config

    cfg = load_config("config/basic/local.yaml")
    run_batch_inference(cfg)
