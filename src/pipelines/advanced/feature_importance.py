"""
Phân tích Feature Importance cho model dự đoán bệnh tim.

Sử dụng 2 phương pháp:
1. Gini Importance (built-in của Random Forest) - nhanh nhưng có thể
   thiên vị các feature có nhiều giá trị unique (biased với high-cardinality)
2. Permutation Importance - đáng tin cậy hơn, đo mức độ accuracy giảm
   khi xáo trộn ngẫu nhiên từng cột trên tập test

Output:
- reports/advanced/figures/feature_importance.png
- reports/advanced/metrics.json (bổ sung phần feature_importance)
"""

import json
import sys
import os

sys.path.append(os.getcwd())

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.pipelines.utils import load_config, get_logger

logger = get_logger(__name__)


def analyze_feature_importance(config: dict):
    data_cfg = config["data"]
    model_cfg = config["model"]

    logger.info("Đang load model và dữ liệu...")
    model = joblib.load(model_cfg["save_path"])
    df = pd.read_csv(data_cfg["features_path"])

    X = df.drop(columns=[data_cfg["target_column"]])
    y = df[data_cfg["target_column"]]

    # Dùng lại đúng tập test như lúc training để đánh giá permutation importance
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,
    )

    # --- 1. Gini Importance ---
    gini_importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    # --- 2. Permutation Importance ---
    logger.info("Đang tính permutation importance (có thể mất chút thời gian)...")
    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=30, random_state=data_cfg["random_state"], scoring="roc_auc"
    )
    perm_importance = pd.Series(
        perm_result.importances_mean, index=X.columns
    ).sort_values(ascending=False)
    perm_std = pd.Series(perm_result.importances_std, index=X.columns)

    # --- In kết quả ---
    logger.info("\n=== TOP 10 Gini Importance ===\n" + gini_importance.head(10).to_string())
    logger.info("\n=== TOP 10 Permutation Importance ===\n" + perm_importance.head(10).to_string())

    # --- Vẽ biểu đồ so sánh ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    top_n = 15
    gini_top = gini_importance.head(top_n).sort_values()
    axes[0].barh(gini_top.index, gini_top.values, color="#4C72B0")
    axes[0].set_title("Gini Importance (Random Forest)", fontsize=13)
    axes[0].set_xlabel("Importance")

    perm_top = perm_importance.head(top_n).sort_values()
    perm_err = perm_std.reindex(perm_top.index)
    axes[1].barh(perm_top.index, perm_top.values, xerr=perm_err.values, color="#DD8452")
    axes[1].set_title("Permutation Importance (ROC-AUC drop)", fontsize=13)
    axes[1].set_xlabel("Importance (mean decrease in ROC-AUC)")

    plt.tight_layout()
    fig_path = "reports/advanced/figures/feature_importance.png"
    plt.savefig(fig_path, dpi=150)
    logger.info(f"Đã lưu biểu đồ vào {fig_path}")

    # --- Lưu kết quả dạng json ---
    result = {
        "gini_importance": gini_importance.to_dict(),
        "permutation_importance": perm_importance.to_dict(),
    }
    with open("reports/advanced/metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu kết quả vào reports/advanced/metrics.json")

    return gini_importance, perm_importance


if __name__ == "__main__":
    cfg = load_config("config/advanced/local.yaml")
    analyze_feature_importance(cfg)
