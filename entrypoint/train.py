"""
Entry point để chạy toàn bộ quy trình huấn luyện.

Cách chạy:
    python entrypoint/train.py --mode basic    --config config/basic/local.yaml
    python entrypoint/train.py --mode advanced --config config/advanced/local.yaml
"""

import argparse
import sys
import os

sys.path.append(os.getcwd())

from src.pipelines.utils import load_config, get_logger


def main():
    parser = argparse.ArgumentParser(description="Train heart disease prediction model")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "advanced"],
        default="basic",
        help="basic: dataset tự đánh giá (BRFSS) | advanced: dataset lâm sàng (UCI)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Đường dẫn config, mặc định config/<mode>/local.yaml",
    )
    args = parser.parse_args()
    config_path = args.config or f"config/{args.mode}/local.yaml"

    config = load_config(config_path)
    logger = get_logger("train_entrypoint", config["logging"]["level"])
    logger.info(f"=== CHẾ ĐỘ: {args.mode.upper()} ===")

    if args.mode == "basic":
        from src.pipelines.basic.preprocess_pipeline import preprocess
        from src.pipelines.basic.feature_eng_pipeline import build_features
        from src.pipelines.basic.training_pipeline import train
    else:
        from src.pipelines.advanced.preprocess_pipeline import preprocess
        from src.pipelines.advanced.feature_eng_pipeline import build_features
        from src.pipelines.advanced.training_pipeline import train

    logger.info("=== BƯỚC 1: Preprocessing ===")
    preprocess(config)

    logger.info("=== BƯỚC 2: Feature Engineering ===")
    build_features(config)

    logger.info("=== BƯỚC 3: Training Model ===")
    model, metrics = train(config)

    logger.info(f"=== HOÀN TẤT === Kết quả: {metrics}")


if __name__ == "__main__":
    main()
