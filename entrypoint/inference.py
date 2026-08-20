"""
Entry point để chạy batch inference với model đã huấn luyện.

Cách chạy:
    python entrypoint/inference.py --mode basic    --config config/basic/local.yaml
    python entrypoint/inference.py --mode advanced --config config/advanced/local.yaml
"""

import argparse
import sys
import os

sys.path.append(os.getcwd())

from src.pipelines.utils import load_config, get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Run inference for heart disease prediction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "advanced"],
        default="basic",
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
    logger = get_logger("inference_entrypoint", config["logging"]["level"])

    if args.mode == "basic":
        from src.pipelines.basic.inference_pipeline import run_batch_inference
    else:
        from src.pipelines.advanced.inference_pipeline import run_batch_inference

    logger.info(f"=== Đang chạy inference (mode={args.mode}) ===")
    result = run_batch_inference(config)
    logger.info(f"=== HOÀN TẤT === Số lượng bản ghi dự đoán: {len(result)}")


if __name__ == "__main__":
    main()
