"""
Các hàm tiện ích dùng chung cho toàn bộ pipeline:
- Load config YAML
- Thiết lập logging
"""

import logging
import os
import yaml


def load_config(config_path: str) -> dict:
    """Đọc file config YAML và trả về dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Khởi tạo logger chuẩn cho toàn bộ project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
