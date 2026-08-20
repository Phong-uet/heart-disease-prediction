"""
Unit test cơ bản cho training pipeline.
Chạy: pytest src/tests/
"""

import pandas as pd
import pytest

from src.pipelines.advanced.training_pipeline import build_model, split_data


def test_build_model_random_forest():
    model = build_model("random_forest", {"n_estimators": 10, "max_depth": 3})
    assert model is not None
    assert model.n_estimators == 10


def test_build_model_invalid_type_raises_error():
    with pytest.raises(ValueError):
        build_model("unknown_model", {})


def test_split_data_shapes():
    df = pd.DataFrame(
        {
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [0, 1] * 50,
        }
    )
    X_train, X_test, y_train, y_test = split_data(
        df, target_col="target", test_size=0.2, random_state=42
    )
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert "target" not in X_train.columns
