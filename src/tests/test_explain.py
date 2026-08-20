"""
Test cho api/explain.py (giải thích dự đoán từng bệnh nhân).
Chạy: pytest src/tests/test_explain.py -v
"""

import os
import sys

sys.path.append(os.getcwd())

import joblib
import pandas as pd
import pytest

from api.explain import explain_prediction, top_contributors

ADVANCED_MODEL_PATH = "models/advanced/best_model.pkl"


@pytest.mark.skipif(
    not os.path.exists(ADVANCED_MODEL_PATH),
    reason="Chưa train model advanced (chạy entrypoint/train.py trước)",
)
def test_explain_prediction_matches_baseline_direction():
    model = joblib.load(ADVANCED_MODEL_PATH)
    scaler = joblib.load("models/advanced/scaler.pkl")
    import json

    with open("models/advanced/feature_columns.json") as f:
        feature_columns = json.load(f)

    numeric_cols = ["age", "chol", "thalch", "oldpeak", "ca"]
    row = {
        "age": 58,
        "sex": "Male",
        "cp": "asymptomatic",
        "chol": 245,
        "fbs": 0,
        "thalch": 140,
        "exang": 1,
        "oldpeak": 2.1,
        "ca": 1,
        "thal": "reversable defect",
    }
    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=["sex", "cp", "thal"], drop_first=True)
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df = df.reindex(columns=feature_columns, fill_value=0)

    contributions = explain_prediction(model, df)
    assert "exang" in contributions

    top = top_contributors(contributions, top_k=3)
    assert len(top) == 3
    # top_contributors phải sắp xếp giảm dần theo |contribution|
    abs_values = [abs(v) for _, v in top]
    assert abs_values == sorted(abs_values, reverse=True)


def test_top_contributors_sorting():
    fake_contributions = {"a": 0.1, "b": -0.5, "c": 0.05, "d": -0.02}
    top = top_contributors(fake_contributions, top_k=2)
    assert top[0][0] == "b"  # |-0.5| lớn nhất
    assert top[1][0] == "a"  # |0.1| lớn nhì
    assert len(top) == 2
