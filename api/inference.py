"""
Chuyển đổi input từ API (PatientInputBasic / PatientInputAdvanced) thành
dataframe feature đúng format model đã học, dự đoán, và giải thích kết quả
(yếu tố nào đang kéo xác suất lên/xuống cho từng bệnh nhân cụ thể).
"""

import json

import joblib
import pandas as pd

from api.schemas import PatientInputBasic, PatientInputAdvanced
from api.explain import explain_prediction, top_contributors

# Phải khớp SCALE_COLS trong src/pipelines/basic/feature_eng_pipeline.py
BASIC_SCALE_COLS = [
    "BMI",
    "MentHlth",
    "PhysHlth",
    "GenHlth",
    "Age",
    "Education",
    "Income",
    "Diabetes",
]

# Phải khớp NUMERIC_COLS trong src/pipelines/advanced/feature_eng_pipeline.py
ADVANCED_NUMERIC_COLS = ["age", "chol", "thalch", "oldpeak", "ca"]

AGE_GROUP_MAP = {
    "18-24": 1,
    "25-29": 2,
    "30-34": 3,
    "35-39": 4,
    "40-44": 5,
    "45-49": 6,
    "50-54": 7,
    "55-59": 8,
    "60-64": 9,
    "65-69": 10,
    "70-74": 11,
    "75-79": 12,
    "80+": 13,
}
DIABETES_MAP = {"Không": 0, "Tiền tiểu đường": 1, "Có": 2}
GEN_HEALTH_MAP = {"Rất tốt": 1, "Tốt": 2, "Khá": 3, "Trung bình": 4, "Kém": 5}
EDUCATION_MAP = {
    "Chưa học/Tiểu học": 1,
    "Trung học cơ sở": 2,
    "Trung học phổ thông (chưa tốt nghiệp)": 3,
    "Tốt nghiệp THPT": 4,
    "Cao đẳng/Đại học (chưa tốt nghiệp)": 5,
    "Tốt nghiệp Đại học": 6,
}
INCOME_MAP = {
    "<10k": 1,
    "10-15k": 2,
    "15-20k": 3,
    "20-25k": 4,
    "25-35k": 5,
    "35-50k": 6,
    "50-75k": 7,
    ">75k": 8,
}

# Nhãn tiếng Việt dễ hiểu cho từng cột feature nội bộ — dùng khi hiển thị giải thích dự đoán
BASIC_FEATURE_LABELS = {
    "HighBP": "Cao huyết áp",
    "HighChol": "Cholesterol cao",
    "CholCheck": "Đã kiểm tra cholesterol (5 năm qua)",
    "BMI": "Chỉ số BMI",
    "Smoker": "Hút thuốc",
    "Stroke": "Từng đột quỵ",
    "Diabetes": "Tiểu đường",
    "PhysActivity": "Vận động thể chất thường xuyên",
    "Fruits": "Ăn trái cây hàng ngày",
    "Veggies": "Ăn rau củ hàng ngày",
    "HvyAlcoholConsump": "Uống rượu bia nhiều",
    "AnyHealthcare": "Có bảo hiểm y tế",
    "NoDocbcCost": "Từng bỏ khám vì chi phí",
    "GenHlth": "Tự đánh giá sức khỏe",
    "MentHlth": "Số ngày không ổn tinh thần (30 ngày)",
    "PhysHlth": "Số ngày không khỏe thể chất (30 ngày)",
    "DiffWalk": "Khó đi bộ/leo cầu thang",
    "Sex": "Giới tính",
    "Age": "Nhóm tuổi",
    "Education": "Trình độ học vấn",
    "Income": "Thu nhập hộ gia đình/năm",
}

ADVANCED_FEATURE_LABELS = {
    "age": "Tuổi",
    "chol": "Cholesterol huyết thanh",
    "thalch": "Nhịp tim tối đa đạt được",
    "oldpeak": "ST depression (oldpeak)",
    "ca": "Số mạch máu chính bị hẹp",
    "fbs": "Đường huyết lúc đói > 120mg/dl",
    "exang": "Đau ngực khi vận động gắng sức",
    "sex_Male": "Giới tính Nam",
    "cp_atypical angina": "Đau ngực không điển hình",
    "cp_non-anginal": "Đau ngực không do tim",
    "cp_typical angina": "Đau thắt ngực điển hình",
    "thal_normal": "Xạ hình thallium: bình thường",
    "thal_reversable defect": "Xạ hình thallium: khiếm khuyết phục hồi được",
    "thal_unknown": "Xạ hình thallium: không có kết quả",
}


def _risk_level(proba: float) -> str:
    if proba < 0.3:
        return "Low"
    elif proba < 0.6:
        return "Medium"
    return "High"


class BasePredictor:
    def __init__(self, model_path: str, scaler_path: str, feature_columns_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(feature_columns_path, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

    def _finalize(self, df: pd.DataFrame, scale_cols: list) -> pd.DataFrame:
        df[scale_cols] = self.scaler.transform(df[scale_cols])
        df = df.reindex(columns=self.feature_columns, fill_value=0)
        return df

    def predict_from_df(
        self, X: pd.DataFrame, labels: dict, display_values: dict
    ) -> dict:
        proba = float(self.model.predict_proba(X)[0, 1])
        prediction = int(proba >= 0.5)

        contributions = explain_prediction(self.model, X)
        top = top_contributors(contributions, top_k=8)
        feature_contributions = [
            {
                "feature": labels.get(col, col),
                "value": display_values.get(col, ""),
                "contribution": round(contrib, 4),
            }
            for col, contrib in top
        ]

        return {
            "prediction": prediction,
            "probability": round(proba, 4),
            "risk_level": _risk_level(proba),
            "feature_contributions": feature_contributions,
        }


class BasicPredictor(BasePredictor):
    """Chế độ tự đánh giá — dataset BRFSS."""

    def transform(self, patient: PatientInputBasic) -> pd.DataFrame:
        bmi = patient.weight_kg / ((patient.height_cm / 100) ** 2)

        row = {
            "HighBP": int(patient.high_bp),
            "HighChol": int(patient.high_chol),
            "CholCheck": int(patient.chol_check),
            "BMI": round(bmi, 1),
            "Smoker": int(patient.smoker),
            "Stroke": int(patient.stroke),
            "Diabetes": DIABETES_MAP[patient.diabetes],
            "PhysActivity": int(patient.phys_activity),
            "Fruits": int(patient.fruits),
            "Veggies": int(patient.veggies),
            "HvyAlcoholConsump": int(patient.heavy_alcohol),
            "AnyHealthcare": int(patient.any_healthcare),
            "NoDocbcCost": int(patient.no_doc_because_cost),
            "GenHlth": GEN_HEALTH_MAP[patient.gen_health],
            "MentHlth": patient.mental_health_days,
            "PhysHlth": patient.phys_health_days,
            "DiffWalk": int(patient.diff_walk),
            "Sex": 1 if patient.sex == "Male" else 0,
            "Age": AGE_GROUP_MAP[patient.age_group],
            "Education": EDUCATION_MAP[patient.education_level],
            "Income": INCOME_MAP[patient.income_level],
        }
        df = pd.DataFrame([row])
        return self._finalize(df, BASIC_SCALE_COLS)

    def _display_values(self, patient: PatientInputBasic) -> dict:

        def yn(b):
            return "Có" if b else "Không"

        return {
            "HighBP": yn(patient.high_bp),
            "HighChol": yn(patient.high_chol),
            "CholCheck": yn(patient.chol_check),
            "BMI": f"{patient.weight_kg / ((patient.height_cm / 100) ** 2):.1f}",
            "Smoker": yn(patient.smoker),
            "Stroke": yn(patient.stroke),
            "Diabetes": patient.diabetes,
            "PhysActivity": yn(patient.phys_activity),
            "Fruits": yn(patient.fruits),
            "Veggies": yn(patient.veggies),
            "HvyAlcoholConsump": yn(patient.heavy_alcohol),
            "AnyHealthcare": yn(patient.any_healthcare),
            "NoDocbcCost": yn(patient.no_doc_because_cost),
            "GenHlth": patient.gen_health,
            "MentHlth": str(patient.mental_health_days),
            "PhysHlth": str(patient.phys_health_days),
            "DiffWalk": yn(patient.diff_walk),
            "Sex": "Nam" if patient.sex == "Male" else "Nữ",
            "Age": patient.age_group,
            "Education": patient.education_level,
            "Income": patient.income_level,
        }

    def predict(self, patient: PatientInputBasic) -> dict:
        X = self.transform(patient)
        return self.predict_from_df(
            X, BASIC_FEATURE_LABELS, self._display_values(patient)
        )


class AdvancedPredictor(BasePredictor):
    """Chế độ nâng cao — dataset UCI (cần kết quả xét nghiệm/ECG)."""

    def transform(self, patient: PatientInputAdvanced) -> pd.DataFrame:
        row = {
            "age": patient.age,
            "sex": patient.sex,
            "cp": patient.cp,
            "chol": patient.chol,
            "fbs": int(patient.fbs),
            "thalch": patient.thalch,
            "exang": int(patient.exang),
            "oldpeak": patient.oldpeak,
            "ca": patient.ca if patient.ca is not None else -1,
            "thal": patient.thal if patient.thal is not None else "unknown",
        }
        df = pd.DataFrame([row])
        df = pd.get_dummies(df, columns=["sex", "cp", "thal"], drop_first=True)
        return self._finalize(df, ADVANCED_NUMERIC_COLS)

    def _display_values(self, patient: PatientInputAdvanced) -> dict:

        def yn(b):
            return "Có" if b else "Không"

        return {
            "age": str(patient.age),
            "chol": f"{patient.chol} mg/dl",
            "thalch": str(patient.thalch),
            "oldpeak": str(patient.oldpeak),
            "ca": str(patient.ca) if patient.ca is not None else "Không có kết quả",
            "fbs": yn(patient.fbs),
            "exang": yn(patient.exang),
            "sex_Male": "Có" if patient.sex == "Male" else "Không",
            "cp_atypical angina": "Có" if patient.cp == "atypical angina" else "Không",
            "cp_non-anginal": "Có" if patient.cp == "non-anginal" else "Không",
            "cp_typical angina": "Có" if patient.cp == "typical angina" else "Không",
            "thal_normal": "Có" if patient.thal == "normal" else "Không",
            "thal_reversable defect": (
                "Có" if patient.thal == "reversable defect" else "Không"
            ),
            "thal_unknown": "Có" if patient.thal is None else "Không",
        }

    def predict(self, patient: PatientInputAdvanced) -> dict:
        X = self.transform(patient)
        return self.predict_from_df(
            X, ADVANCED_FEATURE_LABELS, self._display_values(patient)
        )
