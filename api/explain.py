"""
Giải thích dự đoán cho TỪNG bệnh nhân cụ thể (local explainability) — khác với
feature importance tổng thể của cả model (global, xem src/pipelines/advanced/feature_importance.py).

Dùng phương pháp "ablation" (occlusion): với mỗi feature, thay giá trị của bệnh nhân bằng
giá trị "trung tính" (baseline — vì features đã chuẩn hóa nên baseline = 0 tương đương giá
trị trung bình dân số/nhóm tham chiếu của biến one-hot), đo xác suất dự đoán thay đổi bao
nhiêu. Không cần cài thêm thư viện (chỉ dùng numpy/pandas đã có sẵn), hoạt động với mọi model
scikit-learn có predict_proba, không chỉ riêng Random Forest.

Đây không phải SHAP/Shapley values "chuẩn" về mặt lý thuyết (không đảm bảo tổng các contribution
cộng lại đúng bằng chênh lệch xác suất so với baseline), nhưng đủ tốt để trực quan hóa "yếu tố
nào đang kéo xác suất lên/xuống" cho một bệnh nhân cụ thể.
"""

import pandas as pd


def explain_prediction(model, X_row: pd.DataFrame) -> dict:
    """
    X_row: dataframe 1 dòng, đã qua transform (đúng format model.predict nhận vào).
    Trả về dict {feature_name: contribution}, contribution > 0 nghĩa là giá trị hiện tại
    của bệnh nhân đang làm TĂNG xác suất mắc bệnh so với baseline, < 0 là làm GIẢM.
    """
    baseline = pd.DataFrame([[0.0] * X_row.shape[1]], columns=X_row.columns)

    base_proba = float(model.predict_proba(X_row)[0, 1])

    contributions = {}
    for col in X_row.columns:
        modified = X_row.copy()
        modified[col] = baseline[col].values[0]
        modified_proba = float(model.predict_proba(modified)[0, 1])
        contributions[col] = base_proba - modified_proba

    return contributions


def top_contributors(contributions: dict, top_k: int = 8) -> list[tuple[str, float]]:
    """Sắp xếp theo độ lớn ảnh hưởng (|contribution|) giảm dần, lấy top_k."""
    sorted_items = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return sorted_items[:top_k]
