"""
Test cho Heart Disease Prediction API (2 chế độ: basic, advanced).
Chạy: pytest src/tests/test_api.py -v
"""

import sys
import os

sys.path.append(os.getcwd())

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_BASIC_PATIENT = {
    "age_group": "50-54", "sex": "Male", "height_cm": 170, "weight_kg": 85,
    "high_bp": True, "high_chol": True, "chol_check": True,
    "diabetes": "Không", "stroke": False,
    "smoker": True, "phys_activity": False, "fruits": False, "veggies": True,
    "heavy_alcohol": False, "gen_health": "Trung bình",
    "mental_health_days": 5, "phys_health_days": 10, "diff_walk": True,
    "any_healthcare": True, "no_doc_because_cost": False,
    "education_level": "Tốt nghiệp THPT", "income_level": "25-35k",
}

VALID_ADVANCED_PATIENT = {
    "age": 58, "sex": "Male", "cp": "asymptomatic", "chol": 245,
    "fbs": False, "thalch": 140, "exang": True, "oldpeak": 2.1,
    "ca": 1, "thal": "reversable defect",
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_predict_basic_valid_input():
    response = client.post("/predict/basic", json=VALID_BASIC_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in ["Low", "Medium", "High"]
    assert len(body["feature_contributions"]) > 0
    for fc in body["feature_contributions"]:
        assert "feature" in fc and "value" in fc and "contribution" in fc


def test_predict_advanced_valid_input():
    response = client.post("/predict/advanced", json=VALID_ADVANCED_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert len(body["feature_contributions"]) > 0


def test_predict_advanced_missing_optional_fields():
    patient = dict(VALID_ADVANCED_PATIENT)
    del patient["ca"]
    del patient["thal"]
    response = client.post("/predict/advanced", json=patient)
    assert response.status_code == 200


def test_predict_basic_invalid_age_group_rejected():
    patient = dict(VALID_BASIC_PATIENT)
    patient["age_group"] = "999"
    response = client.post("/predict/basic", json=patient)
    assert response.status_code == 422


def test_predict_missing_required_field():
    patient = dict(VALID_BASIC_PATIENT)
    del patient["sex"]
    response = client.post("/predict/basic", json=patient)
    assert response.status_code == 422


def test_chat_ollama_unavailable_returns_503_or_success():
    """
    Nếu Ollama chưa cài/chưa chạy, endpoint phải trả 503 rõ ràng (không phải lỗi 500
    khó hiểu). Nếu Ollama đã sẵn sàng, phải trả reply hợp lệ.
    """
    predict_response = client.post("/predict/basic", json=VALID_BASIC_PATIENT)
    prediction_result = predict_response.json()

    chat_payload = {
        "mode": "basic",
        "patient": VALID_BASIC_PATIENT,
        "prediction_result": prediction_result,
        "message": "Tôi nên ăn uống thế nào?",
        "history": [],
    }
    response = client.post("/chat", json=chat_payload)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert len(response.json()["reply"]) > 0


def test_chat_stream_returns_text():
    """
    /chat/stream luôn trả 200 (kể cả khi Ollama lỗi — lỗi được nhúng vào nội dung
    text trả về, vì streaming response không đổi status code giữa chừng được).
    """
    predict_response = client.post("/predict/basic", json=VALID_BASIC_PATIENT)
    prediction_result = predict_response.json()

    chat_payload = {
        "mode": "basic",
        "patient": VALID_BASIC_PATIENT,
        "prediction_result": prediction_result,
        "message": "Tôi nên ăn uống thế nào?",
        "history": [],
    }
    with client.stream("POST", "/chat/stream", json=chat_payload) as response:
        assert response.status_code == 200
        full_text = "".join(response.iter_text())
        assert len(full_text) > 0
