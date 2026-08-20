"""
Pydantic schemas cho request/response của API dự đoán bệnh tim.
Có 2 bộ schema tương ứng 2 chế độ: basic (tự đánh giá) và advanced (có xét nghiệm).
"""

from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import Literal


# ============================================================================
# BASIC MODE — chỉ dùng thông tin người dùng tự biết/tự đo tại nhà
# ============================================================================
class PatientInputBasic(BaseModel):
    age_group: Literal[
        "18-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80+",
    ] = Field(..., description="Nhóm tuổi")
    sex: Literal["Male", "Female"] = Field(..., description="Giới tính")
    height_cm: float = Field(..., ge=100, le=250, description="Chiều cao (cm)")
    weight_kg: float = Field(..., ge=20, le=300, description="Cân nặng (kg)")

    high_bp: bool = Field(
        ..., description="Đã từng được bác sĩ chẩn đoán cao huyết áp?"
    )
    high_chol: bool = Field(
        ..., description="Đã từng được bác sĩ chẩn đoán cholesterol cao?"
    )
    chol_check: bool = Field(
        True, description="Có kiểm tra cholesterol trong 5 năm qua?"
    )
    diabetes: Literal["Không", "Tiền tiểu đường", "Có"] = Field(
        "Không", description="Tình trạng tiểu đường"
    )
    stroke: bool = Field(False, description="Đã từng bị đột quỵ?")

    smoker: bool = Field(..., description="Đã hút ít nhất 100 điếu thuốc trong đời?")
    phys_activity: bool = Field(
        ..., description="Có tập thể dục/vận động trong 30 ngày qua?"
    )
    fruits: bool = Field(..., description="Ăn trái cây ít nhất 1 lần/ngày?")
    veggies: bool = Field(..., description="Ăn rau củ ít nhất 1 lần/ngày?")
    heavy_alcohol: bool = Field(
        False, description="Uống rượu bia nhiều (nam >14 ly/tuần, nữ >7 ly/tuần)?"
    )

    gen_health: Literal["Rất tốt", "Tốt", "Khá", "Trung bình", "Kém"] = Field(
        ..., description="Tự đánh giá sức khỏe tổng quát"
    )
    mental_health_days: int = Field(
        0,
        ge=0,
        le=30,
        description="Số ngày cảm thấy không ổn về tinh thần trong 30 ngày qua",
    )
    phys_health_days: int = Field(
        0,
        ge=0,
        le=30,
        description="Số ngày cảm thấy không khỏe về thể chất trong 30 ngày qua",
    )
    diff_walk: bool = Field(..., description="Khó khăn khi đi bộ hoặc leo cầu thang?")

    any_healthcare: bool = Field(
        True, description="Có bảo hiểm y tế/tiếp cận dịch vụ y tế?"
    )
    no_doc_because_cost: bool = Field(
        False, description="Từng không đi khám vì lý do chi phí?"
    )
    education_level: Literal[
        "Chưa học/Tiểu học",
        "Trung học cơ sở",
        "Trung học phổ thông (chưa tốt nghiệp)",
        "Tốt nghiệp THPT",
        "Cao đẳng/Đại học (chưa tốt nghiệp)",
        "Tốt nghiệp Đại học",
    ] = Field(..., description="Trình độ học vấn")
    income_level: Literal[
        "<10k",
        "10-15k",
        "15-20k",
        "20-25k",
        "25-35k",
        "35-50k",
        "50-75k",
        ">75k",
    ] = Field(..., description="Mức thu nhập hộ gia đình hàng năm (USD)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age_group": "50-54",
                "sex": "Male",
                "height_cm": 170,
                "weight_kg": 85,
                "high_bp": True,
                "high_chol": True,
                "chol_check": True,
                "diabetes": "Không",
                "stroke": False,
                "smoker": True,
                "phys_activity": False,
                "fruits": False,
                "veggies": True,
                "heavy_alcohol": False,
                "gen_health": "Trung bình",
                "mental_health_days": 5,
                "phys_health_days": 10,
                "diff_walk": True,
                "any_healthcare": True,
                "no_doc_because_cost": False,
                "education_level": "Tốt nghiệp THPT",
                "income_level": "25-35k",
            }
        }
    }


# ============================================================================
# ADVANCED MODE — dành cho người đã có kết quả xét nghiệm/ECG
# ============================================================================
class PatientInputAdvanced(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Tuổi bệnh nhân")
    sex: Literal["Male", "Female"] = Field(..., description="Giới tính")
    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"] = (
        Field(..., description="Loại đau ngực")
    )
    chol: float = Field(..., ge=0, description="Cholesterol huyết thanh (mg/dl)")
    fbs: bool = Field(..., description="Đường huyết lúc đói > 120 mg/dl?")
    thalch: float = Field(..., ge=0, description="Nhịp tim tối đa đạt được")
    exang: bool = Field(..., description="Đau thắt ngực khi vận động gắng sức?")
    oldpeak: float = Field(..., description="ST depression khi vận động so với nghỉ")
    ca: Optional[int] = Field(
        None,
        ge=0,
        le=3,
        description="Số mạch máu chính bị hẹp (0-3), để trống nếu không có kết quả chụp",
    )
    thal: Optional[Literal["normal", "fixed defect", "reversable defect"]] = Field(
        None, description="Kết quả xạ hình thallium, để trống nếu không có kết quả"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 58,
                "sex": "Male",
                "cp": "asymptomatic",
                "chol": 245,
                "fbs": False,
                "thalch": 140,
                "exang": True,
                "oldpeak": 2.1,
                "ca": 1,
                "thal": "reversable defect",
            }
        }
    }


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Tên yếu tố (nhãn tiếng Việt)")
    value: str = Field(..., description="Giá trị của bệnh nhân cho yếu tố này")
    contribution: float = Field(
        ...,
        description="Mức ảnh hưởng tới xác suất dự đoán: dương = tăng nguy cơ, âm = giảm nguy cơ",
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(
        ..., description="0 = không mắc bệnh tim, 1 = có mắc bệnh tim"
    )
    probability: float = Field(..., description="Xác suất mắc bệnh tim (0-1)")
    risk_level: str = Field(..., description="Mức độ nguy cơ: Low / Medium / High")
    feature_contributions: list[FeatureContribution] = Field(
        default_factory=list,
        description="Top yếu tố ảnh hưởng nhiều nhất tới kết quả dự đoán của bệnh nhân này",
    )


class HealthResponse(BaseModel):
    status: str
    basic_model_loaded: bool
    advanced_model_loaded: bool
    chatbot_available: bool
    rag_enabled: bool


class DayCount(BaseModel):
    date: str
    count: int


class StatsSummary(BaseModel):
    total_predictions: int
    by_mode: dict[str, int]
    by_risk_level: dict[str, int]
    avg_probability: float
    by_day: list[DayCount]


# ============================================================================
# CHATBOT
# ============================================================================
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ..., description="Vai trò trong hội thoại"
    )
    content: str = Field(..., description="Nội dung tin nhắn")


class ChatRequest(BaseModel):
    mode: Literal["basic", "advanced"] = Field(
        ..., description="Chế độ đã dùng để dự đoán"
    )
    patient: dict = Field(
        ..., description="Dữ liệu bệnh nhân đã gửi lúc /predict (nguyên payload)"
    )
    prediction_result: PredictionResponse = Field(
        ..., description="Kết quả trả về từ /predict"
    )
    message: str = Field(
        ..., min_length=1, description="Câu hỏi/tin nhắn mới của người dùng"
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Lịch sử hội thoại trước đó (không gồm message mới nhất)",
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Câu trả lời của chatbot")
