"""
FastAPI app phục vụ dự đoán bệnh tim — hỗ trợ 2 chế độ:
  - /predict/basic    : chỉ cần thông tin tự đánh giá tại nhà
  - /predict/advanced : cần kết quả xét nghiệm/ECG

Chạy local:
    uvicorn api.main:app --reload --port 8000

Sau đó mở http://127.0.0.1:8000/docs để test qua giao diện Swagger.
"""

import sys
import os

sys.path.append(os.getcwd())

from dotenv import load_dotenv

# Tự động đọc file .env ở thư mục gốc project (nếu có) và nạp vào os.environ.
# Nhờ vậy không cần tự "export"/set biến môi trường thủ công trong terminal nữa —
# chỉ cần file .env tồn tại là đủ (xem README phần Chatbot).
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import (
    PatientInputBasic,
    PatientInputAdvanced,
    PredictionResponse,
    HealthResponse,
    ChatRequest,
    ChatResponse,
)
from api.inference import BasicPredictor, AdvancedPredictor
from api.chatbot import HealthChatbot, OllamaConnectionError
from src.pipelines.utils import load_config, get_logger

BASIC_CONFIG_PATH = os.environ.get("BASIC_CONFIG_PATH", "config/basic/local.yaml")
ADVANCED_CONFIG_PATH = os.environ.get("ADVANCED_CONFIG_PATH", "config/advanced/local.yaml")

basic_config = load_config(BASIC_CONFIG_PATH)
advanced_config = load_config(ADVANCED_CONFIG_PATH)
logger = get_logger("api", basic_config["logging"]["level"])

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "API dự đoán khả năng mắc bệnh tim. "
        "Có 2 chế độ: basic (tự đánh giá tại nhà) và advanced (cần kết quả xét nghiệm)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

basic_predictor: BasicPredictor | None = None
advanced_predictor: AdvancedPredictor | None = None
chatbot: HealthChatbot | None = None  # luôn được gán ở startup, không raise lỗi


@app.on_event("startup")
def load_models():
    global basic_predictor, advanced_predictor, chatbot

    try:
        basic_predictor = BasicPredictor(
            model_path=basic_config["model"]["save_path"],
            scaler_path=basic_config["model"]["scaler_path"],
            feature_columns_path=basic_config["model"]["feature_columns_path"],
        )
        logger.info("Basic model đã load thành công.")
    except FileNotFoundError as e:
        logger.error(f"Không load được basic model: {e}")
        basic_predictor = None

    try:
        advanced_predictor = AdvancedPredictor(
            model_path=advanced_config["model"]["save_path"],
            scaler_path=advanced_config["model"]["scaler_path"],
            feature_columns_path=advanced_config["model"]["feature_columns_path"],
        )
        logger.info("Advanced model đã load thành công.")
    except FileNotFoundError as e:
        logger.error(f"Không load được advanced model: {e}")
        advanced_predictor = None

    chatbot = HealthChatbot()
    if chatbot.is_available():
        logger.info(f"Chatbot (Ollama, model={chatbot.model}) đã sẵn sàng.")
    else:
        logger.warning(
            f"Ollama chưa chạy hoặc chưa cài tại {chatbot.base_url}. "
            f"Endpoint /chat sẽ báo lỗi rõ ràng cho tới khi Ollama sẵn sàng "
            f"(không cần khởi động lại API sau khi bật Ollama)."
        )

    if chatbot.rag_ready():
        logger.info(f"RAG đã sẵn sàng ({len(chatbot.retriever._meta)} chunks đã index).")
    else:
        logger.info(
            "RAG chưa được bật (chưa chạy `python rag/build_index.py`). "
            "Chatbot vẫn hoạt động bình thường, chỉ không tra cứu tài liệu."
        )


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        basic_model_loaded=basic_predictor is not None,
        advanced_model_loaded=advanced_predictor is not None,
        chatbot_available=chatbot.is_available() if chatbot else False,
        rag_enabled=chatbot.rag_ready() if chatbot else False,
    )


@app.post("/predict/basic", response_model=PredictionResponse)
def predict_basic(patient: PatientInputBasic):
    if basic_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Basic model chưa được load. Kiểm tra lại models/basic/best_model.pkl.",
        )
    try:
        result = basic_predictor.predict(patient)
        logger.info(f"[basic] input={patient.model_dump()} -> result={result}")
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Lỗi khi predict (basic): {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")


@app.post("/predict/advanced", response_model=PredictionResponse)
def predict_advanced(patient: PatientInputAdvanced):
    if advanced_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Advanced model chưa được load. Kiểm tra lại models/advanced/best_model.pkl.",
        )
    try:
        result = advanced_predictor.predict(patient)
        logger.info(f"[advanced] input={patient.model_dump()} -> result={result}")
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Lỗi khi predict (advanced): {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        history = [{"role": m.role, "content": m.content} for m in request.history]
        reply = chatbot.chat(
            mode=request.mode,
            patient=request.patient,
            prediction_result=request.prediction_result.model_dump(),
            message=request.message,
            history=history,
        )
        logger.info(f"[chat] mode={request.mode} message={request.message!r}")
        return ChatResponse(reply=reply)
    except OllamaConnectionError as e:
        logger.warning(f"Ollama không sẵn sàng: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Lỗi khi chat: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi gọi chatbot: {str(e)}")


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Giống /chat nhưng trả về từng mẩu text ngay khi sinh ra (streaming) thay vì
    đợi xong cả câu — giúp UI hiển thị kiểu "gõ dần", cảm giác nhanh hơn nhiều.
    Response là text thuần, mỗi chunk là 1 đoạn nối tiếp nhau (không phải JSON).
    """
    history = [{"role": m.role, "content": m.content} for m in request.history]

    def generate():
        try:
            for chunk in chatbot.chat_stream(
                mode=request.mode,
                patient=request.patient,
                prediction_result=request.prediction_result.model_dump(),
                message=request.message,
                history=history,
            ):
                yield chunk
        except OllamaConnectionError as e:
            logger.warning(f"Ollama không sẵn sàng (stream): {e}")
            yield f"\n\n⚠️ {str(e)}"
        except Exception as e:
            logger.error(f"Lỗi khi chat (stream): {e}")
            yield f"\n\n⚠️ Lỗi khi gọi chatbot: {str(e)}"

    logger.info(f"[chat/stream] mode={request.mode} message={request.message!r}")
    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API đang chạy.",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/predict/basic", "/predict/advanced", "/chat", "/chat/stream"],
    }
