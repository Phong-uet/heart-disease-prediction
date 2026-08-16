"""
Chatbot tư vấn sức khỏe tim mạch, dùng Ollama (model AI chạy local, miễn phí,
không cần API key/internet sau khi đã tải model).

Yêu cầu: đã cài Ollama (https://ollama.com) và chạy `ollama pull <model>` trước
(xem README phần Chatbot).
"""

import json
import os
import sys

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.retriever import KnowledgeRetriever

# Địa chỉ Ollama server local (mặc định khi cài Ollama trên máy)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Đổi model tại đây hoặc qua biến môi trường OLLAMA_MODEL nếu đã pull model khác
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
REQUEST_TIMEOUT = 120  # Ollama chạy CPU-only trên máy cá nhân có thể chậm, đặc biệt lần đầu (phải nạp model vào RAM)
# Giới hạn số token sinh ra tối đa — câu trả lời ngắn hơn = nhanh hơn tuyến tính.
# Đổi qua biến môi trường OLLAMA_NUM_PREDICT nếu muốn câu trả lời dài/ngắn hơn.
NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "400"))

SYSTEM_PROMPT_TEMPLATE = """Bạn là trợ lý tư vấn sức khỏe tim mạch cho một ứng dụng sàng lọc nguy cơ bệnh tim.
Người dùng vừa nhận được kết quả dự đoán từ model Machine Learning dưới đây. Nhiệm vụ của bạn
là giải thích kết quả và đưa ra lời khuyên lối sống chung chung, DỰA TRÊN chính các yếu tố
nguy cơ mà người dùng đã cung cấp.

=== THÔNG TIN BỆNH NHÂN ({mode_label}) ===
{patient_summary}

=== KẾT QUẢ DỰ ĐOÁN TỪ MODEL ===
- Dự đoán: {prediction_label}
- Xác suất: {probability:.1%}
- Mức độ nguy cơ: {risk_level_label}
{retrieved_context_block}
=== QUY TẮC BẮT BUỘC ===
1. KHÔNG được chẩn đoán bệnh, KHÔNG khẳng định người dùng "chắc chắn bị/không bị" bệnh tim.
   Đây chỉ là kết quả sàng lọc thống kê từ model ML, không phải chẩn đoán y khoa.
2. KHÔNG tư vấn liều lượng thuốc, KHÔNG kê đơn, KHÔNG thay thế bác sĩ.
3. Nếu người dùng mô tả triệu chứng cấp tính (đau ngực dữ dội, khó thở nặng, ngất xỉu,
   tê yếu tay chân...), LẬP TỨC khuyên họ gọi cấp cứu (115 tại Việt Nam) hoặc đến bệnh viện
   ngay, không tiếp tục tư vấn thông thường.
4. Nếu mức độ nguy cơ là "Cao", luôn khuyến khích người dùng đến cơ sở y tế để được khám
   và xét nghiệm chính xác hơn — coi đây là ưu tiên hàng đầu trong câu trả lời.
5. Lời khuyên lối sống (ăn uống, vận động, bỏ thuốc...) phải chung chung, dựa trên khuyến
   cáo y tế công cộng phổ biến (WHO, AHA), KHÔNG cá nhân hóa quá mức như bác sĩ điều trị.
6. Trả lời bằng tiếng Việt, giọng điệu ấm áp, dễ hiểu, không dùng thuật ngữ y khoa phức tạp
   mà không giải thích.
7. Nếu câu hỏi nằm ngoài phạm vi sức khỏe tim mạch, lịch sự từ chối và hướng người dùng quay
   lại chủ đề.
8. Nếu có mục "TÀI LIỆU THAM KHẢO" ở trên, ưu tiên dùng thông tin trong đó để trả lời chính
   xác hơn thay vì chỉ dựa vào kiến thức có sẵn của bạn — nhưng vẫn diễn đạt lại tự nhiên bằng
   lời của bạn, không copy nguyên văn.
"""


def _format_patient_summary_basic(patient: dict) -> str:
    diabetes_label = patient.get("diabetes", "Không")
    lines = [
        f"- Nhóm tuổi: {patient.get('age_group')}, Giới tính: {'Nam' if patient.get('sex') == 'Male' else 'Nữ'}",
        f"- Chiều cao/Cân nặng: {patient.get('height_cm')}cm / {patient.get('weight_kg')}kg",
        f"- Cao huyết áp (đã chẩn đoán): {'Có' if patient.get('high_bp') else 'Không'}",
        f"- Cholesterol cao (đã chẩn đoán): {'Có' if patient.get('high_chol') else 'Không'}",
        f"- Tiểu đường: {diabetes_label}",
        f"- Từng đột quỵ: {'Có' if patient.get('stroke') else 'Không'}",
        f"- Hút thuốc: {'Có' if patient.get('smoker') else 'Không'}",
        f"- Vận động thể chất thường xuyên: {'Có' if patient.get('phys_activity') else 'Không'}",
        f"- Ăn trái cây/rau củ hàng ngày: {'Có' if patient.get('fruits') else 'Không'}/{'Có' if patient.get('veggies') else 'Không'}",
        f"- Tự đánh giá sức khỏe tổng quát: {patient.get('gen_health')}",
        f"- Khó khăn khi đi bộ/leo cầu thang: {'Có' if patient.get('diff_walk') else 'Không'}",
    ]
    return "\n".join(lines)


def _format_patient_summary_advanced(patient: dict) -> str:
    lines = [
        f"- Tuổi: {patient.get('age')}, Giới tính: {'Nam' if patient.get('sex') == 'Male' else 'Nữ'}",
        f"- Loại đau ngực: {patient.get('cp')}",
        f"- Cholesterol huyết thanh: {patient.get('chol')} mg/dl",
        f"- Đường huyết lúc đói > 120mg/dl: {'Có' if patient.get('fbs') else 'Không'}",
        f"- Nhịp tim tối đa đạt được: {patient.get('thalch')}",
        f"- Đau thắt ngực khi vận động: {'Có' if patient.get('exang') else 'Không'}",
        f"- ST depression (oldpeak): {patient.get('oldpeak')}",
    ]
    return "\n".join(lines)


def _format_retrieved_context(retrieved_chunks: list[dict] | None) -> str:
    if not retrieved_chunks:
        return ""
    lines = ["\n=== TÀI LIỆU THAM KHẢO (liên quan tới câu hỏi hiện tại) ==="]
    for chunk in retrieved_chunks:
        lines.append(f"[Nguồn: {chunk['heading']}]\n{chunk['text']}\n")
    return "\n".join(lines)


def build_system_prompt(
    mode: str, patient: dict, prediction_result: dict, retrieved_chunks: list[dict] | None = None
) -> str:
    mode_label = "Sàng lọc nhanh - tự đánh giá" if mode == "basic" else "Nâng cao - có kết quả xét nghiệm"
    patient_summary = (
        _format_patient_summary_basic(patient)
        if mode == "basic"
        else _format_patient_summary_advanced(patient)
    )
    prediction_label = "CÓ nguy cơ mắc bệnh tim" if prediction_result["prediction"] == 1 else "KHÔNG có nguy cơ mắc bệnh tim"
    risk_level_map = {"Low": "Thấp", "Medium": "Trung bình", "High": "Cao"}
    risk_level_label = risk_level_map.get(prediction_result["risk_level"], prediction_result["risk_level"])
    retrieved_context_block = _format_retrieved_context(retrieved_chunks)

    return SYSTEM_PROMPT_TEMPLATE.format(
        mode_label=mode_label,
        patient_summary=patient_summary,
        retrieved_context_block=retrieved_context_block,
        prediction_label=prediction_label,
        probability=prediction_result["probability"],
        risk_level_label=risk_level_label,
    )


class ChatbotConnectionError(Exception):
    """Không kết nối được / không gọi được backend chatbot (Ollama hoặc Groq)."""


# Chọn backend qua biến môi trường CHATBOT_BACKEND=ollama (mặc định, local, miễn phí,
# cần cài Ollama) hoặc =groq (API cloud miễn phí, dùng khi deploy công khai vì server
# không cài được Ollama). Xem README phần Chatbot.
BACKEND = os.environ.get("CHATBOT_BACKEND", "ollama").lower()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


class HealthChatbot:
    def __init__(self, base_url: str = None, model: str = None, backend: str = None):
        self.backend = (backend or BACKEND).lower()
        self.retriever = KnowledgeRetriever()

        if self.backend == "groq":
            self.base_url = GROQ_BASE_URL
            self.model = model or GROQ_MODEL
        else:
            self.backend = "ollama"
            self.base_url = base_url or OLLAMA_BASE_URL
            self.model = model or MODEL_NAME

    def is_available(self) -> bool:
        """Kiểm tra chatbot có sẵn sàng dùng không (dùng lúc startup, không bắt buộc)."""
        if self.backend == "groq":
            # Không gọi thật để tránh tốn quota — chỉ cần đã có API key là coi như sẵn sàng,
            # lỗi thật (key sai, hết quota...) sẽ lộ ra rõ ràng ngay lần chat() đầu tiên.
            return bool(GROQ_API_KEY)
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def rag_ready(self) -> bool:
        return self.retriever.is_ready()

    def _build_messages(
        self, mode: str, patient: dict, prediction_result: dict, message: str, history: list[dict] | None
    ) -> list[dict]:
        # RAG chỉ hoạt động nếu index đã build bằng Ollama embeddings (chỉ khả dụng local) —
        # trên bản deploy dùng backend=groq, retriever tự động không sẵn sàng, bỏ qua RAG,
        # chatbot vẫn hoạt động bình thường bằng kiến thức sẵn có của model.
        retrieved_chunks = self.retriever.retrieve_if_relevant(message) if self.retriever.is_ready() else []

        system_prompt = build_system_prompt(mode, patient, prediction_result, retrieved_chunks)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend((history or [])[-12:])
        messages.append({"role": "user", "content": message})
        return messages

    # ------------------------------------------------------------------
    # Backend: OLLAMA (local, NDJSON streaming, endpoint /api/chat)
    # ------------------------------------------------------------------
    def _ollama_payload(self, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": "30m",
            "options": {"num_predict": NUM_PREDICT},
        }

    def _chat_ollama(self, messages: list[dict]) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=self._ollama_payload(messages, stream=False),
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.ConnectionError:
            raise ChatbotConnectionError(
                f"Không kết nối được Ollama tại {self.base_url}. "
                f"Kiểm tra: đã cài Ollama chưa, Ollama đã chạy chưa (icon khay hệ thống), "
                f"và đã `ollama pull {self.model}` chưa? Xem README phần Chatbot."
            )
        except requests.exceptions.Timeout:
            raise ChatbotConnectionError(
                f"Ollama phản hồi chậm hơn {REQUEST_TIMEOUT}s. Model '{self.model}' có thể đang "
                f"nạp vào bộ nhớ lần đầu, hoặc máy không đủ mạnh — thử lại, hoặc đổi model nhẹ hơn."
            )
        if response.status_code == 404:
            raise ChatbotConnectionError(f"Model '{self.model}' chưa được tải về. Chạy: ollama pull {self.model}")
        if response.status_code != 200:
            try:
                error_detail = response.json().get("error", response.text)
            except ValueError:
                error_detail = response.text
            raise ChatbotConnectionError(f"Ollama trả về lỗi (status {response.status_code}): {error_detail}")
        return response.json()["message"]["content"]

    def _chat_stream_ollama(self, messages: list[dict]):
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=self._ollama_payload(messages, stream=True),
                timeout=REQUEST_TIMEOUT, stream=True,
            )
        except requests.exceptions.ConnectionError:
            raise ChatbotConnectionError(
                f"Không kết nối được Ollama tại {self.base_url}. Kiểm tra Ollama đã chạy chưa."
            )
        except requests.exceptions.Timeout:
            raise ChatbotConnectionError(f"Ollama phản hồi chậm hơn {REQUEST_TIMEOUT}s. Thử lại.")

        if response.status_code != 200:
            try:
                error_detail = response.json().get("error", response.text)
            except ValueError:
                error_detail = response.text
            raise ChatbotConnectionError(f"Ollama trả về lỗi (status {response.status_code}): {error_detail}")

        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
            if chunk.get("done"):
                break

    # ------------------------------------------------------------------
    # Backend: GROQ (cloud, miễn phí, chuẩn OpenAI API, SSE streaming)
    # ------------------------------------------------------------------
    def _groq_headers(self) -> dict:
        return {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    def _groq_payload(self, messages: list[dict], stream: bool) -> dict:
        return {"model": self.model, "messages": messages, "stream": stream, "max_tokens": NUM_PREDICT}

    def _chat_groq(self, messages: list[dict]) -> str:
        if not GROQ_API_KEY:
            raise ChatbotConnectionError(
                "Thiếu GROQ_API_KEY. Lấy key miễn phí tại console.groq.com/keys, "
                "đặt biến môi trường GROQ_API_KEY (xem README phần Chatbot)."
            )
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", headers=self._groq_headers(),
                json=self._groq_payload(messages, stream=False), timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            raise ChatbotConnectionError(f"Không gọi được Groq API: {e}")

        if response.status_code != 200:
            try:
                error_detail = response.json().get("error", {}).get("message", response.text)
            except ValueError:
                error_detail = response.text
            raise ChatbotConnectionError(f"Groq trả về lỗi (status {response.status_code}): {error_detail}")

        return response.json()["choices"][0]["message"]["content"]

    def _chat_stream_groq(self, messages: list[dict]):
        if not GROQ_API_KEY:
            raise ChatbotConnectionError(
                "Thiếu GROQ_API_KEY. Lấy key miễn phí tại console.groq.com/keys, "
                "đặt biến môi trường GROQ_API_KEY (xem README phần Chatbot)."
            )
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", headers=self._groq_headers(),
                json=self._groq_payload(messages, stream=True), timeout=REQUEST_TIMEOUT, stream=True,
            )
        except requests.exceptions.RequestException as e:
            raise ChatbotConnectionError(f"Không gọi được Groq API: {e}")

        if response.status_code != 200:
            try:
                error_detail = response.json().get("error", {}).get("message", response.text)
            except ValueError:
                error_detail = response.text
            raise ChatbotConnectionError(f"Groq trả về lỗi (status {response.status_code}): {error_detail}")

        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: "):]
            if payload.strip() == b"[DONE]":
                break
            chunk = json.loads(payload)
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                yield content

    # ------------------------------------------------------------------
    # API công khai — tự động gọi đúng backend đang cấu hình
    # ------------------------------------------------------------------
    def chat(
        self, mode: str, patient: dict, prediction_result: dict, message: str, history: list[dict] | None = None,
    ) -> str:
        """Gọi chatbot, trả về TOÀN BỘ câu trả lời 1 lần (không streaming)."""
        messages = self._build_messages(mode, patient, prediction_result, message, history)
        if self.backend == "groq":
            return self._chat_groq(messages)
        return self._chat_ollama(messages)

    def chat_stream(
        self, mode: str, patient: dict, prediction_result: dict, message: str, history: list[dict] | None = None,
    ):
        """Giống chat() nhưng trả về generator, yield từng mẩu text ngay khi sinh ra."""
        messages = self._build_messages(mode, patient, prediction_result, message, history)
        if self.backend == "groq":
            yield from self._chat_stream_groq(messages)
        else:
            yield from self._chat_stream_ollama(messages)


# Alias để tương thích ngược với code cũ (api/main.py import OllamaConnectionError)
OllamaConnectionError = ChatbotConnectionError
