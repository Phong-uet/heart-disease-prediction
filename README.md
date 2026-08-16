# Heart Disease Prediction

Dự án Machine Learning end-to-end dự đoán khả năng mắc bệnh tim, với **2 chế độ**:

| Chế độ | Dataset | Đối tượng dùng | Input |
|---|---|---|---|
| **Basic** (mặc định) | BRFSS Heart Disease Health Indicators (CDC, 253k dòng) | Người dùng phổ thông, tự sàng lọc tại nhà | Tuổi, cân nặng/chiều cao, thói quen sống, bệnh nền đã biết... — **không cần xét nghiệm** |
| **Advanced** | UCI Heart Disease (920 dòng) | Người đã có kết quả khám bệnh | Cholesterol máu, ECG, xạ hình thallium... |

Cả 2 chế độ dùng chung 1 kiến trúc pipeline, chỉ khác bộ dữ liệu/feature.

## Cấu trúc dự án

```
heart-disease-prediction/
├── config/
│   ├── basic/{local,prod}.yaml
│   └── advanced/{local,prod}.yaml
├── data/
│   ├── basic/{01-raw,02-preprocessed,03-features,04-predictions}/
│   └── advanced/{01-raw,02-preprocessed,03-features,04-predictions}/
├── entrypoint/
│   ├── train.py          # --mode basic|advanced
│   └── inference.py      # --mode basic|advanced
├── notebooks/
│   ├── EDA_basic.ipynb / EDA_advanced.ipynb
│   └── Baseline_advanced.ipynb
├── src/pipelines/
│   ├── utils.py                  # dùng chung
│   ├── basic/                    # preprocess, feature_eng, training, inference
│   └── advanced/                 # + feature_importance, feature_selection_experiment
├── src/tests/
├── models/{basic,advanced}/      # best_model.pkl, scaler.pkl, feature_columns.json
├── reports/{basic,advanced}/
├── api/                          # FastAPI: /predict/basic, /predict/advanced
├── app/                          # Streamlit demo (chọn mode ở giao diện)
├── Dockerfile / Dockerfile.api / Dockerfile.app / docker-compose.yml
└── requirements-*.txt
```

## Cài đặt

```bash
make install-dev
```

## Chuẩn bị dữ liệu

- **Basic**: tải [Heart Disease Health Indicators (BRFSS)](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset) → đặt vào `data/basic/01-raw/heart.csv`
- **Advanced**: tải [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) → đặt vào `data/advanced/01-raw/heart.csv`

## Huấn luyện

```bash
make train-basic       # hoặc: python entrypoint/train.py --mode basic
make train-advanced    # hoặc: python entrypoint/train.py --mode advanced
```

## Chạy API

```bash
pip install -r requirements-prod.txt
make api
# hoặc: uvicorn api.main:app --reload --port 8000
```

Mở **http://127.0.0.1:8000/docs**. Có 2 endpoint: `POST /predict/basic` và `POST /predict/advanced`.

Ví dụ gọi `/predict/basic`:
```bash
curl -X POST "http://127.0.0.1:8000/predict/basic" \
  -H "Content-Type: application/json" \
  -d '{
    "age_group": "50-54", "sex": "Male", "height_cm": 170, "weight_kg": 85,
    "high_bp": true, "high_chol": true, "chol_check": true,
    "diabetes": "Không", "stroke": false,
    "smoker": true, "phys_activity": false, "fruits": false, "veggies": true,
    "heavy_alcohol": false, "gen_health": "Trung bình",
    "mental_health_days": 5, "phys_health_days": 10, "diff_walk": true,
    "any_healthcare": true, "no_doc_because_cost": false,
    "education_level": "Tốt nghiệp THPT", "income_level": "25-35k"
  }'
```

Response mẫu:
```json
{ "prediction": 1, "probability": 0.7733, "risk_level": "High" }
```

## Chạy Demo UI (Streamlit)

```bash
# Terminal 1
make api
# Terminal 2
pip install -r app/requirements.txt
make app
```

Mở **http://localhost:8501** — chọn chế độ ngay trên giao diện (radio button đầu trang).

## Chạy toàn bộ bằng Docker Compose

```bash
docker-compose up --build heart-disease-api heart-disease-app
```

Mở `http://localhost:8501`.

## Chạy test

```bash
make test
```

## Chatbot tư vấn (2 backend: Ollama local hoặc Groq cloud)

Sau khi có kết quả dự đoán, người dùng có thể **hỏi thêm chatbot** — chatbot trả lời
dựa trên chính hồ sơ và kết quả vừa nhận được (không phải trả lời chung chung).

Chatbot hỗ trợ **2 backend**, chọn bằng biến môi trường `CHATBOT_BACKEND`:

| Backend | Khi nào dùng | Chi phí | Cần gì |
|---|---|---|---|
| `ollama` (mặc định) | Chạy **local** trên máy bạn | Miễn phí 100%, mãi mãi | Cài Ollama (xem bên dưới) |
| `groq` | **Deploy công khai** (server không cài được Ollama) | Miễn phí (14,400 request/ngày, không cần thẻ) | API key tại console.groq.com |

### Dùng Ollama (local)

1. Tải Ollama tại **https://ollama.com/download**, cài như phần mềm bình thường — tự chạy nền
2. Tải model: `ollama pull llama3.1:8b` (~4.7GB, máy yếu hơn dùng `llama3.2:3b`)
3. Không cần cấu hình gì thêm (`CHATBOT_BACKEND=ollama` là mặc định)

### Dùng Groq (khi deploy công khai)

1. Lấy API key miễn phí tại **console.groq.com/keys** (chỉ cần email, không cần thẻ)
2. Đặt biến môi trường (trong `.env` cho local, hoặc mục Environment Variables trên
   Render/nền tảng deploy):
   ```
   CHATBOT_BACKEND=groq
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   ```
3. Model mặc định `llama-3.1-8b-instant` — đổi qua `GROQ_MODEL` nếu muốn model khác
   (xem danh sách tại console.groq.com)

**Lưu ý:** RAG (tra cứu tài liệu) chỉ hoạt động với backend `ollama` vì cần Ollama tính
embedding. Khi dùng `groq`, chatbot vẫn hoạt động bình thường, chỉ không tra cứu tài liệu
(tự động bỏ qua, không lỗi).

Nếu chatbot chưa sẵn sàng (thiếu key/Ollama chưa chạy), giao diện Streamlit **tự ẩn khung
chat** thay vì hiện ra rồi báo lỗi cho người dùng — kiểm tra qua `GET /health`, xem trường
`chatbot_available`.

### Quy tắc an toàn đã thiết lập sẵn cho chatbot

Chatbot (`api/chatbot.py`) được cấu hình với các ràng buộc trong system prompt (áp dụng cho
cả 2 backend):
- **Không chẩn đoán** — chỉ giải thích kết quả model, không khẳng định chắc chắn có/không bị bệnh
- **Không tư vấn thuốc/liều lượng**
- Nếu người dùng mô tả **triệu chứng cấp tính** → khuyên gọi cấp cứu ngay
- Nếu mức độ nguy cơ **Cao** → luôn ưu tiên khuyên đến cơ sở y tế
- Lời khuyên lối sống chỉ ở mức khuyến cáo y tế công cộng chung (WHO/AHA)

> Lưu ý: model nhỏ (3B-8B, dù local hay qua Groq) có thể tuân thủ quy tắc kém nhất quán hơn
> model thương mại lớn — nên xem đây là công cụ demo/học tập, luôn tự kiểm tra output.

## RAG — chatbot tra cứu tài liệu (tùy chọn, mặc định TẮT)

Chatbot có thể tra cứu một bộ tài liệu kiến thức tim mạch (`rag/knowledge_base/*.md` — dinh
dưỡng, vận động, hút thuốc/rượu bia, yếu tố nguy cơ, triệu chứng cảnh báo) trước khi trả lời,
giúp câu trả lời có căn cứ cụ thể hơn thay vì chỉ dựa vào "trí nhớ" của model nhỏ.

**Cơ chế "chỉ dùng khi cần"**: mỗi câu hỏi đều được so khớp ngữ nghĩa với tài liệu trước; nếu
độ liên quan thấp (câu chào hỏi, câu hỏi về chính kết quả dự đoán...) → bỏ qua tài liệu, trả
lời nhanh như bình thường. Chỉ khi câu hỏi thực sự liên quan tới kiến thức y tế (VD: "tôi nên
ăn bao nhiêu muối?") → chèn tài liệu liên quan vào, lúc đó phản hồi sẽ **chậm hơn một chút**
(thêm bước tính embedding + prompt dài hơn) nhưng có căn cứ tốt hơn.

#### Bật RAG (tùy chọn)

1. Tải model embedding (nhẹ, ~274MB, khác với model chat):
   ```bash
   ollama pull nomic-embed-text
   ```
2. Build index (chạy 1 lần, hoặc lại mỗi khi bạn sửa/thêm file trong `rag/knowledge_base/`):
   ```bash
   python rag/build_index.py
   ```
   Sinh ra `rag/index.npy` và `rag/index_meta.json`.
3. Không cần làm gì thêm — `api/main.py` tự phát hiện index đã tồn tại và bật RAG. Kiểm tra
   qua `GET /health`, sẽ thấy `"rag_enabled": true`.

**Chưa build index thì sao?** Chatbot hoạt động **y hệt như khi chưa có RAG** — không lỗi,
không bắt buộc, đây là tính năng hoàn toàn tùy chọn (opt-in).

#### Thêm/sửa tài liệu

Thêm file `.md` mới vào `rag/knowledge_base/`, dùng heading `## Tên phần` để chia nhỏ nội
dung thành từng đoạn dễ tra cứu (mỗi `## heading` sẽ thành 1 chunk riêng). Chạy lại
`python rag/build_index.py` sau khi sửa.

#### Chỉnh độ nhạy

Nếu RAG kích hoạt quá thường xuyên (kể cả câu hỏi không liên quan) hoặc quá hiếm khi kích
hoạt (bỏ sót câu hỏi đáng lẽ nên tra cứu), chỉnh `RAG_SIMILARITY_THRESHOLD` trong `.env`
(mặc định `0.5`, giá trị từ 0 đến 1, cao hơn = khó kích hoạt hơn).

### Tăng tốc chatbot

- **Streaming (đã bật mặc định)** — Streamlit gọi `/chat/stream`, hiển thị câu trả lời
  "gõ dần" thay vì đợi xong cả câu, cảm giác nhanh hơn nhiều dù tổng thời gian không đổi.
- **Giới hạn độ dài trả lời** — mặc định tối đa 400 token (`OLLAMA_NUM_PREDICT` trong `.env`),
  câu trả lời ngắn hơn = nhanh hơn tuyến tính. Tăng lên nếu muốn trả lời chi tiết hơn.
- **Đổi model nhẹ hơn** — `ollama pull llama3.2:3b` rồi sửa `OLLAMA_MODEL=llama3.2:3b`
  trong `.env`, nhanh hơn ~2-3 lần so với `llama3.1:8b`, đánh đổi chất lượng trả lời.
- **Dùng GPU thay vì CPU** — nếu máy có GPU NVIDIA hoạt động bình thường (không lỗi driver),
  Ollama tự động dùng GPU và nhanh hơn CPU 5-10 lần, không cần cấu hình gì thêm.
- Lịch sử hội thoại tự động chỉ giữ **12 tin nhắn gần nhất** gửi lên model — hội thoại dài
  không làm chậm dần theo thời gian.

### Quy tắc an toàn đã thiết lập sẵn cho chatbot

Chatbot (`api/chatbot.py`) được cấu hình với các ràng buộc trong system prompt:
- **Không chẩn đoán** — chỉ giải thích kết quả model, không khẳng định chắc chắn có/không bị bệnh
- **Không tư vấn thuốc/liều lượng**
- Nếu người dùng mô tả **triệu chứng cấp tính** (đau ngực dữ dội, khó thở nặng...) → khuyên gọi cấp cứu ngay
- Nếu mức độ nguy cơ **Cao** → luôn ưu tiên khuyên đến cơ sở y tế
- Lời khuyên lối sống chỉ ở mức khuyến cáo y tế công cộng chung (WHO/AHA), không cá nhân hóa như bác sĩ điều trị

> Lưu ý: model local (đặc biệt bản nhỏ như 3B-8B) có thể tuân thủ quy tắc kém nhất quán hơn
> model thương mại lớn — nên xem đây là công cụ demo/học tập, luôn tự kiểm tra output trước
> khi dùng cho mục đích thực tế.

## Giải thích dự đoán (feature contribution)

Sau mỗi lần dự đoán, giao diện hiển thị thêm biểu đồ **"Yếu tố ảnh hưởng đến kết quả dự đoán
của bạn"** — khác với feature importance tổng thể (đo trên toàn bộ model), biểu đồ này giải
thích **vì sao bệnh nhân cụ thể đó** ra kết quả đó.

**Cách hoạt động:** dùng phương pháp *ablation* (`api/explain.py`) — lần lượt thay từng giá
trị của bệnh nhân bằng giá trị "trung tính" (baseline), đo xác suất dự đoán thay đổi bao
nhiêu. Thanh màu đỏ = yếu tố đó đang làm **tăng** xác suất so với baseline, xanh = làm
**giảm**. Không cần cài thêm thư viện (SHAP, v.v.) — chỉ dùng model đã có sẵn.

Lưu ý: đây không phải Shapley values chuẩn về mặt lý thuyết (tổng các contribution không nhất
thiết bằng đúng chênh lệch xác suất so với baseline), nhưng đủ trực quan để hiểu hướng ảnh
hưởng của từng yếu tố.

## Kết quả model

| Chế độ | Test Accuracy | Test ROC-AUC | Ghi chú |
|---|---|---|---|
| Basic | ~75% | ~0.85 | Dataset mất cân bằng nặng (90.6%/9.4%), ưu tiên Recall (đạt ~80%) để không bỏ sót ca nguy cơ |
| Advanced | ~83% | ~0.92 | Dataset nhỏ, cân bằng hơn (55%/45%) |

## Lưu ý quan trọng

⚠️ Đây là dự án học tập/tham khảo, **không thay thế chẩn đoán y khoa**. Chế độ Basic chỉ mang tính sàng lọc sơ bộ dựa trên yếu tố nguy cơ lối sống (theo khảo sát CDC BRFSS) — nếu kết quả cho thấy nguy cơ cao, nên đến cơ sở y tế để được xét nghiệm và tư vấn chính xác.
