"""
Streamlit demo: giao diện nhập thông tin và gọi API dự đoán bệnh tim.
Hỗ trợ 2 chế độ: Basic (tự đánh giá tại nhà) và Advanced (có kết quả xét nghiệm).

Chạy:
    streamlit run app/streamlit_app.py
"""

import os

import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Dự đoán Bệnh Tim", page_icon="❤️", layout="centered")


def _get_default_api_url() -> str:
    """
    Ưu tiên theo thứ tự: Streamlit Secrets (dùng khi deploy lên Streamlit Cloud)
    -> biến môi trường (dùng khi chạy Docker/local) -> mặc định localhost (chạy local thường).
    """
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:
        pass  # Chưa cấu hình secrets.toml (bình thường khi chạy local) -> bỏ qua, dùng fallback
    return os.environ.get("API_URL", "http://127.0.0.1:8000")


DEFAULT_API_URL = _get_default_api_url()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình")
api_url = st.sidebar.text_input("Địa chỉ API", value=DEFAULT_API_URL)

chatbot_available = False

with st.sidebar:
    st.markdown("---")
    try:
        health = requests.get(f"{api_url}/health", timeout=5).json()
        basic_ok = health.get("basic_model_loaded")
        adv_ok = health.get("advanced_model_loaded")
        chatbot_available = health.get("chatbot_available", False)
        st.write(f"{'✅' if basic_ok else '❌'} Model Basic")
        st.write(f"{'✅' if adv_ok else '❌'} Model Advanced")
        st.write(f"{'✅' if chatbot_available else '⚪'} Chatbot tư vấn")
    except requests.exceptions.RequestException:
        st.error("❌ Không kết nối được API.\nChạy: `uvicorn api.main:app --reload`")

st.title("❤️ Dự đoán Nguy cơ Bệnh Tim")

page = st.sidebar.radio("📄 Trang", ["🔍 Dự đoán", "📊 Dashboard thống kê"])


def render_dashboard():
    st.header("📊 Dashboard thống kê sử dụng")
    st.caption(
        "Số liệu tổng hợp về các lượt dự đoán đã thực hiện trên hệ thống này. "
        "Chỉ lưu số liệu tổng hợp (không lưu thông tin cá nhân của bất kỳ ai)."
    )

    try:
        summary = requests.get(f"{api_url}/stats/summary", timeout=5).json()
    except requests.exceptions.RequestException as e:
        st.error(f"Không lấy được số liệu thống kê: {e}")
        return

    total = summary.get("total_predictions", 0)
    if total == 0:
        st.info("Chưa có lượt dự đoán nào được ghi nhận. Hãy thử dự đoán ở trang '🔍 Dự đoán' trước.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số lượt dự đoán", total)
    col2.metric("Xác suất trung bình", f"{summary.get('avg_probability', 0) * 100:.1f}%")
    by_risk = summary.get("by_risk_level", {})
    col3.metric("Số ca nguy cơ Cao", by_risk.get("High", 0))

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Phân bố mức độ nguy cơ")
        risk_labels_vi = {"Low": "Thấp", "Medium": "Trung bình", "High": "Cao"}
        risk_colors = {"Low": "#2ECC71", "Medium": "#F1C40F", "High": "#E74C3C"}
        labels = [risk_labels_vi.get(k, k) for k in by_risk.keys()]
        values = list(by_risk.values())
        colors = [risk_colors.get(k, "#95A5A6") for k in by_risk.keys()]
        fig1 = go.Figure(go.Pie(labels=labels, values=values, marker_colors=colors, hole=0.4))
        fig1.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.subheader("Số lượt theo chế độ")
        by_mode = summary.get("by_mode", {})
        mode_labels_vi = {"basic": "Sàng lọc nhanh", "advanced": "Nâng cao"}
        labels = [mode_labels_vi.get(k, k) for k in by_mode.keys()]
        fig2 = go.Figure(go.Bar(x=labels, y=list(by_mode.values()), marker_color="#3498DB"))
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    by_day = summary.get("by_day", [])
    if len(by_day) > 1:
        st.subheader("Số lượt dự đoán theo ngày")
        dates = [d["date"] for d in by_day]
        counts = [d["count"] for d in by_day]
        fig3 = go.Figure(go.Scatter(x=dates, y=counts, mode="lines+markers", line=dict(color="#E74C3C")))
        fig3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Ngày", yaxis_title="Số lượt")
        st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "⚠️ Trên bản deploy free tier, số liệu có thể bị reset khi hệ thống khởi động lại/deploy lại "
        "(không dùng ổ đĩa lưu trữ lâu dài)."
    )


if page == "📊 Dashboard thống kê":
    render_dashboard()
    st.stop()


def render_contribution_chart(feature_contributions: list):
    """Vẽ biểu đồ ngang: yếu tố nào đang kéo xác suất TĂNG (đỏ) hay GIẢM (xanh)."""
    if not feature_contributions:
        return

    # Đảo thứ tự để yếu tố ảnh hưởng mạnh nhất nằm TRÊN CÙNG khi vẽ ngang
    items = list(reversed(feature_contributions))
    labels = [f"{fc['feature']} ({fc['value']})" for fc in items]
    values = [fc["contribution"] for fc in items]
    colors = ["#E74C3C" if v > 0 else "#2ECC71" for v in values]
    text = [f"{v:+.1%}" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=text,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Yếu tố ảnh hưởng đến kết quả dự đoán của bạn",
        xaxis_title="Mức ảnh hưởng tới xác suất (đỏ = tăng nguy cơ, xanh = giảm nguy cơ)",
        height=90 + len(items) * 45,
        margin=dict(l=10, r=60, t=60, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Biểu đồ so sánh với một 'bệnh nhân trung tính' (giá trị trung bình/nhóm tham chiếu) — "
        "thanh càng dài, yếu tố đó càng ảnh hưởng nhiều tới kết quả của riêng bạn."
    )



mode = st.radio(
    "Chọn chế độ",
    ["🏠 Sàng lọc nhanh (tự đánh giá tại nhà)", "🏥 Nâng cao (đã có kết quả xét nghiệm/ECG)"],
    horizontal=False,
)
is_basic = mode.startswith("🏠")

if is_basic:
    st.info(
        "Chế độ này chỉ dùng thông tin bạn **tự biết hoặc tự đo được tại nhà** "
        "(cân nặng, chiều cao, thói quen sinh hoạt, bệnh nền đã được bác sĩ chẩn đoán trước đó...). "
        "Phù hợp để **sàng lọc nhanh**, không thay thế chẩn đoán y khoa."
    )
else:
    st.info(
        "Chế độ này cần **kết quả xét nghiệm máu (cholesterol) và điện tâm đồ (ECG)**. "
        "Độ chính xác cao hơn, phù hợp khi bạn đã có hồ sơ khám bệnh."
    )

# ---------------------------------------------------------------------------
# FORM: BASIC MODE
# ---------------------------------------------------------------------------
if is_basic:
    with st.form("basic_form"):
        col1, col2 = st.columns(2)

        with col1:
            age_group = st.selectbox(
                "Nhóm tuổi",
                ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                 "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"],
                index=6,
            )
            sex = st.selectbox("Giới tính", ["Male", "Female"], format_func=lambda x: "Nam" if x == "Male" else "Nữ")
            height_cm = st.number_input("Chiều cao (cm)", min_value=100, max_value=250, value=165)
            weight_kg = st.number_input("Cân nặng (kg)", min_value=20, max_value=300, value=65)

            st.markdown("**Bệnh nền đã từng được bác sĩ chẩn đoán**")
            high_bp = st.checkbox("Cao huyết áp")
            high_chol = st.checkbox("Cholesterol cao")
            chol_check = st.checkbox("Đã kiểm tra cholesterol trong 5 năm qua", value=True)
            diabetes = st.selectbox("Tiểu đường", ["Không", "Tiền tiểu đường", "Có"])
            stroke = st.checkbox("Từng bị đột quỵ")

        with col2:
            st.markdown("**Lối sống**")
            smoker = st.checkbox("Đã hút ít nhất 100 điếu thuốc trong đời")
            phys_activity = st.checkbox("Có tập thể dục/vận động trong 30 ngày qua", value=True)
            fruits = st.checkbox("Ăn trái cây hàng ngày", value=True)
            veggies = st.checkbox("Ăn rau củ hàng ngày", value=True)
            heavy_alcohol = st.checkbox("Uống rượu bia nhiều")

            st.markdown("**Sức khỏe tổng quát**")
            gen_health = st.select_slider(
                "Tự đánh giá sức khỏe", ["Rất tốt", "Tốt", "Khá", "Trung bình", "Kém"], value="Tốt"
            )
            mental_health_days = st.slider("Số ngày không ổn về tinh thần (30 ngày qua)", 0, 30, 0)
            phys_health_days = st.slider("Số ngày không khỏe về thể chất (30 ngày qua)", 0, 30, 0)
            diff_walk = st.checkbox("Khó khăn khi đi bộ/leo cầu thang")

        with st.expander("Thông tin bổ sung (không bắt buộc quan trọng)"):
            any_healthcare = st.checkbox("Có bảo hiểm y tế/tiếp cận dịch vụ y tế", value=True)
            no_doc_because_cost = st.checkbox("Từng không đi khám vì lý do chi phí")
            education_level = st.selectbox(
                "Trình độ học vấn",
                ["Chưa học/Tiểu học", "Trung học cơ sở", "Trung học phổ thông (chưa tốt nghiệp)",
                 "Tốt nghiệp THPT", "Cao đẳng/Đại học (chưa tốt nghiệp)", "Tốt nghiệp Đại học"],
                index=3,
            )
            income_level = st.selectbox(
                "Thu nhập hộ gia đình/năm (USD)",
                ["<10k", "10-15k", "15-20k", "20-25k", "25-35k", "35-50k", "50-75k", ">75k"],
                index=4,
            )

        submitted = st.form_submit_button("🔍 Dự đoán", use_container_width=True)

    if submitted:
        payload = {
            "age_group": age_group, "sex": sex, "height_cm": height_cm, "weight_kg": weight_kg,
            "high_bp": high_bp, "high_chol": high_chol, "chol_check": chol_check,
            "diabetes": diabetes, "stroke": stroke,
            "smoker": smoker, "phys_activity": phys_activity, "fruits": fruits, "veggies": veggies,
            "heavy_alcohol": heavy_alcohol, "gen_health": gen_health,
            "mental_health_days": mental_health_days, "phys_health_days": phys_health_days,
            "diff_walk": diff_walk, "any_healthcare": any_healthcare,
            "no_doc_because_cost": no_doc_because_cost,
            "education_level": education_level, "income_level": income_level,
        }
        endpoint = f"{api_url}/predict/basic"
        bmi_display = weight_kg / ((height_cm / 100) ** 2)
        extra_note = f"BMI tính được: **{bmi_display:.1f}**"
    else:
        endpoint = None

# ---------------------------------------------------------------------------
# FORM: ADVANCED MODE
# ---------------------------------------------------------------------------
else:
    with st.form("advanced_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Tuổi", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Giới tính", ["Male", "Female"], format_func=lambda x: "Nam" if x == "Male" else "Nữ")
            cp = st.selectbox(
                "Loại đau ngực",
                ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
            )
            chol = st.number_input("Cholesterol huyết thanh (mg/dl)", min_value=0, value=200)
            thalch = st.number_input("Nhịp tim tối đa đạt được", min_value=0, value=150)

        with col2:
            fbs = st.checkbox("Đường huyết lúc đói > 120 mg/dl")
            exang = st.checkbox("Đau thắt ngực khi vận động gắng sức")
            oldpeak = st.number_input("ST depression (oldpeak)", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)
            ca_option = st.selectbox("Số mạch máu chính bị hẹp (ca)", ["Không có kết quả", "0", "1", "2", "3"])
            thal_option = st.selectbox(
                "Kết quả xạ hình thallium (thal)",
                ["Không có kết quả", "normal", "fixed defect", "reversable defect"],
            )

        submitted = st.form_submit_button("🔍 Dự đoán", use_container_width=True)

    if submitted:
        payload = {
            "age": age, "sex": sex, "cp": cp, "chol": chol, "fbs": fbs,
            "thalch": thalch, "exang": exang, "oldpeak": oldpeak,
            "ca": None if ca_option == "Không có kết quả" else int(ca_option),
            "thal": None if thal_option == "Không có kết quả" else thal_option,
        }
        endpoint = f"{api_url}/predict/advanced"
        extra_note = None
    else:
        endpoint = None

# ---------------------------------------------------------------------------
# Gọi API và hiển thị kết quả
# ---------------------------------------------------------------------------
if submitted and endpoint:
    try:
        with st.spinner("Đang dự đoán..."):
            response = requests.post(endpoint, json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            probability = result["probability"]
            risk_level = result["risk_level"]

            st.markdown("---")
            st.subheader("Kết quả dự đoán")
            if extra_note:
                st.caption(extra_note)

            risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            risk_labels_vi = {"Low": "Thấp", "Medium": "Trung bình", "High": "Cao"}

            col_a, col_b = st.columns(2)
            with col_a:
                if prediction == 1:
                    st.error("**Kết quả: CÓ nguy cơ mắc bệnh tim**")
                else:
                    st.success("**Kết quả: KHÔNG có nguy cơ mắc bệnh tim**")
            with col_b:
                st.metric(
                    "Mức độ nguy cơ",
                    f"{risk_colors.get(risk_level, '')} {risk_labels_vi.get(risk_level, risk_level)}",
                )

            st.write(f"**Xác suất mắc bệnh:** {probability * 100:.1f}%")
            st.progress(probability)

            render_contribution_chart(result.get("feature_contributions", []))

            if is_basic:
                st.caption(
                    "⚠️ Đây là kết quả **sàng lọc sơ bộ** dựa trên yếu tố lối sống, "
                    "KHÔNG thay thế chẩn đoán y khoa. Nếu kết quả cho thấy nguy cơ, "
                    "hãy đến cơ sở y tế để được xét nghiệm và tư vấn chính xác hơn "
                    "(có thể dùng chế độ 'Nâng cao' nếu đã có kết quả xét nghiệm)."
                )
            else:
                st.caption(
                    "⚠️ Đây là công cụ demo cho mục đích học tập/tham khảo, "
                    "KHÔNG thay thế chẩn đoán y khoa. Vui lòng tham khảo ý kiến bác sĩ."
                )

            # Lưu lại để dùng cho chatbot, và reset hội thoại cũ (nếu có) vì đây là kết quả mới
            st.session_state["last_mode"] = "basic" if is_basic else "advanced"
            st.session_state["last_payload"] = payload
            st.session_state["last_result"] = result
            st.session_state["chat_history"] = []
        else:
            st.error(f"API trả về lỗi ({response.status_code}): {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(
            f"Không gọi được API tại `{api_url}`.\n\n"
            f"Kiểm tra API đã chạy chưa: `uvicorn api.main:app --reload`\n\n"
            f"Chi tiết lỗi: {e}"
        )

# ---------------------------------------------------------------------------
# Chatbot tư vấn — chỉ hiện nếu API báo chatbot khả dụng, và đã có kết quả dự đoán
# ---------------------------------------------------------------------------
if "last_result" in st.session_state and not chatbot_available:
    st.markdown("---")
    st.info(
        "💬 Chatbot tư vấn hiện không khả dụng trên bản demo này "
        "(cần cấu hình Ollama hoặc Groq API cho backend chatbot)."
    )
elif "last_result" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Hỏi thêm chatbot tư vấn")
    st.caption(
        "Chatbot trả lời dựa trên chính kết quả và thông tin bạn vừa nhập ở trên. "
        "Ví dụ: \"Tôi nên thay đổi gì trong chế độ ăn?\", \"Kết quả này có đáng lo không?\""
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_message = st.chat_input("Nhập câu hỏi của bạn...")
    if user_message:
        st.session_state["chat_history"].append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            chat_payload = {
                "mode": st.session_state["last_mode"],
                "patient": st.session_state["last_payload"],
                "prediction_result": st.session_state["last_result"],
                "message": user_message,
                # gửi lịch sử TRƯỚC message vừa hỏi (đã append ở trên) -> bỏ phần tử cuối
                "history": st.session_state["chat_history"][:-1],
            }
            placeholder = st.empty()
            reply = ""
            try:
                with requests.post(
                    f"{api_url}/chat/stream", json=chat_payload, timeout=120, stream=True
                ) as chat_response:
                    if chat_response.status_code == 200:
                        for chunk in chat_response.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                reply += chunk
                                placeholder.markdown(reply + "▌")  # con trỏ nhấp nháy kiểu đang gõ
                        placeholder.markdown(reply)
                    else:
                        reply = f"⚠️ Lỗi từ API ({chat_response.status_code}): {chat_response.text}"
                        placeholder.markdown(reply)
            except requests.exceptions.RequestException as e:
                reply = f"⚠️ Không gọi được chatbot: {e}"
                placeholder.markdown(reply)

            st.session_state["chat_history"].append({"role": "assistant", "content": reply})

