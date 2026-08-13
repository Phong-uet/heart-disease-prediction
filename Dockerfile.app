FROM python:3.10-slim

WORKDIR /app

COPY app/requirements.txt ./app-requirements.txt
RUN pip install --no-cache-dir -r app-requirements.txt

COPY app/ app/

EXPOSE 8501

# Trong docker-compose, API chạy ở service riêng "heart-disease-api"
# nên dùng tên service làm hostname thay vì 127.0.0.1
ENV API_URL=http://heart-disease-api:8000

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
