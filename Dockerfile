FROM python:3.10-slim

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY config/ config/
COPY src/ src/
COPY entrypoint/ entrypoint/
COPY models/ models/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "entrypoint/inference.py"]
CMD ["--mode", "basic", "--config", "config/basic/prod.yaml"]
