#!/bin/bash
# Khởi động cả FastAPI (nội bộ, port 8000) và Streamlit (public, port 7860)
# trong CÙNG 1 container — Streamlit gọi API qua localhost, không cần cấu hình
# URL/secrets phức tạp giữa 2 dịch vụ riêng biệt.

set -e

echo "=== Khởi động FastAPI (nội bộ, port 8000) ==="
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Đợi API sẵn sàng trước khi Streamlit khởi động (tối đa 30s)
echo "=== Đợi API sẵn sàng... ==="
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "API đã sẵn sàng sau ${i}s"
    break
  fi
  sleep 1
done

echo "=== Khởi động Streamlit (public, port 7860) ==="
streamlit run app/streamlit_app.py \
  --server.port=7860 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
