"""
Lưu và truy vấn thống kê sử dụng (số lượt dự đoán, phân bố risk level, xu hướng theo ngày...).
Dùng SQLite — nhẹ, không cần cài server database riêng.

Lưu ý quan trọng: trên môi trường deploy free tier (Render...), ổ đĩa KHÔNG persistent —
dữ liệu có thể bị reset mỗi khi container build lại/deploy lại. Phù hợp cho mục đích demo/
thống kê tương đối, không phù hợp nếu cần lưu trữ lâu dài đáng tin cậy.

CHỦ Ý VỀ QUYỀN RIÊNG TƯ: chỉ lưu số liệu tổng hợp (mode, prediction, probability, risk_level,
thời gian) — KHÔNG lưu bất kỳ thông tin cá nhân/lâm sàng nào của bệnh nhân (tuổi cụ thể, giới
tính, các chỉ số...).
"""

import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.environ.get("STATS_DB_PATH", "data/stats/usage.db")


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            mode TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL NOT NULL,
            risk_level TEXT NOT NULL
        )
        """)
    conn.commit()
    conn.close()


def log_prediction(mode: str, prediction: int, probability: float, risk_level: str):
    """Ghi lại 1 lượt dự đoán. Không raise lỗi ra ngoài — nếu ghi log thất bại,
    không được làm hỏng luồng dự đoán chính của người dùng."""
    try:
        conn = _get_connection()
        conn.execute(
            "INSERT INTO predictions (timestamp, mode, prediction, probability, risk_level) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                mode,
                prediction,
                probability,
                risk_level,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # Thống kê là tính năng phụ — lỗi ghi log không được ảnh hưởng tới /predict


def get_summary() -> dict:
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM predictions")
    total = cur.fetchone()[0]

    cur.execute("SELECT mode, COUNT(*) FROM predictions GROUP BY mode")
    by_mode = {row[0]: row[1] for row in cur.fetchall()}

    cur.execute("SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level")
    by_risk_level = {row[0]: row[1] for row in cur.fetchall()}

    cur.execute("SELECT AVG(probability) FROM predictions")
    avg_row = cur.fetchone()[0]
    avg_probability = round(avg_row, 4) if avg_row is not None else 0.0

    cur.execute("""
        SELECT DATE(timestamp) as day, COUNT(*)
        FROM predictions
        GROUP BY day
        ORDER BY day
        """)
    by_day = [{"date": row[0], "count": row[1]} for row in cur.fetchall()]

    conn.close()

    return {
        "total_predictions": total,
        "by_mode": by_mode,
        "by_risk_level": by_risk_level,
        "avg_probability": avg_probability,
        "by_day": by_day,
    }
