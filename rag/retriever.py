"""
Retriever cho RAG — load index đã build sẵn (rag/index.npy + rag/index_meta.json),
tìm các chunk liên quan nhất tới câu hỏi bằng cosine similarity.

Nếu chưa build index (chưa chạy rag/build_index.py), retriever tự vô hiệu hóa
(is_ready() = False) — chatbot sẽ hoạt động bình thường như khi chưa có RAG,
không lỗi, không bắt buộc.
"""

import json
import os

import numpy as np
import requests

INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.npy")
META_PATH = os.path.join(os.path.dirname(__file__), "index_meta.json")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Ngưỡng cosine similarity để quyết định có "đủ liên quan" để dùng RAG không.
# Câu hỏi thường/chào hỏi/hỏi về chính kết quả dự đoán sẽ có similarity thấp -> bỏ qua RAG,
# trả lời nhanh như bình thường. Có thể chỉnh qua biến môi trường nếu thấy chưa hợp lý.
SIMILARITY_THRESHOLD = float(os.environ.get("RAG_SIMILARITY_THRESHOLD", "0.5"))
TOP_K = int(os.environ.get("RAG_TOP_K", "3"))


class KnowledgeRetriever:
    def __init__(self):
        self._embeddings = None
        self._meta = None
        self._load()

    def _load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self._embeddings = np.load(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                self._meta = json.load(f)

    def is_ready(self) -> bool:
        return self._embeddings is not None and len(self._meta) > 0

    def _embed_query(self, query: str) -> np.ndarray:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": query},
            timeout=15,
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)

    def retrieve(self, query: str, top_k: int = None) -> tuple[list[dict], float]:
        """
        Trả về (danh sách chunk liên quan nhất, similarity score cao nhất).
        Nếu retriever chưa sẵn sàng hoặc gọi embedding lỗi -> trả về ([], 0.0)
        thay vì raise lỗi, để không làm gián đoạn luồng chat chính.
        """
        if not self.is_ready():
            return [], 0.0

        top_k = top_k or TOP_K
        try:
            query_emb = self._embed_query(query)
        except requests.exceptions.RequestException:
            return [], 0.0

        # Cosine similarity giữa query và toàn bộ chunk
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb)
        norms[norms == 0] = 1e-10
        scores = (self._embeddings @ query_emb) / norms

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [self._meta[i] | {"score": float(scores[i])} for i in top_indices]
        best_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 0.0

        return results, best_score

    def retrieve_if_relevant(self, query: str, top_k: int = None) -> list[dict]:
        """Chỉ trả về chunk nếu similarity cao nhất vượt ngưỡng, ngược lại trả về []."""
        results, best_score = self.retrieve(query, top_k)
        if best_score >= SIMILARITY_THRESHOLD:
            return results
        return []
