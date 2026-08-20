"""
Xây dựng index vector cho RAG từ các file .md trong rag/knowledge_base/.
Chạy 1 lần (hoặc mỗi khi sửa/thêm tài liệu):

    python rag/build_index.py

Yêu cầu: đã `ollama pull nomic-embed-text` (model embedding, nhẹ, ~274MB).

Output:
    rag/index.npy        — ma trận embedding (n_chunks x embedding_dim)
    rag/index_meta.json  — text + nguồn của từng chunk (cùng thứ tự với index.npy)
"""

import json
import os
import re
import sys

import numpy as np
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.npy")
META_PATH = os.path.join(os.path.dirname(__file__), "index_meta.json")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def split_into_chunks(filepath: str) -> list[dict]:
    """Tách file markdown thành các chunk theo từng heading '## '."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(filepath)
    # Bỏ heading cấp 1 (# Tiêu đề file), tách theo heading cấp 2 (## ...)
    sections = re.split(r"\n(?=## )", content)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            # phần đầu file (chỉ có tiêu đề chính, chưa có heading con) -> bỏ qua
            if not section.startswith("## "):
                continue
        heading_match = re.match(r"## (.+)", section)
        heading = heading_match.group(1).strip() if heading_match else filename
        chunks.append({"source": filename, "heading": heading, "text": section})
    return chunks


def embed_text(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def build_index():
    md_files = sorted(f for f in os.listdir(KB_DIR) if f.endswith(".md"))
    if not md_files:
        print(f"Không tìm thấy file .md nào trong {KB_DIR}")
        return

    all_chunks = []
    for filename in md_files:
        chunks = split_into_chunks(os.path.join(KB_DIR, filename))
        all_chunks.extend(chunks)
        print(f"  {filename}: {len(chunks)} chunks")

    print(
        f"\nTổng {len(all_chunks)} chunks. Đang tính embedding qua Ollama ({EMBED_MODEL})..."
    )

    embeddings = []
    for i, chunk in enumerate(all_chunks):
        emb = embed_text(chunk["text"])
        embeddings.append(emb)
        print(f"  [{i + 1}/{len(all_chunks)}] {chunk['source']} - {chunk['heading']}")

    embeddings_matrix = np.array(embeddings, dtype=np.float32)
    np.save(INDEX_PATH, embeddings_matrix)

    meta = [
        {"source": c["source"], "heading": c["heading"], "text": c["text"]}
        for c in all_chunks
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu index: {INDEX_PATH} (shape={embeddings_matrix.shape})")
    print(f"Đã lưu metadata: {META_PATH}")


if __name__ == "__main__":
    try:
        build_index()
    except requests.exceptions.ConnectionError:
        print(
            f"Không kết nối được Ollama tại {OLLAMA_BASE_URL}. "
            f"Đảm bảo Ollama đang chạy và đã `ollama pull {EMBED_MODEL}`."
        )
        sys.exit(1)
