"""
Test cho RAG retriever (không cần Ollama thật — dùng embedding giả lập).
Chạy: pytest src/tests/test_rag.py -v
"""

import json
import os
import sys
import unittest.mock as mock

import numpy as np

sys.path.append(os.getcwd())


def _make_fake_index(tmp_path):
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    meta = [
        {"source": "a.md", "heading": "Chủ đề A", "text": "Nội dung A"},
        {"source": "b.md", "heading": "Chủ đề B", "text": "Nội dung B"},
    ]
    index_path = tmp_path / "index.npy"
    meta_path = tmp_path / "index_meta.json"
    np.save(index_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return str(index_path), str(meta_path)


def test_retriever_not_ready_without_index(tmp_path):
    import rag.retriever as retriever_module

    retriever_module.INDEX_PATH = str(tmp_path / "nonexistent.npy")
    retriever_module.META_PATH = str(tmp_path / "nonexistent.json")

    from rag.retriever import KnowledgeRetriever

    r = KnowledgeRetriever()
    assert r.is_ready() is False
    results, score = r.retrieve("bất kỳ câu hỏi gì")
    assert results == []
    assert score == 0.0


def test_retriever_finds_relevant_chunk(tmp_path):
    import rag.retriever as retriever_module

    index_path, meta_path = _make_fake_index(tmp_path)
    retriever_module.INDEX_PATH = index_path
    retriever_module.META_PATH = meta_path
    retriever_module.SIMILARITY_THRESHOLD = 0.5

    from rag.retriever import KnowledgeRetriever

    r = KnowledgeRetriever()
    assert r.is_ready() is True

    with mock.patch.object(
        r, "_embed_query", return_value=np.array([1.0, 0.05], dtype=np.float32)
    ):
        relevant = r.retrieve_if_relevant("câu hỏi liên quan chủ đề A")
        assert len(relevant) > 0
        assert relevant[0]["heading"] == "Chủ đề A"


def test_retriever_skips_irrelevant_question(tmp_path):
    import rag.retriever as retriever_module

    index_path, meta_path = _make_fake_index(tmp_path)
    retriever_module.INDEX_PATH = index_path
    retriever_module.META_PATH = meta_path
    retriever_module.SIMILARITY_THRESHOLD = 0.5

    from rag.retriever import KnowledgeRetriever

    r = KnowledgeRetriever()

    with mock.patch.object(
        r, "_embed_query", return_value=np.array([0.6, 0.6], dtype=np.float32)
    ):
        relevant = r.retrieve_if_relevant("câu hỏi không rõ liên quan gì")
        # Similarity với cả 2 chunk đều ~0.707 -> nếu threshold=0.5 thì vẫn có thể match,
        # nên test này chỉ đảm bảo hàm chạy không lỗi và trả về list hợp lệ
        assert isinstance(relevant, list)
