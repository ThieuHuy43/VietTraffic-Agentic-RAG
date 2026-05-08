import pytest
import os
import sys

# Thêm path để import từ backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parsers.utils import LegalStateTracker, roman_to_int

def test_roman_to_int():
    assert roman_to_int("I") == 1
    assert roman_to_int("IV") == 4
    assert roman_to_int("IX") == 9
    assert roman_to_int("X") == 10
    assert roman_to_int("XIII") == 13

def test_legal_state_tracker():
    metadata = {
        "doc_id": "test_doc",
        "doc_type": "luat",
        "status": "active",
        "effective_date": "2024-01-01"
    }
    tracker = LegalStateTracker(metadata)
    
    lines = [
        "Chương I QUY ĐỊNH CHUNG",
        "Dòng text rác không nằm trong cấu trúc",
        "Điều 1. Phạm vi điều chỉnh",
        "Luật này quy định về ABC.",
        "Điều 2. Đối tượng áp dụng",
        "1. Tổ chức cá nhân.",
        "2. Người nước ngoài."
    ]
    
    for line in lines:
        tracker.process_line(line)
    tracker.flush()
    
    chunks = tracker.chunks
    assert len(chunks) == 4 # Chương I intro, Điều 1, Điều 2 khoản 1, Điều 2 khoản 2
    
    # Kiểm tra chunk "Điều 1"
    chunk_d1 = next(c for c in chunks if c["dieu"] == 1)
    assert chunk_d1["chuong"] == 1
    assert "Phạm vi điều chỉnh Luật này quy định" in chunk_d1["text"]
    assert chunk_d1["khoan"] is None
    
    # Kiểm tra chunk "Điều 2 Khoản 1"
    chunk_d2_k1 = next(c for c in chunks if c["dieu"] == 2 and c["khoan"] == 1)
    assert chunk_d2_k1["chuong"] == 1
    assert chunk_d2_k1["text"] == "1. Tổ chức cá nhân."
    
    # Metadata kiểm tra
    assert chunk_d2_k1["doc_id"] == "test_doc"
    assert chunk_d2_k1["status"] == "active"
