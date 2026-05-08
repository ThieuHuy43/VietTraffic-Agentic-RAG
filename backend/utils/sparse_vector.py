import hashlib
from collections import Counter
from pyvi import ViTokenizer
from qdrant_client import models

def get_deterministic_hash(term: str) -> int:
    """Tạo mã băm cố định (deterministic) cho một chuỗi."""
    # Sử dụng hashlib.md5 thay cho hàm hash() built-in vì hash() thay đổi theo process trong Python 3 (bảo mật PYTHONHASHSEED)
    m = hashlib.md5(term.encode('utf-8'))
    # Lấy 8 byte đầu và ép kiểu int, chia dư để lấy chỉ số vector
    return int(m.hexdigest()[:16], 16) % (10**8)

def create_sparse_vector(text: str) -> models.SparseVector:
    """Tạo sparse vector sử dụng pyvi tách từ và đếm tần suất (TF)."""
    tokens = ViTokenizer.tokenize(text).split()
    term_frequencies = Counter(tokens)
    
    indices = []
    values = []
    for term, freq in term_frequencies.items():
        idx = get_deterministic_hash(term)
        indices.append(idx)
        values.append(float(freq))
        
    return models.SparseVector(indices=indices, values=values)
