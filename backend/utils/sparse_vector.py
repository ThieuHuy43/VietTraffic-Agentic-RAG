from typing import Optional
from pyvi import ViTokenizer
from qdrant_client import models
from fastembed import SparseTextEmbedding

_bm25_model: Optional[SparseTextEmbedding] = None

def _get_bm25_model() -> SparseTextEmbedding:
    """Singleton cho model BM25 built-in của fastembed/Qdrant, thay cho sparse vector tự chế
    bằng MD5 hash trước đây (không có term-saturation/length-norm chuẩn, dễ đụng hash collision).
    Tiếng Việt không nằm trong danh sách ngôn ngữ hỗ trợ stemmer/stopwords sẵn có của thư viện
    -> tắt stemmer (disable_stemmer=True, ngôn ngữ "english" chỉ là placeholder bắt buộc, không
    được dùng khi đã tắt stemmer) và tự tách từ tiếng Việt trước bằng pyvi (xem _segment)."""
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(
            model_name="Qdrant/bm25", language="english", disable_stemmer=True
        )
    return _bm25_model

def _segment(text: str) -> str:
    # pyvi ghép từ ghép tiếng Việt bằng "_" (VD: "giao thông" -> "giao_thông"); tokenizer nội bộ
    # của BM25 chỉ tách theo ký tự không phải \w nên giữ nguyên các cụm đã nối "_" thành 1 token.
    return ViTokenizer.tokenize(text)

def _to_sparse_vector(embedding) -> models.SparseVector:
    return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())

def create_sparse_vector(text: str) -> models.SparseVector:
    """Sparse vector cho DOCUMENT (lúc ingest): dùng embed() để lấy tần suất đã qua BM25
    term-saturation (k1/b). Qdrant áp IDF (Modifier.IDF, cấu hình ở tầng collection) lúc query."""
    embedding = next(_get_bm25_model().embed([_segment(text)]))
    return _to_sparse_vector(embedding)

def create_sparse_query_vector(text: str) -> models.SparseVector:
    """Sparse vector cho QUERY: dùng query_embed() (giá trị nhị phân đánh dấu token có mặt),
    đúng theo pattern mã hoá bất đối xứng doc/query chuẩn của BM25 trong Qdrant."""
    embedding = next(_get_bm25_model().query_embed([_segment(text)]))
    return _to_sparse_vector(embedding)
