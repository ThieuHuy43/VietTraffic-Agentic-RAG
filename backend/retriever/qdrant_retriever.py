import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import sys

# Thêm đường dẫn để import custom sparse vector utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.sparse_vector import create_sparse_vector

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "viet_traffic_laws")

class QdrantHybridRetriever:
    """
    Module tìm kiếm theo chuẩn Hybrid Search kết hợp Reciprocal Rank Fusion (RRF)
    và Payload Filtering.
    """
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        print("Loading retriever dense model e5-small-v2...")
        # Load mô hình dense e5-small-v2
        self.dense_model = SentenceTransformer('intfloat/e5-small-v2')
        
    def _build_filter(self, preferred_doc_types: List[str] | None = None) -> models.Filter:
        must = []
        if preferred_doc_types:
            must.append(
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchAny(any=preferred_doc_types)
                )
            )

        return models.Filter(
            must=must,
            must_not=[
                models.FieldCondition(
                    key="status",
                    match=models.MatchAny(any=["repealed", "superseded"])
                )
            ]
        )

    def _query(self, query: str, query_filter: models.Filter, limit: int):
        dense_vec = self.dense_model.encode(f"query: {query}").tolist()
        sparse_vec = create_sparse_vector(query)

        return self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=limit * 2
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=limit * 2
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit
        )

    def retrieve(self, query: str, preferred_doc_types: List[str] | None = None) -> List[Dict[str, Any]]:
        """
        Thực hiện truy vấn lấy các chunks liên quan.
        - Tìm kiếm Dense (Cosine Similarity) bằng e5-small-v2.
        - Tìm kiếm Sparse (BM25 custom) bằng pyvi.
        - Kết hợp bằng thuật toán RRF.
        - Lọc bỏ các chunk từ các văn bản đã bị bãi bỏ (status="repealed").
        """
        try:
            query_filter = self._build_filter(preferred_doc_types)
            response = self._query(query, query_filter, self.top_k)

            if preferred_doc_types and len(response.points) < max(2, self.top_k // 2):
                response = self._query(query, self._build_filter(), self.top_k)
            
            # Trích xuất payload từ points trả về
            results = []
            for point in response.points:
                if point.payload:
                    results.append(point.payload)
                    
            return results
        except Exception as e:
            print(f"Lỗi truy vấn Qdrant: {e}")
            return []
