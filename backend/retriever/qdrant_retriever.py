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
        
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Thực hiện truy vấn lấy các chunks liên quan.
        - Tìm kiếm Dense (Cosine Similarity) bằng e5-small-v2.
        - Tìm kiếm Sparse (BM25 custom) bằng pyvi.
        - Kết hợp bằng thuật toán RRF.
        - Lọc bỏ các chunk từ các văn bản đã bị bãi bỏ (status="repealed").
        """
        # Prefix "query:" bắt buộc đối với các mô hình e5
        dense_vec = self.dense_model.encode(f"query: {query}").tolist()
        sparse_vec = create_sparse_vector(query)
        
        # Payload Filter: Loại bỏ chunk có status="repealed"
        query_filter = models.Filter(
            must_not=[
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value="repealed")
                )
            ]
        )
        
        try:
            # Dùng Qdrant Query API (hỗ trợ Fusion RRF từ bản 1.10+)
            response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=self.top_k * 2
                    ),
                    models.Prefetch(
                        query=sparse_vec,
                        using="sparse",
                        limit=self.top_k * 2
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=self.top_k
            )
            
            # Trích xuất payload từ points trả về
            results = []
            for point in response.points:
                if point.payload:
                    results.append(point.payload)
                    
            return results
        except Exception as e:
            print(f"Lỗi truy vấn Qdrant: {e}")
            return []
