import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
import sys

# Thêm đường dẫn để import custom sparse vector utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.sparse_vector import create_sparse_query_vector

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "viet_traffic_laws")

class QdrantHybridRetriever:
    """
    Module tìm kiếm theo chuẩn Hybrid Search kết hợp Reciprocal Rank Fusion (RRF)
    và Payload Filtering, sau đó rerank bằng cross-encoder để đưa đúng chunk liên quan
    lên đầu trước khi cắt xuống top_k cuối cùng (cho phép dùng top_k/context budget nhỏ
    mà vẫn đủ recall, do RRF một mình không đủ chính xác phân biệt các đoạn luật gần giống nhau).
    """
    def __init__(self, top_k: int = 8, candidate_k: int = 25):
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        print("Loading retriever dense model e5-small-v2...")
        # Load mô hình dense e5-small-v2
        self.dense_model = SentenceTransformer('intfloat/e5-small-v2')

        print("Loading reranker cross-encoder mmarco-mMiniLMv2-L12-H384-v1...")
        self.reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

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
        sparse_vec = create_sparse_query_vector(query)

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

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank candidate chunks bằng cross-encoder (query, chunk_text) rồi cắt xuống top_k.
        RRF một mình dễ xếp nhầm các đoạn "sửa đổi, bổ sung" na ná nhau lên trên đoạn thực sự
        liên quan; cross-encoder chấm điểm trực tiếp mức độ khớp ngữ nghĩa giữa query và từng
        đoạn nên đáng tin cậy hơn nhiều cho việc xếp hạng cuối cùng."""
        if not candidates:
            return candidates
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[: self.top_k]]

    def retrieve(self, query: str, preferred_doc_types: List[str] | None = None) -> List[Dict[str, Any]]:
        """
        Thực hiện truy vấn lấy các chunks liên quan.
        - Tìm kiếm Dense (Cosine Similarity) bằng e5-small-v2.
        - Tìm kiếm Sparse (BM25 custom) bằng pyvi.
        - Kết hợp bằng thuật toán RRF để lấy một pool ứng viên rộng (candidate_k).
        - Rerank bằng cross-encoder rồi cắt xuống top_k cuối cùng.
        - Lọc bỏ các chunk từ các văn bản đã bị bãi bỏ (status="repealed").
        """
        return self.retrieve_multi([query], preferred_doc_types)

    def retrieve_multi(
        self, queries: List[str], preferred_doc_types: List[str] | None = None
    ) -> List[Dict[str, Any]]:
        """Truy vấn với NHIỀU biến thể câu hỏi (VD: câu hỏi gốc + bản diễn giải thuật ngữ pháp
        lý) rồi GỘP candidate pool, thay vì nối chuỗi các câu hỏi thành 1 query dài duy nhất.

        Đã xác minh thực tế: nối chuỗi câu hỏi gốc (ngắn, khớp tốt) với bản diễn giải dài hơn
        (do expand_query sinh ra) có thể pha loãng dense embedding, khiến chunk đúng bị rớt khỏi
        cả candidate pool (candidate_k) dù truy riêng câu hỏi gốc vẫn tìm thấy đúng ở hạng #1.
        Truy riêng từng biến thể rồi gộp giữ được tín hiệu mạnh của từng câu, tránh bị pha loãng."""
        try:
            merged: Dict[Any, Dict[str, Any]] = {}
            for query in queries:
                query_filter = self._build_filter(preferred_doc_types)
                response = self._query(query, query_filter, self.candidate_k)

                if preferred_doc_types and len(response.points) < max(2, self.candidate_k // 2):
                    response = self._query(query, self._build_filter(), self.candidate_k)

                for point in response.points:
                    if point.payload:
                        merged.setdefault(point.id, point.payload)

            # Rerank bằng biến thể câu hỏi cuối cùng (thường là bản đã diễn giải/chuẩn hoá
            # follow-up, đầy đủ ngữ cảnh nhất) để chấm điểm liên quan nhất quán.
            rerank_query = queries[-1]
            return self._rerank(rerank_query, list(merged.values()))
        except Exception as e:
            print(f"Lỗi truy vấn Qdrant: {e}")
            return []
