import json
import os
import uuid
from typing import Any, Dict
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from parsers import HtmlParser, DocxParser, PdfParser

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils.sparse_vector import create_sparse_vector

# Load biến môi trường
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "viet_traffic_laws")

# Namespace cố định để sinh point ID xác định (deterministic) theo doc_id + vị trí chunk
# trong văn bản, giúp ingest lại 1 văn bản là idempotent (không tạo point trùng/rác).
POINT_ID_NAMESPACE = uuid.UUID("6f6e1f68-9a2b-4c1d-8e3f-2b6a7d4c9e10")

_dense_model = None

def get_dense_model():
    global _dense_model
    if _dense_model is None:
        print("Loading e5-small-v2 model...")
        _dense_model = SentenceTransformer('intfloat/e5-small-v2')
    return _dense_model

def force_reingest() -> bool:
    return os.getenv("FORCE_REINGEST", "").lower() in {"1", "true", "yes", "y"}

def get_parser(file_path: str):
    ext = file_path.lower().split('.')[-1]
    if ext == 'html':
        return HtmlParser()
    elif ext == 'docx':
        return DocxParser()
    elif ext == 'pdf':
        return PdfParser()
    else:
        raise ValueError(f"Không hỗ trợ định dạng: {ext}")

def load_metadata_map() -> Dict[str, Dict[str, Any]]:
    metadata_path = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_point_id(doc_id: str, local_index: int) -> str:
    """Sinh point ID xác định theo doc_id + vị trí chunk trong văn bản (không phụ thuộc thứ tự
    duyệt các file khác), để re-ingest 1 văn bản không làm xáo trộn ID của các văn bản khác."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"{doc_id}::{local_index}"))

def collection_exists(client: QdrantClient) -> bool:
    collections = [c.name for c in client.get_collections().collections]
    return COLLECTION_NAME in collections

def init_qdrant(client: QdrantClient, recreate: bool = False) -> QdrantClient:
    if recreate and collection_exists(client):
        print(f"Xóa collection {COLLECTION_NAME} để ingest lại từ đầu...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    # Kiểm tra và tạo collection cho Hybrid Search
    if not collection_exists(client):
        dense_model = get_dense_model()
        print(f"Tạo collection {COLLECTION_NAME}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )
    return client

def delete_doc_points(client: QdrantClient, doc_id: str):
    """Xóa các point cũ thuộc doc_id này trước khi upsert lại, tránh rác point khi parser
    đổi ranh giới chunk giữa các lần ingest."""
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        ),
    )

def ingest_file(
    client: QdrantClient,
    dense_model: SentenceTransformer,
    file_path: str,
    metadata: Dict[str, Any],
) -> int:
    """Parse + upsert 1 văn bản. Idempotent: xóa point cũ của doc_id này rồi upsert lại,
    nên chạy lại ingest cho 1 file không cần FORCE_REINGEST và không đụng tới văn bản khác."""
    parser = get_parser(file_path)
    chunks = parser.parse(file_path, metadata)
    if not chunks:
        return 0

    doc_id = metadata["doc_id"]
    points = []
    for local_index, chunk in enumerate(chunks):
        text = chunk["text"]
        # Prefix cho e5 model
        dense_vec = dense_model.encode(f"passage: {text}").tolist()
        sparse_vec = create_sparse_vector(text)

        points.append(models.PointStruct(
            id=make_point_id(doc_id, local_index),
            vector={
                "dense": dense_vec,
                "sparse": sparse_vec
            },
            payload=chunk
        ))

    delete_doc_points(client, doc_id)
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client = init_qdrant(client, recreate=force_reingest())

    metadata_map = load_metadata_map()
    dense_model = get_dense_model()

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw_laws")
    total_files = 0
    total_points = 0

    # Đọc, parse và ingest từng văn bản (hỗ trợ quét đệ quy các thư mục con).
    for root, _, files in os.walk(raw_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = filename.lower().split('.')[-1]
            if ext not in ['html', 'docx', 'pdf']:
                print(f"Bỏ qua file không hỗ trợ: {filename}")
                continue

            metadata = metadata_map.get(filename, {
                "doc_id": filename, "doc_type": "unknown", "status": "active",
                "effective_date": "unknown", "amends": None, "supersedes": None
            })

            point_count = ingest_file(client, dense_model, file_path, metadata)
            total_files += 1
            total_points += point_count
            print(f"Đã ingest {filename} (doc_id={metadata['doc_id']}): {point_count} chunks")

    if total_files == 0:
        print("Không có văn bản nào để ingest.")
        return

    print(f"Ingest hoàn tất! {total_files} văn bản, {total_points} points.")

if __name__ == "__main__":
    main()
