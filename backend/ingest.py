import os
from typing import List, Dict, Any
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

def collection_exists(client: QdrantClient) -> bool:
    collections = [c.name for c in client.get_collections().collections]
    return COLLECTION_NAME in collections

def should_skip_ingest(client: QdrantClient) -> bool:
    if force_reingest() or not collection_exists(client):
        return False

    point_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    if point_count > 0:
        print(
            f"Collection {COLLECTION_NAME} đã có {point_count} points. "
            "Bỏ qua ingest. Đặt FORCE_REINGEST=1 nếu muốn nạp lại."
        )
        return True

    return False

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

def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if should_skip_ingest(client):
        return

    # Metadata map giả định để thỏa mãn yêu cầu DOD
    # Cập nhật Metadata Map thực tế theo các file bạn vừa tải lên
    METADATA_MAP = {
        "Luat_23_2008_GiaoThongDuongBo.html": {
            "doc_id": "Luat_23_2008", "doc_type": "luat", "status": "superseded", 
            "effective_date": "2009-07-01", "amends": None, "supersedes": None
        },
        "Luat_36_2024_TratTuAnToanGiaoThongDuongBo.docx": {
            "doc_id": "Luat_36_2024", "doc_type": "luat", "status": "active", 
            "effective_date": "2025-01-01", "amends": None, "supersedes": "Luat_23_2008"
        },
        "Luat_35_2024_DuongBo.html": {
            "doc_id": "Luat_35_2024", "doc_type": "luat", "status": "active", 
            "effective_date": "2025-01-01", "amends": None, "supersedes": "Luat_23_2008"
        },
        "NGHỊ ĐỊNH 100_2019_NĐ-CP về QUY ĐỊNH XỬ PHẠT VI PHẠM HÀNH CHÍNH TRONG LĨNH VỰC GIAO THÔNG ĐƯỜNG BỘ VÀ ĐƯỜNG SẮT.html": {
            "doc_id": "NghiDinh_100_2019", "doc_type": "nghi_dinh", "status": "active", 
            "effective_date": "2020-01-01", "amends": None, "supersedes": None
        },
        "NGHỊ ĐỊNH 123_2021_NĐ-CP SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA CÁC NGHỊ ĐỊNH QUY ĐỊNH XỬ PHẠT VI PHẠM HÀNH CHÍNH TRONG HÀNG HẢI; GIAO THÔNG Đường bộ, đường sắt mới nhất.html": {
            "doc_id": "NghiDinh_123_2021", "doc_type": "nghi_dinh", "status": "active", 
            "effective_date": "2022-01-01", "amends": "NghiDinh_100_2019", "supersedes": None
        },
        "Thông tư 12_2017_TT-BGTVT đào tạo sát hạch cấp giấy phép lái xe cơ giới đường bộ.html": {
            "doc_id": "ThongTu_12_2017", "doc_type": "thong_tu", "status": "active", 
            "effective_date": "2017-06-01", "amends": None, "supersedes": None
        },
        "Thông tư 31_2019_TT-BGTVT quy định về tốc độ và khoảng cách an toàn của xe cơ giới.html": {
            "doc_id": "ThongTu_31_2019", "doc_type": "thong_tu", "status": "active", 
            "effective_date": "2019-10-15", "amends": None, "supersedes": None
        },
        "quy-chuan-ky-thuat-qcvn-41-2019-bgtvt-bao-hieu-duong-bo.pdf": {
            "doc_id": "QCVN_41_2019", "doc_type": "quy_chuan", "status": "active", 
            "effective_date": "2020-07-01", "amends": None, "supersedes": None
        }
    }

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw_laws")
    all_chunks = []
    
    # Đọc và Parse văn bản (hỗ trợ quét đệ quy các thư mục con)
    for root, _, files in os.walk(raw_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = filename.lower().split('.')[-1]
            if ext not in ['html', 'docx', 'pdf']:
                print(f"Bỏ qua file không hỗ trợ: {filename}")
                continue

            metadata = METADATA_MAP.get(filename, {
                "doc_id": filename, "doc_type": "unknown", "status": "active",
                "effective_date": "unknown", "amends": None, "supersedes": None
            })
            
            parser = get_parser(file_path)
            chunks = parser.parse(file_path, metadata)
            all_chunks.extend(chunks)
            print(f"Đã parse {filename}: {len(chunks)} chunks")

    if not all_chunks:
        print("Không có chunks nào để ingest.")
        return

    # Ingest vào Qdrant
    client = init_qdrant(client, recreate=force_reingest())
    dense_model = get_dense_model()
    points = []
    
    for i, chunk in enumerate(all_chunks):
        text = chunk["text"]
        # Prefix cho e5 model
        dense_vec = dense_model.encode(f"passage: {text}").tolist()
        sparse_vec = create_sparse_vector(text)
        
        points.append(models.PointStruct(
            id=i,
            vector={
                "dense": dense_vec,
                "sparse": sparse_vec
            },
            payload=chunk
        ))
        
    print(f"Bắt đầu upsert {len(points)} points vào Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("Ingest hoàn tất!")

if __name__ == "__main__":
    main()
