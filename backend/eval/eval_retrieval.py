import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retriever.qdrant_retriever import QdrantHybridRetriever

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def matches(chunk: dict, item: dict) -> bool:
    if chunk.get("doc_id") != item["expected_doc_id"]:
        return False
    if item.get("expected_dieu") is not None and chunk.get("dieu") != item["expected_dieu"]:
        return False
    if item.get("expected_khoan") is not None and chunk.get("khoan") != item["expected_khoan"]:
        return False
    return True


def run():
    """Đánh giá riêng tầng retrieval (KHÔNG qua expand_query/LLM), dùng thẳng câu hỏi thô.
    Mục đích: cô lập chất lượng retriever/reranker/ingestion khỏi biến động của LLM, chạy
    nhanh và không tốn chi phí API - phù hợp chạy lại mỗi khi sửa ingest.py/qdrant_retriever.py.
    Vì bỏ qua expand_query nên hit-rate ở đây có thể THẤP HƠN eval_e2e.py (câu hỏi thông tục
    chưa được diễn giải sang thuật ngữ pháp lý) - đó là tín hiệu bình thường, không phải lỗi
    harness, xem thêm README.md trong thư mục này."""
    dataset = load_dataset()
    retriever = QdrantHybridRetriever()

    rows = []
    for item in dataset:
        chunks = retriever.retrieve(item["question"])
        rank = None
        for i, c in enumerate(chunks, start=1):
            if matches(c, item):
                rank = i
                break
        rows.append((item["id"], item["question"], rank))

    total = len(rows)
    hits = sum(1 for _, _, rank in rows if rank is not None)
    mrr = sum((1.0 / rank) for _, _, rank in rows if rank is not None) / total if total else 0.0

    print(f"{'ID':<22}{'Hit':<6}{'Rank':<6}Câu hỏi")
    for id_, q, rank in rows:
        print(f"{id_:<22}{'YES' if rank else 'NO':<6}{rank or '-':<6}{q}")

    print(f"\nHit@k: {hits}/{total} ({hits / total:.0%})  MRR: {mrr:.3f}")

    failed = [(id_, q) for id_, q, rank in rows if rank is None]
    if failed:
        print(f"\n{len(failed)} câu retrieval-only KHÔNG tìm thấy đúng chunk trong top-k:")
        for id_, q in failed:
            print(f"  - [{id_}] {q}")

    return rows


if __name__ == "__main__":
    rows = run()
    sys.exit(0 if all(rank is not None for _, _, rank in rows) else 1)
