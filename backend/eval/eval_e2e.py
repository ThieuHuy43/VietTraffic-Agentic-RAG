import json
import os
import sys
import time
import urllib.request

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
BASE_URL = os.environ.get("EVAL_BASE_URL", "http://localhost:8000")


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def call_chat(question: str, timeout: int = 150) -> str:
    body = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def extract_events(raw: str):
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                events.append(json.loads(line[len("data:"):].strip()))
            except json.JSONDecodeError:
                pass
    return events


def check_answer(answer: str, item: dict) -> bool:
    if item["expected_doc_id"] not in answer:
        return False
    if item.get("expected_dieu") is not None and f"Điều {item['expected_dieu']}" not in answer:
        return False
    if item.get("expected_khoan") is not None and f"Khoản {item['expected_khoan']}" not in answer:
        return False
    return True


def run():
    """Đánh giá end-to-end qua /chat thật (router -> expand_query -> retrieve -> grade ->
    generate). Chậm và tốn chi phí LLM hơn eval_retrieval.py, nhưng phản ánh đúng trải nghiệm
    người dùng thật, bao gồm cả trường hợp rơi vào HITL/web_search fallback (tính là FAIL vì
    câu hỏi trong golden dataset lẽ ra phải trả lời được từ dữ liệu đã ingest).
    Yêu cầu: server đang chạy tại EVAL_BASE_URL (mặc định http://localhost:8000)."""
    dataset = load_dataset()
    rows = []
    for item in dataset:
        t0 = time.time()
        try:
            raw = call_chat(item["question"])
        except Exception as e:
            rows.append((item["id"], item["question"], False, f"ERROR: {e}", 0.0))
            continue
        elapsed = time.time() - t0
        events = extract_events(raw)
        final = next((e for e in events if e.get("status") == "done"), None)
        pending = next((e for e in events if e.get("status") == "pending"), None)

        if final:
            ok = check_answer(final["answer"], item)
            rows.append((item["id"], item["question"], ok, final["answer"], elapsed))
        elif pending:
            rows.append((item["id"], item["question"], False, "FALLBACK: rơi vào HITL/web_search", elapsed))
        else:
            rows.append((item["id"], item["question"], False, "Không có kết quả (lỗi/timeout)", elapsed))

    passed = sum(1 for r in rows if r[2])
    total = len(rows)

    print(f"{'ID':<22}{'Kết quả':<8}{'T.gian':<8}Câu hỏi")
    for id_, q, ok, detail, elapsed in rows:
        print(f"{id_:<22}{'PASS' if ok else 'FAIL':<8}{elapsed:>5.1f}s  {q}")
        if not ok:
            print(f"    -> {detail}")

    print(f"\nTổng: {passed}/{total} pass")
    return rows


if __name__ == "__main__":
    rows = run()
    sys.exit(0 if all(r[2] for r in rows) else 1)
