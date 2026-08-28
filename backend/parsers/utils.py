import re
from typing import Optional, Dict, Any, List

MAX_CHUNK_CHARS = 3500
CHUNK_OVERLAP_CHARS = 300

def roman_to_int(s: str) -> int:
    """Chuyển đổi số La Mã sang số nguyên."""
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    try:
        for i in range(len(s)):
            if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
                int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
            else:
                int_val += rom_val[s[i]]
        return int_val
    except KeyError:
        return 0

class LegalStateTracker:
    """
    Quản lý trạng thái phân mảnh (chunking) văn bản theo cấu trúc:
    Chương -> Điều -> Khoản -> Điểm.
    Bảo đảm schema chứa đủ 10 trường theo yêu cầu.
    """
    def __init__(self, metadata: Dict[str, Any]):
        self.doc_id = metadata.get("doc_id", "unknown")
        self.doc_type = metadata.get("doc_type", "unknown")
        self.status = metadata.get("status", "active")
        self.effective_date = metadata.get("effective_date", "")
        self.amends = metadata.get("amends", None)
        self.supersedes = metadata.get("supersedes", None)
        
        self.current_chuong: Optional[int] = None
        self.current_dieu: Optional[int] = None
        self.current_khoan: Optional[int] = None
        self.current_chuong_title: Optional[str] = None
        self.current_dieu_title: Optional[str] = None
        
        self.chunks: List[Dict[str, Any]] = []
        self.current_text: List[str] = []

    def _split_long_text(self, text: str) -> List[str]:
        if len(text) <= MAX_CHUNK_CHARS:
            return [text]

        parts = []
        start = 0
        while start < len(text):
            end = min(start + MAX_CHUNK_CHARS, len(text))
            if end < len(text):
                split_at = max(
                    text.rfind(". ", start, end),
                    text.rfind("; ", start, end),
                    text.rfind("\n", start, end),
                    text.rfind(" ", start, end),
                )
                if split_at > start + int(MAX_CHUNK_CHARS * 0.6):
                    end = split_at + 1

            part = text[start:end].strip()
            if part:
                parts.append(part)

            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP_CHARS, 0)

        return parts

    def _build_citation(self) -> str:
        parts = [self.doc_id]
        if self.current_dieu is not None:
            parts.append(f"Điều {self.current_dieu}")
        if self.current_khoan is not None:
            parts.append(f"Khoản {self.current_khoan}")
        return ", ".join(parts)

    def _add_parent_context(self, text: str) -> str:
        context_parts = []
        if self.current_dieu_title and not text.startswith("Điều "):
            context_parts.append(self.current_dieu_title)
        elif self.current_chuong_title and not text.startswith("Chương "):
            context_parts.append(self.current_chuong_title)

        if not context_parts:
            return text
        return " ".join(context_parts + [text])

    def _append_chunk(self, text: str, part_index: int, total_parts: int, is_table: bool = False):
        text_with_context = self._add_parent_context(text)
        citation = self._build_citation()
        if is_table:
            citation += " (Bảng)"
        self.chunks.append({
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "chuong": self.current_chuong,
            "dieu": self.current_dieu,
            "khoan": self.current_khoan,
            "chuong_title": self.current_chuong_title,
            "dieu_title": self.current_dieu_title,
            "text": text_with_context,
            "status": self.status,
            "effective_date": self.effective_date,
            "amends": self.amends,
            "supersedes": self.supersedes,
            "chunk_part": part_index,
            "chunk_total_parts": total_parts,
            "citation": citation,
            "chunk_type": "table" if is_table else "text",
        })

    def add_table_chunk(self, table_text: str):
        """Thêm 1 bảng (đã được chuyển sang text có cấu trúc "Header: value") thành chunk riêng,
        kế thừa ngữ cảnh Chương/Điều/Khoản hiện tại của tracker. Dùng cho các văn bản có bảng
        dữ liệu kỹ thuật (VD: QCVN 41 - bảng kích thước, màu sắc biển báo) mà nếu để lẫn vào
        luồng text thường sẽ bị extract_text() làm rối thứ tự cột/hàng."""
        if not table_text or not table_text.strip():
            return
        # Đóng đoạn text thường đang dở trước khi chèn chunk bảng, để không lẫn 2 loại nội dung.
        self.flush()
        text = f"Bảng dữ liệu: {table_text}"
        parts = self._split_long_text(text)
        total_parts = len(parts)
        for index, part in enumerate(parts, start=1):
            self._append_chunk(part, index, total_parts, is_table=True)

    def flush(self):
        """Gộp text hiện hành và tạo chunk, sau đó clear text."""
        if self.current_text:
            text = " ".join(self.current_text).strip()
            # Bỏ qua các chunk quá ngắn hoặc chỉ chứa whitespace
            if text and len(text) > 5:
                parts = self._split_long_text(text)
                total_parts = len(parts)
                for index, part in enumerate(parts, start=1):
                    self._append_chunk(part, index, total_parts)
            self.current_text = []
            
    def process_line(self, line: str):
        """
        Đọc từng dòng và update trạng thái (Chương, Điều, Khoản)
        Nếu gặp mốc mới, flush chunk cũ đi.
        """
        line = line.strip()
        if not line:
            return
        normalized_line = line.lower()
            
        # Detect Chương (VD: Chương I, CHƯƠNG IV)
        chuong_match = re.match(r'^chương\s+([ivxlcdm]+)', normalized_line)
        if chuong_match:
            self.flush()
            chuong_num = roman_to_int(chuong_match.group(1).upper())
            self.current_chuong = chuong_num if chuong_num > 0 else None
            self.current_dieu = None
            self.current_khoan = None
            self.current_chuong_title = line
            self.current_dieu_title = None
            self.current_text.append(line)
            return
            
        # Detect Điều (VD: Điều 1., Điều 2)
        dieu_match = re.match(r'^điều\s+(\d+)', normalized_line)
        if dieu_match:
            self.flush()
            self.current_dieu = int(dieu_match.group(1))
            self.current_khoan = None
            self.current_dieu_title = line
            self.current_text.append(line)
            return
            
        # Detect Khoản (VD: 1. , 2. )
        khoan_match = re.match(r'^(\d+)\.\s', line)
        if khoan_match and self.current_dieu is not None:
            self.flush()
            self.current_khoan = int(khoan_match.group(1))
            self.current_text.append(line)
            return
            
        # Dòng bình thường hoặc Điểm (vd: a), b)) được gộp chung vào Khoản/Điều hiện tại
        self.current_text.append(line)
