import re
from typing import Optional, Dict, Any, List

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
        
        self.chunks: List[Dict[str, Any]] = []
        self.current_text: List[str] = []
        
    def flush(self):
        """Gộp text hiện hành và tạo chunk, sau đó clear text."""
        if self.current_text:
            text = " ".join(self.current_text).strip()
            # Bỏ qua các chunk quá ngắn hoặc chỉ chứa whitespace
            if text and len(text) > 5:
                self.chunks.append({
                    "doc_id": self.doc_id,
                    "doc_type": self.doc_type,
                    "chuong": self.current_chuong,
                    "dieu": self.current_dieu,
                    "khoan": self.current_khoan,
                    "text": text,
                    "status": self.status,
                    "effective_date": self.effective_date,
                    "amends": self.amends,
                    "supersedes": self.supersedes,
                })
            self.current_text = []
            
    def process_line(self, line: str):
        """
        Đọc từng dòng và update trạng thái (Chương, Điều, Khoản)
        Nếu gặp mốc mới, flush chunk cũ đi.
        """
        line = line.strip()
        if not line:
            return
            
        # Detect Chương (VD: Chương I, CHƯƠNG IV)
        chuong_match = re.match(r'^Chương\s+([IVXLCDM]+)', line, re.IGNORECASE)
        if chuong_match:
            self.flush()
            chuong_num = roman_to_int(chuong_match.group(1).upper())
            self.current_chuong = chuong_num if chuong_num > 0 else None
            self.current_dieu = None
            self.current_khoan = None
            self.current_text.append(line)
            return
            
        # Detect Điều (VD: Điều 1., Điều 2)
        dieu_match = re.match(r'^Điều\s+(\d+)', line, re.IGNORECASE)
        if dieu_match:
            self.flush()
            self.current_dieu = int(dieu_match.group(1))
            self.current_khoan = None
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
