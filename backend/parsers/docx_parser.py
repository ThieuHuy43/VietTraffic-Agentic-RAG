import docx
from typing import List, Dict, Any
from .base import BaseParser
from .utils import LegalStateTracker

class DocxParser(BaseParser):
    """Parser dành cho các file DOCX (giữ nguyên cấu trúc)."""
    def parse(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = docx.Document(file_path)
        tracker = LegalStateTracker(metadata)
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                tracker.process_line(text)
                
        tracker.flush()
        return tracker.chunks
